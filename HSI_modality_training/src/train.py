from __future__ import annotations

import copy, math, os, warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from config import CONFIG
from model import ModelEMA, SpectralQuadNet, tta_predict
from utils import (
    FocalLoss, SupConLoss, ProtoNCELoss, SAM,
    _load_data_into_ram, _DATA_ON_GPU, _USING_MMAP, _GPU_PATCHES,
    build_splits, build_loaders,
    build_optimizer_s1, build_optimizer_s2,
    sgdr_scheduler, arcface_margin,
    _wd_groups,
    train_one_epoch, train_one_epoch_sam,
    evaluate, compute_class_difficulty,
    update_bn_stats,
    stage_ckpt_path, latest_completed_stage,
    save_ckpt, load_ckpt, load_stage_meta,
    RiceSeedDataset, BlockSortedSampler,
)

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — 3-PHASE PROGRESSIVE AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, ema, loaders_by_phase, val_ldr, device, best_ckpt: str) -> float:
    """
    Phase 1 (0–40%):  heavy aug + mixup + high label-smooth → explore
    Phase 2 (40–70%): medium aug + mixup + decaying label-smooth → consolidate
    Phase 3 (70–100%): light aug + NO mixup + Focal+LS → discriminate hard classes

    Saves and does early stopping on macro-F1 (not accuracy).
    """
    model.use_arcface(False)
    model.unfreeze_head("linear"); model.freeze_head("arcface")

    ep_total = CONFIG["s1_epochs"]
    p1_end   = int(ep_total * CONFIG["s1_phase1_frac"])
    p2_end   = int(ep_total * (CONFIG["s1_phase1_frac"] + CONFIG["s1_phase2_frac"]))

    optimizer = build_optimizer_s1(model, CONFIG["s1_max_lr"] / 25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG["s1_max_lr"], epochs=ep_total,
            steps_per_epoch=math.ceil(len(loaders_by_phase[1]) / CONFIG["s1_accum"]),
            pct_start=0.25, div_factor=25, final_div_factor=1e4, anneal_strategy="cos")

    scaler       = GradScaler()
    ls_hi        = CONFIG["s1_label_smooth_hi"]
    ls_lo        = CONFIG["s1_label_smooth_lo"]
    best_f1      = 0.0
    no_improve   = 0
    ema_reinited = [False, False]

    w = 66
    print(f"\n{'═'*w}\n  Stage 1 — 3-Phase Progressive Augmentation  [{ep_total} epochs max]\n{'═'*w}")
    print(f"  Phase 1: ep 1–{p1_end}    heavy aug + mixup")
    print(f"  Phase 2: ep {p1_end+1}–{p2_end}  medium aug + mixup")
    print(f"  Phase 3: ep {p2_end+1}–{ep_total}  light aug, NO mixup, Focal+LS")
    print(f"  Label smooth: {ls_hi} → {ls_lo}  |  Primary metric: macro-F1")

    for ep in range(1, ep_total + 1):
        if   ep <= p1_end: phase = 1; cur_ldr = loaders_by_phase[1]; use_mx = True
        elif ep <= p2_end: phase = 2; cur_ldr = loaders_by_phase[2]; use_mx = True
        else:              phase = 3; cur_ldr = loaders_by_phase[3]; use_mx = False

        if phase == 2 and not ema_reinited[0] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 2 (ep {ep})")
            ema_reinited[0] = True
        if phase == 3 and not ema_reinited[1] and CONFIG["s1_ema_reinit_phases"]:
            ema.reinit_from(model)
            print(f"[INFO] EMA re-init at Phase 3 (ep {ep})")
            ema_reinited[1] = True

        t      = (ep - 1) / max(ep_total - 1, 1)
        ls_now = ls_hi * (1 - t) + ls_lo * t

        # Phase 3: Focal loss with label smoothing — keeps regularisation while
        # sharpening focus on hard examples. Phases 1-2 use standard CE+LS.
        if phase == 3:
            crit = FocalLoss(gamma=CONFIG["s1_focal_gamma"], label_smoothing=ls_now)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=ls_now)

        tl, ta = train_one_epoch(
            model, cur_ldr, optimizer, crit, scaler, ema, device,
            scheduler=scheduler, use_mixup=use_mx,
            mixup_alpha=CONFIG["s1_mixup"], accum_steps=CONFIG["s1_accum"])

        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1  = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        lr_now      = optimizer.param_groups[0]["lr"]
        saved       = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1, _cdws = compute_class_difficulty(ema.shadow, val_ldr, device, "S1")
            save_ckpt(best_ckpt, ep, "Stage 1", model, ema,
                      val_f1=best_ep_f1, val_acc=best_ep_acc,
                      class_f1=_cf1, cdws_weights=_cdws, arcface_init_done=False)
            saved = "  ✓"
        else:
            no_improve += 1

        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
              f"LR {lr_now:.2e}  LS {ls_now:.3f} [P{phase}]{saved}")

        if no_improve >= CONFIG["s1_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("arcface")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR
# ══════════════════════════════════════════════════════════════════════

def run_stage2(model, ema, train_ldr, val_ldr, device, best_ckpt: str,
               class_f1=None) -> float:
    model.set_dropout(CONFIG["s2_dropout"])
    model.use_arcface(True)
    model.freeze_head("linear"); model.unfreeze_head("arcface")

    ema.reinit_from(model)
    ema.set_dropout(CONFIG["s2_dropout"]); ema.shadow.use_arcface(True)

    if class_f1 is not None:
        model.arcface_head.update_margins_from_f1(class_f1)
        ema.shadow.arcface_head.update_margins_from_f1(class_f1)

    focal  = FocalLoss(gamma=CONFIG["s2_focal_gamma"])
    supcon = SupConLoss(temperature=CONFIG["supcon_temp"])
    proto  = ProtoNCELoss(temperature=CONFIG["proto_temp"])

    optimizer = build_optimizer_s2(model, CONFIG["s2_head_lr"], CONFIG["s2_back_lr"])
    scheduler = sgdr_scheduler(
        optimizer,
        warmup_ep=CONFIG["s2_warmup_ep"],
        T_0=CONFIG["s2_sgdr_T0"],
        T_mult=CONFIG["s2_sgdr_Tmult"],
        eta_min_frac=CONFIG["s2_min_lr"] / CONFIG["s2_head_lr"])

    sc_w     = CONFIG["supcon_weight"];  pt_w = CONFIG["proto_weight"]
    ep_total = CONFIG["s2_epochs"]
    best_f1  = 0.0; no_improve = 0

    r1 = CONFIG["s2_warmup_ep"] + CONFIG["s2_sgdr_T0"]
    r2 = r1 + CONFIG["s2_sgdr_T0"] * CONFIG["s2_sgdr_Tmult"]

    w = 66
    print(f"\n{'═'*w}\n  Stage 2 — Sub-ctr ArcFace + SupCon + ProtoNCE + CDWS + SGDR  [{ep_total} ep]\n{'═'*w}")
    print(f"  hLR={CONFIG['s2_head_lr']:.1e}  bLR={CONFIG['s2_back_lr']:.1e}  "
          f"SGDR T0={CONFIG['s2_sgdr_T0']} Tmult={CONFIG['s2_sgdr_Tmult']} "
          f"→ restarts ep {r1} & {r2}")
    print(f"  ArcFace K={CONFIG['subcenter_K']}  "
          f"m={CONFIG['s2_arcface_m0']}→{CONFIG['s2_arcface_m']}+Δ{CONFIG['s2_arcface_m_delta']}")
    print(f"  Losses: Focal(γ={CONFIG['s2_focal_gamma']}) + SupCon(w={sc_w}) + ProtoNCE(w={pt_w})")
    print(f"  Batch: {CONFIG['bal_n_cls']} cls × {CONFIG['bal_n_spc']} spc = "
          f"{CONFIG['bal_n_cls']*CONFIG['bal_n_spc']} | Primary metric: macro-F1")

    for ep in range(1, ep_total + 1):
        warmup_done = (ep - 1) >= CONFIG["s2_margin_warmup_ep"]
        m_now       = (CONFIG["s2_arcface_m"] if warmup_done
                       else arcface_margin(ep - 1,
                                           CONFIG["s2_arcface_m0"],
                                           CONFIG["s2_arcface_m"],
                                           CONFIG["s2_margin_warmup_ep"]))
        arc_m  = None if warmup_done else m_now
        ramp   = min(1.0, ep / 10.0)
        sc_now = sc_w * ramp; pt_now = pt_w * ramp

        tl, ta = train_one_epoch(
            model, train_ldr, optimizer, focal, scaler=None, ema=ema,
            device=device, scheduler=None,
            use_mixup=False, supcon=supcon, supcon_weight=sc_now,
            proto=proto, proto_weight=pt_now, arc_m=arc_m)
        scheduler.step()

        f1_live, acc_live = evaluate(model,      val_ldr, device)
        f1_ema,  acc_ema  = evaluate(ema.shadow, val_ldr, device)
        best_ep_f1  = max(f1_live, f1_ema)
        best_ep_acc = max(acc_live, acc_ema)
        head_lr     = optimizer.param_groups[0]["lr"]
        back_lr     = optimizer.param_groups[2]["lr"]
        saved       = ""

        if best_ep_f1 > best_f1:
            best_f1, no_improve = best_ep_f1, 0
            _cf1_s2, _cdws_s2   = compute_class_difficulty(ema.shadow, val_ldr, device, "S2")
            save_ckpt(best_ckpt, ep, "Stage 2", model, ema,
                      val_f1=best_ep_f1, val_acc=best_ep_acc,
                      class_f1=_cf1_s2, cdws_weights=_cdws_s2, s2_val_f1=best_ep_f1)
            saved = "  ✓"
        else:
            no_improve += 1

        rf = " ↻R1" if ep == r1 else (" ↻R2" if ep == r2 else "")
        print(f"Ep {ep:03d}/{ep_total} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}/{f1_ema:.3f}  Acc {acc_live:.1%}/{acc_ema:.1%} │ "
              f"hLR {head_lr:.1e} bLR {back_lr:.1e}  m={m_now:.3f}{saved}{rf}")

        if no_improve >= CONFIG["s2_patience"]:
            print(f"\nEarly stopping at epoch {ep}."); break

    model.unfreeze_head("linear")
    return best_f1


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — SAM + GREEDY SWA
# ══════════════════════════════════════════════════════════════════════

def run_stage3_swa(model, ema, train_ldr, val_ldr, device,
                   best_ckpt: str, prev_best_f1: float) -> float:
    if hasattr(torch, "_dynamo"): torch._dynamo.disable()

    model.set_dropout(CONFIG["s2_dropout"])
    model.branch_drop_prob = 0.0
    ema.shadow.branch_drop_prob = 0.0
    model.use_arcface(True); ema.shadow.use_arcface(True)

    params = list(_wd_groups(model.named_parameters(), CONFIG["s3_swa_lr"]))
    sam    = SAM(params, optim.AdamW,
                 rho=CONFIG["s3_sam_rho"],
                 lr=CONFIG["s3_swa_lr"],
                 weight_decay=CONFIG["weight_decay"])

    focal_s3  = FocalLoss(gamma=CONFIG["s3_focal_gamma"])
    supcon_s3 = SupConLoss(temperature=CONFIG["s3_supcon_temp"])
    proto_s3  = ProtoNCELoss(temperature=CONFIG["s3_proto_temp"])

    swa_state     = None
    n_snap        = 0; n_rejected = 0; best_live_f1 = 0.0

    w = 66
    print(f"\n{'═'*w}\n  Stage 3 — SAM + Greedy SWA  [{CONFIG['s3_epochs']} epochs]\n{'═'*w}")
    print(f"  SAM ρ={CONFIG['s3_sam_rho']}  Cycle={CONFIG['s3_cycle_len']} ep  "
          f"Peak LR={CONFIG['s3_swa_lr']:.0e}")

    def _s3_margin(ep):
        return 0.25 + 0.05 * math.cos(math.pi * ep / CONFIG["s3_epochs"])

    for ep in range(1, CONFIG["s3_epochs"] + 1):
        cycle_ep = (ep - 1) % CONFIG["s3_cycle_len"]
        lr_now   = CONFIG["s3_swa_lr"] * (0.3 + 0.7 * 0.5 * (
            1 + math.cos(math.pi * cycle_ep / CONFIG["s3_cycle_len"])))
        for pg in sam.param_groups: pg["lr"] = lr_now

        tl, ta = train_one_epoch_sam(
            model, train_ldr, sam, focal_s3, device,
            supcon=supcon_s3, supcon_weight=CONFIG["s3_supcon_weight"],
            proto=proto_s3,   proto_weight=CONFIG["s3_proto_weight"],
            arc_m=_s3_margin(ep))

        f1_live, acc_live = evaluate(model, val_ldr, device)
        best_live_f1      = max(best_live_f1, f1_live)

        snap_info = ""
        if ep % CONFIG["s3_cycle_len"] == 0:
            if not CONFIG["s3_greedy"] or f1_live >= best_live_f1 * 0.98:
                n_snap += 1
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = copy.deepcopy(sd)
                else:
                    beta = 1.0 / float(n_snap)
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k].mul_(1.0 - beta).add_(sd[k], alpha=beta)
                        else:
                            swa_state[k].copy_(sd[k])
                snap_info = f"  ★ snap {n_snap}"
            else:
                n_rejected += 1
                snap_info   = f"  ✗ rejected (F1 {f1_live:.3f} < {best_live_f1*0.98:.3f})"

        print(f"Ep {ep:03d}/{CONFIG['s3_epochs']} │ Loss {tl:.4f}  Tr {ta:.1%} │ "
              f"F1 {f1_live:.3f}  Acc {acc_live:.1%} │ LR {lr_now:.2e}{snap_info}")

    print(f"\nUpdating BN stats ({n_snap} accepted, {n_rejected} rejected) ...")
    if swa_state is None:
        print("[WARN] No snapshots accepted — using final live model.")
        swa_state = copy.deepcopy(model.state_dict())

    swa_model = copy.deepcopy(model)
    swa_model.load_state_dict(swa_state); swa_model.use_arcface(True)
    update_bn_stats(train_ldr, swa_model, device)
    f1_swa, acc_swa = evaluate(swa_model, val_ldr, device)
    print(f"SWA val: F1={f1_swa:.3f}  Acc={acc_swa:.1%}")

    ema.shadow.load_state_dict(swa_model.state_dict())
    ema.shadow.use_arcface(True)

    note = ""
    if f1_swa <= prev_best_f1:
        note = "val_f1 did not beat Stage 2; Stage 2 ckpt preferred for eval"
        print(f"Stage 3 F1 {f1_swa:.3f} ≤ Stage 2 best {prev_best_f1:.3f} — Stage 2 preferred.")
    else:
        print(f"Stage 3 F1 {f1_swa:.3f} > Stage 2 best {prev_best_f1:.3f} → saving.")

    save_ckpt(best_ckpt, CONFIG["s3_epochs"], "Stage 3",
              swa_model, ema, val_f1=f1_swa, val_acc=acc_swa,
              swa_n_snapshots=n_snap, swa_n_rejected=n_rejected,
              **({"note": note} if note else {}))
    return f1_swa


# ══════════════════════════════════════════════════════════════════════
#  FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════

def final_evaluation(model, ema, test_ldr, device, best_ckpt: str) -> None:
    w = 66
    print(f"\n{'═'*w}\n  FINAL TEST EVALUATION\n{'═'*w}")
    ckpt       = load_ckpt(best_ckpt, model, ema, device)
    eval_model = ema.shadow; eval_model.eval()

    print(f"  ArcFace: {eval_model._use_arcface}  |  "
          f"Checkpoint: ep {ckpt['epoch']} | {ckpt['stage']} | "
          f"F1={ckpt.get('val_f1',0):.3f}  Acc={ckpt.get('val_acc',0):.1%}")
    print(f"  TTA: {CONFIG['tta_spatial']} spatial + {CONFIG['tta_spectral']} spectral "
          f"= {CONFIG['tta_spatial']+CONFIG['tta_spectral']} total views")

    results = {}
    for tag, use_tta in [("No TTA", False), ("TTA   ", True)]:
        preds, targets = [], []
        for x, y in test_ldr:
            x      = x.to(device, non_blocking=True)
            logits = (tta_predict(eval_model, x,
                                  CONFIG["tta_spatial"], CONFIG["tta_spectral"])
                      if use_tta else eval_model(x))
            preds.append(logits.argmax(1).cpu()); targets.append(y)
        p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        results[tag] = (p, t)
        print(f"\n  [{tag}]  F1(macro)={f1_score(t,p,average='macro',zero_division=0):.4f}  "
              f"F1(wt)={f1_score(t,p,average='weighted',zero_division=0):.4f}  "
              f"Acc={accuracy_score(t,p):.1%}")

    p_tta, t_tta = results["TTA   "]
    print(f"\nClassification Report (TTA):\n")
    print(classification_report(t_tta, p_tta, zero_division=0))

    out = CONFIG["output_dir"]
    np.save(f"{out}/test_preds_noTTA.npy", results["No TTA"][0])
    np.save(f"{out}/test_preds_TTA.npy",   p_tta)
    np.save(f"{out}/test_targets.npy",     t_tta)
    print(f"\nOutputs saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  BEST CHECKPOINT SELECTION
# ══════════════════════════════════════════════════════════════════════

def _pick_best_checkpoint(*ckpt_paths: str) -> str:
    """Select checkpoint with highest val_f1 (primary) across all stages."""
    best_val, best_path = -1.0, ckpt_paths[-1]
    for p in ckpt_paths:
        if not os.path.isfile(p): continue
        try:
            sn   = int(os.path.basename(p).replace("best_stage", "").replace(".pth", ""))
            meta = load_stage_meta(sn)
            v    = meta.get("val_f1", meta.get("val_acc", None))
        except (ValueError, KeyError):
            v = None
        if v is None:
            try:
                v = torch.load(p, map_location="cpu",
                               weights_only=False).get("val_f1", 0.0)
            except Exception:
                v = 0.0
        if v > best_val:
            best_val, best_path = v, p
    return best_path


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # Import module-level globals after possible updates from _load_data_into_ram
    from utils import _DATA_ON_GPU, _USING_MMAP, _GPU_PATCHES

    device     = CONFIG["device"]
    ckpt_s1    = stage_ckpt_path(1)
    ckpt_s2    = stage_ckpt_path(2)
    ckpt_s3    = stage_ckpt_path(3)
    done_stage = latest_completed_stage()

    labels_map = {0: "starting fresh", 1: "Stage 1 done",
                  2: "Stages 1–2 done", 3: "all done"}
    print(f"\n{'─'*66}")
    print(f"  Auto-resume: {labels_map.get(done_stage, f'stage {done_stage} done')}")
    print(f"  Output dir : {CONFIG['output_dir']}")
    print(f"{'─'*66}")
    print(f"[INFO] Latest completed stage: {done_stage}")

    _load_data_into_ram(CONFIG["patches_data"], CONFIG["labels_path"])

    # Re-import after load so the module globals are refreshed
    import utils as _utils
    if _utils._DATA_ON_GPU:
        free = torch.cuda.mem_get_info(device)[0] / 1e9
        print(f"[DATA] ✓ GPU mode: {_utils._GPU_PATCHES.nelement()*4/1e9:.1f} GB in VRAM  "
              f"| {free:.1f} GB free | num_workers=0")
    elif _utils._USING_MMAP:
        f32_need = os.path.getsize(CONFIG["patches_data"]) / 1e9
        print(f"\n{'═'*66}\n  ⚠  MMAP MODE — BlockSortedSampler active")
        print(f"  Need {f32_need*1.1:.0f} GB GPU VRAM or {f32_need*1.2:.0f} GB CPU RAM for full speed")
        print(f"{'═'*66}\n")
    else:
        print("[DATA] ✓ CPU RAM mode.")

    all_labels, train_idx, val_idx, test_idx = build_splits()
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    print(f"Samples/class (train): ~{len(train_idx)//CONFIG['num_classes']}")

    model = SpectralQuadNet(
        num_classes=CONFIG["num_classes"],
        num_bands=CONFIG["num_bands"],
        dropout=CONFIG["s1_dropout"],
        wl_embed_dim=CONFIG["wl_embed_dim"],
        cfg=CONFIG).to(device)

    ema = ModelEMA(model, decay=CONFIG["ema_decay"])

    print(f"Model  : SpectralQuadNet v10")
    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Device : {device}")

    if hasattr(torch, "compile"):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.recompile_limit        = 64
        warnings.filterwarnings("ignore", message=".*networkx backend.*")
        print("[INFO] Applying torch.compile(mode='default') ...")
        model      = torch.compile(model,      mode="default", fullgraph=False)
        ema.shadow = torch.compile(ema.shadow, mode="default", fullgraph=False)
    else:
        print("[WARN] torch.compile unavailable (PyTorch < 2.0)")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU]  {props.name}  |  VRAM {props.total_memory//1024**3} GB  |  "
              f"TF32={torch.backends.cuda.matmul.allow_tf32}")

    def _s1_ldr(aug_str):
        ds = RiceSeedDataset(CONFIG["patches_data"], CONFIG["labels_path"],
                             train_idx, aug_strength=aug_str)
        bs = CONFIG["s1_batch"]
        if _utils._DATA_ON_GPU:
            return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=False)
        nw = CONFIG["num_workers"]; pf = CONFIG["prefetch_factor"]
        if _utils._USING_MMAP:
            bss = BlockSortedSampler(train_idx, CONFIG["mmap_block_size"])
            return DataLoader(ds, batch_size=bs, sampler=bss, drop_last=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=True, prefetch_factor=pf)
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                          num_workers=nw, pin_memory=True,
                          persistent_workers=True, prefetch_factor=pf)

    if done_stage < 1:
        print("\n[RUN] Stage 1")
        phase_loaders = {1: _s1_ldr("heavy"), 2: _s1_ldr("medium"), 3: _s1_ldr("light")}
        _, val_ldr1, _ = build_loaders(train_idx, val_idx, test_idx,
                                       CONFIG["s1_batch"], train_aug="none")
        run_stage1(model, ema, phase_loaders, val_ldr1, device, ckpt_s1)
        print("[INFO] Reloading best Stage 1 checkpoint ...")
        load_ckpt(ckpt_s1, model, ema, device)
    else:
        print("\n[SKIP] Stage 1 → loading checkpoint")
        load_ckpt(ckpt_s1, model, ema, device)

    meta_s1      = load_stage_meta(1)
    class_f1_s1  = meta_s1.get("class_f1",    {})
    cdws_wts_s1  = meta_s1.get("cdws_weights", {})
    arcface_done = meta_s1.get("arcface_init_done", False)
    s1_best_f1   = meta_s1.get("val_f1", meta_s1.get("val_acc", 0.0))
    print(f"[INFO] Stage 1 → F1={s1_best_f1:.3f}  "
          f"hard classes={sum(1 for f in class_f1_s1.values() if f<0.5)}")

    if done_stage < 2:
        if not arcface_done:
            print("\n[INFO] Bootstrapping ArcFace from linear head")
            lw = model.linear_head[-1].weight.data.clone()
            model.arcface_head.init_from_linear(lw)
            ema.shadow.arcface_head.init_from_linear(lw)

        if not class_f1_s1:
            print("[WARN] No class_f1 in Stage 1 meta — recomputing")
            _, val_cd, _ = build_loaders(train_idx, val_idx, test_idx,
                                         CONFIG["eval_batch_size"])
            class_f1_s1, cdws_wts_s1 = compute_class_difficulty(
                ema.shadow, val_cd, device, "Stage 1 (recomputed)")

        print("\n[RUN] Stage 2")
        tr2, va2, _ = build_loaders(train_idx, val_idx, test_idx,
                                    CONFIG["s2_batch"],
                                    balanced=True, all_labels=all_labels,
                                    train_aug="light", class_weights=cdws_wts_s1)
        run_stage2(model, ema, tr2, va2, device, ckpt_s2, class_f1_s1)
        print("[INFO] Reloading best Stage 2 checkpoint ...")
        load_ckpt(ckpt_s2, model, ema, device)
    else:
        print("\n[SKIP] Stage 2 → loading checkpoint")
        load_ckpt(ckpt_s2, model, ema, device)

    meta_s2     = load_stage_meta(2)
    class_f1_s2 = meta_s2.get("class_f1",    {})
    cdws_wts_s2 = meta_s2.get("cdws_weights", {})
    s2_best_f1  = meta_s2.get("val_f1", meta_s2.get("s2_val_f1", meta_s2.get("val_acc", 0.0)))
    print(f"[INFO] Stage 2 → F1={s2_best_f1:.3f}")

    if hasattr(torch, "_dynamo"):
        print("[INFO] Disabling torch.compile for Stage 3 stability")
        torch._dynamo.reset()

    if done_stage < 3:
        if not cdws_wts_s2:
            print("[WARN] No cdws_weights in Stage 2 meta — falling back to Stage 1")
            cdws_wts_s2 = cdws_wts_s1

        print("\n[RUN] Stage 3 (SAM + Greedy SWA)")
        tr3, va3, _ = build_loaders(train_idx, val_idx, test_idx,
                                    CONFIG["s2_batch"],
                                    balanced=True, all_labels=all_labels,
                                    train_aug="light", class_weights=cdws_wts_s2)
        run_stage3_swa(model, ema, tr3, va3, device, ckpt_s3, prev_best_f1=s2_best_f1)
    else:
        print("\n[SKIP] Stage 3 → loading checkpoint")
        load_ckpt(ckpt_s3, model, ema, device)
        meta_s3 = load_stage_meta(3)
        print(f"[INFO] Stage 3 → snaps={meta_s3.get('swa_n_snapshots','?')}  "
              f"rejected={meta_s3.get('swa_n_rejected','?')}  "
              f"F1={meta_s3.get('val_f1', meta_s3.get('val_acc',0)):.3f}")

    best_final_ckpt = _pick_best_checkpoint(ckpt_s1, ckpt_s2, ckpt_s3)
    print(f"\n[INFO] Best checkpoint (by val_f1): {best_final_ckpt}")

    _, _, test_ldr = build_loaders(train_idx, val_idx, test_idx,
                                   CONFIG["eval_batch_size"])
    final_evaluation(model, ema, test_ldr, device, best_final_ckpt)


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback, sys, logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(CONFIG["output_dir"], "training.log")),
            logging.StreamHandler(sys.stdout),
        ])
    try:
        main()
    except Exception:
        logging.critical("FATAL:\n" + traceback.format_exc())
        sys.exit(1)