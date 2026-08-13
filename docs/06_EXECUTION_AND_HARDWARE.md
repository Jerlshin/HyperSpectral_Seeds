# 6 · Execution Engine and Hardware Performance

> **CHANGES.md IC-8 restored this document's own invariant.** The rule stated below is that
> *every* field under `cfg.runtime` is a throughput knob and changing one must never change a
> reported metric — which is why `allow_tf32` defaults to `false`, since it cuts matmul
> mantissas from 24 bits to 11.
>
> The audited run set `allow_tf32=True`, `num_workers=0` (auto would give 8) and `compile=off`
> (auto would give on). The two knobs that only change speed were disabled and the one that
> changes numerics was enabled, so that run is **not comparable to a run at the shipped
> defaults**. `configs/experiment/seednet_grouped.yaml` sets all three back;
> `configs/experiment/quadnet_audited.yaml` deliberately keeps the run's actual values, because
> its job is to reproduce what was audited.
>
> Two related fixes documented in CHANGES §9.2 land here:
>
> * **IC-7** — passing a SupCon module no longer disables autocast for the whole epoch. Phase 3
>   cost 190–405 s/epoch against Phases 1–2's 39 s on identical data, because fp32 roughly
>   doubles activation memory on a working set already at 91% of an 11 GB card and the allocator
>   starts paging against a host page cache holding the 5.65 GB mmapped cube. The fp32 region is
>   now confined to the similarity matrix.
> * **IC-6** — `grad_clip` moves 1.0 → 5.0, and `grad_norm/clip_fraction` is logged so "does the
>   clip bind?" is a measurement rather than an inference. At the old threshold the backbone's
>   pre-clip norm was 25–50 for the entire run.

Everything in this document is governed by one invariant, stated in `RuntimeConfig`'s own
docstring: **every field under `cfg.runtime` is a throughput knob, and changing one must never
change a reported metric.** That is why `runtime` carries real defaults instead of
`omegaconf.MISSING` like every other config group — its values are not part of an experiment's
identity, so a config that never mentions them is not under-specified. The two fields that would
*materially* change a number if flipped (`allow_tf32`, dropping matmul mantissas 24 → 11 bits;
`channels_last`, re-selecting convolution kernels with a different reduction order) default to
off for exactly that reason.

Four fields sit close enough to that line to be named rather than left implicit:

- **`amp_dtype`** is the one field here that *admits* to changing numerics rather than defaulting
  away from it. Stage 1 trains under autocast either way, and the fp16 this group used to inherit
  from torch's per-device default is what put `NaN` in the loss (§6.2) — so the knob exists to
  make the choice recorded and overridable instead of implicit.
- **`fused_optimizer`** and **`decompose_conv3d`** change a *summation order*, not an arithmetic:
  the fused AdamW step accumulates in the same precision but not the same order, and the `Conv3d`
  decomposition moves the logits by $1.9\times10^{-7}$. Both are re-associations of the same
  sum, not a precision trade.
- **`checkpoint_branch_a`** is bit-exact by construction — Branch A's towers hold no dropout, no
  BatchNorm and nothing that reads the RNG, so recomputing them in the backward reproduces the
  gradients exactly.

---

## 6.1 `train.py` — entrypoint orchestration

### RNG ordering invariant

`register_configs()` registers every dataclass schema with Hydra's `ConfigStore` before
`@hydra.main` composes anything, so a malformed YAML field fails at startup. Inside `_run(cfg,
tracker, dist)`, construction follows one required order, stated verbatim in the source:

$$
\text{config} \;\to\; \texttt{set\_seed(cfg.seed)} \;\to\; \texttt{DataStore} \;\to\; \texttt{SpectralQuadNet} \;\to\; \texttt{ModelEMA}
$$

Every `_init_weights` call draws from the same global torch RNG stream, in the order
`SpectralQuadNet.__init__` constructs its sub-modules (§3.8) — so the seed must be set before
model construction, and nothing between `set_seed` and `ModelEMA` may consume the global RNG.
`build_split_bundle` runs after `set_seed` but is RNG-safe on both its paths: the `stratified`
split uses `sklearn`'s own `random_state=42` (a private `RandomState`), and the `grouped` split
uses a private `np.random.default_rng(SPLIT_SEED)` — neither touches the global torch/NumPy
stream. Building the tracker in `main()` happens *before* `set_seed` for the same
non-interference reason.

### Process bootstrap (`main`)

1. `dist = init_distributed(cfg.runtime, fallback=resolve_device(cfg.device))` — **every rank
   joins the process group and owns its device before anything else exists** (model, data store,
   tracker), since a rank must own its CUDA device before allocating on it.
2. Logging: a `StreamHandler` always; a `FileHandler(training.log)` prepended **only on
   `dist.is_main`**. Log level `INFO` on rank 0, `WARNING` elsewhere; the format includes
   `rank{n}` when DDP is enabled.
3. `tracker = build_tracker(cfg) if dist.is_main else NullTracker()` — **only rank 0 gets a real
   tracker**, so two GPUs never interleave two copies of the same progress output.
4. `_run(...)` wrapped so any escaping exception is logged critical and exits `1`.
5. `finally`: `tracker.close()`, `empty_cache()`, `shutdown(dist)` — always runs.

### The training pipeline (`_run`)

- **Device and runtime plan**: `device = dist.device` (already resolved by `init_distributed`);
  `plan = resolve_runtime(cfg.runtime, device, dist.world_size)`;
  `backend_notes = configure_backend(device, cfg.runtime)`.
- **Auto-resume detection**: `latest_completed_stage(cfg)` probes stages **3 → 2 → 1**; a stage
  counts as complete only when **both** `best_stage{n}.pth` and `stage{n}_meta.json` exist, so a
  crash between the two writes replays the stage rather than silently skipping it.
- **DataStore**, then a data-size/VRAM log line — `torch.cuda.mem_get_info` is explicitly
  guarded by `device.type == "cuda"` (it raises off-CUDA), so the VRAM figure is simply omitted
  elsewhere.
- **Splits and morphometrics**: `build_split_bundle(cfg)`; `standardised_morphometrics` fitted
  **once** on `train_idx` and threaded to every loader that needs it.
- **Model construction → `apply_runtime_optimisations` → EMA** — the RNG-critical section
  (§6.1 above). `apply_runtime_optimisations(model, plan)` sits between the two on purpose: it
  selects the per-device execution paths *inside* the model (§6.2), and it must run **before**
  the EMA deep-copy, the DDP wrap, `torch.compile` and Stage 3's two SWA copies, so every later
  copy of the module inherits the same paths. It sets no parameter, no buffer, no state-dict key
  and draws nothing from the RNG, so it neither perturbs the initialisation stream nor changes
  what a checkpoint contains. It returns banner lines, printed as the startup block's
  `Paths :` row beside `Kernels :`.
- **Hardware dispatch, in this exact order** (out-of-order breaks the compiled graph or the
  BatchNorm invariant):

$$
\texttt{channels\_last cast} \;\to\; \texttt{SyncBatchNorm + DDP wrap (wrap\_for\_training)} \;\to\; \texttt{torch.compile (maybe\_compile)}
$$

  SyncBatchNorm conversion and the DDP wrap must precede `torch.compile`, or the compiled graph
  captures the un-converted BatchNorm. The EMA shadow is **never** wrapped — it takes no
  gradient, so it needs neither replication nor a compiled graph, which is what lets `save_ckpt`
  write one checkpoint schema regardless of how the live model was dispatched.
- **Stage 1 → 2 → 3**, each either run fresh or skip-loaded from an existing checkpoint,
  reloading the best checkpoint after Stages 1 and 2 (not after Stage 3 — that stage's own
  averaging logic is trusted to leave the best state, §4.3).
- **Stage boundaries `del` their loaders, then sweep.** `release_memory(device)` alone would not
  free them: `run_stage1` returning does not drop its loaders while `_run`'s own names still hold
  them, and a loader holds its `persistent_workers` pool for exactly as long as something holds
  the loader. Without the explicit `del phase_loaders, val_ldr1` (and `del tr2, va2`, `del tr3,
  va3` at the later boundaries) the Stage-1 workers stay resident through Stages 2 and 3 —
  processes that will never serve another batch, each a `spawn`-ed interpreter with torch
  imported and its own mapping of the patch cube, which on a unified-memory accelerator competes
  with the activations.
- **Compile reset at the Stage 2 → 3 boundary**: `reset_compilation()` drops every
  `torch._dynamo` graph and its guards, because Stage 3's SAM double-backward and per-cycle
  margin vector would otherwise trigger continual recompilation; `run_stage3_swa` disables
  compilation for its own run.
- **Final evaluation**: `best_final_ckpt = _pick_best_checkpoint(cfg, ckpt_s1, ckpt_s2,
  ckpt_s3)` — **chosen by recorded validation macro-F1, not by which stage ran last** — then
  `empty_cache(device)` (frees Stage 3's two extra model copies before TTA activations
  allocate) before the test loader and `final_evaluation` run.

Pointing `output_dir` at a directory with all three stages present makes `latest_completed_stage
== 3`, which skips every stage body and goes straight to checkpoint selection and final
evaluation:

```bash
python train.py output_dir=<existing_run_dir>
```

---

## 6.2 Device selection

`cfg.device = "auto"` resolves in this order — **Metal (MPS) → CUDA → CPU** — and an explicit
`cuda`/`cuda:1`/`cpu`/`mps` is never overridden:

```python
if torch.backends.mps.is_available(): return torch.device("mps")
if torch.cuda.is_available():         return torch.device("cuda")
return torch.device("cpu")
```

Under `torchrun`, this function is not consulted for the choice at all — `init_distributed`
resolves `cuda:${LOCAL_RANK}` itself and hands the device down before `resolve_device` would
matter.

### Mixed precision: dtype, scaler, and the one module that opts out

```python
amp_dtype = resolve_amp_dtype(device, cfg.runtime.amp_dtype)   # bf16 by default
scaler    = make_grad_scaler(amp_dtype, device)                 # paired with it
```

Stage 1 Phases 1–2 are the only place AMP is used at all (§4.5's `use_amp` rule — a SupCon loss
disables it, so Phase 3 and every later stage are fp32 throughout). That one place used to run at
torch's per-device autocast default, **fp16**, and diverged: fp16 carries a 5-bit exponent, so it
saturates at 65 504 and its smallest normal is $6.1\times10^{-5}$, and once an activation or a
gradient leaves that window the loss comes back non-finite and `train_one_epoch` skips the batch.

Three things had to change, and only the first is the dtype:

1. **`resolve_amp_dtype` picks bf16** wherever the device can run it. bf16 has fp32's 8-bit
   exponent — neither bound exists — and pays for it in mantissa bits. `runtime.amp_dtype`
   accepts `auto` / `bf16` / `fp16` / `off`, and the resolved value is printed in the startup
   block and the Stage-1 banner, because it is part of what a reported number means. On Turing
   (sm_75, the T4) bf16 has no Tensor Core path and torch converts instead, so the banner says
   `emulated` — a slower run, chosen deliberately over a `NaN` one.
2. **`make_grad_scaler` pairs the scaler to the dtype.** Loss scaling exists only to lift fp16
   gradients off that underflow floor, so it is *enabled* under fp16 and constructed **disabled**
   under bf16 — a documented pass-through on `scale`/`unscale_`/`step`/`update`, which lets both
   dtypes share one code path in the epoch loop. The explicit `device=` is still what makes any of
   it real: a bare `GradScaler()` binds to CUDA and silently disables itself everywhere else, so
   an fp16 run on Metal would have had autocast without the protection it requires.
3. **The ArcFace head opts out entirely.** Every path in `models/heads.py` runs under
   `autocast(enabled=False)` on an upcast embedding, whatever the ambient dtype (the objectives
   themselves are `04_CURRICULUM_AND_LOSSES.md` §4.4). It is a `(B,256)×(256,270)` matmul
   against a four-branch backbone, the cheapest part of
   the forward and the only part that cannot be traded down: `sqrt(1-\cos^2)` at the $10^{-3}$
   cosine clamp is cancellation that bf16's 8 mantissa bits (resolution $3.9\times10^{-3}$ near
   1.0) cannot represent, `cos/\tau` at $\tau=0.02$ reaches ±50, and the balance term's $10^{-8}$
   probability floor is *below fp16's smallest subnormal*, so it flushed to zero and put
   $0\log 0 = \texttt{NaN}$ straight into the loss.

A fourth change is what turned a transient failure into a permanent one. `GradNormAuxWeights`
reads each epoch's **mean per-branch gradient norm**, which the loop accumulates *before*
`GradScaler` vetoes the step — so a routine fp16 overflow put `inf` in the mean,
$(\infty/\infty)^{\alpha}$ put `NaN` in an auxiliary weight, and `min`/`max` bounds do not catch a
`NaN` (every comparison with it is false). From that epoch on, every loss was `NaN` and every
batch was skipped, while the epoch line kept reporting a mean over `len(loader)`. Non-finite norms
are now filtered out and the branch holds its weight for that epoch. Pinned by
`tests/unit/test_amp_precision.py`.

### Metal-specific handling

Backend tuning lives in `utils/device.py::configure_backend`:

- **SDPA watermark**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` is set to `0.0` (unbounded) while the
  low watermark stays `0.8`, because Branch A's per-cell processing (`batch × grid_size_a² =
  128×64 = 8{,}192` sequences through three towers at the shipped config) hits the default `1.0`
  ratio mid-backward; raising only the high watermark lets the allocator keep going and still
  garbage-collect, rather than aborting the run. Uses `os.environ.setdefault`, so a user-set
  value is respected.
- **BatchNorm re-estimation and Metal's fused attention kernel.** Metal's fused
  `scaled_dot_product_attention` kernel raises `NotImplementedError:
  ... does not support dropout` when a `train()`-mode `nn.MultiheadAttention` is run under
  `no_grad()`. `engine/checkpoint.py::update_bn_stats` (§4.3) resolves this on **every**
  accelerator identically, not with an accelerator-conditional branch: it forces every stochastic
  module (`nn.Dropout*` and `nn.MultiheadAttention`) to `.eval()` before the pass, so nothing
  stochastic remains and plain `no_grad()` works the same way everywhere — dropout under `eval()`
  is also the numerically correct setting for the statistics being re-estimated, since inverted
  dropout would inflate their variance. `tests/unit/test_device.py` keeps a direct reproduction of
  the underlying Metal limitation as a tripwire, so a future torch release that lifts it is
  noticed rather than silently made irrelevant.

### In-model execution paths (`apply_runtime_optimisations`)

`configure_backend` tunes the backend; `apply_runtime_optimisations(model, plan)` is its
companion one level down, and tunes the two modules whose default path is wrong on some
accelerator. It walks the unwrapped model, sets a flag on every matching module, and returns the
banner lines that become the startup block's `Paths :` row. Both settings are execution choices
and nothing else — no parameter, no buffer, no state-dict key, no RNG draw — so a checkpoint
written under either loads under the other, and a resumed run may legitimately change device and
change paths with it. It is idempotent, safe on a model that has neither module, and warns if a
flag is set while its target module is absent (i.e. it was handed something that is not a
`SpectralQuadNet`).

| Flag | Target | Effect |
|---|---|---|
| `plan.decompose_conv3d` | `SpectralSpatialStem3D.decompose_conv3d` | Branch C's three `Conv3d` stages evaluate as stacks of `Conv2d` calls — identical arithmetic in a different summation order ($1.9\times10^{-7}$ on the logits), but a different *kernel*. Metal's `Conv3d` backward is a bad one: **3.12× on the stem, 2.12× on the whole step** (2103 → 994 ms at batch 32), for 9 % more activation memory. |
| `plan.checkpoint_branch_a` | `SpectralProfileBranch.grad_checkpoint` | Branch A's three towers are recomputed in the backward instead of stored: **2.13× less activation memory** (4054 → 1901 MB at batch 32) for 4.8 % of step time, with bit-exact gradients — the towers hold no dropout, no BatchNorm and nothing that reads the RNG. |

Both default to Metal-only, for two different reasons: the first is a kernel-quality defect
specific to that backend (cuDNN's 3-D kernels have no such flaw, and `torch.compile` would rather
be handed the fused operator), the second is unified memory making an activation peak the host's
problem.

At the batch this config actually trains at, the batch-32 figures above badly understate the
second one. Measured on an M5 (16 GB, 12.7 GB recommended working set) at `stage1.batch = 128`:
**49.6 ms/sample holding 9.4 GB with the recompute on, against 780.1 ms/sample holding 14.6 GB
with it off** — 15.7× slower, because without it the step's working set clears the ceiling and
the run pages. There it is not a memory optimisation but the thing that makes the batch
runnable; turning it off on Metal is a decision to lower the batch too.

**The same treatment does not pay for Branch C's stem**, and that has been measured rather than
assumed — full training step on an M5, plain vs `torch.utils.checkpoint` around the stem:

| `stage1.batch` | plain | stem recomputed |
|---:|---:|---:|
| 32 | 32.3 ms/sample | 39.7 ms/sample |
| 64 | 34.3 ms/sample | 47.5 ms/sample |
| 128 | 90.4 ms/sample | 173.8 ms/sample |

and the allocator's high-water mark went **up** in each case rather than down (4041 → 4268 MB at
batch 32). The reason is the decomposition itself: its `kd` gathered depth-slices are transient
within the forward, but a recompute rebuilds all of them *during* the backward while the rest of
the backward's buffers are live — so the peak the checkpoint was meant to lower is the one it
raises. Gradients are bit-identical either way, so this is purely a cost question and the answer
is no. For context on where the time goes: Branch C is **59 % of the forward pass** at batch 128
on an M5 (1363 ms of 2323 ms), with Branch A second at 29 %.

### Reporting a Metal memory overrun, which is otherwise silent

Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` removes the allocator's refusal, so a large
`stage1.batch` cannot abort a multi-hour stage. What it converts that abort into is **paging**,
and a run in that state looks entirely healthy: the loss falls, the epoch line prints, nothing
complains. It is simply several times slower per sample than the same run one batch size down,
and nothing else in the log lets a reader attribute that. Measured on the same M5 at
`stage1.batch`: 32 → 32.3 ms/sample holding 4.0 GB, 64 → 34.3 ms/sample holding 7.9 GB, 128 →
**90.4 ms/sample** holding 9.6 GB. The arithmetic per sample is identical at all three; the knee
is entirely a memory effect.

Three functions put that number where it can be read:

| Function | Returns |
|---|---|
| `memory_budget(device)` | Bytes the device will hold before it pages — `torch.mps.recommended_max_memory()` on Metal, total VRAM on CUDA, `None` elsewhere and on a torch too old to expose the query. |
| `memory_allocated(device)` | What the caching allocator currently holds from the driver — `torch.mps.driver_allocated_memory()` / `torch.cuda.memory_reserved()`. |
| `report_memory_use(device)` | One `[MEM] peak … of a … working set (…%)` line, logged **once** per process (`reset_memory_report()` re-arms it), at `WARNING` at or above `MEMORY_WARN_FRACTION = 0.90` and `INFO` below. |

`release_memory` calls it **before** the sweep, deliberately: that function runs at the epoch and
phase boundaries, so the allocation standing there is the closest thing to the epoch's high-water
mark that can be sampled without a per-step probe — after the sweep it is the floor, which would
say nothing. `describe_hardware` additionally puts the budget in the startup banner next to the
batch size (`[GPU] Apple Silicon (Metal / MPS)  |  12.7 GB recommended working set`).

**It reports rather than adjudicates, and that is the design.** Note where the batch-128 figure
sits: 9.6 GB is **76 %** of a 12.7 GB budget. That run was paging badly and was still nowhere
near the ceiling, so a threshold tuned to catch it would have to fire well below the budget — and
would then fire on a larger machine where the same allocation is fine. Allocation understates the
pressure because on unified memory the budget is not the process's alone: the host's page cache
competes for it, and this pipeline memory-maps a 5.6 GB patch cube that it reads randomly. There
is no honest threshold here, so the line states the peak and the budget and leaves the inference
to a reader comparing `train/epoch_s` across two batch sizes.

### Architecture-gated kernel selection (CUDA)

```python
if capability_major &gt;= 8:   # Ampere and later
    enable flash + mem-efficient + math SDPA
else:                        # Turing (sm_75, e.g. T4)
    enable mem-efficient + math SDPA only  — flash SDPA does not exist pre-Ampere
```

`cudnn.benchmark` defaults `True` (re-applied here so it is visible/overridable, though
`set_seed` already sets it); `allow_tf32` defaults `False` and, when flipped, sets **both**
`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` together with
`torch.set_float32_matmul_precision`, so a later torch version cannot re-enable TF32 behind the
flag's back.

---

## 6.3 Runtime performance knobs (`cfg.runtime`)

`resolve_runtime(cfg.runtime, device, world_size)` turns every `-1`/`"auto"` sentinel into a
concrete value once, producing a frozen `RuntimePlan`. `-1` means "decide from the hardware."

| Field | Default | Resolution |
|---|---|---|
| `num_workers` | `-1` | Half the usable CPU cores (capped at `MAX_AUTO_WORKERS = 8`) on CUDA; `NON_CUDA_AUTO_WORKERS = 2` on Metal/CPU, where the accelerator — not host augmentation — is the bottleneck and `spawn` worker start-up is not free. Measured on an M5 at `stage1.batch = 128`, `heavy` profile: the loader delivers a batch every **187 ms at two workers** (87 ms at eight) against a **9.9 s** training step, so the feed is hidden ~50× over. The per-sample host cost is 1.86 ms, of which 1.41 ms is the mmap page-in and only ~0.45 ms is augmentation — an I/O number, not a CPU one, so more workers buy latency that is already invisible while costing RAM. **Divided again by `world_size` under DDP**, even when set explicitly, floored at 0. |
| `eval_num_workers` | `-1` | `min(⟨resolved num_workers⟩, 4)` — derived from the already-DDP-divided worker count. |
| `pin_memory` | `-1` | Auto-enables on CUDA only. An **explicit** `pin_memory=1` off-CUDA is refused with a logged warning and coerced back to `False` rather than raising — `DataLoader`'s pinning thread would raise with a less legible message. |
| `persistent_workers` | `-1` | Defaults to `num_workers > 0`, then unconditionally re-ANDed with it — an explicit `True` at `num_workers=0` still collapses to `False`. |
| `prefetch_factor` | `4` | Omitted from loader kwargs entirely (not passed as `None`) at `num_workers=0`, since torch raises rather than ignoring it there. |
| `compile` | `auto` | `auto` → on for CUDA, off for Metal/CPU. Measured: on this model at batch 32, the Metal inductor backend produced **983 ms/forward against eager's 437 ms** (2.25× *slower*), because it cannot fuse Branch C's 3-D stem. Compile failure is **never fatal** — any exception (missing Triton, backend incompatibility) falls back to eager with a logged warning. |
| `compile_backend` / `compile_mode` | `inductor` / `default` | Passed straight to `torch.compile`; `dynamic=False` is hardcoded. |
| `channels_last` | `false` | **Not auto-resolved** — a hard opt-in, since NHWC re-selects convolution kernels with a different reduction order (a precision-adjacent change the runtime group is not allowed to make silently). |
| `allow_tf32` | `false` | Off on purpose: cuts a matmul's mantissa from 24 bits to 11, and Turing (the T4) has no TF32 path at all. |
| `amp_dtype` | `auto` | `auto` → **bf16** on any device that can autocast to it, fp16 only where none can; `bf16`/`fp16` force one, `off` trains in fp32. The field that admits to changing numerics rather than defaulting away from it — Stage 1 trains under autocast either way, and the fp16 this group used to inherit from torch is what produced the non-finite losses (§6.2). An unrecognised value raises at resolution time. |
| `cudnn_benchmark` | `true` | Autotuned convolution algorithm selection — already what `set_seed` leaves set. |
| `fused_optimizer` | `auto` | `auto` → CUDA only. AdamW's fused multi-tensor kernel folds the whole step into one launch; it accumulates in the same precision but **not the same order** — the one runtime default here that is not bit-exact against eager AdamW. |
| `decompose_conv3d` | `auto` | `auto` → **Metal only**. Evaluates Branch C's three `Conv3d` stages as stacks of `Conv2d` calls: same arithmetic, different summation order ($1.9\times10^{-7}$ on the logits), different kernel — **3.12× on the stem, 2.12× on the whole step** there, for 9 % more activation memory. Off on CUDA, whose 3-D kernels have no such defect and where `torch.compile` would rather see the fused operator. Applied by `apply_runtime_optimisations` (§6.2). |
| `checkpoint_branch_a` | `auto` | `auto` → **Metal only**. Recomputes Branch A's three towers in the backward: **2.13× less activation memory** (4054 → 1901 MB at batch 32) for 4.8 % of step time, bit-exact gradients. At `stage1.batch = 128` on a 16 GB M5 it is the difference between 49.6 ms/sample and 780.1 ms/sample, because without it the working set clears the ceiling and the run pages — there it is what makes the batch runnable, not a memory optimisation. Off by default on CUDA, where dedicated VRAM makes the recompute a cost; turn it `on` there for a small card or a large batch. |
| `multi_gpu` | `auto` | `auto`/`ddp`/`off` — full semantics in §6.4. |
| `sync_batchnorm` | `true` | Converts every BatchNorm to SyncBatchNorm under DDP — the numerical-equivalence invariant (§6.4). |
| `dist_timeout_s` | `1800` | NCCL/gloo rendezvous timeout. |
| `empty_cache_interval` | `0` | Periodic allocator sweep every N epochs; `0` disables it. Stage boundaries always sweep regardless, since that is where Stage 3's two extra model copies are freed. |
| `progress` | `auto` | Whether the per-epoch line is rendered; `off` suppresses it. One appended line per epoch on every stdout — there is no redrawing mode to select, and `bar`/`rows` are legacy spellings that both render the line (§5.3). |
| `diagnostics_interval` | `50` | Epoch stride for the *rendering* of the hardest-class block and the branch-influence ablation; a new best checkpoint renders them off-stride too. The underlying numbers (per-class F1, CDWS weights) are computed whenever a checkpoint needs them; only the display and the ablation are throttled. |

### Worker residency on the evaluation loaders

`RuntimePlan.loader_kwargs` takes its residency from the plan. Evaluation loaders do not: they
call `eval_loader_kwargs_for(persistent=…)`, and `eval_loader_kwargs` is simply the
`persistent=False` case. The default is no resident workers, because a pool that outlives its
loader is a process leak and most evaluation loaders here are built for exactly one pass —
`build_natural_prior_loader`, the $k$-means seeding loader, the final test loader.

Two are not throwaway, and `build_loaders`/`build_calib_loader` pass `persistent=True` for them:

- the **validation** loader, constructed once per stage and then iterated *twice per epoch* (the
  live model and the EMA shadow) for as many epochs as the stage runs;
- the **calibration** loader, built once in `train.py` and read by every stage's per-class F1 and
  CDWS fit — the longest-lived loader in the run.

At `persistent_workers=False` every one of those passes builds a worker pool and tears it down
again, and on macOS building one is a `spawn`: a fresh interpreter that re-imports torch before
it can read its first patch. Measured on an M5 over the 1,294-patch validation split — six
batches of actual work — **11.47 s per pass non-persistent against 0.08 s persistent**. The
teardown is the larger half: the same pool costs 1.37 s to build on the persistent loader's first
pass, so most of the 11.47 s is the shutdown join at the *end* of every pass, which is pure waste
when the next epoch immediately wants the pool back.

Residency does not raise the peak process count, which is what makes this free rather than a
trade: the training loader's own persistent workers are idle-but-resident during evaluation
either way, so the four-process peak is the same — persistence only stops the two evaluation
workers being rebuilt between epochs. Nothing about the data changes either: an evaluation
dataset runs the `none` augmentation profile, so `__getitem__` draws no randomness and the worker
count cannot move a single value.

### `unwrap_model`

Every checkpoint write, EMA update, and per-branch gradient split addresses parameters by
unprefixed name. `unwrap_model` strips up to 4 layers of `torch.compile`'s `_orig_mod.` and
DDP's `module.` prefix, in either nesting order, so the schema in §3.8 is what every run
produces regardless of how the live model was wrapped for training.

---

## 6.4 Distributed training (DDP)

There is **no `DataParallel` option**. The model carries five `BatchNorm1d` layers in fusion
plus one each in Branches C and D; a split-batch replication scheme where each replica
normalises by its own shard's statistics is a *different function*, and no amount of gradient
averaging repairs a difference that happens inside the forward pass.
`torch.nn.SyncBatchNorm` — available for DDP and not for `DataParallel` — is the whole reason
for the choice, ahead of `DataParallel`'s secondary costs (single-process scatter/gather,
per-step Python replication).

### Activation

Nothing engages unless launched by `torchrun` (or equivalent) with `WORLD_SIZE > 1`. A plain
`python train.py` produces a `DistContext(enabled=False)` whose every collective
(`barrier`/`all_reduce_mean`/`all_reduce_sum`/`broadcast_object`) is an identity no-op — the
single-device path executes exactly the code it executed before this module existed.

`runtime.multi_gpu` has three modes:

| Mode | Behaviour |
|---|---|
| `off` | Never joins the process group, even under `torchrun` — if launched under `torchrun` anyway, logs a warning that every rank will independently train the same single-device job. |
| `auto` (default) | Joins only if launched by `torchrun` with `WORLD_SIZE>1`; else silently degrades to single-device. |
| `ddp` | **Demands** a launcher — raises `RuntimeError` if `RANK`/`WORLD_SIZE`/`LOCAL_RANK` aren't set, so a mistyped launch fails immediately instead of quietly training on one GPU for hours. |

```bash
torchrun --standalone --nproc_per_node=2 train.py runtime.multi_gpu=ddp
```

`LOCAL_RANK` exceeding the visible CUDA device count raises immediately, naming the mismatch
against `--nproc_per_node`/`CUDA_VISIBLE_DEVICES`. Backend is `nccl` on CUDA, `gloo` otherwise
(CPU-only DDP plumbing testing, explicitly not a training configuration). `shutdown` barriers
before destroying the process group, so a rank finishing early does not tear down while another
is mid-collective.

### SyncBatchNorm + DDP wrap

Conversion happens **before** the DDP wrap (§6.1's ordering). `find_unused_parameters=False` is
deliberate, not an oversight — every branch and auxiliary head receives gradient on every step
(branch dropout scales a branch's fused contribution to zero but still routes gradient through
its auxiliary head), so the extra graph traversal DDP would otherwise perform finds nothing.
`gradient_as_bucket_view=True` lets gradients alias the bucket buffer directly.
`SyncBatchNorm` conversion is CUDA-only — a `gloo`/CPU DDP group skips it and logs that the run
is *not* BN-invariant against single-device, exactly why that configuration is testing-only. The
EMA shadow is never wrapped (deep-copied before this step, stays plain BatchNorm) — correct,
since it is only ever evaluated, and in `.eval()` mode SyncBatchNorm and plain BatchNorm are the
same function.

### Per-rank responsibilities

| Concern | Handling |
|---|---|
| Batch size | Global batch stays the configured one: `batch // world_size`, refused (`ValueError`) rather than rounded if it does not divide evenly. |
| Balanced batches | `ClassBalancedBatchSampler` is given an explicit `seed` under DDP so **every rank composes the identical global batch**; `DistributedBatchShardSampler` then gives each rank a contiguous slice of it, preserving $n_{\text{cls}}\times n_{\text{spc}}$ balance across the shards rather than each rank drawing an independent (and therefore larger-effective-batch) sample. `DistributedIndexShardSampler` does the analogous round-robin split for Phase 3's flat weighted stream. |
| Gradients | DDP's mean all-reduce over equal shards is exactly the global-batch gradient. |
| Evaluation | Sharded via `DistributedSampler(shuffle=False, drop_last=False)` — which pads an unevenly-divisible split by repeating early samples — then re-joined by `gather_concat` (a two-phase all-gather: exchange lengths, pad to the max, gather, trim back to true length), so a split whose size doesn't divide by `world_size` is still scored on its true size. Macro-F1 is computed once, on rank 0, over the whole split — never on a shard. |
| Checkpoints, console | Rank 0 only. `save_ckpt` is a no-op on every other rank (every rank holds identical weights, so a second writer would only race the first); other ranks get a `NullTracker`. |
| Decisions | Anything one rank computes that every rank must agree on is broadcast via `broadcast_object` — Stage 3's greedy SWA accept/reject, and both Stage 1's and Stage 2's early-stopping decision. A rank that decided differently would deadlock the next collective. |

---

## 6.5 Checkpointing and auto-resume

Full checkpoint bundle schema and schema versioning (`SCHEMA_VERSION = 3`) are documented at the
architecture level in §3.8. The execution-facing rules:

- `stage_ckpt_path(cfg, s)` → `{output_dir}/best_stage{s}.pth`;
  `stage_meta_path(cfg, s)` → `{output_dir}/stage{s}_meta.json`.
- `stage_exists(cfg, s)` requires **both** files.
- `latest_completed_stage(cfg)` probes `(3, 2, 1)`, returns the highest complete stage, `0` if
  none.
- `_pick_best_checkpoint` ranks each candidate by sidecar `val_f1` → sidecar `val_acc` → the
  full `.pth` bundle's `val_f1` → `0.0`, falling back to the last path argument (Stage 3's) if
  nothing exists on disk.
- `save_ckpt`/`load_ckpt` both route through `unwrap_model` (§6.3) so a compiled/DDP-wrapped run
  writes and reads the same unprefixed key schema as a plain one.

---

## 6.6 Experiment tracking

The `ExperimentTracker` Protocol (`tracking/base.py`, `runtime_checkable`, no shared base class)
splits into two channels because the two audiences want different things from the same run:

| Channel | Methods | Purpose |
|---|---|---|
| Machine | `log_scalar`, `log_scalars`, `log_table`, `log_hyperparams`, `watch`, `close` | plottable series, tabular records, the composed config, gradient histograms |
| Human | `banner`, `log_message`, `log_row`, `progress_start`/`progress_stop` | stage headers, one-line notices, pre-formatted table rows, terminal progress |

`log_row` takes **pre-formatted strings**, not raw numbers — the call site owns formatting
(`.4f` for a loss, `.1%` for accuracy). The engine's default tracker parameter is `tracker=None`,
not an implicit `NullTracker()`: `train_one_epoch(..., tracker=None)` skips diagnostic
accumulation entirely rather than accumulating and discarding, keeping the no-tracker path free
of tracking overhead.

| `tracking.backend` | Class | Human channel | Machine channel |
|---|---|---|---|
| `none` | `NullTracker` | no-op | no-op |
| `console` *(default)* | `ConsoleTracker` | full `rich` rendering (§5.3) | quiet unless `show_diagnostics` |
| `wandb` | `WandbTracker` | **inert** | `wandb.log`, `wandb.Table`, `wandb.watch` (if `watch_model=true`) |
| `tensorboard` | `TensorBoardTracker` | **inert** | `add_scalar`; tables and hyperparameters as Markdown `add_text` (TensorBoard has no tabular primitive, and `add_hparams` insists on a metric dict and creates a nested run) |
| `multi` | `MultiTracker` | fan-out | fan-out |

Both remote backends leave the human channel inert by design — a banner has no useful W&B/
TensorBoard representation, and the numbers behind it already arrive as plottable series. To keep
a readable terminal *and* stream remotely:

```bash
python train.py tracking.backend=multi tracking.backends=[console,wandb]
```

`MultiTracker.close()` closes every child even if one raises, collecting exceptions and
re-raising only the first — a failing backend cannot strand another's file handle or leave a
W&B run unfinished. Nesting `multi` inside `backends`, or an empty `backends` list, is rejected
at construction. Both optional SDKs (`wandb`, `torch.utils.tensorboard`) are imported **inside**
their tracker's constructor, so `import spectralquadnet` never requires either.

`flatten_hyperparams` converts the nested composed config to dotted keys
(`stage1.max_lr → 0.0005`) for W&B/TensorBoard, stringifying lists since `hparams` accepts only
scalars.

Console rendering (the append-only per-epoch line, its span-derived prefix and clocks, the
`training.log` mirror, scalar quieting) is covered alongside the diagnostics it renders in §5.3,
since the two are read together.
