from __future__ import annotations

import copy, math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG, WL_MIN, WL_MAX


# ══════════════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════════════

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = CONFIG["ema_decay"]) -> None:
        self.max_decay    = decay
        self._num_updates = 0
        self.shadow       = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @property
    def current_decay(self) -> float:
        n = self._num_updates
        return min(self.max_decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._num_updates += 1
        d  = self.current_decay
        lp = dict(model.named_parameters())
        for n, sp in self.shadow.named_parameters():
            if n in lp: sp.copy_(d * sp + (1.0 - d) * lp[n])
        lb = dict(model.named_buffers())
        for n, sb in self.shadow.named_buffers():
            if n in lb and sb.dtype.is_floating_point: sb.copy_(lb[n])

    def reinit_from(self, model: nn.Module) -> None:
        self.shadow.load_state_dict(copy.deepcopy(model.state_dict()))
        self._num_updates = 0

    def set_dropout(self, p: float) -> None:
        for m in self.shadow.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def state_dict(self) -> dict:         return self.shadow.state_dict()
    def load_state_dict(self, sd: dict):  self.shadow.load_state_dict(sd)


# ══════════════════════════════════════════════════════════════════════
#  ADAPTIVE SUB-CENTER ARCFACE HEAD
# ══════════════════════════════════════════════════════════════════════

class AdaptiveSubcenterArcFaceHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int,
                 K: int          = CONFIG["subcenter_K"],
                 s: float        = CONFIG["s2_arcface_s"],
                 m_base: float   = CONFIG["s2_arcface_m"],
                 m_delta: float  = CONFIG["s2_arcface_m_delta"]):
        super().__init__()
        self.K = K; self.C = num_classes
        self.s = s; self.m_base = m_base; self.m_delta = m_delta
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("margins", torch.full((num_classes,), m_base))

    def update_margins_from_f1(self, class_f1: Dict[int, float]):
        for c, f1 in class_f1.items():
            self.margins[c] = self.m_base + self.m_delta * (1.0 - min(float(f1), 1.0))
        print(f"[INFO] ArcFace margins  mean={self.margins.mean():.3f}  "
              f"min={self.margins.min():.3f}  max={self.margins.max():.3f}")

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                global_m: Optional[float] = None) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        cosine = (F.linear(x_n, w_n).clamp(-1 + 1e-6, 1 - 1e-6)
                  .view(-1, self.C, self.K).max(dim=2).values)
        if labels is None or not self.training:
            return cosine * self.s
        m_per  = (torch.full((x.shape[0],), global_m, device=x.device)
                  if global_m is not None else self.margins[labels])
        cosm   = torch.cos(m_per); sinm = torch.sin(m_per)
        th     = torch.cos(math.pi - m_per); mm = torch.sin(math.pi - m_per) * m_per
        sine   = torch.sqrt(torch.clamp(1 - cosine ** 2, min=1e-6))
        tgt_c  = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        tgt_s  = sine.gather(1,   labels.view(-1, 1)).squeeze(1)
        phi    = tgt_c * cosm - tgt_s * sinm
        phi    = torch.where(tgt_c > th, phi, tgt_c - mm)
        oh     = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((oh * phi.unsqueeze(1)) + ((1 - oh) * cosine)) * self.s

    def init_from_linear(self, linear_w: torch.Tensor):
        with torch.no_grad():
            wn = F.normalize(linear_w, dim=1)
            for k in range(self.K):
                noise = torch.randn_like(wn) * 0.01 * k
                self.weight[k::self.K].copy_(wn + noise)
        print(f"[INFO] ArcFace (K={self.K}) bootstrapped from linear head.")


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════

class SpectralSE(nn.Module):
    """Channel attention using both mean and max pooling (stronger than mean-only)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, mid, bias=False), nn.GELU(),
            nn.Linear(mid, channels, bias=False),      nn.Sigmoid())

    def forward(self, x):
        g = torch.cat([x.mean([2, 3]), x.amax([2, 3])], dim=1)
        return x * self.gate(g).view(x.shape[0], x.shape[1], 1, 1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return F.gelu(self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x))))) + self.skip(x))


class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(c // r, 8)
        self.ch = nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.GELU(),
                                 nn.Conv2d(mid, c, 1, bias=False))
        self.sp = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * torch.sigmoid(self.ch(x.mean([2, 3], keepdim=True))
                               + self.ch(x.amax([2, 3], keepdim=True)))
        return x * self.sp(torch.cat([x.mean(1, keepdim=True),
                                      x.amax(1, keepdim=True)], 1))


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid = max(out_ch // 2, in_ch)
        self.c1 = nn.Conv2d(in_ch, mid, 1, bias=False);     self.n1 = nn.GroupNorm(min(8, mid), mid)
        self.c2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False); self.n2 = nn.GroupNorm(min(8, mid), mid)
        self.c3 = nn.Conv2d(mid, out_ch, 1, bias=False);    self.n3 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                   nn.GroupNorm(min(8, out_ch), out_ch))
                     if (stride != 1 or in_ch != out_ch) else nn.Identity())

    def forward(self, x):
        return F.gelu(self.n3(self.c3(
            F.gelu(self.n2(self.c2(F.gelu(self.n1(self.c1(x)))))))) + self.skip(x))


class WavelengthPositionalEncoding(nn.Module):
    def __init__(self, num_bands: int = CONFIG["num_bands"],
                 embed_dim: int = CONFIG["wl_embed_dim"]):
        super().__init__()
        wl   = torch.linspace(0.0, 1.0, num_bands)
        half = embed_dim // 2
        freq = torch.exp(torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1)))
        enc  = torch.zeros(num_bands, embed_dim)
        enc[:, :half] = torch.sin(wl.unsqueeze(1) * freq.unsqueeze(0))
        enc[:, half:] = torch.cos(wl.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("enc", enc)
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.01); nn.init.zeros_(self.proj.bias)

    def forward(self): return self.proj(self.enc).squeeze(-1).view(1, 1, -1)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH A — SPECTRAL PROFILE
#  (Signal + 1st + 2nd derivatives with independent encoding + fusion)
# ══════════════════════════════════════════════════════════════════════

class SpectralProfileBranch(nn.Module):
    """
    Spectral modeling branch.

    Features:
    - Independent encoding of signal / d1 / d2
    - Learnable derivative scaling
    - Concatenation fusion (no information bottleneck)
    - Multi-scale spectral receptive fields
    - 1D channel attention (spectral SE)
    - Stable for 90-class fine-grained discrimination
    """

    def __init__(self, out_dim: int = 256, tower_ch: int = 96, wl_enc=None):
        super().__init__()
        self.wl_enc   = wl_enc
        dropout       = CONFIG["branch_internal_drop"]

        # Learnable derivative scaling (chemometric stabilizer)
        self.alpha_d1 = nn.Parameter(torch.tensor(1.0))
        self.alpha_d2 = nn.Parameter(torch.tensor(1.0))

        # Independent stream projections (1×1 conv)
        self.proj_s  = nn.Sequential(nn.Conv1d(1, tower_ch // 3, 1, bias=False),
                                     nn.BatchNorm1d(tower_ch // 3), nn.GELU())
        self.proj_d1 = nn.Sequential(nn.Conv1d(1, tower_ch // 3, 1, bias=False),
                                     nn.BatchNorm1d(tower_ch // 3), nn.GELU())
        self.proj_d2 = nn.Sequential(nn.Conv1d(1, tower_ch // 3, 1, bias=False),
                                     nn.BatchNorm1d(tower_ch // 3), nn.GELU())

        fused_ch = tower_ch

        # Multi-scale spectral towers
        def make_tower(kernel):
            return nn.Sequential(ResBlock1D(fused_ch, fused_ch, kernel),
                                 ResBlock1D(fused_ch, fused_ch, kernel))

        self.tower_s = make_tower(3)
        self.tower_m = make_tower(7)
        self.tower_l = make_tower(15)

        # Spectral Channel Attention (1D SE variant)
        self.se_1d = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(fused_ch, fused_ch // 4, 1, bias=False), nn.GELU(),
            nn.Conv1d(fused_ch // 4, fused_ch, 1, bias=False), nn.Sigmoid())

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(fused_ch * 6, out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    @staticmethod
    def _gp(f):
        return torch.cat([f.mean(dim=2), f.max(dim=2).values], dim=1)

    def forward(self, ms):
        s  = ms.unsqueeze(1)              # (B,1,L)
        d1 = F.pad(torch.diff(s,  dim=2), (0, 1))
        d2 = F.pad(torch.diff(d1, dim=2), (0, 1))
        d1 = self.alpha_d1 * d1
        d2 = self.alpha_d2 * d2
        fs  = self.proj_s(s); fd1 = self.proj_d1(d1); fd2 = self.proj_d2(d2)
        x   = torch.cat([fs, fd1, fd2], dim=1)
        if self.wl_enc is not None: x = x + self.wl_enc()
        xs  = self.tower_s(x); xm = self.tower_m(x); xl = self.tower_l(x)
        attn = self.se_1d(x)
        xs = xs * attn; xm = xm * attn; xl = xl * attn
        feat = torch.cat([self._gp(xs), self._gp(xm), self._gp(xl)], dim=1)
        return self.proj(feat)


# ══════════════════════════════════════════════════════════════════════
#  BRANCH B — SPECTRAL STATISTICS  (mean, std, max across pixels)
# ══════════════════════════════════════════════════════════════════════

class SpectralStatsBranch(nn.Module):
    def __init__(self, out_dim: int = 256, tower_ch: int = 80, wl_enc=None):
        super().__init__()
        self.wl_enc = wl_enc
        dropout     = CONFIG["branch_internal_drop"]
        mk = lambda k: nn.Sequential(ResBlock1D(3, tower_ch // 2, k),
                                     ResBlock1D(tower_ch // 2, tower_ch, k),
                                     ResBlock1D(tower_ch, tower_ch, k))
        self.tower_s = mk(3); self.tower_m = mk(7); self.tower_l = mk(15)
        self.proj    = nn.Sequential(nn.Linear(tower_ch * 6, out_dim),
                                     nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout))

    @staticmethod
    def _gp(f): return torch.cat([f.mean(2), f.max(2).values], 1)

    def forward(self, ms, ss, mx):
        x = torch.stack([ms, ss, mx], 1)
        if self.wl_enc: x = x + self.wl_enc()
        return self.proj(torch.cat([self._gp(self.tower_s(x)),
                                    self._gp(self.tower_m(x)),
                                    self._gp(self.tower_l(x))], 1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH C — SPATIAL CNN
# ══════════════════════════════════════════════════════════════════════

class SpatialCNNBranch(nn.Module):
    def __init__(self, num_bands: int = CONFIG["num_bands"], out_dim: int = 256):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(num_bands, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU())
        self.stages = nn.Sequential(
            ResBlock2D(64,  128, 2),  CBAM(128),
            ResBlock2D(128, 192, 2),  CBAM(192),
            ResBlock2D(192, 256, 2),  CBAM(256),
            ResBlock2D(256, out_dim, 2))
        self.proj = nn.Sequential(nn.Linear(out_dim * 2, out_dim),
                                  nn.BatchNorm1d(out_dim), nn.GELU())

    @staticmethod
    def _pn(x): return x.sign() * x.abs().clamp(1e-8).sqrt()

    def forward(self, x):
        h = self.stages(self.band_reduce(x))
        return self.proj(F.normalize(
            torch.cat([self._pn(h.mean([2, 3])), self._pn(h.amax([2, 3]))], 1), dim=1))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH D — SPECFORMER
# ══════════════════════════════════════════════════════════════════════

class _PreLNBlock(nn.Module):
    def __init__(self, d: int, heads: int, d_ff: int, drop: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(d_ff, d), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        lx  = self.ln1(x)
        h, _ = self.attn(lx, lx, lx, need_weights=False)
        x   = x + self.drop(h)
        return x + self.drop(self.ff(self.ln2(x)))


class SpecFormerBranch(nn.Module):
    def __init__(self,
                 num_bands: int  = CONFIG["num_bands"],
                 patch_size: int = CONFIG["specf_patch"],
                 d_model: int    = CONFIG["specf_dim"],
                 n_heads: int    = CONFIG["specf_heads"],
                 n_layers: int   = CONFIG["specf_layers"],
                 out_dim: int    = 256,
                 dropout: float  = CONFIG["specf_drop"]):
        super().__init__()
        n_p = num_bands // patch_size
        self.patch_size = patch_size; self.n_patches = n_p
        self.patch_proj = nn.Sequential(nn.Linear(patch_size, d_model, bias=False),
                                        nn.LayerNorm(d_model))
        wl_n = (torch.linspace(WL_MIN, WL_MAX, n_p) - WL_MIN) / (WL_MAX - WL_MIN)
        half = d_model // 2
        freq = torch.exp(torch.arange(half).float() * -(math.log(1e4) / max(half - 1, 1)))
        pe   = torch.zeros(n_p, d_model)
        pe[:, :half] = torch.sin(wl_n.unsqueeze(1) * freq.unsqueeze(0))
        pe[:, half:] = torch.cos(wl_n.unsqueeze(1) * freq.unsqueeze(0))
        self.register_buffer("wl_pe", pe)
        self.cls    = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([_PreLNBlock(d_model, n_heads, d_model * 2, dropout)
                                     for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.proj   = nn.Sequential(nn.Linear(d_model, out_dim),
                                    nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, ms):
        B = ms.shape[0]
        x = ms.float().view(B, self.n_patches, self.patch_size)
        x = self.patch_proj(x) + self.wl_pe.unsqueeze(0)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        for blk in self.blocks: x = blk(x)
        return self.proj(self.norm(x)[:, 0])



# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL STATISTICS HELPER
# ══════════════════════════════════════════════════════════════════════

def masked_spectral_stats(x: torch.Tensor):
    x32  = x.float(); B, C, H, W = x32.shape
    flat = x32.reshape(B, C, H * W)
    mask = (flat.abs().sum(1, keepdim=True) > 1e-5).float()
    cnt  = mask.sum(2).clamp(min=1.0)
    mean = (flat * mask).sum(2) / cnt
    std  = ((flat ** 2 * mask).sum(2) / cnt - mean ** 2).clamp(min=1e-6).sqrt()
    mx   = flat.masked_fill(mask.expand_as(flat) == 0, -1e4).max(2).values
    mx   = mx.masked_fill(mx < -9999.0, 0.0)
    return (torch.nan_to_num(mean, 0), torch.nan_to_num(std, 0),
            torch.nan_to_num(mx, 0))


# ══════════════════════════════════════════════════════════════════════
#  BRANCH FUSION
# ══════════════════════════════════════════════════════════════════════

class SpectralFusion(nn.Module):
    def __init__(self, d=256, heads=4, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)

        self.ff   = nn.Sequential(
            nn.Linear(d, d*2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d*2, d),
            nn.Dropout(drop)
        )

    def forward(self, spectral_branches):
        B = spectral_branches[0].shape[0]

        x = torch.stack(spectral_branches, dim=1)   # (B,3,256)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # (B,4,256)

        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + h
        x = x + self.ff(self.ln2(x))

        return x[:, 0]

class CrossModalFusion(nn.Module):
    def __init__(self, d=256, heads=4, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1,1,d))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)

        self.ff   = nn.Sequential(
            nn.Linear(d, d*2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d*2, d),
            nn.Dropout(drop)
        )

    def forward(self, spectral_token, spatial_token):
        B = spectral_token.shape[0]

        tokens = torch.stack([spectral_token, spatial_token], dim=1)  # (B,2,256)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B,3,256)

        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + h
        x = x + self.ff(self.ln2(x))

        return x[:, 0]

# ══════════════════════════════════════════════════════════════════════
#  SPECTRALQUADNET
# ══════════════════════════════════════════════════════════════════════

class SpectralQuadNet(nn.Module):
    def __init__(self,
                 num_classes: int  = CONFIG["num_classes"],
                 num_bands: int    = CONFIG["num_bands"],
                 dropout: float    = CONFIG["s1_dropout"],
                 wl_embed_dim: int = CONFIG["wl_embed_dim"],
                 cfg: dict         = None):
        super().__init__()
        cfg = cfg or CONFIG
        self.branch_drop_prob = cfg.get("branch_drop_prob", 0.0)

        self.se       = SpectralSE(num_bands, 16)
        self.wl_enc   = WavelengthPositionalEncoding(num_bands, wl_embed_dim)
        self.branch_a = SpectralProfileBranch(out_dim=256, tower_ch=96, wl_enc=self.wl_enc)
        self.branch_b = SpectralStatsBranch(256, 80, self.wl_enc)
        self.branch_c = SpatialCNNBranch(num_bands, 256)
        self.branch_d = SpecFormerBranch(num_bands,
                                         cfg["specf_patch"], cfg["specf_dim"],
                                         cfg["specf_heads"], cfg["specf_layers"],
                                         256, cfg["specf_drop"])
        self.spectral_fusion = SpectralFusion(d=256, heads=cfg["fusion_heads"], drop=cfg["fusion_drop"])
        self.cross_modal_fusion = CrossModalFusion(d=256, heads=cfg["fusion_heads"], drop=cfg["fusion_drop"])
        self.embed_net  = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,  256), nn.LayerNorm(256))
        self.linear_head  = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout * 0.4), nn.Linear(256, num_classes))
        self.arcface_head = AdaptiveSubcenterArcFaceHead(
            256, num_classes,
            K       = cfg.get("subcenter_K",       CONFIG["subcenter_K"]),
            s       = cfg["s2_arcface_s"],
            m_base  = cfg["s2_arcface_m"],
            m_delta = cfg.get("s2_arcface_m_delta", CONFIG["s2_arcface_m_delta"]))
        self._use_arcface = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def set_dropout(self, p: float):
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.p = p

    def use_arcface(self, flag: bool): self._use_arcface = flag

    def freeze_head(self, which: str):
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(False)

    def unfreeze_head(self, which: str):
        h = self.linear_head if which == "linear" else self.arcface_head
        for p in h.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_embed: bool = False,
                arc_m: Optional[float] = None) -> torch.Tensor:
        x = self.se(x)
        ms, ss, mx = masked_spectral_stats(x)

        ba = self.branch_a(ms)
        bb = self.branch_b(ms, ss, mx)
        bc = self.branch_c(x)
        bd = self.branch_d(ms)

        if self.training and self.branch_drop_prob > 0:
            do_drop  = torch.bernoulli(
                torch.tensor(self.branch_drop_prob, device=ba.device))
            drop_idx = torch.randint(0, 4, (), device=ba.device)
            one_hot  = F.one_hot(drop_idx, num_classes=4).float()
            keep     = 1.0 - one_hot * do_drop
            ba = ba * keep[0]; bb = bb * keep[1]
            bc = bc * keep[2]; bd = bd * keep[3]

        spectral_token = self.spectral_fusion([ba, bb, bd])
        joint_token = self.cross_modal_fusion(spectral_token, bc)
        emb = self.embed_net(joint_token)
        
        if self._use_arcface:
            emb_n  = F.normalize(emb, dim=1)
            logits = self.arcface_head(emb_n, labels, global_m=arc_m)
        else:
            logits = self.linear_head(emb)

        if return_embed:
            return logits, F.normalize(F.gelu(emb), dim=1)
        return logits


# ══════════════════════════════════════════════════════════════════════
#  TTA — 8 spatial + 4 spectral
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                n_spatial: int = CONFIG["tta_spatial"],
                n_spectral: int = CONFIG["tta_spectral"]) -> torch.Tensor:
    from torch.amp import autocast
    device = x.device; logits = []
    for k, flip in [(k, f) for k in range(4) for f in (False, True)][:n_spatial]:
        aug = torch.rot90(x, k, [2, 3])
        if flip: aug = torch.flip(aug, [3])
        with autocast(device_type=device.type): logits.append(model(aug))
    step   = max(CONFIG["num_bands"] // (max(n_spectral, 1) * 2), 1)
    shifts = ([-step * i for i in range(1, n_spectral // 2 + 1)] +
              [ step * i for i in range(1, n_spectral // 2 + 1)])[:n_spectral]
    for sh in shifts:
        with autocast(device_type=device.type):
            logits.append(model(torch.roll(x, sh, dims=1)))
    return torch.stack(logits).mean(0)