"""One pass over the 36 GB cube -> every feature the band analytics need.

Outputs (in OUT):
  mean_snv, sd_snv          (N,256)  masked mean / spatial sd of the per-pixel-SNV cube
  mean_raw                  (N,256)  masked mean of dark-corrected radiance (snv*sd+mu from gain.npy)
  px_snv                    (N,P,256) P random foreground pixels (SNV space)
  px_gain                   (N,P,2)   their (mu, sd) so raw = px_snv*sd + mu
  pool16                    (N,256,16,16) float16 masked 4x4 average pool (SNV space)
  fill16                    (N,16,16) float16 foreground fraction per pooled cell
  noise_sp_raw, noise_sp_snv (N,256) horizontal-neighbour difference variance / 2 (spatial noise upper bound)
  noise_sd2_raw             (N,256)  spectral 2nd-difference variance / 6 (spectral noise estimate)
"""
import sys, time
import numpy as np

ROOT = "/Users/jerlshin/FieldOfInterest/ResearchWork/HSI_RGB_seeds/Code/dataset"
OUT = sys.argv[1]
P = 48
CH = 192
rng = np.random.default_rng(0)

patches = np.load(f"{ROOT}/patches.npy", mmap_mode="r")
gain = np.load(f"{ROOT}/gain.npy", mmap_mode="r")
N, C, H, W = patches.shape

mean_snv = np.zeros((N, C), np.float32)
sd_snv = np.zeros((N, C), np.float32)
mean_raw = np.zeros((N, C), np.float32)
px_snv = np.zeros((N, P, C), np.float32)
px_gain = np.zeros((N, P, 2), np.float32)
pool16 = np.lib.format.open_memmap(f"{OUT}/pool16.npy", mode="w+", dtype=np.float16, shape=(N, C, 16, 16))
fill16 = np.zeros((N, 16, 16), np.float16)
noise_sp_raw = np.zeros((N, C), np.float32)
noise_sp_snv = np.zeros((N, C), np.float32)
noise_sd2_raw = np.zeros((N, C), np.float32)

t0 = time.time()
for s in range(0, N, CH):
    e = min(s + CH, N)
    x = np.asarray(patches[s:e], dtype=np.float32)          # B,C,H,W
    g = np.asarray(gain[s:e], dtype=np.float32)              # B,2,H,W
    B = e - s
    m = (np.abs(x).sum(1) > 1e-5)                            # B,H,W
    mf = m.astype(np.float32)
    cnt = mf.reshape(B, -1).sum(1).clip(min=1)
    raw = x * g[:, 1:2] + g[:, 0:1]
    raw *= mf[:, None]
    xf = x.reshape(B, C, -1)
    rf = raw.reshape(B, C, -1)
    mu = xf.sum(2) / cnt[:, None]
    mean_snv[s:e] = mu
    sd_snv[s:e] = np.sqrt(np.clip((xf ** 2).sum(2) / cnt[:, None] - mu ** 2, 0, None))
    mean_raw[s:e] = rf.sum(2) / cnt[:, None]
    # pixel samples
    for b in range(B):
        idx = np.flatnonzero(m[b].ravel())
        pick = rng.choice(idx, size=P, replace=len(idx) < P)
        px_snv[s + b] = xf[b][:, pick].T
        px_gain[s + b, :, 0] = g[b, 0].ravel()[pick]
        px_gain[s + b, :, 1] = g[b, 1].ravel()[pick]
    # 4x4 masked average pool
    xs = x.reshape(B, C, 16, 4, 16, 4).sum((3, 5))
    ms = mf.reshape(B, 16, 4, 16, 4).sum((2, 4))
    pool16[s:e] = (xs / ms.clip(min=1)[:, None]).astype(np.float16)
    fill16[s:e] = (ms / 16.0).astype(np.float16)
    # spatial neighbour noise
    m2 = (m[:, :, 1:] & m[:, :, :-1]).astype(np.float32)
    n2 = m2.reshape(B, -1).sum(1).clip(min=1)
    d = (raw[:, :, :, 1:] - raw[:, :, :, :-1]) * m2[:, None]
    noise_sp_raw[s:e] = (d ** 2).reshape(B, C, -1).sum(2) / n2[:, None] / 2
    d = (x[:, :, :, 1:] - x[:, :, :, :-1]) * m2[:, None]
    noise_sp_snv[s:e] = (d ** 2).reshape(B, C, -1).sum(2) / n2[:, None] / 2
    # spectral second difference (interior bands; edges copied)
    sd2 = rf[:, 2:] - 2 * rf[:, 1:-1] + rf[:, :-2]
    v = (sd2 ** 2).sum(2) / cnt[:, None] / 6
    noise_sd2_raw[s:e, 1:-1] = v
    noise_sd2_raw[s:e, 0] = v[:, 0]
    noise_sd2_raw[s:e, -1] = v[:, -1]
    el = time.time() - t0
    print(f"{e}/{N}  {el:.0f}s  eta {el / e * (N - e):.0f}s", flush=True)

pool16.flush()
for k, v in dict(mean_snv=mean_snv, sd_snv=sd_snv, mean_raw=mean_raw, px_snv=px_snv, px_gain=px_gain,
                 fill16=fill16, noise_sp_raw=noise_sp_raw, noise_sp_snv=noise_sp_snv,
                 noise_sd2_raw=noise_sd2_raw).items():
    np.save(f"{OUT}/{k}.npy", v)
print("done", time.time() - t0)
