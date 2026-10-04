"""Descriptive S19 audit: metadata, saved results and archive headers; no model fitting.

Run from any directory with the repository's Python/numpy/pandas/matplotlib/Pillow.
Only writes this study's evidence and figures. Does not read new held-out predictions.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/s19-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/s19-cache')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
E = ROOT / 'docs/research/evidence/S19_next_generation_strategy'
F = ROOT / 'docs/research/figures/S19_next_generation_strategy'
E.mkdir(exist_ok=True, parents=True)
F.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(ROOT / 'src'))

def dump(name, value):
    (E / name).write_text(json.dumps(value, indent=2, default=str) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Hash only small provenance inputs, never read all multi-GB arrays just to list them.
inventory = []
for base in ['docs', 'src', 'configs', 'scripts', 'tests', 'notebooks', 'outputs', 'dataset', 'dataset_u430k32']:
    for p in sorted((ROOT / base).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'S19_next_generation_strategy' in p.parts:
            continue
        item = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size)
        if p.stat().st_size < 2_000_000 and p.suffix in {'.md','.py','.yaml','.json','.csv','.sha256'}:
            item['sha256'] = sha(p)
        inventory.append(item)
pd.DataFrame(inventory).to_csv(E / 'inventory.csv', index=False)
dump('provenance.json', dict(date='2026-10-03',
    commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
    working_tree=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
    python=sys.version, scope='descriptive metadata and existing summary audit; no new training or scoring'))

scans = pd.read_csv(ROOT / 'dataset/scan_table.csv')
y = np.load(ROOT / 'dataset/labels.npy')
groups = np.load(ROOT / 'dataset/groups.npy')
assert len(y) == scans.n_patches.sum() == 8624
assert np.array_equal(y, np.load(ROOT / 'dataset_u430k32/labels.npy'))
assert np.array_equal(groups, np.load(ROOT / 'dataset_u430k32/groups.npy'))
assert all(np.sum(groups == r.scan_id) == r.n_patches for r in scans.itertuples())
support = pd.crosstab(scans.label, scans.session_id)
support.to_csv(E / 'class_session_support.csv')
session_table = scans.groupby(['session_id','session']).agg(scans=('scan_id','size'),
    classes=('label','nunique'), kernels=('n_patches','sum')).reset_index()
session_table.to_csv(E / 'session_summary.csv', index=False)
pairs = []
for label, g in scans.groupby('label'):
    sessions = sorted(g.session_id.unique())
    pairs.append(dict(label=int(label), variety=g.variety.iloc[0],
                      sessions='|'.join(map(str,sessions)), cross=len(sessions)>1,
                      kernels=int(g.n_patches.sum()), lost_vs_96=int(96-g.n_patches.sum())))
pd.DataFrame(pairs).to_csv(E / 'variety_support.csv', index=False)
cross = [p for p in pairs if p['cross']]
edges = {}
for p in cross:
    key = p['sessions']
    edges[key] = edges.get(key, 0) + 1

from spectralquadnet.data.loaders import grouped_split
split_checks = []
for fold in [0,1]:
    s = grouped_split(y, groups, eval_frac=.3, calib_frac=.15, fold=fold)
    # Attribute names verified against the repository Splits dataclass.
    tr = s.train
    train_scans = scans[scans.scan_id.isin(groups[tr])]
    A = np.eye(90)[train_scans.label.to_numpy()]
    B = np.eye(9)[train_scans.session_id.to_numpy()]
    split_checks.append(dict(fold=fold, report=vars(s.report),
        train_class_session_design_rank=int(np.linalg.matrix_rank(np.concatenate([A,B],axis=1))),
        design_columns=99, training_classes_in_multiple_sessions=int((train_scans.groupby('label').session_id.nunique()>1).sum())))
dump('split_audit.json', split_checks)
full_design = np.concatenate([np.eye(90)[scans.label], np.eye(9)[scans.session_id]], axis=1)
dump('identifiability.json', dict(full_class_session_design_rank=int(np.linalg.matrix_rank(full_design)),
    design_columns=99, connected_components=2,
    interpretation='Additive class plus session model only; full design connects all sessions except 6. Each grouped training design has nine components and one session per class.'))

arrays = {}
for root in ['dataset','dataset_u430k32']:
    for name in ['patches.npy','labels.npy','groups.npy','masks.npy','morphology.npy']:
        p = ROOT/root/name
        arrays[f'{root}/{name}'] = {'exists':p.exists()}
        if p.exists():
            a = np.load(p, mmap_mode='r')
            arrays[f'{root}/{name}'].update(shape=list(a.shape),dtype=str(a.dtype),bytes=p.stat().st_size)
dump('array_availability.json',arrays)
wl = pd.read_csv(ROOT/'dataset/wavelengths.csv').iloc[:,-1].to_numpy()
band_rows = []
band_arrays = {}
for k in [32,64,215]:
    ids = np.arange(215) if k == 215 else np.load(ROOT/f'outputs/band_finalists/uniform430_k{k}.npy')
    band_arrays[k] = ids
    w = wl[ids]
    band_rows.append(dict(k=k,minimum_nm=float(w.min()),maximum_nm=float(w.max()),
        max_gap_nm=float(np.diff(w).max()),median_gap_nm=float(np.median(np.diff(w))),
        below_430=int(np.sum(w<430)),float16_cube_GiB=len(y)*k*64*64*2/2**30))
pd.DataFrame(band_rows).to_csv(E/'band_geometry.csv',index=False)
white = np.load(ROOT/'dataset/white_spectra.npz')
radiometry = json.loads((ROOT/'dataset/radiometry.json').read_text())
kept = np.ones(256,dtype=bool)
# Instrument indices in radiometry.json are ZERO based (checked against saved nm).
kept[radiometry['bands']['dropped_instrument_index']] = False
assert np.allclose(white['wavelengths'][kept], wl)
calibration = {}
for k, ids in band_arrays.items():
    src = white['source'][:,kept][:,ids]
    calibration[str(k)] = dict(zip(map(str,np.unique(src)), map(int,np.unique(src,return_counts=True)[1])))
dump('calibration_sources.json', dict(legend={'0':'own scan','1':'session shape','2':'session gain','3':'unresolved'},
    retained_sources=calibration,
    session_filled_scan_ids=white['scan_id'][np.where(white['source'][:,kept]==1)[0]].tolist(),
    session_filled_nm=wl[np.where(white['source'][:,kept]==1)[1]].tolist(),
    note='Session references pool scan white tiles, not seed labels. Audit fit/apply boundary before strict inductive full-band comparison.'))
run_rows = []
for p in sorted((ROOT/'outputs').rglob('run.json')):
    r = json.loads(p.read_text())
    run_rows.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p), top_level_keys='|'.join(sorted(r))))
pd.DataFrame(run_rows).to_csv(E/'run_inventory.csv',index=False)
frozen = []
for p in sorted((ROOT/'docs/research/evidence').rglob('*preregistration*.json')):
    if 'S19_next_generation_strategy' not in p.parts:
        frozen.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
pd.DataFrame(frozen).to_csv(E/'frozen_plan_hashes.csv',index=False)

archive = ROOT/'dataset/rice_hsi.zip'
archive_info = dict(exists=archive.exists(),full_crc_or_checksum_verified=False)
if archive.exists():
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        pd.DataFrame([dict(member=m.filename,bytes=m.file_size,compressed=m.compress_size,crc32=f'{m.CRC:08x}')
                      for m in members]).to_csv(E/'archive_inventory.csv',index=False)
        names = {m.filename for m in members}
        images = [m for m in members if m.filename.lower().endswith(('.jpg','.jpeg')) and not m.filename.startswith('__MACOSX/')]
        rgb_pairs = []
        for row in scans.itertuples():
            stem = str(Path(row.member).with_suffix(''))
            candidates = [n for n in names if n.rsplit('.',1)[0] == stem and n.lower().endswith(('.jpg','.jpeg'))]
            rec = dict(scan_id=row.scan_id,label=row.label,session_id=row.session_id,hsi_member=row.member,rgb_candidates=len(candidates))
            if len(candidates)==1:
                with z.open(candidates[0]) as f:
                    im = Image.open(f)
                    rec.update(rgb_member=candidates[0],width=im.width,height=im.height)
            rgb_pairs.append(rec)
        pd.DataFrame(rgb_pairs).to_csv(E/'rgb_scan_pairs.csv',index=False)
        archive_info.update(bytes=archive.stat().st_size,members=len(members),jpeg_members=len(images),
            paired_scans=sum(r['rgb_candidates']==1 for r in rgb_pairs),
            chessboard_members=[n for n in sorted(names) if 'chessboard' in n.lower() and not n.startswith('__MACOSX/')],
            index_members=[n for n in sorted(names) if n.endswith('index.csv')])
dump('archive_availability.json',archive_info)

# Only aggregate result files that have already been reported by S16.
v5 = pd.read_csv(ROOT/'docs/research/evidence/S16_replication_reading/arm_summary.csv')
v5.to_csv(E/'s16_summary_reference.csv', index=False)
summary = dict(kernels=len(y),classes=len(pairs),scans=len(scans),sessions=len(session_table),
    same_session_classes=90-len(cross),cross_session_classes=len(cross),cross_session_pairs=edges,
    lost_vs_original_8640=8640-len(y),all_cross_pairs_touch_session8=all('8' in p['sessions'].split('|') for p in cross),
    cube215_available=(ROOT/'dataset/patches.npy').exists(),rgb_archive=archive_info,
    band32_in_band64=int(len(set(band_arrays[32])&set(band_arrays[64]))),
    recall_ceiling_if_cross_fixed_at_v5=(73+17*.199401)/90,
    interpretation='Ceiling expression is conditional macro-recall arithmetic, not a dataset ceiling or macro-F1 bound.')
dump('audit_summary.json',summary)

dump('band_axes.json',dict(valid_wavelengths_nm=wl.tolist(),
    selected_indices={str(k):v.tolist() for k,v in band_arrays.items()},
    dropped_nm=radiometry['bands']['dropped_nm']))
import runpy
runpy.run_path(str(Path(__file__).with_name('draw_figures.py')),run_name='__main__')
print(json.dumps(summary,indent=2))
