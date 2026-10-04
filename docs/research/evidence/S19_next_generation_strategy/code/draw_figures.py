"""Regenerate S19 figures from tracked evidence only; no dataset required."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/s19-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/s19-cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = Path(__file__).resolve().parents[1]
F = E.parent.parent / 'figures' / E.name
F.mkdir(parents=True, exist_ok=True)
support = pd.read_csv(E/'class_session_support.csv',index_col=0)
axes = json.loads((E/'band_axes.json').read_text())
wl = np.array(axes['valid_wavelengths_nm'])
band_arrays = {int(k):np.array(v) for k,v in axes['selected_indices'].items()}
radiometry = {'bands':{'dropped_nm':axes['dropped_nm']}}
plt.rcParams.update({'font.size':10,'savefig.dpi':160})
fig, ax = plt.subplots(figsize=(10,5))
im=ax.imshow(support.to_numpy().T, aspect='auto',cmap='Blues',vmin=0,vmax=2)
ax.set(xlabel='Variety label (90 classes)',ylabel='Acquisition session',yticks=range(9),
    title='Only 17 varieties bridge sessions; every bridge touches session 8')
fig.colorbar(im,ax=ax,label='Scans per class/session',ticks=[0,1,2])
fig.text(.01,.015,'Source: class_session_support.csv; dataset/scan_table.csv. Counts, not model performance.',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(F/'class_session_support.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.6))
for row,k in enumerate([32,64,215]):
    ax.scatter(wl[band_arrays[k]],np.full(k,row),s=12 if k<215 else 5,label=f'{k} bands')
ax.axvspan(min(radiometry['bands']['dropped_nm']),max(radiometry['bands']['dropped_nm']),color='gray',alpha=.2,label='No common valid white reference')
ax.set(yticks=range(3),yticklabels=['uniform430 k32','uniform430 k64','all valid 215'],
    xlabel='Wavelength (nm)',title='Band count also changes coverage and physical sampling')
ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
fig.text(.01,.01,'Source: band_geometry.csv; dataset/wavelengths.csv; outputs/band_finalists/*.npy.',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(F/'band_geometry.png');plt.close(fig)
