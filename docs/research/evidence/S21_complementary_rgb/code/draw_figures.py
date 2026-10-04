"""Rebuild scientific figures from the tracked S20 evidence only."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/s20-matplotlib')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[5];E=ROOT/'docs/research/evidence/S21_complementary_rgb';F=ROOT/'docs/research/figures/S21_complementary_rgb';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
s=pd.DataFrame(json.loads((E/'summary.json').read_text())).set_index('arm')
choices=['rgb_shape','rgb_all','dino_silhouette','dino_rgb32','dino_gray','dino_rgb','hsi_q32','hsi_q214_own','fusion32_equal','fusion32_calib','concat32_dino']
labels=['RGB morphology','RGB hand descriptors','DINO silhouette','DINO RGB at 32px','DINO grayscale','DINO RGB','HSI k32 quantiles + shape','HSI own-white 214 + shape','Equal RGB + HSI32','Calib-weight RGB + HSI32','Concatenated features']
fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True)
for ax,key,ci,title in zip(axes,['f1','cross_recall'],['f1_ci','cross_ci'],['Grouped macro-F1 (90 varieties)','Cross-session recall (17 varieties)']):
 values=s.loc[choices,key].to_numpy();bounds=np.array(s.loc[choices,ci].tolist());colors=['#336b87' if 'dino' in a or a.startswith('rgb') else '#a55b2a' if a.startswith('hsi') else '#347a52' for a in choices]
 ax.barh(np.arange(len(choices)),values,color=colors,height=.65);ax.errorbar(values,np.arange(len(choices)),xerr=np.maximum(np.stack([values-bounds[:,0],bounds[:,1]-values]),0),fmt='none',ecolor='#222',capsize=2,lw=.8);ax.set_title(title);ax.set_xlim(0,1);ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(np.arange(len(choices)),labels);axes[0].invert_yaxis();fig.suptitle('RGB evidence on complementary acquisition folds',fontsize=14);fig.text(.02,.014,'Source: S21 summary.json; two grouped folds; 95% variety-bootstrap intervals; CPU fixed-feature screen.',fontsize=8);fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(F/'rgb_and_fusion.png');plt.close(fig)
choices=['hsi_mean32','hsi_mean214_own','hsi_snvmean32','hsi_snvmean214_own','hsi_q32','hsi_q64','hsi_qnested32','hsi_qnested64','hsi_q195','hsi_q215','hsi_q214_own'];fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(choices));ax.bar(x-.18,s.loc[choices,'f1'],.36,label='Macro-F1',color='#336b87');ax.bar(x+.18,s.loc[choices,'cross_recall'],.36,label='Cross-session recall',color='#a55b2a');ax.set_xticks(x,[a.removeprefix('hsi_').replace('_','\n') for a in choices],rotation=35,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Mean over both grouped folds');ax.set_title('Controlled HSI coverage and summary probes');ax.legend();ax.grid(axis='y',alpha=.15);fig.text(.02,.015,'Source: S21 summary.json; fixed shrinkage LDA; all arms include original HSI morphology.',fontsize=8);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(F/'spectral_controls.png');plt.close(fig)
