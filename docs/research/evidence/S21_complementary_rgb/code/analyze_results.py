"""Offline S20 analysis from frozen-run saved predictions; no fitting or rescoring."""
from pathlib import Path
import json,sys,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src'))
from spectralquadnet.data.prep.multimodal import sha256,write_json
from spectralquadnet.experiments.rgb_probe import cluster_interval
O=ROOT/'outputs/s21_rgb_study';E=ROOT/'docs/research/evidence/S21_complementary_rgb'
assert (O/'COMPLETED.json').exists()
for name in ['metrics.csv','per_class.csv','predictions.csv.gz','calibration.csv','summary.json','deltas.json','STARTED.json','COMPLETED.json']:
 shutil.copy2(O/name,E/name)
c=pd.read_csv(E/'per_class.csv');p=pd.read_csv(E/'predictions.csv.gz');meta=pd.read_csv(ROOT/'docs/research/evidence/S20_rgb_pathway/kernel_manifest.csv').set_index('index');rng=np.random.default_rng(20261004);summary=json.loads((E/'summary.json').read_text());extra=[]
for r in summary:
 arm=r['arm'];block=c[c.arm==arm];correct=block.pivot(index='label',columns='fold',values='correct').to_numpy();n=block.pivot(index='label',columns='fold',values='support').to_numpy();boot=rng.integers(90,size=(2000,90));accuracy=(correct[boot].sum(1)/n[boot].sum(1)).mean(1);r['accuracy_ci']=np.quantile(accuracy,[.025,.975]).tolist()
write_json(E/'summary_with_accuracy_ci.json',summary)
cross=c.groupby('label')['cross'].first().to_numpy()
# Directional session result, class-macro recall over the cross-session subset.
keys=['dino_rgb','hsi_q32','hsi_q214_own','fusion32_equal']
d=c[c.cross & c.arm.isin(keys)].groupby(['arm','session']).agg(classes=('label','nunique'),recall=('recall','mean')).reset_index();d.to_csv(E/'destination_sessions.csv',index=False)
# Error complementarity for the primary descriptor pair and each historical v5 seed.
records=[]
for fold in [0,1]:
 rows=p[p.fold==fold];r=rows[rows.arm=='dino_rgb'].set_index('index').sort_index();rr=(r.prediction==r.target).to_numpy();target=r.target.to_numpy()
 for h in ['hsi_q32','hsi_q214_own']:
  hsi=rows[rows.arm==h].set_index('index').sort_index();assert np.array_equal(hsi.index,r.index);hh=(hsi.prediction==hsi.target).to_numpy()
  for group,mask in [('all',np.ones(len(rr),bool)),('same',~np.isin(target,np.flatnonzero(cross))),('cross',np.isin(target,np.flatnonzero(cross)))]:
   records.append({'fold':fold,'hsi':h,'subset':group,'n':int(mask.sum()),'both_correct':float((hh[mask]&rr[mask]).mean()),'rgb_only_correct':float((~hh[mask]&rr[mask]).mean()),'hsi_only_correct':float((hh[mask]&~rr[mask]).mean()),'both_wrong':float((~hh[mask]&~rr[mask]).mean()),'oracle_accuracy':float((hh[mask]|rr[mask]).mean()),'disagreement':float((hsi.prediction.to_numpy()[mask]!=r.prediction.to_numpy()[mask]).mean())})
pd.DataFrame(records).to_csv(E/'complementarity.csv',index=False)
# Quality diagnostics from saved predictions only; do not refit/select/exclude.
quality=meta.residual_hsi_px;cuts=[('all',quality.index),('residual_gt2',quality[quality>2].index)]
quality_rows=[]
for arm in ['dino_rgb','hsi_q32','fusion32_equal']:
 for label,indices in cuts:
  sub=p[(p.arm==arm)&p['index'].isin(indices)];quality_rows.append({'arm':arm,'subset':label,'n':len(sub),'accuracy':float((sub.target==sub.prediction).mean()) if len(sub) else None})
write_json(E/'quality_sensitivity.json',quality_rows)
print(pd.DataFrame(summary)[['arm','f1','same_recall','cross_recall']].to_string(index=False))
