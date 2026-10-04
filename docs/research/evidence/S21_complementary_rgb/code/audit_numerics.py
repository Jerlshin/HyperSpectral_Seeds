"""Investigate NumPy matmul warnings without refitting or rescoring held-out rows."""
from pathlib import Path
import json,sys,warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src'))
from spectralquadnet.experiments.rgb_probe import make_features
from spectralquadnet.data.prep.multimodal import write_json
E=ROOT/'docs/research/evidence/S21_complementary_rgb';O=ROOT/'outputs/s21_rgb_study';plan=json.loads((E/'preregistration.json').read_text());features=make_features(ROOT/plan['data'],ROOT/plan['embeddings'],plan['axes']);records=[];probabilities=[]
for fold in [0,1]:
 # Only within-training acquisition rows; this is numerical validation, not a repeat test.
 ids=np.asarray(plan['splits'][str(fold)]['train'][:32]+plan['splits'][str(fold)]['calib'][:32])
 for path in sorted(O.glob(f'probe_f{fold}_*.npz')):
  arm=path.stem.split(f'probe_f{fold}_')[1]
  with np.load(path,allow_pickle=False) as state:
   assert all(np.isfinite(state[key]).all() for key in state.files)
   assert (state['scale']>0).all()
   x=(features[arm][ids].astype(np.float64)-state['mean'])/state['scale']
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');a=x@state['coef'].T+state['intercept']
   b=np.einsum('ij,kj->ik',x,state['coef'],optimize=False)+state['intercept']
   np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-9)
   assert np.isfinite(a).all()
   records.append({'fold':fold,'arm':arm,'max_logit_magnitude':float(np.abs(a).max()),'blas_vs_explicit_max_abs':float(np.abs(a-b).max()),'warnings':[str(w.message) for w in caught]})
 with np.load(O/f'probabilities_f{fold}.npz') as p:
  for arm in plan['arms']:
   a=p[arm];assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all();np.testing.assert_allclose(a.sum(1),1,atol=1e-12)
  probabilities.append({'fold':fold,'arms_finite_normalized':len(plan['arms'])})
write_json(E/'numerical_validation.json',{'purpose':'Investigate recorded matmul floating-point warnings. No refit or held-out feature rescoring.','train_calib_explicit_sum_checks':records,'saved_probability_checks':probabilities,'interpretation':'All coefficients/scales and saved probabilities are finite; direct scalar-sum and BLAS logits agree on train/calib samples. Warnings alone are not evidence of a nonfinite prediction; their low-level runtime cause was not established.'})
print('Verified',len(records),'exported probes and 48 saved probability matrices; max difference',max(r['blas_vs_explicit_max_abs'] for r in records))
