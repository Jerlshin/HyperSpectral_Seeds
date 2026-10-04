"""Explain differences between original v5 decisions and float16 cached logits."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[5];E=ROOT/'docs/research/evidence/S20_rgb_pathway'
plan=json.loads((E/'preregistration.json').read_text());records=[]
for key,paths in plan['historical_v5'].items():
 path=ROOT/paths['val_test'];folder=path.parent
 with np.load(path) as x:
  rows=np.load(folder/'rows_val_test_tta.npy');pred=np.load(folder/'preds_val_test_tta.npy');assert np.array_equal(rows,x['rows'])
  a=x['logits'].argmax(1);bad=np.flatnonzero(a!=pred)
  records.append({'fold_seed':key,'logit_dtype':str(x['logits'].dtype),'changed_decisions':len(bad),'rows':rows[bad].tolist(),'all_changes_are_quantized_top_ties':bool(all(x['logits'][i,a[i]]==x['logits'][i,pred[i]] for i in bad))})
(E/'historical_logit_quantization.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
