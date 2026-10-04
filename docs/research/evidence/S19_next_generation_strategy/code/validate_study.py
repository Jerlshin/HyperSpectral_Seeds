"""Validate S19 documentary links, saved invariants, bibliography and frozen history.
No model predictions are read and no training tests are run.
"""
from pathlib import Path
import ast,csv,hashlib,json,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[5]
E=Path(__file__).resolve().parents[1]
S=ROOT/'docs/research/studies/S19_next_generation_strategy'
checks={}
def read(name):return json.loads((E/name).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in E.glob('*.json'):json.loads(p.read_text())
checks['json_files_parse']=True
for p in (E/'code').glob('*.py'):ast.parse(p.read_text())
ast.parse((ROOT/'docs/research/tools/build_assets.py').read_text())
checks['python_sources_parse']=True
summary=read('audit_summary.json')
assert [summary[x] for x in ['kernels','classes','scans','sessions']]==[8624,90,180,9]
assert summary['cross_session_classes']==17 and summary['all_cross_pairs_touch_session8']
assert summary['band32_in_band64']==8 and summary['rgb_archive']['paired_scans']==180
assert summary['cube215_available'] is False
checks['audit_invariants']=True
axes=read('band_axes.json');wl=np.array(axes['valid_wavelengths_nm'])
assert len(wl)==215 and np.sum(wl<430)==20
assert np.isclose(np.max(np.diff(wl)),102.666667)
assert all(len(v)==int(k) for k,v in axes['selected_indices'].items())
cal=read('calibration_sources.json')['retained_sources']
assert cal['32']=={'0':5760} and cal['64']=={'0':11517,'1':3} and cal['215']=={'0':38697,'1':3}
assert [x['train_class_session_design_rank'] for x in read('split_audit.json')]==[90,90]
assert read('identifiability.json')['full_class_session_design_rank']==97
checks['bands_calibration_design']=True
lit=read('literature.json');ids=[r['id'] for r in lit]
assert len(ids)==len(set(ids))==39
assert all(r['url'].startswith('https://') and r['verified_on']=='2026-10-03' for r in lit)
assert len(list(csv.DictReader((E/'literature.csv').open())))==len(lit)
assert (E/'references.bib').read_text().count('@misc{')==len(lit)
checks['literature_records']=len(lit)
mds=list(S.glob('*.md'))+[ROOT/'docs/research/MASTER_RESEARCH_PLAN.md',ROOT/'docs/research/RESEARCH_PROGRESS.md',ROOT/'docs/research/README.md']
links=0
for p in mds:
 for url in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in url or url.startswith('#'):continue
  target=url.split('#')[0]
  if target:
   assert (p.parent/target).exists(),(p,url)
   links+=1
checks['local_markdown_links_valid']=links
plans=list(csv.DictReader((E/'frozen_plan_hashes.csv').open()))
for r in plans:assert sha(ROOT/r['path'])==r['sha256'],r['path']
assert sha(ROOT/'docs/research/evidence/S16_replication_reading/preregistration_s16.json')=='3b623c45c559962c36b59383ddf6a636da9a087e41523d8d5c455e81954b9d3e'
checks['frozen_plan_hashes_unchanged']=len(plans)
# Existing study documents were inventoried before S19 integration; they are preserved.
old=[]
for r in csv.DictReader((E/'inventory.csv').open()):
 if re.match(r'docs/research/studies/S\d\d_',r['path']) and r.get('sha256'):
  assert sha(ROOT/r['path'])==r['sha256'],r['path']
  old.append(r['path'])
checks['historical_study_documents_unchanged_since_audit']=len(old)
checks['figures']={}
for name in ['class_session_support.png','band_geometry.png']:
 p=ROOT/'docs/research/figures/S19_next_generation_strategy'/name
 with Image.open(p) as im:im.verify()
 with Image.open(p) as im:checks['figures'][name]=list(im.size)
checks['scope']='Document/metadata validation only; does not validate proposed predictive performance.'
(E/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
