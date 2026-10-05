from pathlib import Path
import json,io,zipfile,hashlib,base64,subprocess
root=Path.cwd();plan=Path('configs/research/s22_screening_amendment05.json');p=json.loads(plan.read_text())
files={Path(k) for k in p['input_hashes'] if not k.startswith('dataset_u430k32/')}
files|={plan,plan.with_suffix('.sha256'),Path('pyproject.toml')}
manifest={'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'repo_working_tree':'Contains explicit S22 screening/compatibility amendments; individual source hashes govern identity.','training_plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 for f in sorted(files):
  assert not f.is_absolute() and '..' not in f.parts
  archive.writestr(str(f),f.read_bytes())
 archive.writestr('bundle_origin.json',json.dumps(manifest,indent=2)+'\n')
blob=buf.getvalue();encoded=base64.b64encode(blob).decode()
push=Path('outputs/s22_kaggle_push');push.mkdir(exist_ok=False)
code='''# S22 amended development screen; private execution, fixed seed0 and both directions.
import base64,hashlib,io,json,os,subprocess,sys,tarfile,time,zipfile
from pathlib import Path
BUNDLE = """ENCODED_BUNDLE"""
EXPECTED_BUNDLE = "BUNDLE_HASH"
blob=base64.b64decode(BUNDLE)
assert hashlib.sha256(blob).hexdigest()==EXPECTED_BUNDLE
root=Path('/kaggle/working/s22_repo');root.mkdir()
with zipfile.ZipFile(io.BytesIO(blob)) as archive:archive.extractall(root)
origin=json.loads((root/'bundle_origin.json').read_text())
for name,digest in origin['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
sources=[p.parent for p in Path('/kaggle/input').rglob('MANIFEST.json') if (p.parent/'patches.npy').exists()]
assert len(sources)==1,sources
(root/'dataset_u430k32').symlink_to(sources[0],target_is_directory=True)
out=root/'outputs';out.mkdir()
env=os.environ.copy();env['PYTHONPATH']=str(root/'src');env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
started=time.monotonic()
try:
 subprocess.run([sys.executable,'-m','pip','install','-q','-e','.[prep]'],cwd=root,env=env,check=True)
 import torch
 assert torch.cuda.is_available() and torch.cuda.device_count()==2, 'S22 requires two CUDA GPUs'
 facts={'torch':torch.__version__,'python':sys.version,'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'origin':origin,'data_source':str(sources[0])}
 (out/'cuda_launch_receipt.json').write_text(json.dumps(facts,indent=2)+'\\n')
 for fold in [0,1]:
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/run_complementary_v5.py','train','--plan','configs/research/s22_screening_amendment05.json','--fold',str(fold),'--seed','0']
  print('S22 launching fixed cell',fold,0,flush=True)
  with (out/f'cuda_f{fold}_s0.log').open('w') as log:
   subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('S22 completed fixed cell',fold,0,flush=True)
 (out/'CUDA_COMPLETE.json').write_text(json.dumps({'plan_sha256':origin['training_plan_sha256'],'cells':[{'fold':f,'seed':0} for f in [0,1]],'seconds':time.monotonic()-started},indent=2)+'\\n')
finally:
 with tarfile.open('/kaggle/working/s22_cuda_outputs.tar.gz','w:gz') as archive:archive.add(out,arcname='outputs')
 print('S22 output archive preserved',flush=True)
'''.replace('ENCODED_BUNDLE',encoded).replace('BUNDLE_HASH',hashlib.sha256(blob).hexdigest())
(push/'s22_screen.py').write_text(code)
metadata={'id':'jgfreak/s22-complementary-screen-20261005','title':'S22 corrected-fold single-seed development screen','code_file':'s22_screen.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'dataset_sources':['jerlshinjg/dataset-u430k32'],'competition_sources':[],'kernel_sources':[],'machine_shape':'NvidiaTeslaT4'}
(push/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
record={'training_plan_sha256':manifest['training_plan_sha256'],'source_zip_sha256':hashlib.sha256(blob).hexdigest(),'script_sha256':hashlib.sha256(code.encode()).hexdigest(),'files':manifest['files'],'kernel_metadata':metadata,'source_bundle_bytes':len(blob),'script_bytes':len(code.encode()),'credential_inclusion':'none: only explicit frozen source/config/evidence files'}
Path('docs/research/evidence/S22_complementary_v5/kaggle_dispatch_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print('Bundle bytes',len(blob),'script bytes',len(code.encode()),'files',len(files))
