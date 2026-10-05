#!/usr/bin/env python
"""Build (never push) the private Kaggle script for the S39 HSI cells, following the S22 bundle pattern.

The bundle holds only the frozen plan's own input files (no dataset, no credentials). The script
verifies every file hash after extraction, requires two CUDA GPUs, trains each authorized cell with
the S22 runner and archives outputs. Pushing it spends the owner's GPU quota and requires explicit
authorization: `kaggle kernels push -p outputs/s39_kaggle_push`.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

PLAN = Path("configs/research/s39_hsi_seeds_amendment06.json")
PUSH = Path("outputs/s39_kaggle_push")
RECORD = Path("docs/research/evidence/S39_final_confirmation/kaggle_dispatch_manifest.json")


def main() -> None:
    plan = json.loads(PLAN.read_text())
    files = {Path(k) for k in plan["input_hashes"] if not k.startswith("dataset_u430k32/")}
    files |= {PLAN, PLAN.with_suffix(".sha256"), Path("pyproject.toml")}
    manifest = {"repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "training_plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                "files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):
            assert not f.is_absolute() and ".." not in f.parts
            archive.writestr(str(f), f.read_bytes())
        archive.writestr("bundle_origin.json", json.dumps(manifest, indent=2) + "\n")
    blob = buf.getvalue()
    cells = plan["cells"]
    code = f'''# S39 HSI confirmation cells (v5 seeds 1/2 x folds 0/1); private execution.
import base64,hashlib,io,json,os,subprocess,sys,tarfile,time,zipfile
from pathlib import Path
blob=base64.b64decode("""{base64.b64encode(blob).decode()}""")
assert hashlib.sha256(blob).hexdigest()=="{hashlib.sha256(blob).hexdigest()}"
root=Path('/kaggle/working/s39_repo');root.mkdir()
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
 assert torch.cuda.is_available() and torch.cuda.device_count()==2,'S39 requires two CUDA GPUs'
 for cell in {json.dumps(cells)}:
  f,s=cell['fold'],cell['seed']
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/run_complementary_v5.py','train',
       '--plan','{PLAN}','--fold',str(f),'--seed',str(s),'--output',f'outputs/s22_complementary_v5/f{{f}}_s{{s}}']
  with (out/f'cuda_f{{f}}_s{{s}}.log').open('w') as log:
   subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('S39 completed cell',f,s,flush=True)
 (out/'CUDA_COMPLETE.json').write_text(json.dumps({{'plan_sha256':origin['training_plan_sha256'],'cells':{json.dumps(cells)},'seconds':time.monotonic()-started}},indent=2)+'\\n')
finally:
 with tarfile.open('/kaggle/working/s39_cuda_outputs.tar.gz','w:gz') as archive:archive.add(out,arcname='outputs')
'''
    PUSH.mkdir(parents=True, exist_ok=False)
    (PUSH / "s39_hsi_seeds.py").write_text(code)
    metadata = {"id": "jgfreak/s39-hsi-confirmation-seeds", "title": "S39 v5 confirmation seeds 1-2", "code_file": "s39_hsi_seeds.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": ["jerlshinjg/dataset-u430k32"], "competition_sources": [],
                "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (PUSH / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    RECORD.write_text(json.dumps({"training_plan_sha256": manifest["training_plan_sha256"],
                                  "source_zip_sha256": hashlib.sha256(blob).hexdigest(),
                                  "script_sha256": hashlib.sha256(code.encode()).hexdigest(), "files": manifest["files"],
                                  "kernel_metadata": metadata, "pushed": False,
                                  "credential_inclusion": "none: only explicit frozen source/config files"}, indent=2) + "\n")
    print("Built", PUSH, "bundle bytes", len(blob), "files", len(files), "(not pushed)")


if __name__ == "__main__":
    main()
