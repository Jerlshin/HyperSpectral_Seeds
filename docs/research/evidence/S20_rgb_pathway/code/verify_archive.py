from pathlib import Path
import hashlib,zipfile,json,time
p=Path('dataset/rice_hsi.zip'); start=time.monotonic();md5=hashlib.md5();sha=hashlib.sha256()
with p.open('rb') as f:
 while b:=f.read(8*1024*1024):md5.update(b);sha.update(b)
r={'bytes':p.stat().st_size,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'expected_md5':'aeb8e4b9bfbf80550d368c13c766d761','source':'https://zenodo.org/records/3241923','verified_date':'2026-10-04'}
assert r['md5']==r['expected_md5']; print('checksum matched',flush=True)
with zipfile.ZipFile(p) as z:
 r['first_bad_crc_member']=z.testzip();r['members']=len(z.infolist())
assert r['first_bad_crc_member'] is None
r['elapsed_seconds']=time.monotonic()-start
Path('docs/research/evidence/S20_rgb_pathway/archive_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
