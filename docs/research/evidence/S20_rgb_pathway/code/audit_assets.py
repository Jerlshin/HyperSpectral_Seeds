"""S20 descriptive identity audit; never fits or scores a predictor."""
from pathlib import Path
import json,sys,hashlib,shutil,ast
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw
from skimage.measure import regionprops
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src'))
from spectralquadnet.data.prep.multimodal import sha256,write_json
from spectralquadnet.data.prep.rgb import grid_cells
D=ROOT/'dataset_rgb_hsi_v3';E=ROOT/'docs/research/evidence/S20_rgb_pathway';F=ROOT/'docs/research/figures/S20_rgb_pathway';F.mkdir(exist_ok=True)
man=json.loads((D/'MANIFEST.json').read_text());table=pd.read_csv(D/'kernel_manifest.csv');qc=pd.read_csv(D/'scan_qc.csv');scans=pd.read_csv(D/'scan_table.csv');ex=pd.read_csv(D/'exclusions.csv')
checks={p:sha256(D/p)==v['sha256'] for p,v in man['files'].items()}
assert all(checks.values()); assert len(table)==table.kernel_id.nunique()==8624
assert np.array_equal(table['index'],np.arange(8624))
assert np.array_equal(table.label,np.load(D/'labels.npy'))
assert np.array_equal(table.scan_id,np.load(D/'groups.npy'))
assert (qc.k32_different_values==0).all()
for name in ['kernel_manifest.csv','scan_qc.csv','exclusions.csv','MANIFEST.json','build_config.json']:
 shutil.copy2(D/name,E/('asset_'+name if name.endswith('.json') else name))
# Exact duplicate images/foreground crops. This cannot exclude biological reimaging.
crops=np.load(D/'rgb.npy',mmap_mode='r'); hashes=[hashlib.sha256(c.tobytes()).hexdigest() for c in crops]
assert len(set(hashes))==len(hashes)
missing=[]
for sid,sub in ex.groupby('scan_id'):
 meta=json.loads((D/'scans'/f'{sid:03d}.json').read_text());pairs=table[table.scan_id==sid]
 with np.load(D/'scans'/f'{sid:03d}_masks.npz') as masks:
  rr=regionprops(masks['rgb']);hr=regionprops(masks['hsi']); cells=grid_cells(np.array([r.centroid for r in rr]));look={tuple(c):r for c,r in zip(cells,rr)}
  H,_=cv2.findHomography(pairs[['rgb_x','rgb_y']].to_numpy(),pairs[['hsi_x','hsi_y']].to_numpy(),0)
  for rec in sub.itertuples():
   r=look[(rec.grid_row,rec.grid_col)];center=cv2.perspectiveTransform(np.array([[[r.centroid[1],r.centroid[0]]]],float),H)[0,0]
   closest=min(hr,key=lambda r:np.linalg.norm(np.array(r.centroid)[::-1]-center));distance=float(np.linalg.norm(np.array(closest.centroid)[::-1]-center))
   reasons=[]
   if not 300<closest.area<800:reasons.append('area')
   if closest.eccentricity<=.6:reasons.append('eccentricity')
   if closest.solidity<=.85:reasons.append('solidity')
   missing.append({'scan_id':int(sid),'row':int(rec.grid_row)+1,'col':int(rec.grid_col)+1,'nearest_hsi_component':int(closest.label),'center_distance_hsi_px':distance,'area':float(closest.area),'eccentricity':float(closest.eccentricity),'solidity':float(closest.solidity),'failed_gate':'|'.join(reasons),'rgb_area':float(r.area)})
pd.DataFrame(missing).to_csv(E/'missing_component_review.csv',index=False)
# Atlas: each session's first scan, all omission scans, worst residual scans, random sample.
chosen=sorted(set(scans.groupby('session_id').scan_id.first().tolist()+ex.scan_id.unique().tolist()+qc.nlargest(4,'max_hsi_px').scan_id.tolist()+np.random.default_rng(20).choice(180,8,replace=False).tolist()))
for page,start in enumerate(range(0,len(chosen),9)):
 ids=chosen[start:start+9];canvas=Image.new('RGB',(1800,1500),'white');draw=ImageDraw.Draw(canvas)
 for pos,sid in enumerate(ids):
  overlay=Image.open(D/'overlays'/f'{sid:03d}.jpg');panel=Image.new('RGB',(600,480),'white')
  # Keep the RGB seed grid and entire HSI, excluding label/hardware in the montage.
  rgb=overlay.crop((400,25,920,840));hsi=overlay.crop((1224,24,1672,824))
  rgb.thumbnail((320,440));hsi.thumbnail((260,440));panel.paste(rgb,(0,28));panel.paste(hsi,(330,28))
  ImageDraw.Draw(panel).text((8,6),f'scan {sid} / session {int(scans.set_index("scan_id").loc[sid,"session_id"])}',(0,0,0));canvas.paste(panel,((pos%3)*600,(pos//3)*490))
 draw.text((10,1480),'source: dataset_rgb_hsi_v3/overlays; kernel_manifest.csv; scan_qc.csv',(0,0,0));canvas.save(F/f'pairing_atlas_{page+1}.jpg',quality=90)
# Inspect exceptional kernels at native crop size beside matching HSI patch.
worst=table.nlargest(12,'residual_hsi_px');legacy=np.load(ROOT/'dataset_u430k32/patches.npy',mmap_mode='r');canvas=Image.new('RGB',(1800,960),'white');draw=ImageDraw.Draw(canvas)
for p,r in enumerate(worst.itertuples()):
 i=int(r.index);a=Image.fromarray(crops[i]);v=legacy[i].astype(float).mean(0);v=(v/max(v.max(),1e-9)*255).astype('uint8');b=Image.fromarray(v).convert('RGB').resize((224,224))
 x=(p%4)*450;y=(p//4)*315;canvas.paste(a,(x,y+35));canvas.paste(b,(x+224,y+35));draw.text((x+5,y+5),f'idx {i}, scan {r.scan_id}, r{r.grid_row+1}c{r.grid_col+1}; residual {r.residual_hsi_px:.2f}px',(0,0,0))
draw.text((10,945),'source: kernel_manifest.csv; masked RGB crops / historical k32 mean image',(0,0,0));canvas.save(F/'largest_residual_pairs.jpg',quality=90)
summary={'kernels':len(table),'scans':len(qc),'rgb_detected':8640,'hsi_omitted':len(ex),'unique_kernel_ids':table.kernel_id.nunique(),'unique_rgb_payload_hashes':qc.rgb_sha256.nunique(),'unique_crop_hashes':len(set(hashes)),'k32_different_values':int(qc.k32_different_values.sum()),'masks_equal':True,'morphology_equal':True,'asset_hashes_verified':checks,'scan_rmse_mean_hsi_px':qc.rmse_hsi_px.mean(),'scan_rmse_max_hsi_px':qc.rmse_hsi_px.max(),'max_pair_residual_hsi_px':qc.max_hsi_px.max(),'max_pair_residual_row_pitch':qc.max_fraction_row_pitch.max(),'min_orientation_rmse_ratio':min(min(ast.literal_eval(r.alternative_rmse_hsi_px))/r.rmse_hsi_px for r in qc.itertuples()),'native_rgb_foreground_area_median':table.rgb_area.median(),'native_bbox_height_median':np.median([ast.literal_eval(b)[2]-ast.literal_eval(b)[0] for b in table.rgb_bbox]),'review_atlas_scans':chosen,'calibration_note':'214 own-source bands are the strict reference. Historical 215/195/64 include the 605.583333nm pooled-white band; full-band numeric reconstruction is retained as compact float32 summaries and 4x4 region means, no dense full cube was materialized.'}
write_json(E/'asset_validation.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k!='asset_hashes_verified'},indent=2));print(pd.DataFrame(missing).to_string(index=False))
