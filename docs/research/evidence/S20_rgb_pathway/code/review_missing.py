"""Account for all 16 historical exclusions before clear_border/shape rejection."""
from pathlib import Path
import json,sys,zipfile
import numpy as np,pandas as pd,cv2
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label,regionprops
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src'))
from spectralquadnet.data.prep.multimodal import read_envi,write_json
from spectralquadnet.data.prep.segmentation import dark_frame
from spectralquadnet.data.prep.rgb import grid_cells
D=ROOT/'dataset_rgb_hsi_v3';E=ROOT/'docs/research/evidence/S20_rgb_pathway';scans=pd.read_csv(D/'scan_table.csv').set_index('scan_id');pairs=pd.read_csv(D/'kernel_manifest.csv');ex=pd.read_csv(D/'exclusions.csv');wl=np.load(D/'white_spectra.npz')['wavelengths'];records=[]
with zipfile.ZipFile(ROOT/'dataset/rice_hsi.zip') as z:
 for sid,omitted in ex.groupby('scan_id'):
  member=scans.loc[sid,'member'];raw=read_envi(z,member);dark=dark_frame(read_envi(z,str(Path(member).parent/'black.hdr')));rad=np.maximum(raw[:600].astype(np.float32)-dark,0);vis=rad[:,:,(wl>450)&(wl<700)].mean(2);lab=label(binary_fill_holes(vis>.4*threshold_otsu(vis)));hr=regionprops(lab)
  with np.load(D/'scans'/f'{sid:03d}_masks.npz') as data:rr=regionprops(data['rgb'])
  cells=grid_cells(np.array([r.centroid for r in rr]));lookup={tuple(c):r for c,r in zip(cells,rr)};p=pairs[pairs.scan_id==sid];H,_=cv2.findHomography(p[['rgb_x','rgb_y']].to_numpy(),p[['hsi_x','hsi_y']].to_numpy(),0)
  for o in omitted.itertuples():
   rgb=lookup[(o.grid_row,o.grid_col)];center=cv2.perspectiveTransform(np.array([[[rgb.centroid[1],rgb.centroid[0]]]],float),H)[0,0]
   # Component containing the expected center, or nearest centroid for a merged seed.
   x,y=np.rint(center).astype(int);identity=lab[np.clip(y,0,599),np.clip(x,0,335)]
   region=next((r for r in hr if r.label==identity),None)
   if region is None:region=min(hr,key=lambda r:np.linalg.norm(np.array(r.centroid)[::-1]-center))
   border=region.bbox[0]==0 or region.bbox[1]==0 or region.bbox[2]==600 or region.bbox[3]==336
   reasons=[]
   if border:reasons.append('border')
   if not 300<region.area<800:reasons.append('area')
   if region.eccentricity<=.6:reasons.append('eccentricity')
   if region.solidity<=.85:reasons.append('solidity')
   records.append({'scan_id':int(sid),'row':int(o.grid_row)+1,'col':int(o.grid_col)+1,'hsi_component':int(region.label),'area':float(region.area),'eccentricity':float(region.eccentricity),'solidity':float(region.solidity),'bbox':list(map(int,region.bbox)),'border':border,'failed_gate':'|'.join(reasons),'center_distance_px':float(np.linalg.norm(np.array(region.centroid)[::-1]-center))})
assert len(records)==16 and all(r['failed_gate'] for r in records)
pd.DataFrame(records).to_csv(E/'missing_component_review.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
