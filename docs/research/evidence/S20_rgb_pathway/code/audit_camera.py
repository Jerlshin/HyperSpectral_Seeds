"""Descriptive RGB acquisition metadata; never used as classifier features."""
from pathlib import Path
import io,json,zipfile
import pandas as pd
from PIL import Image,ExifTags
ROOT=Path(__file__).resolve().parents[5]
D=ROOT/'dataset_rgb_hsi_v3';E=ROOT/'docs/research/evidence/S20_rgb_pathway'
scans=pd.read_csv(D/'scan_table.csv');records=[]
fields=['Make','Model','Orientation','ExposureTime','FNumber','ISOSpeedRatings','PhotographicSensitivity','FocalLength','WhiteBalance','ExposureMode','ExposureBiasValue','Flash','MeteringMode','ColorSpace','DateTimeOriginal']
with zipfile.ZipFile(ROOT/'dataset/rice_hsi.zip') as archive:
 for scan in scans.itertuples():
  member=json.loads((D/'scans'/f'{scan.scan_id:03d}.json').read_text())['rgb_member']
  with Image.open(io.BytesIO(archive.read(member))) as image:
   exif=image.getexif();values={ExifTags.TAGS.get(k,str(k)):v for k,v in exif.items()}
   try:values.update({ExifTags.TAGS.get(k,str(k)):v for k,v in exif.get_ifd(34665).items()})
   except (KeyError,TypeError):pass
   records.append({'scan_id':scan.scan_id,'session_id':scan.session_id,**{key:str(values[key]) if key in values else '' for key in fields}})
frame=pd.DataFrame(records);frame.to_csv(E/'rgb_capture_metadata.csv',index=False)
summary={field:frame.groupby('session_id')[field].agg(lambda s:sorted(set(s))).to_dict() for field in fields if field!='DateTimeOriginal'}
(E/'rgb_capture_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(frame.drop(columns=['scan_id','DateTimeOriginal']).drop_duplicates().to_string(index=False))
