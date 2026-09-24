import numpy as np, rasterio, sys
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pyproj import Transformer
D="/data/P-Prosjekter2/154001_nyvest/landcover_Geethen/landcover_2018_2024/"
out={}
with rasterio.open(D+"classified_2024.tif") as s:
    f=25
    a=s.read(1,out_shape=(s.height//f,s.width//f),resampling=Resampling.mode)
    out["full"]=a; out["full_bounds"]=np.array(s.bounds)
    x,y=Transformer.from_crs(4326,32633,always_xy=True).transform(6.85,61.87)
    b=(x-8000,y-8000,x+8000,y+8000)
    w=from_bounds(*b,transform=s.transform).round_offsets().round_lengths()
    out["zoom_cls"]=s.read(1,window=w); out["zoom_bounds"]=np.array(rasterio.windows.bounds(w,s.transform))
with rasterio.open(D+"uq_2024_pcal.tif") as s:
    p=s.read(window=w); out["zoom_pmax"]=p.max(0)
    print(s.tags(), s.tags(1))
with rasterio.open(D+"uq_2024_setsize.tif") as s:
    out["zoom_set"]=s.read(1,window=w)
np.savez_compressed(sys.argv[1],**out)
print({k:(v.shape,v.dtype) for k,v in out.items()}, np.unique(out["zoom_cls"]), out["zoom_pmax"].max())
