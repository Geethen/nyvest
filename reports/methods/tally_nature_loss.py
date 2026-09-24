import numpy as np, rasterio, json
from rasterio.windows import Window
D="/data/P-Prosjekter2/154001_nyvest/landcover_Geethen/landcover_2018_2024/"
M=np.zeros((13,13),np.int64)
with rasterio.open(D+"nature_loss_2018_2024.tif") as s, rasterio.open(D+"classified_2024.tif") as c:
    H,W=s.height,s.width
    for r in range(0,H,4096):
        w=Window(0,r,W,min(4096,H-r))
        t=s.read(1,window=w); m=t==4
        if not m.any(): continue
        f=s.read(2,window=w)[m]; to=c.read(1,window=w)[m]
        np.add.at(M,(f,to),1)
out={f"{i}->{j}":int(M[i,j]) for i in range(13) for j in range(13) if M[i,j]}
print(out, M.sum())
import sys; json.dump(out, open(sys.argv[1], "w"))
