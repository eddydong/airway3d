"""Asymmetric, offset volume: catches XYZ/ZYX mistakes a square duct cannot."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from cfd_geometry import mesh_arrays

mask=np.zeros((9,12,8),bool)
mask[2:6,1:10,3:6]=True
mask[2,4:7,3]=False
ports=np.zeros_like(mask,dtype='u1')
ports[2:4,9,3:6]=1;ports[4:6,9,3:6]=2;ports[:,1,:][mask[:,1,:]]=3
exterior=np.zeros_like(mask);exterior[:,10,:]=True
origin=np.array([-17.,-29.,41.]);h=.7
domain=dict(mask=mask,ports=ports,exterior=exterior,origin=origin,h=h,audit={})
a=mesh_arrays(domain)
expected=(np.argwhere(mask)+.5)*h+origin
assert np.allclose(a['centres'],expected), 'mesh cell centers disagree with native occupancy'
field=np.zeros((*mask.shape,3));field[mask]=[.001,-.003,.002]
print(json.dumps(dict(dims=list(mask.shape),boxMin=origin.tolist(),spacingMm=h,
                     centres=a['centres'].tolist(),occupancy=mask.astype(int).ravel().tolist(),
                     velocity=field.ravel().tolist(),vertices=a['surface'].vertices.tolist(),
                     faces=a['surface'].faces.tolist(),bounds=a['surface'].bounds.tolist())))
