"""Reproducible screenshot/CFD-domain comparison and display topology repair."""
import json
import shutil
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import config as C
from mesh import mask_to_mesh, world_centre, export_mesh
from cfd_geometry import ROOT, prepare_source


def repair_display():
    records=[]
    for dataset,suffix in [('pre',''),('post','_post')]:
        meta_path=C.DATA_DIR/dataset/'meta.json';meta=json.loads(meta_path.read_text())
        vm=json.loads((C.WORK_DIR/'volume_meta.json').read_text())
        for key,pid in [('airway_L',1),('airway_R',2),('airway_common',3)]:
            path=C.DATA_DIR/dataset/(key+'.glb');m=trimesh.load(path,force='mesh')
            record=dict(dataset=dataset,mesh=key,originalWatertight=bool(m.is_watertight),originalFaces=len(m.faces))
            if not m.is_watertight:
                backup=ROOT/'original-display'/dataset;backup.mkdir(parents=True,exist_ok=True)
                if not (backup/path.name).exists(): shutil.copy2(path,backup/path.name)
                sides=np.load(C.WORK_DIR/f'airway_side_fine{suffix}.npy')
                m=mask_to_mesh(sides==pid,vm['grids']['fine']['spacing_mm'],sides.shape,world_centre(sides.shape[0]),(.5,.9,.9),200000)
                if not m.is_watertight: raise ValueError(f'Original extraction of {key} is not watertight')
                meta['meshes'][key]=export_mesh(m,path)
                meta_path.write_text(json.dumps(meta,indent=1))
            record.update(watertight=bool(m.is_watertight),finalFaces=len(m.faces));records.append(record)
    (ROOT/'display-audit.json').write_text(json.dumps(records,indent=2))
    return records


def montage(case):
    with np.load(prepare_source()) as z: src={k:z[k] for k in z.files}
    with np.load(case/'domain.npz') as z: dom={k:z[k] for k in z.files}
    # Coronal planes, anterior to posterior; gray screenshot with source cyan
    # and actual CFD boundary orange. No smoothing of the displayed mask.
    count=12; width=360;height=330
    image=Image.new('RGB',(width*4,height*3+50),(19,24,31));draw=ImageDraw.Draw(image)
    draw.text((15,12),'CT reconstruction / CFD boundary: cyan = source; orange = solver domain (world mm)',fill='white')
    klist=np.linspace(5,src['mask'].shape[2]-8,count).astype(int)[::-1]
    for i,k in enumerate(klist):
        gray=src['gray'][:,:,k].T;a=src['mask'][:,:,k].T
        xyz=np.indices(gray.shape,dtype=float)
        world=np.stack([(xyz[1]+.5)*src['spacing'][0]+src['origin'][0],
                        (xyz[0]+.5)*src['spacing'][1]+src['origin'][1],
                        np.full(gray.shape,(k+.5)*src['spacing'][2]+src['origin'][2])])
        coords=(world-dom['origin'][:,None,None])/float(dom['h'])-.5
        b=ndi.map_coordinates(dom['mask'].astype('u1'),coords,order=0,mode='constant')>0
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        rgb[a&~ndi.binary_erosion(a)]=[20,225,220]
        rgb[b&~ndi.binary_erosion(b)]=[255,166,70]
        panel=Image.fromarray(rgb[::-1]);panel.thumbnail((width-15,height-30))
        left=(i%4)*width;top=(i//4)*height+50
        image.paste(panel,(left,top+20));draw.text((left+8,top),f'Anterior Z = {world[2,0,0]:.1f} mm',fill='white')
    image.save(case/'geometry-review.png')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repair-display',action='store_true');p.add_argument('--case',default='baseline-070')
    args=p.parse_args()
    if args.repair_display: print(json.dumps(repair_display(),indent=2))
    montage(ROOT/args.case)
