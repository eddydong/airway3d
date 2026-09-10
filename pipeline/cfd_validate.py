"""Independent square-duct analytic benchmark and patient grid comparison.

The fully-developed laminar square-duct solution has dp/dx =
28.45415377 * mu * Q / width^4. Fit pressure in the middle half to exclude
the developing inlet and compare 8, 12 and 16 cells across the same duct.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cfd_geometry import ROOT,digest,mesh_arrays,write_poly_mesh
from cfd_solver import write_case,read_internal,RHO,NU


def prepare(n):
    h=4/n;mask=np.ones((n,15*n,n),bool);ports=np.zeros_like(mask,dtype='u1')
    ports[:,0,:]=3;ports[:,-1,:n//2]=2;ports[:,-1,n//2:]=1
    # Only the end faces open; the four sides of the inlet collar stay walls.
    exterior=np.zeros(np.array(mask.shape)+[0,2,0],bool)
    mask=np.pad(mask,((0,0),(1,1),(0,0)))
    ports=np.pad(ports,((0,0),(1,1),(0,0)))
    exterior[:,-1,:]=True
    # Generic mesher restricts outlet to -Y faces, exactly as for the airway.
    audit=dict(version='analytic-square-duct',geometryHash=digest(mask.tobytes()),sourceHash='analytic',
               cells=int(mask.sum()),spacingMm=h,volumeCc=float(mask.sum()*h**3/1000))
    domain=dict(mask=mask,ports=ports,exterior=exterior,origin=np.array([0,-h,0]),h=h,audit=audit)
    arrays=mesh_arrays(domain);case=ROOT/f'duct-{n}';case.mkdir(parents=True,exist_ok=True)
    write_poly_mesh(domain,arrays,case);write_case(case,q=1,iterations=1000)
    return case


def verify():
    rows=[]
    for n in [8,12,16]:
        case=ROOT/f'duct-{n}';result=json.loads((case/'result.json').read_text())
        with np.load(case/'domain.npz') as z: centres=z['centres']
        p=read_internal(case/str(result['iterations'])/'p',len(centres))[:,0]*RHO
        y=centres[:,1]/1000;middle=(y>.02)&(y<.04)
        slope=float(np.polyfit(y[middle],p[middle],1)[0]);exact=28.45415377*(RHO*NU)*1e-6/.004**4
        error=abs(slope-exact)/exact
        rows.append(dict(cellsAcross=n,computedGradientPaM=slope,analyticGradientPaM=exact,relativeError=error,
                         converged=result['status']=='converged',wallLeakMlS=result['wallLeakMlS']))
    passed=all(r['converged'] and r['wallLeakMlS']<1e-9 for r in rows) and rows[-1]['relativeError']<.03 and rows[-1]['relativeError']<rows[0]['relativeError']
    report=dict(benchmark='fully developed square duct',passed=passed,results=rows,
                validates='viscous pressure gradient, no-slip walls, finite-volume mesh and field export; not patient anatomy')
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not passed: raise SystemExit('Analytic duct validation failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:
        for n in [8,12,16]: print(prepare(n))
    else: verify()
