"""Offline transient sensible heat transport driven by accepted CFD recordings.

Solves rho*cp*(dT/dt + u.grad(T)) = k*laplacian(T) in the 3-D fluid cells,
conjugately coupled to one finite-capacity tissue slab per wall face. The deep
slab boundary is held at body temperature; the exposed surface is NOT fixed.
First-order upwind advection, central diffusion, backward Euler time integration.
No browser endpoint invokes this script. No evaporation or calibrated tissue model.
"""
from __future__ import annotations
import os
# Avoid BLAS oversubscription on a laptop. Set before importing NumPy/SciPy.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab,LinearOperator

VERSION='sensible-conjugate-thermal-3'
PERIODIC_TOLERANCE_C=1e-4
DEFAULTS=dict(ambientC=22.,bodyC=37.,exhaledC=34.,initialC=34.,tissueThicknessMm=1.,
              tissueConductivityWmK=.5,tissueDensityKgM3=1000.,tissueCpJKgK=3600.,
              airConductivityWmK=.026,airDensityKgM3=1.2,airCpJKgK=1005.,maxStepS=.02)
DIRECTIONS=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
CORNERS=np.array([[[1,0,0],[1,1,0],[1,1,1],[1,0,1]],[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
 [[0,1,0],[0,1,1],[1,1,1],[1,1,0]],[[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
 [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],[[0,0,0],[0,1,0],[1,1,0],[1,0,0]]])

def topology(mask,origin,h_mm,openings):
    """Face order shared by the solver, wall geometry and wall scalar arrays."""
    xyz=np.argwhere(mask);n=len(xyz);index=np.full(mask.shape,-1,dtype=np.int32);index[mask]=np.arange(n)
    padded=np.pad(index,1,constant_values=-1)
    opens={}
    for opening in openings:
        normal=np.asarray(opening['normal']);axis=int(np.argmax(abs(normal)));direction=axis*2+(normal[axis]<0)
        for face in opening['facesMm']:
            owner=np.rint((np.asarray(face)-origin)/h_mm-.5-DIRECTIONS[direction]*.5).astype(int)
            if np.any(owner<0) or np.any(owner>=mask.shape) or index[tuple(owner)]<0:raise ValueError('Opening does not map to a fluid cell')
            opens[(int(index[tuple(owner)]),direction)]=opening['role']
    owners=[];neighbours=[];axes=[];wall=[];wall_dirs=[];ports=[];port_dirs=[];port_roles=[]
    for direction,delta in enumerate(DIRECTIONS):
        nb=padded[tuple((xyz+1+delta).T)]
        if direction%2==0:
            keep=nb>=0;owners.extend(np.flatnonzero(keep));neighbours.extend(nb[keep]);axes.extend(np.full(keep.sum(),direction//2))
        for owner in np.flatnonzero(nb<0):
            role=opens.get((int(owner),direction))
            if role:ports.append(owner);port_dirs.append(direction);port_roles.append(role)
            else:wall.append(owner);wall_dirs.append(direction)
    if len(ports)!=len(opens):raise ValueError('Opening face is not on the boundary')
    wall=np.asarray(wall,dtype=np.int32);wall_dirs=np.asarray(wall_dirs,dtype=np.int32)
    vertices=(xyz[wall,None,:]+CORNERS[wall_dirs])*h_mm+origin
    return dict(n=n,owner=np.asarray(owners,dtype=np.int32),neighbour=np.asarray(neighbours,dtype=np.int32),
                axis=np.asarray(axes,dtype=np.int32),wall=wall,wallDirections=wall_dirs,vertices=vertices,
                ports=np.asarray(ports,dtype=np.int32),portDirections=np.asarray(port_dirs,dtype=np.int32),
                portRoles=np.asarray(port_roles),xyz=xyz)

class ThermalSystem:
    def __init__(self,top,h_mm,settings=None):
        self.top=top;self.p=dict(DEFAULTS,**(settings or {}));p=self.p
        self.n=top['n'];self.h=h_mm/1000;h=self.h;area=h*h
        self.ca=p['airDensityKgM3']*p['airCpJKgK']*h**3
        d=p['tissueThicknessMm']/1000
        self.cw=p['tissueDensityKgM3']*p['tissueCpJKgK']*d*area
        self.ga=2*p['airConductivityWmK']*area/h
        self.gb=2*p['tissueConductivityWmK']*area/d
        self.g=1/(1/self.ga+1/self.gb)
        self.diff=p['airConductivityWmK']*h/self.ca
        self.air=np.full(self.n,p['initialC'],dtype=np.float64)
        self.wall=np.full(len(top['wall']),p['initialC'],dtype=np.float64)
        self.maxResidual=0.;self.maxBalanceResidual=0.;self.maxDivergenceHeatW=0.
        self.steps=0
        self.rows=np.r_[top['owner'],top['neighbour'],np.arange(self.n)]
        self.cols=np.r_[top['neighbour'],top['owner'],np.arange(self.n)]

    def step(self,velocity,dt):
        t=self.top;p=self.p;n=self.n;o=t['owner'];nb=t['neighbour'];w=t['wall']
        # Incoming face transport form of u.grad(T); exactly preserves uniform
        # temperature even when cell-centred recorded velocities have small
        # divergence. The implied energy correction is measured, never hidden.
        speed=(velocity[o,t['axis']]+velocity[nb,t['axis']])*.5
        incoming_o=np.maximum(-speed,0)/self.h;incoming_n=np.maximum(speed,0)/self.h
        ao=self.diff+incoming_o;an=self.diff+incoming_n
        rate=(np.bincount(o,weights=ao,minlength=n)+np.bincount(nb,weights=an,minlength=n)).astype(float)
        den=self.cw/dt+self.gb+self.g
        wall_rhs=self.g*(self.cw/dt*self.wall+self.gb*p['bodyC'])/den
        wall_diag=self.g*(self.cw/dt+self.gb)/den
        rate+=np.bincount(w,minlength=n)*wall_diag/self.ca
        rhs=self.air/dt+np.bincount(w,weights=wall_rhs,minlength=n)/self.ca
        ports=t['ports'];dirs=t['portDirections'];normal=DIRECTIONS[dirs]
        out_speed=np.sum(velocity[ports]*normal,axis=1)
        inlet=np.maximum(-out_speed,0)/self.h
        tin=np.where(t['portRoles']=='nostril',p['ambientC'],p['exhaledC'])
        rate+=np.bincount(ports,weights=inlet,minlength=n)
        rhs+=np.bincount(ports,weights=inlet*tin,minlength=n)
        diag=1/dt+rate
        matrix=sparse.csr_matrix((np.r_[-ao,-an,diag],(self.rows,self.cols)),shape=(n,n))
        preconditioner=LinearOperator((n,n),matvec=lambda x:x/diag)
        air,info=bicgstab(matrix,rhs,x0=self.air,rtol=1e-10,atol=1e-10,maxiter=400,M=preconditioner)
        if info!=0:raise ValueError(f'Thermal linear solver did not converge ({info})')
        residual=matrix@air-rhs
        self.maxResidual=max(self.maxResidual,float(np.max(abs(residual)/diag)))
        wall=(self.cw/dt*self.wall+self.gb*p['bodyC']+self.g*air[w])/den
        lower=min(p['ambientC'],p['bodyC'],p['exhaledC'],p['initialC'])
        upper=max(p['ambientC'],p['bodyC'],p['exhaledC'],p['initialC'])
        if not np.isfinite(air).all() or not np.isfinite(wall).all() or min(air.min(),wall.min())<lower-1e-4 or max(air.max(),wall.max())>upper+1e-4:
            raise ValueError('Thermal maximum principle failed; recording withheld')
        # Combined air+tissue energy ledger. The divergence correction is the
        # difference between advective and conservative transport on these data.
        divergence=np.bincount(o,weights=speed/self.h,minlength=n)-np.bincount(nb,weights=speed/self.h,minlength=n)
        divergence+=np.bincount(ports,weights=out_speed/self.h,minlength=n)
        correction=self.ca*np.sum(divergence*(air-p['bodyC']))
        port_heat=self.ca*np.sum((inlet*(tin-p['bodyC'])-np.maximum(out_speed,0)/self.h*(air[ports]-p['bodyC'])))
        body_heat=np.sum(self.gb*(p['bodyC']-wall))
        stored=self.ca*np.sum(air-self.air)/dt+self.cw*np.sum(wall-self.wall)/dt
        self.maxBalanceResidual=max(self.maxBalanceResidual,float(abs(stored-port_heat-body_heat-correction)))
        self.maxDivergenceHeatW=max(self.maxDivergenceHeatW,float(abs(correction)))
        self.air=air;self.wall=wall;self.steps+=1

    def wall_fields(self):
        flux=self.g*(self.wall-self.air[self.top['wall']])/self.h**2
        surface=self.air[self.top['wall']]+flux*self.h**2/self.ga
        return surface,flux

def run(result_path,settings=None,output=None,max_cycles=32):
    result_path=Path(result_path);result=json.loads(result_path.read_text());base=result_path.parent
    required=['mesh','massBalance','coordinateAlignment','completed','frames']
    required+=['stability'] if result.get('solver',{}).get('backend')=='gpu-lbm' else ['wallLeak','requestedFlow','residuals','courant']
    if result.get('status')!='converged' or not all(result.get('gates',{}).get(k) is True for k in required) or not all(result['gates'].values()):raise ValueError('Accepted CFD recording required')
    if result['field'].get('storageOrder')!='fluid-C' or result['solver'].get('temporal')!='transient' or not result.get('openings'):
        raise ValueError('Thermal transport currently requires a sparse transient recording with explicit openings')
    p=dict(DEFAULTS,**(settings or {}));field=result['field'];h=field['spacingMm']
    if not all(math.isfinite(v) for v in p.values()) or any(p[k]<=0 for k in p if k not in ['ambientC','bodyC','exhaledC','initialC']):raise ValueError('Invalid thermal parameters')
    identity=dict(version=VERSION,flowRequestId=result['requestId'],geometryHash=result['geometryHash'],settings=p)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    rootdest=Path(output) if output else base/'thermal';rootdest.mkdir(exist_ok=True,parents=True)
    dest=rootdest/key;dest.mkdir(exist_ok=True)
    if (rootdest/'result.json').exists() and json.loads((rootdest/'result.json').read_text()).get('id')==key:
        print('Thermal recording already prepared:',dest,flush=True);return json.loads((rootdest/'result.json').read_text())
    mask=np.fromfile(base/field['occupancy'],dtype='u1').reshape(field['dims']).astype(bool)
    top=topology(mask,np.asarray(field['boxMin']),h,result['openings'])
    excluded=0
    if result['geometry'].get('flatNostrils'):
        from cfd_geometry import build_domain
        request=result['request']
        original=build_domain(request['settings'],spacing_mm=request['spacingMm'],subdivisions=request.get('refinement',2),flat_nostrils=False)
        if original['mask'].shape!=mask.shape or not np.allclose(original['origin'],field['boxMin']) or original['h']!=h:raise ValueError('Unextended airway grid differs; cannot classify virtual collar walls')
        physical=original['mask'][tuple(top['xyz'][top['wall']].T)]
        excluded=int((~physical).sum())
        for name in ['wall','wallDirections','vertices']:top[name]=top[name][physical]
    system=ThermalSystem(top,h,p)
    # A wall-only mesh: nostril and throat caps are omitted. Every 4 vertices
    # correspond to exactly one thermal face; no smoothing across the septum.
    top['vertices'].astype('<f4').tofile(dest/'wall-quads.f32')
    frames=field['frames'];period=result['solver']['periodS'];start=frames[-1]['timeS']-period
    cycle=[f for f in frames if f['timeS']>=start-1e-7]
    velocities=[np.fromfile(base/f['velocity'],dtype='<f4').reshape(-1,3).astype(np.float64) for f in cycle]
    if any(v.shape!=(top['n'],3) or not np.isfinite(v).all() for v in velocities):raise ValueError('Velocity recording size/values invalid')
    started=time.monotonic();output_frames=[];periodic_error=None
    for cycle_number in range(1,max_cycles+1):
        wall_start=system.wall.copy();air_start=system.air.copy();surface,flux=system.wall_fields()
        samples=[(0.,surface.copy(),flux.copy())]
        for i in range(len(cycle)-1):
            duration=cycle[i+1]['timeS']-cycle[i]['timeS'];steps=math.ceil(duration/p['maxStepS']);dt=duration/steps
            for j in range(steps):
                weight=(j+1)/steps;system.step(velocities[i]*(1-weight)+velocities[i+1]*weight,dt)
            surface,flux=system.wall_fields();samples.append((cycle[i+1]['timeS']-start,surface.copy(),flux.copy()))
        periodic_error=float(max(np.max(abs(system.wall-wall_start)),np.max(abs(system.air-air_start))))
        print(f'{base.name[:10]} thermal cycle {cycle_number}: periodic error {periodic_error:.4g} C, wall {surface.min():.2f}–{surface.max():.2f} C, {time.monotonic()-started:.1f}s computing',flush=True)
        if periodic_error<PERIODIC_TOLERANCE_C:break
    if periodic_error>=PERIODIC_TOLERANCE_C:raise ValueError('Thermal cycle did not reach periodic tolerance; increase max_cycles')
    for i,(time_s,surface,flux) in enumerate(samples):
        # Both channels in one small file: [temperature C, outward heat flux W/m2] per wall face.
        name=f'{i}.wall.f32';np.stack([surface,flux],axis=1).astype('<f4').tofile(dest/name)
        output_frames.append(dict(timeS=time_s,file=f'{key}/{name}',minC=float(surface.min()),maxC=float(surface.max()),meanC=float(surface.mean()),
                                  heatLossW=float(flux.sum()*(h/1000)**2),cooledAreaAbove50Cm2=float((flux>50).sum()*(h/10)**2)))
    gates=dict(finiteAndBounded=True,linearResidual=system.maxResidual<1e-4,energyLedger=system.maxBalanceResidual<1e-4,periodic=periodic_error<PERIODIC_TOLERANCE_C)
    if not all(gates.values()):raise ValueError(f'Thermal acceptance gates failed: {gates}')
    final=dict(**identity,id=key,schemaVersion=1,status='complete',kind='3-D sensible energy transport + finite-capacity tissue slab',
      units=dict(temperature='C',heatFlux='W/m2',positiveHeatFlux='wall to air (cooling)'),gates=gates,
      wall=dict(geometry=f'{key}/wall-quads.f32',faces=len(top['wall']),verticesPerFace=4,spacingMm=h),frames=output_frames,periodS=period,
      diagnostics=dict(excludedVirtualCollarFaces=excluded,steps=system.steps,cycles=cycle_number,computeSeconds=time.monotonic()-started,periodicMaxDifferenceC=periodic_error,periodicToleranceC=PERIODIC_TOLERANCE_C,
                       maxLinearResidualC=system.maxResidual,maxEnergyLedgerResidualW=system.maxBalanceResidual,maxDivergenceCorrectionW=system.maxDivergenceHeatW),
      limitations=['Sensible heat only; evaporation and humidity are not solved.',
        'One tissue slab per wall face; thickness, conductivity and deep temperature are assumed, not measured.',
        'No lateral tissue conduction, perfusion field, radiation or thermal feedback into airflow.',
        'First-order transport on the existing CFD grid; patient mesh/time-step independence is not established.',
        'Recorded cell velocities use advective transport; the discrete divergence energy correction is reported.',
        'Virtual nostril collar walls are adiabatic and excluded from the heated wall surface and summaries.',
        'Wall temperatures are model predictions, not measured patient temperatures or a congestion score.'],
      sources=['https://pubmed.ncbi.nlm.nih.gov/28499215/','https://pmc.ncbi.nlm.nih.gov/articles/PMC8450908/'])
    temp=rootdest/f'result.{os.getpid()}.tmp';temp.write_text(json.dumps(final,indent=2,allow_nan=False));temp.replace(rootdest/'result.json')
    return final

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result',type=Path);parser.add_argument('--all',action='store_true')
    parser.add_argument('--ambient',type=float,default=22);parser.add_argument('--body',type=float,default=37)
    parser.add_argument('--dt',type=float,default=.02);parser.add_argument('--output',type=Path);parser.add_argument('--max-cycles',type=int,default=32)
    args=parser.parse_args()
    from cfd_catalog import build_catalog
    catalog=build_catalog();root=Path(__file__).resolve().parents[1]/'viewer'
    paths=[args.result] if args.result else [root/e['resultUrl'].lstrip('/') for e in catalog['entries']] if args.all else []
    if not paths:parser.error('Choose --result path or --all')
    for path in paths:
        run(path,dict(ambientC=args.ambient,bodyC=args.body,maxStepS=args.dt),args.output,args.max_cycles)
        build_catalog()
    build_catalog()
if __name__=='__main__':main()
