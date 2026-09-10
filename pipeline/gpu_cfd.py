"""GPU lattice-Boltzmann recordings of breathing, with explicit units and gates.

The geometry and request identity are shared with the OpenFOAM path. This
backend runs FluidX3D (D3Q19 TRT, FP32, Smagorinsky-Lilly eddy viscosity) on
the same voxel domain, driven by a smooth asymmetric throat pressure against
ambient nostrils. Nothing prescribes a flow rate or a left/right split.
Results are large-eddy simulations on a rigid stair-step wall, not DNS.
"""
from pathlib import Path
import json
import math
import subprocess
import time
import numpy as np
from scipy import ndimage as ndi
from gpu_cfd_build import BINARY, ROOT, COMMIT, VERSION

DIRECTIONS=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
SIX=ndi.generate_binary_structure(3,1)
RHO=1.2
NU=1.5e-5
FRAMES_PER_CYCLE=64
DESIGN_SPEED=10.  # m/s; sets the lattice time step
DESIGN_LATTICE_SPEED=.1  # lattice units at the design speed (Mach 0.17 there; 0.09 at the 5 m/s seen in the airway)
MOMENTUM_FILTER=.05  # per-step damping of period-2 momentum oscillations in thin passages; unsteady inertia becomes (1+alpha)*rho


def available():
    return BINARY.exists() and (ROOT/'build-id.txt').exists()


def boundary_faces(domain):
    """Open faces per side: owner cell (mask coordinates), outward direction, side.

    Nostril faces must face exterior air; the outlet only faces the inferior
    cut. This mirrors mesh_arrays() so both backends open the same faces.
    """
    mask,ports=domain['mask'],domain['ports']
    ext=domain.get('exterior',np.zeros_like(mask))
    pm=np.pad(mask,1);pp=np.pad(ports,1);pe=np.pad(ext,1)
    xyz=np.argwhere(mask)
    owners=[];directions=[];sides=[]
    for direction,d in enumerate(DIRECTIONS):
        nxt=xyz+1+d
        boundary=~pm[tuple(nxt.T)]
        pid=pp[tuple((xyz+1).T)].copy()
        pid[((pid==1)|(pid==2))&~pe[tuple(nxt.T)]]=0
        if direction!=3:pid[pid==3]=0
        keep=boundary&(pid>0)
        owners.append(xyz[keep]);directions.append(np.full(int(keep.sum()),direction));sides.append(pid[keep])
    owners=np.concatenate(owners);directions=np.concatenate(directions);sides=np.concatenate(sides)
    for side in [1,2,3]:
        if not np.any(sides==side):raise ValueError('An airway opening is blocked on the GPU grid')
    return owners,directions,sides


def input_arrays(domain,pad=2):
    """Flags, fluid cell ids and opening faces for the padded lattice.

    Every fluid cell is TYPE 0; solid padding prevents FluidX3D's implicit
    periodic box connections. One ghost cell outside each open face holds
    TYPE_E | (D3Q19 index of its fluid neighbour)<<2 | TYPE_X (nostril) or
    TYPE_Y (throat). D3Q19 indices 1..6 are +x,-x,+y,-y,+z,-z; the neighbour
    lies opposite to the face direction. Lattice indexing is F-order; fluid ids
    are listed in the viewer's C-order over the unpadded mask.
    """
    mask=domain['mask']
    shape=np.array(mask.shape)+2*pad
    flags=np.ones(shape,dtype='u1')
    xyz=np.argwhere(mask)+pad
    flags[tuple(xyz.T)]=0
    owners,directions,sides=boundary_faces(domain)
    src=owners+pad;dst=src+DIRECTIONS[directions]
    code=2|(((directions^1)+1)<<2)|np.where(sides<3,64,128)
    flags[tuple(dst.T)]=code.astype('u1')
    owner=np.ravel_multi_index(src.T,shape,order='F');ghost=np.ravel_multi_index(dst.T,shape,order='F')
    faces=np.stack([owner,ghost,directions//2,np.where(directions%2==0,1,-1),sides],axis=1).astype('<i4')
    ids=np.ravel_multi_index(xyz.T,shape,order='F').astype('<u4')
    return flags,ids,faces


def time_step(domain):
    # Larger steps raise lattice Mach (O(Ma^2) compressibility error) and
    # lattice viscosity together; 0.1 at the design speed keeps the error
    # under 1 % at airway speeds while cutting the step count by 40 %.
    return DESIGN_LATTICE_SPEED*(domain['h']/1000)/DESIGN_SPEED


def write_input(case,domain,pressure=30,period=4,cycles=2,dt=None,intervals=None):
    case=Path(case);case.mkdir(exist_ok=True,parents=True)
    flags,ids,faces=input_arrays(domain)
    h=domain['h']/1000
    # Physical air viscosity is kept; the time step follows the design speed
    # so lattice Mach stays small. Stability then relies on the LES closure.
    desired_dt=dt or time_step(domain)
    intervals=intervals or cycles*FRAMES_PER_CYCLE
    duration=period*cycles if period else .5
    stride=math.ceil(duration/intervals/desired_dt)
    dt=duration/intervals/stride
    peak=pressure*3/RHO*dt**2/h**2
    if peak>.04: raise ValueError('Requested pressure exceeds GPU density limit')
    flags.ravel(order='F').tofile(case/'flags.u8');ids.tofile(case/'cells.u32');faces.tofile(case/'faces.i32')
    values=[*flags.shape,len(ids),intervals,stride,h,dt,NU*dt/h**2,period,peak,len(faces),MOMENTUM_FILTER]
    (case/'input.txt').write_text(' '.join(map(str,values)))
    metadata=dict(backend='gpu-lbm',solver=VERSION,dependency='FluidX3D',dependencyCommit=COMMIT,
                  temporal='transient' if period else 'steady',direction='cycle' if period else 'inspiration',
                  pressureAmplitudePa=pressure,periodS=period,cycles=cycles,durationS=duration,dtS=dt,stepsPerFrame=stride,
                  framesPerCycle=FRAMES_PER_CYCLE if period else intervals,spacingMm=domain['h'],cells=len(ids),totalLatticeCells=int(flags.size),
                  densityKgM3=RHO,kinematicViscosityM2S=NU,latticeViscosity=NU*dt/h**2,
                  collision='D3Q19 TRT, FP32',momentumFilter=MOMENTUM_FILTER,
                  momentumFilterNote='Guo force -alpha*rho*(u - u_previous) per step: damps the period-2 wall-normal momentum mode of one- and two-cell passages; unsteady inertia is (1+alpha)*rho, steady flow unchanged',regime='LES: Smagorinsky-Lilly eddy viscosity on molecular air viscosity',
                  walls='rigid, halfway bounce-back on the voxel surface',initialCondition='at rest',
                  nostrils='ambient-pressure planar openings (collar through exterior air, see geometry nostrilCollars)',
                  throat='planar opening at ambient minus a smooth asymmetric resting-breath pressure drive',
                  waveform='resting-beta-1: inspiration 40%, expiration 60%; C1 smooth lobes, earlier expiratory peak and longer tail',
                  inspiratoryFraction=.4,breathModel='resting-beta-1',
                  waveformNote='Model boundary condition, not a patient breathing measurement. Equal signed pressure impulse does not prescribe tidal volume; flow remains a CFD result.',
                  boundaryMethod='planar openings: anti-bounce-back reservoir pressure on inflow, equilibrium outflow with the extrapolated velocity; flows and pressures read at the adjacent cells',
                  pressureNote='A is the nominal reservoir amplitude. Reported pressure differences are measured between the cells adjacent to the openings; the outflow opening loses part of A.',
                  surfaceRefinement=False,clinicalValidation=False)
    (case/'solver.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def run(case,on_progress=print,cancel=None,timeout=None):
    if not available():raise ValueError('GPU backend is not built. Run .venv/bin/python pipeline/gpu_cfd_build.py')
    started=time.monotonic()
    with (Path(case)/'log.gpu').open('w') as log:
        process=subprocess.Popen([str(BINARY),str(Path(case).resolve())],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                if cancel and cancel():raise RuntimeError('Cancelled')
                if timeout and time.monotonic()-started>timeout:raise RuntimeError('GPU benchmark time limit')
                if line.startswith('AIRWAY_FRAME '):on_progress(line.strip())
                if line.startswith('AIRWAY_ERROR '):on_progress(line.strip())
            if process.wait()!=0:raise ValueError('GPU numerical run failed; see log.gpu')
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
    return [json.loads(s) for s in (Path(case)/'history.jsonl').read_text().splitlines()]


def duct(width=6,length=60,h=2/3):
    mask=np.zeros((width+4,length+4,width+4),bool)
    mask[2:-2,2:-2,2:-2]=True
    ports=np.zeros(mask.shape,'u1');ports[:,2,:][mask[:,2,:]]=3
    # Split the single upstream patch into two labels for shared recording IO.
    ports[2:2+width//2,-3,2:-2]=1;ports[2+width//2:-2,-3,2:-2]=2
    exterior=np.zeros_like(mask);exterior[:,-2:,:]=True
    return dict(mask=mask,ports=ports,exterior=exterior,h=h,origin=np.zeros(3))


def read_frame(case,frame,n):
    f=Path(case)/'frames'/str(frame)
    v=np.fromfile(str(f)+'.velocity.f32','<f4').reshape(-1,3);p=np.fromfile(str(f)+'.pressure.f32','<f4')
    if len(v)!=n or len(p)!=n:raise ValueError('GPU frame size does not match the fluid domain')
    return v,p


def opening_flow_ml_s(v,owner_index,directions,h):
    """Volume flow into the airway through one planar opening, from the
    face-normal velocity of each cell adjacent to it."""
    axis=directions//2;inward=np.where(directions%2==0,-1.,1.)
    return float(np.sum(v[owner_index,axis]*inward)*h*h*1e6)


def frame_metrics(case,domain,frame,openings=None):
    """Flows (positive into the airway at the nostrils, out at the throat)
    and the nose-to-throat pressure difference, both read at the cells
    adjacent to the planar openings."""
    n=int(domain['mask'].sum());v,p=read_frame(case,frame,n);h=domain['h']/1000
    owners,directions,sides=openings or boundary_faces(domain)
    index=np.full(domain['mask'].shape,-1,np.int64);index[domain['mask']]=np.arange(n)
    owner_index=index[tuple(owners.T)]
    flows={s:opening_flow_ml_s(v,owner_index[sides==k],directions[sides==k],h)*(1 if s!='outlet' else -1) for s,k in [('L',1),('R',2),('outlet',3)]}
    pressure=float(p[owner_index[sides<3]].mean()-p[owner_index[sides==3]].mean())
    return dict(pressureDropPa=pressure,flowMlS=flows),v,p


def opening_summary(domain,openings=None):
    """Where each planar opening is, for the viewer's air outside the domain:
    face-centre centroid (mm, world), mean outward normal, area and the
    equivalent radius. Positions only; no flow is prescribed here."""
    owners,directions,sides=openings or boundary_faces(domain);h=domain['h'];origin=np.asarray(domain['origin'])
    out=[]
    for name,side,k in [('left_nostril','L',1),('right_nostril','R',2),('outlet','outlet',3)]:
        sel=sides==k;d=DIRECTIONS[directions[sel]]
        centres=(owners[sel]+.5+.5*d)*h+origin;normal=d.mean(0);normal=normal/np.linalg.norm(normal);area=float(h*h*sel.sum())
        out.append(dict(name=name,side=side,role='outlet' if k==3 else 'nostril',faces=int(sel.sum()),centreMm=centres.mean(0).tolist(),
                        normal=normal.tolist(),areaMm2=area,radiusMm=float(np.sqrt(area/np.pi)),facesMm=np.round(centres,3).tolist()))
    return out


def collect(case,domain,geometry):
    """Build the viewer result (field schema 3, fluid-C order) with GPU gates."""
    import trimesh
    case=Path(case);request=json.loads((case/'solver.json').read_text())
    history=[json.loads(s) for s in (case/'history.jsonl').read_text().splitlines()]
    mask=domain['mask'];n=int(mask.sum());h=domain['h']
    centres=(np.argwhere(mask)+.5)*h+domain['origin']
    bounds=np.array([centres.min(0)-h/2,centres.max(0)+h/2])
    aligned=np.allclose(trimesh.load(case/'airway.glb',force='mesh').bounds,bounds,atol=1e-4,rtol=0)
    if geometry['geometryHash']!=domain['audit']['geometryHash']:raise ValueError('Rebuilt GPU domain differs from the displayed geometry')
    openings=boundary_faces(domain)
    target=np.linspace(0,request['durationS'],request['cycles']*request['framesPerCycle']+1)
    frames=[];peak=0.;p99=0.;imbalance=0.;finite=True
    for i,record in enumerate(history):
        metrics,v,p=frame_metrics(case,domain,record['frame'],openings)
        finite&=bool(np.all(np.isfinite(v))and np.all(np.isfinite(p)))
        speed=np.linalg.norm(v,axis=1);peak=max(peak,float(speed.max()));p99=max(p99,float(np.percentile(speed,99)))
        flows=metrics['flowMlS']
        imbalance=max(imbalance,abs(flows['L']+flows['R']-flows['outlet']))
        frames.append(dict(timeS=float(record['timeS']),velocity=f'frames/{record["frame"]}.velocity.f32',flowMlS=flows,
                           pressureDropPa=metrics['pressureDropPa'],peakSpeedMS=float(speed.max()),p99SpeedMS=float(np.percentile(speed,99)),
                           maxLatticeMach=record['maxMach'],computeSeconds=record['computeSeconds']))
    times=np.array([f['timeS'] for f in frames])
    complete=len(times)==len(target) and np.allclose(times,target,atol=request['dtS'],rtol=0)
    peak_flow=max(abs(f['flowMlS']['outlet']) for f in frames) or 1e-9
    mass=imbalance/peak_flow
    # Mass: the lattice conserves mass exactly; the residual is the
    # second-order face-velocity quadrature at the openings (1-2 % on the
    # 0.7 mm patient lattice) plus O(Ma^2) compressibility. 3 % of peak flow
    # separates that from a lost or leaking opening.
    max_mach=max(r['maxMach'] for r in history);max_drho=max(r['maxDensityDeviation'] for r in history)
    gates=dict(mesh=bool(geometry.get('watertight'))and bool(geometry.get('windingConsistent')),coordinateAlignment=bool(aligned),
               completed=bool(complete) and history[-1]['frame']==len(target)-1,frames=bool(complete),
               stability=bool(finite and max_mach<.3 and max_drho<.05),massBalance=mass<.03)
    mask.astype('u1').tofile(case/'occupancy.u8')
    result=dict(geometryHash=geometry['geometryHash'],geometry=geometry,solver=request,gates=gates,
                status='converged' if all(gates.values()) else 'unconverged',
                acceptanceMeaning='numerical checks on a finite LES trajectory; not periodic convergence, mesh independence or validation',
                diagnostics=dict(steps=int(round(request['durationS']/request['dtS'])),maxLatticeMach=max_mach,maxDensityDeviation=max_drho,
                                 computeSeconds=history[-1]['computeSeconds']),
                massImbalanceFraction=mass,peakSpeedMS=peak,p99SpeedMS=p99,flowMlS=frames[0]['flowMlS'],
                pressureDropPa=max(f['pressureDropPa'] for f in frames),openings=opening_summary(domain,openings),
                meshIndependence=False,timeStepIndependence=False,periodicConvergence=False,clinicalValidation=False,thermalCFD=False,
                field=dict(schemaVersion=3,axisOrder='XYZ',storageOrder='fluid-C',dims=list(map(int,mask.shape)),
                           boxMin=np.asarray(domain['origin']).tolist(),spacingMm=float(h),fluidBoundsMm=bounds.tolist(),
                           occupancy='occupancy.u8',fluidCells=n,frames=frames,
                           format='little-endian float32 XYZ velocity per occupied cell, occupancy C-order (Z fastest)'),
                history=dict(timeS=times.tolist(),outletFlowMlS=[f['flowMlS']['outlet'] for f in frames],
                             pressureDropPa=[f['pressureDropPa'] for f in frames]))
    (case/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def solve(case,domain,pressure,period,cycles,on_progress=print,cancel=None):
    case=Path(case);geometry=json.loads((case/'geometry.json').read_text())
    write_input(case,domain,pressure=pressure,period=period,cycles=cycles)
    on_progress('Recording breathing on the GPU lattice; physical time is not wall-clock time')
    def progress(line):
        if line.startswith('AIRWAY_FRAME '):
            _,frame,t,elapsed=line.split();on_progress(f'Time = {float(t):.5f}')
        else:on_progress(line)
    run(case,on_progress=progress,cancel=cancel)
    on_progress('Checking and exporting recorded breathing frames')
    return collect(case,domain,geometry)


def poiseuille_check(pressure=.08,dt=2e-4,duration=.5,frames=10):
    """Pressure-driven square duct against the analytic laminar solution.

    Returns the measured/analytic flow ratio and the interior pressure
    gradient consistency, which separates opening losses from bulk physics.
    """
    d=duct();case=ROOT/'validation-poiseuille'
    import shutil;shutil.rmtree(case,ignore_errors=True)
    write_input(case,d,pressure=pressure,period=0,intervals=frames,dt=dt)
    run(case,on_progress=lambda s:None)
    metrics,v,p=frame_metrics(case,d,frames)
    flags,ids,_=input_arrays(d);xyz=np.array(np.unravel_index(ids,flags.shape,order='F')).T
    ys=sorted(set(xyz[:,1]));profile=np.array([p[xyz[:,1]==y].mean() for y in ys])
    interior_drop=float(profile[-1]-profile[0])  # nostril end (high y) minus throat end
    a=(d['mask'].shape[0]-4)*d['h']/1000;length=(len(ys)-1)*d['h']/1000
    analytic=lambda dp:dp/(RHO*NU*length)*a**4*.035144*1e6  # square duct, Shah & London
    # The duct's two "nostril" labels share one planar opening, so only the
    # throat cut is a meaningful flow measurement here.
    q=metrics['flowMlS']['outlet']
    out=dict(flowMlS=q,analyticAtNominalMlS=analytic(pressure*len(ys)/(len(ys)+2)),analyticAtInteriorDropMlS=analytic(interior_drop),
             interiorDropPa=interior_drop,nominalPa=pressure)
    out['bulkError']=q/out['analyticAtInteriorDropMlS']-1;out['openingLoss']=1-interior_drop/(pressure*len(ys)/(len(ys)+2))
    (case/'check.json').write_text(json.dumps(out,indent=2));return out


if __name__=='__main__':
    print(json.dumps(poiseuille_check(),indent=2))
