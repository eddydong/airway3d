"""Time-resolved rigid-wall breathing CFD. Frames are actual pimpleFoam outputs.

Signed prescribed throat flux; ambient static pressure at both nostrils.
A completed transient trajectory is not a periodic, mesh/time independent or
clinically validated solution. Numerical checks are separate from those claims.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import numpy as np
import trimesh
from cfd_geometry import ROOT, foam_header
from cfd_solver import IMAGE, RHO, NU, write_case, run_foam, read_internal, read_patch, series

TRANSIENT_VERSION = 'OpenFOAM-v2412-resting-breath-2'
FRAMES_PER_CYCLE = 32


def waveform(t, q, period):
    """Smooth, asymmetric quiet-breath volume-flow multiplier.

    Inspiration occupies 40% of the cycle; passive expiration occupies 60%.
    Beta-shaped lobes are zero-slope at both reversals and have equal signed
    integrals, so the cycle returns to the same lung volume.
    """
    p=np.mod(np.asarray(t,dtype=float),period)/period;fi=.4
    x=np.where(p<fi,p/fi,(p-fi)/(1-fi));tail=1-x
    return q*np.where(p<fi,16*x*x*tail*tail,-56*fi/(1-fi)*x*x*tail**4)


def write_transient_case(case, q=250, period=4, cycles=2, max_co=.5, max_dt=None):
    write_case(case, q)
    duration=period*cycles
    # flowRateInletVelocity with extrapolateProfile=false uses signed Function1
    # values directly. Negative inlet rate gives positive outward throat flux.
    samples=np.linspace(0,duration,cycles*256+1)
    table='\n'.join(f'({t:.10g} {-waveform(t,q,period)*1e-6:.12g})' for t in samples)
    (case/'0/U').write_text(foam_header('U','volVectorField')+f'''dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField {{
 walls {{ type noSlip; }}
 left_nostril {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
 right_nostril {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
 outlet {{ type flowRateInletVelocity; extrapolateProfile false;
 volumetricFlowRate table ( {table} ); value uniform (0 0 0); }}
}}
''')
    functions=[]
    for patch in ['walls','left_nostril','right_nostril','outlet']:
        functions.append(f'''flux_{patch} {{ type surfaceFieldValue; libs (fieldFunctionObjects);
 regionType patch; name {patch}; operation sum; fields (phi); writeFields false;
 writeControl timeStep; writeInterval 1; log false; }}''')
    functions.append('''pressure_outlet { type surfaceFieldValue; libs (fieldFunctionObjects);
 regionType patch; name outlet; operation areaAverage; fields (p); writeFields false;
 writeControl timeStep; writeInterval 1; log false; }''')
    step=period/FRAMES_PER_CYCLE
    max_dt=max_dt or period/400
    (case/'system/controlDict').write_text(foam_header('controlDict')+f'''application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {duration:.10g};
deltaT {min(.0001,max_dt):.10g}; adjustTimeStep yes; maxCo {max_co}; maxDeltaT {max_dt:.10g};
writeControl adjustableRunTime; writeInterval {step:.10g}; purgeWrite 0;
writeFormat ascii; writePrecision 10; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable false;
functions {{ {chr(10).join(functions)} }}
''')
    (case/'system/fvSchemes').write_text(foam_header('fvSchemes')+'''
ddtSchemes { default backward; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss upwind;
 div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
    (case/'system/fvSolution').write_text(foam_header('fvSolution')+'''
solvers {
 p { solver GAMG; tolerance 1e-9; relTol 0; smoother GaussSeidel;
     agglomerator faceAreaPair; nCellsInCoarsestLevel 20; mergeLevels 1; }
 pFinal { $p; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; }
 UFinal { $U; }
}
PIMPLE { nOuterCorrectors 12; nCorrectors 2; nNonOrthogonalCorrectors 0;
 momentumPredictor yes;
 residualControl { p { tolerance 1e-5; relTol 0; } U { tolerance 1e-5; relTol 0; } }
}
''')
    request=dict(solver=TRANSIENT_VERSION,image=IMAGE,qMlS=q,periodS=period,cycles=cycles,
                 durationS=duration,direction='cycle',temporal='transient',waveform='resting-beta-1 throat volume flux; positive = inspiration',
                 inspiratoryFraction=.4,breathModel='resting-beta-1',
                 densityKgM3=RHO,kinematicViscosityM2S=NU,regime='laminar',walls='rigid, no slip',
                 initialCondition='at rest',nostrils='0 Pa gauge static pressure',
                 throat='signed uniform normal velocity prescribed by volume flux',
                 maxCo=max_co,courantLimit=1.0,maxDeltaTS=max_dt,framesPerCycle=FRAMES_PER_CYCLE,
                 convection='Gauss upwind (first order)',timeScheme='backward (second order after startup)')
    (case/'solver.json').write_text(json.dumps(request,indent=2))


def transient_diagnostics(log,duration):
    blocks=re.split(r'^Time = ([\d.eE+-]+)\s*$',log,flags=re.M)
    last=[];times=[];max_final=0.;max_outer=0.;outer_ok=True
    for i in range(1,len(blocks)-1,2):
        t=float(blocks[i]);block=blocks[i+1];times.append(t)
        # Last outer corrector, not the first linear solve or a later time.
        outer=re.split(r'PIMPLE: iteration \d+',block)[-1]
        values=re.findall(r'Solving for (\w+), Initial residual = ([\deE.+-]+), Final residual = ([\deE.+-]+)',outer)
        by={}
        for field,initial,final in values:
            by.setdefault(field,[]).append(float(initial));max_final=max(max_final,float(final))
        ok=all(k in by for k in ['p','Ux','Uy','Uz'])
        if ok:
            # Use the first pressure solve in the final outer loop; a second
            # pressure correction alone cannot establish outer convergence.
            residual=max(by[k][0] for k in ['p','Ux','Uy','Uz']);max_outer=max(max_outer,residual)
            ok=residual<1e-5
        outer_ok &= ok
        last.append(dict(timeS=t,passed=bool(ok)))
    co=[float(x) for x in re.findall(r'Courant Number mean: [\deE.+-]+ max: ([\deE.+-]+)',log)]
    return dict(completed='\nEnd\n' in log and bool(times) and abs(times[-1]-duration)<1e-6,
                steps=len(times),outerResidualsPassed=bool(times) and outer_ok,
                failedSteps=sum(not x['passed'] for x in last),maxOuterResidual=max_outer,
                maxLinearResidual=max_final,maxCourant=max(co,default=1e30))


def collect_transient(case):
    geometry=json.loads((case/'geometry.json').read_text());request=json.loads((case/'solver.json').read_text())
    log=(case/'log.pimpleFoam').read_text();meshlog=(case/'log.checkMesh').read_text()
    diagnostics=transient_diagnostics(log,request['durationS'])
    with np.load(case/'domain.npz') as z: a={k:z[k] for k in z.files}
    mask=a['mask'];n=int(mask.sum());centres=(np.argwhere(mask)+.5)*a['h']+a['origin']
    bounds=np.array([centres.min(0)-a['h']/2,centres.max(0)+a['h']/2])
    aligned=np.allclose(a['centres'],centres,atol=1e-8,rtol=0) and np.allclose(trimesh.load(case/'airway.glb',force='mesh').bounds,bounds,atol=1e-4,rtol=0)
    flux={k:series(case,'flux_'+k) for k in ['walls','left_nostril','right_nostril','outlet']}
    t=flux['outlet'][:,0];qpeak=request['qMlS'];qout=flux['outlet'][:,1]*1e6
    if not all(np.array_equal(v[:,0],t) for v in flux.values()): raise ValueError('Diagnostic times disagree')
    mass=float(np.max(abs(sum(v[:,1] for v in flux.values())))*1e6/qpeak)
    leak=float(np.max(abs(flux['walls'][:,1]))*1e6)
    rate_error=float(np.max(abs(qout-waveform(t,qpeak,request['periodS'])))/qpeak)
    frame_dirs=sorted([p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9.]+',p.name)],key=lambda p:float(p.name))
    times=np.array([float(p.name) for p in frame_dirs])
    target=np.linspace(0,request['durationS'],request['cycles']*request['framesPerCycle']+1)
    complete_frames=len(times)==len(target) and np.allclose(times,target,atol=1e-6,rtol=0)
    out=case/'frames';out.mkdir(exist_ok=True)
    owners=a['boundaryOwner'][a['boundaryPatch']==3];counts={p['name']:p['nFaces'] for p in geometry['patches']}
    frames=[];peak=0.;p99=0.
    for i,folder in enumerate(frame_dirs):
        u=read_internal(folder/'U',n,3);p=read_internal(folder/'p',n)[:,0]*RHO
        speed=np.linalg.norm(u,axis=1);peak=max(peak,float(speed.max()));p99=max(p99,float(np.percentile(speed,99)))
        name=f'frames/u-{i:04d}.f32';u.astype('<f4').tofile(case/name)
        instant={k:0. if float(folder.name)==0 else float(read_patch(folder/'phi',patch,counts[patch]).sum())*1e6
                 for k,patch in [('L','left_nostril'),('R','right_nostril'),('outlet','outlet')]}
        frames.append(dict(timeS=float(folder.name),velocity=name,flowMlS=dict(L=-instant['L'],R=-instant['R'],outlet=instant['outlet']),
                           pressureDropPa=-float(p[owners].mean()),peakSpeedMS=float(speed.max()),p99SpeedMS=float(np.percentile(speed,99))))
    gates=dict(mesh='Mesh OK.' in meshlog and 'Failed' not in meshlog,coordinateAlignment=bool(aligned),
               completed=diagnostics['completed'],frames=bool(complete_frames),residuals=diagnostics['outerResidualsPassed'],
               courant=diagnostics['maxCourant']<=request['courantLimit'],massBalance=mass<.001,wallLeak=leak<1e-9,requestedFlow=rate_error<.001)
    mask.astype('u1').tofile(case/'occupancy.u8')
    result=dict(geometryHash=geometry['geometryHash'],geometry=geometry,solver=request,gates=gates,
                status='converged' if all(gates.values()) else 'unconverged',
                acceptanceMeaning='numerical checks on a finite transient trajectory; not periodic convergence',
                diagnostics=diagnostics,massImbalanceFraction=mass,wallLeakMlS=leak,requestedFlowErrorFraction=rate_error,
                peakSpeedMS=peak,p99SpeedMS=p99,flowMlS=frames[0]['flowMlS'],
                meshIndependence=False,timeStepIndependence=False,periodicConvergence=False,clinicalValidation=False,thermalCFD=False,
                field=dict(schemaVersion=3,axisOrder='XYZ',storageOrder='fluid-C',dims=list(map(int,mask.shape)),
                           boxMin=a['origin'].tolist(),spacingMm=float(a['h']),fluidBoundsMm=bounds.tolist(),
                           occupancy='occupancy.u8',fluidCells=n,frames=frames,
                           format='little-endian float32 XYZ velocity per occupied cell, occupancy C-order (Z fastest)'),
                history=dict(timeS=t.tolist(),outletFlowMlS=qout.tolist()))
    (case/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def solve_transient(case,on_progress=print,cancel=None):
    log=run_foam(case,'checkMesh',['-allGeometry','-allTopology'],cancel=cancel)
    if 'Mesh OK.' not in log or 'Failed' in log: raise ValueError('Mesh quality/topology check failed')
    n=json.loads((case/'geometry.json').read_text())['cells'];parallel=6 if n>50000 else 1
    if parallel>1:
        (case/'system/decomposeParDict').write_text(foam_header('decomposeParDict')+'numberOfSubdomains 6; method scotch;\n')
        run_foam(case,'decomposePar',['-force'],cancel=cancel)
    on_progress('Solving time-dependent Navier–Stokes; physical time is not wall-clock time')
    run_foam(case,'pimpleFoam',on_progress=on_progress,cancel=cancel,parallel=parallel)
    if parallel>1:
        on_progress('Reconstructing velocity frames');run_foam(case,'reconstructPar',cancel=cancel)
    on_progress('Checking and exporting recorded breathing frames')
    return collect_transient(case)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--copy-mesh')
    p.add_argument('--q',type=float,default=250);p.add_argument('--period',type=float,default=4)
    p.add_argument('--cycles',type=int,default=2);p.add_argument('--max-dt',type=float);p.add_argument('--collect',action='store_true')
    args=p.parse_args();case=ROOT/args.case
    if args.copy_mesh:
        src=ROOT/args.copy_mesh;case.mkdir(exist_ok=True,parents=True)
        shutil.copytree(src/'constant/polyMesh',case/'constant/polyMesh',dirs_exist_ok=True)
        for name in ['domain.npz','geometry.json','airway.glb']:shutil.copy2(src/name,case/name)
    if args.collect:r=collect_transient(case)
    else:
        write_transient_case(case,args.q,args.period,args.cycles,max_dt=args.max_dt);r=solve_transient(case)
    print(json.dumps({k:r[k] for k in ['status','gates','diagnostics']},indent=2))
