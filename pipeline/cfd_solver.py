"""Pinned OpenFOAM 2412 viscous incompressible finite-volume cases.

The solver runs locally, without networking, and can access only work/cfd.
No model result is published as converged unless mesh, residual, flow balance
and pressure stability checks pass. Rigid walls and laminar flow are assumptions.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
import numpy as np
import trimesh
from cfd_geometry import ROOT, foam_header, build_domain, mesh_arrays, write_poly_mesh

IMAGE = 'microfluidica/openfoam@sha256:61157ed95443ff7965f5697c50144de4e2b9853ddd610b490d6e138b5871e6cf'
RHO, NU = 1.2, 1.5e-5
SOLVER_VERSION = 'OpenFOAM-v2412-laminar-2'


def write_case(case, q=250, iterations=2400, direction='inspiration'):
    for folder in ['0','system','constant']: (case/folder).mkdir(parents=True,exist_ok=True)
    def write(folder,name,body,cls='dictionary'):
        (case/folder/name).write_text(foam_header(name,cls)+body)
    rate=q*1e-6
    outlet_type='flowRateOutletVelocity' if direction=='inspiration' else 'flowRateInletVelocity'
    write('0','U',f'''dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField {{
 walls {{ type noSlip; }}
 left_nostril {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
 right_nostril {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
 outlet {{ type {outlet_type}; volumetricFlowRate {rate:.10g}; value uniform (0 0 0); }}
}}
''','volVectorField')
    write('0','p','''dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField {
 walls { type zeroGradient; }
 left_nostril { type fixedValue; value uniform 0; }
 right_nostril { type fixedValue; value uniform 0; }
 outlet { type zeroGradient; }
}
''','volScalarField')
    write('constant','transportProperties',f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {NU};\n')
    write('constant','turbulenceProperties','simulationType laminar;\n')
    functions=[]
    for patch in ['walls','left_nostril','right_nostril','outlet']:
        functions.append(f'''flux_{patch} {{ type surfaceFieldValue; libs (fieldFunctionObjects);
 regionType patch; name {patch}; operation sum; fields (phi); writeFields false;
 writeControl timeStep; writeInterval 10; log false; }}''')
        if patch!='walls':
            functions.append(f'''pressure_{patch} {{ type surfaceFieldValue; libs (fieldFunctionObjects);
 regionType patch; name {patch}; operation areaAverage; fields (p); writeFields false;
 writeControl timeStep; writeInterval 10; log false; }}''')
    functions.append('''shear { type wallShearStress; libs (fieldFunctionObjects); patches (walls);
 writeControl writeTime; log false; }''')
    write('system','controlDict',f'''application simpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {iterations}; deltaT 1;
writeControl timeStep; writeInterval 100; purgeWrite 2;
writeFormat ascii; writePrecision 10; writeCompression off;
timeFormat general; timePrecision 8; runTimeModifiable false;
functions {{ {chr(10).join(functions)} }}
''')
    write('system','fvSchemes','''
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss upwind;
 div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
    write('system','fvSolution','''
solvers {
 p { solver GAMG; tolerance 1e-8; relTol 0.01; smoother GaussSeidel;
     agglomerator faceAreaPair; nCellsInCoarsestLevel 20; mergeLevels 1; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.01; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent no;
 residualControl { p 1e-5; U 1e-5; } }
relaxationFactors { fields { p 0.3; } equations { U 0.7; } }
''')
    request=dict(solver=SOLVER_VERSION,image=IMAGE,qMlS=q,direction=direction,iterations=iterations,
                 densityKgM3=RHO,kinematicViscosityM2S=NU,regime='laminar',walls='rigid, no slip',
                 inlet='both nostrils: 0 Pa gauge static pressure',outlet='prescribed total volumetric flow',
                 convection='bounded Gauss upwind (first order)',temporal='steady')
    (case/'solver.json').write_text(json.dumps(request,indent=2))


def run_foam(case,command,extra=(),on_progress=None,cancel=None,parallel=1):
    relative=case.resolve().relative_to(ROOT.resolve())
    executable=['mpirun','--allow-run-as-root','-np',str(parallel),command,'-parallel'] if parallel>1 else [command]
    container='airway-cfd-'+uuid.uuid4().hex[:16]
    args=['docker','run','--name',container,'--rm','--network','none','-v',f'{ROOT.resolve()}:/cases',IMAGE,
          *executable,'-case',f'/cases/{relative}',*extra]
    with (case/f'log.{command}').open('w') as log:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                if cancel and cancel():
                    subprocess.run(['docker','stop','-t','1',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
                    process.terminate(); raise RuntimeError('Cancelled')
                if on_progress and (line.startswith('Time =') or 'FOAM FATAL' in line):
                    on_progress(line.strip())
            if process.wait()!=0:
                raise RuntimeError(f'{command} failed; see {case/f"log.{command}"}')
        finally:
            if process.poll() is None: process.terminate(); process.wait()
    return (case/f'log.{command}').read_text()


def series(case,name):
    files=sorted((case/'postProcessing'/name).glob('*/surfaceFieldValue.dat'))
    if not files: raise ValueError(f'Missing solver diagnostic: {name}')
    rows=[]
    for f in files:
        for line in f.read_text().splitlines():
            if line and not line.startswith('#'):
                rows.append([float(x) for x in line.split()])
    return np.array(rows)


def read_internal(path,n,components=1):
    text=path.read_text()
    uniform=re.search(r'internalField\s+uniform\s+([^;]+);',text)
    if uniform:
        value=np.fromstring(uniform[1].strip('() '),sep=' ')
        return np.tile(value,(n,1)).reshape((n,components))
    match=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not match or int(match[1])!=n: raise ValueError(f'Invalid field size in {path}')
    data=np.fromstring(match[2].replace('(',' ').replace(')',' '),sep=' ')
    if len(data)!=n*components or not np.isfinite(data).all(): raise ValueError(f'Invalid field values in {path}')
    return data.reshape(n,components)


def read_patch(path,name,n,components=1):
    text=path.read_text()
    patch=re.search(r'\b'+re.escape(name)+r'\s*\{([^{}]*)\}',text,re.S)
    if not patch: raise ValueError(f'Missing patch {name} in {path}')
    uniform=re.search(r'\bvalue\s+uniform\s+([^;]+);',patch[1])
    if uniform:
        value=np.fromstring(uniform[1].strip('() '),sep=' ')
        return np.tile(value,(n,1)).reshape(n,components)
    match=re.search(r'\bvalue\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',patch[1],re.S)
    if not match or int(match[1])!=n: raise ValueError(f'Invalid patch size in {path}: {name}')
    data=np.fromstring(match[2].replace('(',' ').replace(')',' '),sep=' ')
    if len(data)!=n*components or not np.isfinite(data).all(): raise ValueError('Invalid patch values')
    return data.reshape(n,components)


def iteration_diagnostics(log,export_time):
    """Diagnostics for exactly the saved fields, with SIMPLE outer residuals."""
    blocks=re.split(r'^Time = ([\d.eE+-]+)\s*$',log,flags=re.M)
    block=next((blocks[i+1] for i in range(1,len(blocks)-1,2)
                if float(blocks[i])==float(export_time)), '')
    residuals={};linear_residuals={}
    for field,initial,final in re.findall(r'Solving for (\w+), Initial residual = ([\deE.+-]+), Final residual = ([\deE.+-]+)',block):
        residuals[field]=float(initial);linear_residuals[field]=float(final)
    return dict(residuals=residuals,linearResiduals=linear_residuals,
                residualsPassed=all(residuals.get(k,1)<1e-5 for k in ['p','Ux','Uy','Uz']),
                completed='\nEnd\n' in log)


def collect(case):
    geometry=json.loads((case/'geometry.json').read_text()); request=json.loads((case/'solver.json').read_text())
    log=(case/'log.simpleFoam').read_text(); meshlog=(case/'log.checkMesh').read_text()
    latest=max((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9.]+',p.name)),key=lambda p:float(p.name))
    with np.load(case/'domain.npz') as z: arrays={k:z[k] for k in z.files}
    n=int(arrays['mask'].sum());p=read_internal(latest/'p',n)[:,0]*RHO
    flows={p:series(case,'flux_'+p) for p in ['walls','left_nostril','right_nostril','outlet']}
    patch_counts={p['name']:p['nFaces'] for p in geometry['patches']}
    q={p:float(read_patch(latest/'phi',p,patch_counts[p]).sum())*1e6 for p in flows}
    pressure=series(case,'pressure_outlet')
    pressure=pressure[pressure[:,0]<=float(latest.name)]
    # zeroGradient pressure patches omit 'value' when written. Their boundary
    # value equals the adjacent cell, which is exact for this orthogonal mesh.
    outlet_owners=arrays['boundaryOwner'][arrays['boundaryPatch']==3]
    dp=abs(float(p[outlet_owners].mean()))
    # Five samples span 40 SIMPLE iterations; exclude startup history.
    tail=pressure[-min(5,len(pressure)):,1]*RHO
    stability=float(np.ptp(tail)/max(abs(np.mean(tail)),1e-8))
    # SIMPLE convergence uses initial (outer-iteration) residuals. A tiny final
    # linear-solver residual alone does not establish nonlinear convergence.
    # Use only diagnostics at the exported time, never a later unsaved iterate.
    diagnostics=iteration_diagnostics(log,latest.name)
    residuals=diagnostics['residuals'];linear_residuals=diagnostics['linearResiduals']
    expected_centres=(np.argwhere(arrays['mask'])+.5)*arrays['h']+arrays['origin']
    aligned=bool(np.allclose(arrays['centres'],expected_centres,atol=1e-8,rtol=0))
    bounds=np.array([expected_centres.min(axis=0)-arrays['h']/2,
                     expected_centres.max(axis=0)+arrays['h']/2])
    surface=trimesh.load(case/'airway.glb',force='mesh')
    aligned=aligned and bool(np.allclose(surface.bounds,bounds,atol=1e-4,rtol=0))
    balance=abs(sum(q.values()))/max(abs(q['outlet']),1e-12)
    gates=dict(mesh='Mesh OK.' in meshlog and 'Failed' not in meshlog,
               residuals=diagnostics['residualsPassed'],
               massBalance=balance<.001,wallLeak=abs(q['walls'])<1e-9,
               pressureStable=len(tail)>=5 and stability<.01,
               requestedFlow=abs(abs(q['outlet'])-request['qMlS'])/request['qMlS']<.001,
               coordinateAlignment=aligned,completed=diagnostics['completed'])
    velocity=read_internal(latest/'U',n,3)
    speed=np.linalg.norm(velocity,axis=1)
    shear=read_patch(latest/'wallShearStress','walls',patch_counts['walls'],3)*RHO
    result=dict(geometryHash=geometry['geometryHash'],geometry=geometry,solver=request,
                status='converged' if all(gates.values()) else 'unconverged',gates=gates,
                iterations=int(float(latest.name)),residuals=residuals,linearResiduals=linear_residuals,massImbalanceFraction=balance,
                wallLeakMlS=abs(q['walls']),pressureVariationFraction=stability,
                pressureDropPa=dp,resistancePaSMl=dp/request['qMlS'],
                flowMlS={'L':abs(q['left_nostril']),'R':abs(q['right_nostril']),'outlet':abs(q['outlet'])},
                peakSpeedMS=float(speed.max()),p99SpeedMS=float(np.percentile(speed,99)),
                peakWallShearPa=float(np.linalg.norm(shear,axis=1).max()),
                pressureRangePa=[float(p.min()),float(p.max())],
                meshIndependence=False,clinicalValidation=False,thermalCFD=False,
                history={'iteration':pressure[:,0].tolist(),'pressureDropPa':(abs(pressure[:,1])*RHO).tolist()})
    # Exact native cell values and occupancy; renderer may interpolate but cannot warp.
    shape=arrays['mask'].shape; field=np.zeros((*shape,3),dtype='<f4'); field[arrays['mask']]=velocity
    field.tofile(case/'velocity.f32'); arrays['mask'].astype('u1').tofile(case/'occupancy.u8')
    pf=np.zeros(shape,dtype='<f4');pf[arrays['mask']]=p;pf.tofile(case/'pressure.f32')
    ports=[]
    for pid,name in [(1,'L'),(2,'R'),(3,'outlet')]:
        owners=arrays['boundaryOwner'][arrays['boundaryPatch']==pid]; unique=np.unique(owners)
        ports.append(dict(name=name,centres=arrays['centres'][unique].round(5).tolist(),
                          flowMlS=abs(q[{'L':'left_nostril','R':'right_nostril','outlet':'outlet'}[name]])))
    result['field']=dict(dims=list(map(int,shape)),boxMin=arrays['origin'].tolist(),spacingMm=float(arrays['h']),
                         velocity='velocity.f32',occupancy='occupancy.u8',pressure='pressure.f32',ports=ports,
                         axisOrder='XYZ',storageOrder='C',schemaVersion=2,fluidBoundsMm=bounds.tolist(),
                         format='little-endian float32 XYZ vectors; cells ordered X,Y,Z (Z fastest)')
    (case/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def solve(case,on_progress=print,cancel=None):
    on_progress('Checking finite-volume mesh')
    log=run_foam(case,'checkMesh',['-allGeometry','-allTopology'],cancel=cancel)
    if 'Mesh OK.' not in log or 'Failed' in log: raise ValueError('Mesh quality/topology check failed')
    n=json.loads((case/'geometry.json').read_text())['cells']
    parallel=6 if n>50000 else 1
    if parallel>1:
        (case/'system/decomposeParDict').write_text(foam_header('decomposeParDict')+'numberOfSubdomains 6; method scotch;\n')
        on_progress('Partitioning mesh across 6 CPUs')
        run_foam(case,'decomposePar',['-force'],cancel=cancel)
    on_progress('Solving viscous Navier–Stokes')
    run_foam(case,'simpleFoam',on_progress=on_progress,cancel=cancel,parallel=parallel)
    if parallel>1:
        on_progress('Reconstructing solved fields')
        run_foam(case,'reconstructPar',['-latestTime'],cancel=cancel)
    result=collect(case)
    on_progress(f"{result['status']}: {result['pressureDropPa']:.3f} Pa; mass error {result['massImbalanceFraction']:.3g}")
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='baseline-070');parser.add_argument('--spacing',type=float,default=.7);parser.add_argument('--subdivisions',type=int,default=2)
    parser.add_argument('--q',type=float,default=250);parser.add_argument('--iterations',type=int,default=2400)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--collect',action='store_true')
    parser.add_argument('--solve',action='store_true',help='Run OpenFOAM after preparing a case')
    parser.add_argument('--copy-mesh',help='Reuse an existing immutable mesh for another boundary condition')
    args=parser.parse_args();case=ROOT/args.case
    if args.copy_mesh:
        import shutil
        source=ROOT/args.copy_mesh;case.mkdir(exist_ok=True,parents=True)
        shutil.copytree(source/'constant/polyMesh',case/'constant/polyMesh',dirs_exist_ok=True)
        for name in ['domain.npz','geometry.json','airway.glb']: shutil.copy2(source/name,case/name)
    if args.prepare:
        domain=build_domain(spacing_mm=args.spacing,subdivisions=args.subdivisions);arrays=mesh_arrays(domain)
        case.mkdir(parents=True,exist_ok=True);write_poly_mesh(domain,arrays,case)
        if not args.solve: raise SystemExit(0)
    if args.collect: collect(case)
    elif args.solve or not args.prepare:
        write_case(case,args.q,args.iterations);solve(case)
