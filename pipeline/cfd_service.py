"""Single-worker local CFD jobs; immutable request/result identity and atomic status."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
from cfd_transient import TRANSIENT_VERSION, write_transient_case, solve_transient
from cfd_geometry import ROOT, VERSION, prepare_source, build_domain, mesh_arrays, write_poly_mesh
from cfd_solver import SOLVER_VERSION, IMAGE, write_case, solve
import gpu_cfd
import config as C

DEFAULTS=dict(position='supine',elapsed=10,tau=5,responseL=.6,responseR=.6,cycle=0,gravity=.15,relief=0,
              operations={s:{k:0 for k in ['head','body','valve']} for s in ['L','R']})
LIMITS=dict(elapsed=(0,30),tau=(.5,15),responseL=(0,5),responseR=(0,5),cycle=(-2,2),gravity=(0,1),relief=(0,100))


def normalized(body):
    if not isinstance(body,dict): raise ValueError('Expected a JSON object')
    s=body.get('settings',{})
    if not isinstance(s,dict): raise ValueError('Invalid settings')
    out=json.loads(json.dumps(DEFAULTS))
    position=s.get('position','supine')
    if position not in ['supine','left','right','upright']: raise ValueError('Invalid position')
    out['position']=position
    def number(value,lo,hi):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError(f'Expected a finite number between {lo} and {hi}')
        return float(value)
    for key,(lo,hi) in LIMITS.items(): out[key]=number(s.get(key,out[key]),lo,hi)
    ops=s.get('operations',{})
    if not isinstance(ops,dict): raise ValueError('Invalid operations')
    for side in ['L','R']:
        if not isinstance(ops.get(side,{}),dict): raise ValueError('Invalid side operations')
        for key in ['head','body','valve']: out['operations'][side][key]=number(ops.get(side,{}).get(key,0),0,2)
    spacing=number(body.get('spacingMm',.7),.5,.7)
    if spacing not in [.5,.7]: raise ValueError('Choose a supported geometry grid')
    direction=body.get('direction','inspiration')
    if direction not in ['inspiration','expiration','cycle']: raise ValueError('Invalid breathing direction')
    mode=body.get('mode','steady')
    if direction=='cycle' and mode!='transient': raise ValueError('Cycle requires transient mode')
    if mode not in ['steady','transient']: raise ValueError('Invalid solver mode')
    backend=body.get('backend','openfoam')
    if backend not in ['openfoam','gpu-lbm']: raise ValueError('Unknown CFD backend')
    if backend=='gpu-lbm':
        # Pressure-driven breathing on the GPU lattice: no flow rate is
        # prescribed, so qMlS is not part of the request identity.
        if mode!='transient': raise ValueError('The GPU lattice backend records breathing cycles only')
        cycles=body.get('cycles',2)
        if cycles not in [1,2]: raise ValueError('Record one or two breathing cycles')
        refinement=body.get('refinement',1)
        if refinement not in [1,2]: raise ValueError('Lattice refinement must be 1 or 2 cells per geometry voxel')
        return dict(settings=out,backend=backend,pressurePa=number(body.get('pressurePa',30),5,200),spacingMm=spacing,
                    refinement=refinement,direction='cycle',mode=mode,periodS=number(body.get('periodS',4),2,10),cycles=cycles)
    request=dict(settings=out,qMlS=number(body.get('qMlS',250),10,500),spacingMm=spacing,direction=direction)
    if mode=='transient':
        request.update(mode=mode,direction='cycle',periodS=number(body.get('periodS',4),2,10),cycles=2)
    return request


def solver_version(request):
    if request.get('backend')=='gpu-lbm': return gpu_cfd.VERSION
    return TRANSIENT_VERSION if request.get('mode')=='transient' else SOLVER_VERSION


def request_key(request,source_hash):
    data=dict(request=request,sourceHash=source_hash,geometryVersion=VERSION,solverVersion=solver_version(request),
              image=None if request.get('backend')=='gpu-lbm' else IMAGE)
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class CFDService:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='airway-cfd')
        self.lock=threading.RLock();self.jobs={};self.cancelled=set()
        self.mesh_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='airway-mesh')
        self.mesh_futures={};self.job_futures={}
        self.public=C.DATA_DIR/'cfd';self.public.mkdir(parents=True,exist_ok=True)
        self.source_hash=None

    def source(self):
        # Hash the reconstruction once per server process; restart after replacing
        # input scans/segmentation. The source hash is part of every job key.
        if self.source_hash is None:
            import numpy as np
            with np.load(prepare_source()) as z: self.source_hash=str(z['source_hash'])
        return self.source_hash

    def set_status(self,key,**fields):
        with self.lock:
            self.jobs.setdefault(key,{}).update(fields,updatedAt=time.time())
            value=dict(self.jobs[key])
        folder=ROOT/key;folder.mkdir(exist_ok=True,parents=True)
        # The HTTP server is threaded; two simultaneous lookups may register
        # the same saved baseline. Give each atomic writer its own temp path.
        temp=folder/f'status.json.{threading.get_ident()}.tmp'
        temp.write_text(json.dumps(value,allow_nan=False));temp.replace(folder/'status.json')
        return value

    def publish(self,case,key,request,result):
        dest=self.public/key;dest.mkdir(parents=True,exist_ok=True)
        result=dict(result,requestId=key,request=request)
        names=['airway.glb','geometry.json','occupancy.u8']
        if result['solver'].get('temporal')=='transient':
            names += [f['velocity'] for f in result['field']['frames']]
        else: names += ['velocity.f32','pressure.f32']
        for name in names:
            (dest/name).parent.mkdir(exist_ok=True,parents=True)
            shutil.copy2(case/name,dest/name)
        (dest/'result.json').write_text(json.dumps(result,allow_nan=False))
        return '/data/cfd/'+key+'/result.json'

    def geometry(self,body):
        request=normalized(body)
        identity=dict(settings=request['settings'],spacingMm=request['spacingMm'],source=self.source(),version=VERSION)
        # The default two-fold subdivision keeps existing geometry keys stable.
        if request.get('refinement',2)!=2: identity['subdivisions']=request['refinement']
        if request.get('backend')=='gpu-lbm': identity['flatNostrils']=True
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        case=ROOT/'geometries'/key
        with self.lock:
            future=self.mesh_futures.get(key)
            if (case/'ready.json').exists():
                return json.loads((case/'ready.json').read_text()),None
            if future is None:
                if sum(not f.done() for f in self.mesh_futures.values())>=3:
                    raise ValueError('Geometry builder busy; wait for the current edits')
                future=self.mesh_executor.submit(self.build_geometry,case,key,request)
                self.mesh_futures[key]=future
        if future.done(): return future.result(),future
        return dict(id=key,state='meshing',message='Building exact CFD geometry'),future

    def build_geometry(self,case,key,request):
        domain=build_domain(request['settings'],spacing_mm=request['spacingMm'],subdivisions=request.get('refinement',2),
                            flat_nostrils=request.get('backend')=='gpu-lbm')
        arrays=mesh_arrays(domain);write_poly_mesh(domain,arrays,case)
        dest=self.public/'geometry'/key;dest.mkdir(parents=True,exist_ok=True)
        for name in ['airway.glb','geometry.json']: shutil.copy2(case/name,dest/name)
        result=dict(id=key,state='ready',geometryHash=domain['audit']['geometryHash'],
                    geometryUrl='/data/cfd/geometry/'+key+'/airway.glb',
                    metadataUrl='/data/cfd/geometry/'+key+'/geometry.json',case=str(case),
                    message='Exact solver geometry; no flow solved yet')
        (case/'ready.json').write_text(json.dumps(result));return result

    def register_baseline(self):
        # Existing locally verified runs can be consumed without another solve.
        candidates=['baseline-070-r4','baseline-050-r4']
        for name in candidates:
            case=ROOT/name
            if not (case/'result.json').exists(): continue
            result=json.loads((case/'result.json').read_text())
            if result['status'] not in ['converged','unconverged'] or result['geometry']['sourceHash']!=self.source(): continue
            if result['geometry'].get('version')!=VERSION: continue
            if result['solver']['solver']!=SOLVER_VERSION or result.get('field',{}).get('schemaVersion')!=2: continue
            if not result['gates'].get('coordinateAlignment') or not result['gates'].get('completed'): continue
            request=normalized(dict(spacingMm=result['geometry']['reconstructionSpacingMm'],
                                    qMlS=result['solver']['qMlS'],direction=result['solver']['direction']))
            key=request_key(request,self.source())
            if key in self.jobs: continue
            url=self.publish(case,key,request,result)
            accepted=result['status']=='converged' and all(result['gates'].values())
            self.set_status(key,id=key,state='complete' if accepted else 'unconverged',
                            message='Matching saved CFD result' if accepted else 'Matching numerical field did not converge',request=request,resultUrl=url)

    def lookup(self,body):
        request=normalized(body);key=request_key(request,self.source())
        self.register_baseline()
        if key not in self.jobs and (ROOT/key/'status.json').exists():
            saved=json.loads((ROOT/key/'status.json').read_text())
            if saved.get('state') in ['queued','meshing','solving']:
                # No worker for this job survived this service restart. Never
                # leave clients watching a persisted "solving" state forever.
                saved=dict(saved,state='failed',message='CFD service restarted before this job finished. Queue it again to retry.')
            if saved.get('state') in ['complete','failed','cancelled','unconverged']: self.jobs[key]=saved
        return key,request,self.jobs.get(key)

    def submit(self,body,queue_limit=3):
        key,request,existing=self.lookup(body)
        with self.lock:
            existing=self.jobs.get(key,existing)
            future=self.job_futures.get(key)
            if future is not None and not future.done(): return existing
            if existing and existing.get('state')=='complete': return existing
            pending=sum(not f.done() for f in self.job_futures.values())
            if pending>=queue_limit: raise ValueError('CFD queue is full; cancel or wait for a result')
            self.cancelled.discard(key)
            value=self.set_status(key,id=key,state='queued',request=request,message='Queued for local CFD')
            self.job_futures[key]=self.executor.submit(self.worker,key,request)
            return value

    def submit_batch(self,body):
        requests=body.get('requests')
        if not isinstance(requests,list) or not 1<=len(requests)<=20:
            raise ValueError('A comparison requires 1–20 requests')
        requests=[normalized(r) for r in requests]
        # Validate the entire batch before creating any job.
        with self.lock:
            pending=sum(j.get('state') in ['queued','meshing','solving'] for j in self.jobs.values())
            if pending+len(requests)>24: raise ValueError('Too many queued CFD runs; wait or cancel first')
            jobs=[self.submit(r,queue_limit=24) for r in requests]
        return dict(jobs=jobs)

    def worker(self,key,request):
        case=ROOT/key
        cancelled=lambda:key in self.cancelled
        try:
            if cancelled(): raise RuntimeError('Cancelled')
            self.set_status(key,state='meshing',message='Building the actual airway domain')
            mesh,future=self.geometry(request)
            if future is not None:
                while True:
                    if cancelled(): raise RuntimeError('Cancelled')
                    try:
                        mesh=future.result(timeout=.25)
                        break
                    except TimeoutError:
                        if future.done(): raise
            if cancelled(): raise RuntimeError('Cancelled')
            src=Path(mesh['case']);case.mkdir(exist_ok=True,parents=True)
            shutil.copytree(src/'constant/polyMesh',case/'constant/polyMesh',dirs_exist_ok=True)
            for name in ['domain.npz','geometry.json','airway.glb']: shutil.copy2(src/name,case/name)
            self.set_status(key,geometryUrl=mesh['geometryUrl'],geometryHash=mesh['geometryHash'])
            (case/'request.json').write_text(json.dumps(request,indent=2))
            if request.get('backend')=='gpu-lbm': return self.gpu_worker(case,key,request,cancelled)
            transient=request.get('mode')=='transient'
            if transient:
                cells=json.loads((case/'geometry.json').read_text())['cells']
                required=4*1024**3+cells*65*300
                if shutil.disk_usage(case).free<required: raise ValueError('Not enough free disk space for raw transient frames')
            if transient: write_transient_case(case,request['qMlS'],request['periodS'],request['cycles'])
            else: write_case(case,request['qMlS'],direction=request['direction'])
            if cancelled(): raise RuntimeError('Cancelled')
            self.set_status(key,state='solving',message='Checking the mesh')
            started=time.monotonic()
            def progress(text):
                if transient and text.startswith('Time ='):
                    t=float(text.split('=')[1]);elapsed=time.monotonic()-started
                    self.set_status(key,state='solving',message=f'Breathing CFD: {t:.5f} / {request["periodS"]*request["cycles"]:g} simulated seconds · {elapsed/60:.1f} min computing',
                                    simulatedTimeS=t,computeSeconds=elapsed)
                else: self.set_status(key,state='solving',message=text)
            result=(solve_transient if transient else solve)(case,on_progress=progress,cancel=cancelled)
            if cancelled(): raise RuntimeError('Cancelled')
            url=self.publish(case,key,request,result)
            self.set_status(key,state='complete' if result['status']=='converged' else 'unconverged',
                            message=('Transient numerical checks passed' if transient else 'CFD converged') if result['status']=='converged' else 'Solver checks did not all pass',resultUrl=url)
        except Exception as exc:
            self.set_status(key,state='cancelled' if cancelled() else 'failed',message=str(exc))

    def gpu_worker(self,case,key,request,cancelled):
        if not gpu_cfd.available(): raise ValueError('GPU backend is not built; run .venv/bin/python pipeline/gpu_cfd_build.py')
        geometry=json.loads((case/'geometry.json').read_text())
        required=2*1024**3+geometry['cells']*16*(request['cycles']*gpu_cfd.FRAMES_PER_CYCLE+1)*2
        if shutil.disk_usage(case).free<required: raise ValueError('Not enough free disk space for recorded lattice frames')
        self.set_status(key,state='solving',message='Preparing the GPU lattice')
        # The lattice needs the exterior-air labels, which the mesh files do
        # not store; rebuild the same domain and verify its hash before use.
        # Lattice geometry gives each nostril a planar opening (see
        # cfd_geometry.flatten_nostrils); the displayed mesh is built the same way.
        domain=build_domain(request['settings'],spacing_mm=request['spacingMm'],subdivisions=request['refinement'],flat_nostrils=True)
        if domain['audit']['geometryHash']!=geometry['geometryHash']: raise ValueError('Rebuilt lattice domain differs from the displayed geometry')
        started=time.monotonic();duration=request['periodS']*request['cycles']
        def progress(text):
            if text.startswith('Time ='):
                t=float(text.split('=')[1]);elapsed=time.monotonic()-started
                self.set_status(key,state='solving',message=f'GPU breathing LES: {t:.3f} / {duration:g} simulated seconds · {elapsed/60:.1f} min computing',
                                simulatedTimeS=t,computeSeconds=elapsed)
            else: self.set_status(key,state='solving',message=text)
        result=gpu_cfd.solve(case,domain,request['pressurePa'],request['periodS'],request['cycles'],on_progress=progress,cancel=cancelled)
        if cancelled(): raise RuntimeError('Cancelled')
        url=self.publish(case,key,request,result)
        self.set_status(key,state='complete' if result['status']=='converged' else 'unconverged',
                        message='GPU lattice checks passed' if result['status']=='converged' else 'Lattice checks did not all pass',resultUrl=url)

    def cancel(self,key):
        if key not in self.jobs: raise ValueError('Unknown job')
        if self.jobs[key].get('state') in ['complete','unconverged','failed','cancelled']:
            return dict(self.jobs[key])
        self.cancelled.add(key)
        return self.set_status(key,state='cancelled',message='Cancellation requested')
