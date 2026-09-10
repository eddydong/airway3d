"""Precompute the CFD scenario library so the viewer replays it instantly.

The library is the CFD comparison grid: every body position times the
no-intervention reference plus any saved plans, all sharing physiology,
grid, waveform and driving pressure. Each scenario is a separate recorded
breathing run keyed by its full request, so the viewer's lookup finds it
without solving. This offline process owns its worker queue. The viewer is
read-only and cannot submit jobs. Nothing here estimates or interpolates between scenarios.

    .venv/bin/python pipeline/cfd_library.py                 # queue and wait
    .venv/bin/python pipeline/cfd_library.py --plans plans.json --positions supine,upright
    .venv/bin/python pipeline/cfd_library.py --status        # report only

`plans.json` is a list of {"name", "settings"} objects, the same shape the
viewer keeps under localStorage key `airway3d-plans-v1`.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

POSITIONS=['supine','left','right','upright']


def scenario_requests(base,plans,positions):
    """No-intervention reference first, then each distinct plan, per position."""
    from cfd_service import DEFAULTS
    reference=dict(name='Pre · no intervention',settings=dict(operations=json.loads(json.dumps(DEFAULTS['operations'])),relief=0))
    seen=set();unique=[]
    for plan in [reference,*plans]:
        s=plan['settings'];identity=json.dumps([s.get('operations'),s.get('relief',0)],sort_keys=True)
        if identity in seen:continue
        seen.add(identity);unique.append(plan)
    out=[]
    for position in positions:
        for i,plan in enumerate(unique):
            settings=dict(base.get('settings',{}),position=position,operations=plan['settings']['operations'],relief=plan['settings'].get('relief',0))
            out.append(dict(name=plan['name'],position=position,baseline=i==0,request=dict(base,settings=settings)))
    return out


class LocalService:
    def __init__(self):
        from cfd_service import CFDService
        self.service=CFDService()
    def lookup(self,request):
        key,normalized,value=self.service.lookup(request)
        return value or dict(id=key,state='missing',request=normalized)
    def submit(self,requests):return self.service.submit_batch(dict(requests=requests))['jobs']
    def poll(self,key):return self.service.jobs.get(key) or {'state':'missing'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend',default='gpu-lbm',choices=['gpu-lbm','openfoam'])
    p.add_argument('--refinement',type=int,default=1,choices=[1,2],help='lattice cells per geometry voxel (GPU backend)')
    p.add_argument('--pressure',type=float,default=30,help='nominal throat pressure amplitude, Pa (GPU backend)')
    p.add_argument('--q',type=float,default=250,help='peak throat flow, mL/s (OpenFOAM backend)')
    p.add_argument('--period',type=float,default=4);p.add_argument('--cycles',type=int,default=2,choices=[1,2])
    p.add_argument('--grid',type=float,default=.7,choices=[.5,.7],help='reconstruction grid, mm')
    p.add_argument('--positions',default=','.join(POSITIONS))
    p.add_argument('--plans',type=Path,help='JSON list of {name, settings} saved plans')
    p.add_argument('--status',action='store_true',help='report library state without queueing')
    args=p.parse_args(argv)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    base=dict(backend='gpu-lbm',mode='transient',periodS=args.period,cycles=args.cycles,pressurePa=args.pressure,spacingMm=args.grid,refinement=args.refinement) \
        if args.backend=='gpu-lbm' else dict(mode='transient',periodS=args.period,qMlS=args.q,spacingMm=args.grid)
    plans=json.loads(args.plans.read_text()) if args.plans else []
    positions=[s.strip() for s in args.positions.split(',') if s.strip()]
    unknown=[s for s in positions if s not in POSITIONS]
    if unknown:p.error(f'unknown positions {unknown}; choose from {POSITIONS}')
    scenarios=scenario_requests(base,plans,positions)
    service,where=LocalService(),'offline process'
    print(f'{len(scenarios)} scenarios via {where}')
    for s in scenarios:
        s['job']=service.lookup(s['request']);s['id']=s['job']['id']
        print(f"  {s['position']:8s} {s['name']:32s} {s['job'].get('state','missing'):12s} {s['job'].get('message','')}")
    if args.status:
        from cfd_catalog import build_catalog
        build_catalog();return 0
    missing=[s for s in scenarios if s['job'].get('state') in ['missing','cancelled','failed']]
    if missing:
        # Unconverged recordings stay for review; identical numerics would only repeat them.
        jobs=service.submit([s['request'] for s in missing])
        print(f'queued {len(jobs)} runs')
    else:print('nothing to queue')
    pending=set(s['id'] for s in scenarios)
    while pending:
        time.sleep(15)
        for s in scenarios:
            if s['id'] not in pending:continue
            job=service.poll(s['id']);state=job.get('state')
            if state in ['complete','unconverged','failed','cancelled','missing']:
                pending.discard(s['id']);print(f"  {s['position']:8s} {s['name']:32s} {state:12s} {job.get('message','')}",flush=True)
            elif state=='solving':print(f"  … {s['position']} {s['name']}: {job.get('message','')}",end='\r',flush=True)
    accepted=sum(service.poll(s['id']).get('state')=='complete' for s in scenarios)
    print(f'{accepted}/{len(scenarios)} scenarios accepted and ready for instant replay')
    from cfd_catalog import build_catalog
    build_catalog()
    return 0 if accepted==len(scenarios) else 1


if __name__=='__main__':sys.exit(main())
