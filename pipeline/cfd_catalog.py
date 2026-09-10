"""Publish a static, version-matched recording catalog. No viewer-side jobs.
Run after offline CFD/thermal batches: .venv/bin/python pipeline/cfd_catalog.py
"""
import json
import os
from pathlib import Path
from cfd_service import CFDService,request_key,normalized

def build_catalog():
    service=CFDService();source=service.source();entries=[]
    for path in sorted(service.public.glob('*/result.json')):
        r=json.loads(path.read_text());request=r.get('request')
        if not request or r.get('status')!='converged' or not all(r.get('gates',{}).values()):continue
        key=request_key(normalized(request),source)
        if key!=path.parent.name or r.get('requestId')!=key or r.get('geometryHash')!=r.get('geometry',{}).get('geometryHash'):continue
        entry=dict(id=key,state='complete',request=request,message='Precomputed CFD recording',resultUrl=f'/data/cfd/{key}/result.json')
        thermal=path.parent/'thermal'/'result.json'
        if thermal.exists():
            t=json.loads(thermal.read_text())
            from cfd_thermal import VERSION as THERMAL_VERSION
            thermal_gates=t.get('gates',{})
            if t.get('version')==THERMAL_VERSION and t.get('flowRequestId')==key and t.get('geometryHash')==r['geometryHash'] and t.get('status')=='complete' and all(thermal_gates.get(k) is True for k in ('finiteAndBounded','linearResidual','energyLedger','periodic')) and all(thermal_gates.values()):
                entry['thermalUrl']=f'/data/cfd/{key}/thermal/result.json'
        entries.append(entry)
    data=dict(schemaVersion=1,sourceHash=source,entries=entries)
    target=service.public/'library.json';temp=target.with_suffix(f'.{os.getpid()}.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(target)
    service.executor.shutdown();service.mesh_executor.shutdown()
    print(f'Published {len(entries)} recordings; {sum("thermalUrl" in e for e in entries)} thermal recordings',flush=True)
    return data

if __name__=='__main__':build_catalog()
