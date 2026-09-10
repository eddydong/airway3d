"""Offline rebuild of the currently served library, including thermal cycles.

Keep the existing static catalog usable until every replacement is accepted.
No HTTP request can invoke this process.
"""
import gc
import json
from pathlib import Path
from cfd_service import CFDService
from cfd_thermal import run as thermal_run
from cfd_catalog import build_catalog


def main():
    service=CFDService()
    catalog=json.loads((service.public/'library.json').read_text())
    try:
        for entry in catalog['entries']:
            request=entry['request']
            if request.get('backend')!='gpu-lbm':continue
            job=service.submit(request)
            print('Rebuilding',request['settings']['position'],request['periodS'],'s',job['id'],flush=True)
            future=service.job_futures.get(job['id'])
            if future:future.result()
            job=service.jobs[job['id']]
            if job['state']!='complete':raise RuntimeError(job)
            thermal_run(service.public/job['id']/'result.json')
            gc.collect()
            print('Airflow and thermal accepted:',job['id'],flush=True)
        build_catalog()
    finally:
        service.executor.shutdown()
        service.mesh_executor.shutdown()


if __name__=='__main__':main()
