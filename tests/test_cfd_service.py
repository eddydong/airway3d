import sys
import threading
import unittest
from concurrent.futures import Future,ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock,patch
import json
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_service import CFDService,normalized

class ServiceChecks(unittest.TestCase):
    def service(self):
        s=CFDService.__new__(CFDService);s.lock=threading.RLock();s.jobs={};s.job_futures={};s.cancelled=set()
        s.executor=Mock();s.executor.submit.side_effect=lambda *args:Future()
        s.lookup=lambda body:('key',normalized(body),s.jobs.get('key'))
        def status(key,**kwargs):
            s.jobs.setdefault(key,{}).update(kwargs);return dict(s.jobs[key])
        s.set_status=status;return s

    def test_concurrent_identical_submissions_only_create_one_worker(self):
        s=self.service()
        with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(lambda _:s.submit({}),range(8)))
        self.assertEqual(s.executor.submit.call_count,1)
        self.assertEqual(len(s.job_futures),1)

    def test_resubmit_does_not_clear_cancellation_while_worker_still_active(self):
        s=self.service();s.submit({});s.cancel('key');s.submit({})
        self.assertIn('key',s.cancelled);self.assertEqual(s.executor.submit.call_count,1)

    def test_invalid_batch_is_rejected_before_any_job_is_started(self):
        s=self.service()
        with self.assertRaises(ValueError):s.submit_batch({'requests':[{}, {'mode':'transient','periodS':0}]})
        self.assertEqual(s.executor.submit.call_count,0)

    def test_cancel_does_not_overwrite_a_completed_recording(self):
        s=self.service();s.jobs['key']={'state':'complete','resultUrl':'/result.json'}
        self.assertEqual(s.cancel('key')['state'],'complete')
        self.assertNotIn('key',s.cancelled)

    def test_restart_exposes_interrupted_job_as_retryable_failure(self):
        from cfd_service import request_key
        s=self.service();s.source=lambda:'source';s.register_baseline=lambda:None
        key=request_key(normalized({}),'source')
        with tempfile.TemporaryDirectory() as temp,patch('cfd_service.ROOT',Path(temp)):
            folder=Path(temp)/key;folder.mkdir()
            (folder/'status.json').write_text(json.dumps({'id':key,'state':'solving'}))
            _,_,job=CFDService.lookup(s,{})
            self.assertEqual(job['state'],'failed');self.assertIn('restarted',job['message'])




class GpuRequests(unittest.TestCase):
    def test_lattice_requests_are_pressure_driven_and_omit_flow_rate(self):
        r=normalized({'backend':'gpu-lbm','mode':'transient','periodS':4,'pressurePa':30,'refinement':1,'qMlS':400})
        self.assertNotIn('qMlS',r);self.assertEqual(r['pressurePa'],30);self.assertEqual(r['refinement'],1);self.assertEqual(r['cycles'],2)
        self.assertEqual(r['direction'],'cycle')
        with self.assertRaises(ValueError):normalized({'backend':'gpu-lbm','mode':'steady'})
        with self.assertRaises(ValueError):normalized({'backend':'gpu-lbm','mode':'transient','refinement':3})
        with self.assertRaises(ValueError):normalized({'backend':'gpu-lbm','mode':'transient','pressurePa':1000})
        with self.assertRaises(ValueError):normalized({'backend':'magic'})

    def test_backends_never_share_a_result_identity(self):
        from cfd_service import request_key
        foam=normalized({'mode':'transient','periodS':4});gpu=normalized({'backend':'gpu-lbm','mode':'transient','periodS':4})
        self.assertNotEqual(request_key(foam,'src'),request_key(gpu,'src'))
        self.assertNotEqual(request_key(gpu,'src'),request_key(dict(gpu,refinement=2),'src'))
        self.assertEqual(request_key(gpu,'src'),request_key(normalized({'backend':'gpu-lbm','mode':'transient','periodS':4,'qMlS':100}),'src'))
if __name__=='__main__':unittest.main()
