import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_transient import waveform, write_transient_case, transient_diagnostics
from cfd_service import normalized, request_key

class TransientChecks(unittest.TestCase):
    def test_signed_cycle_and_identity(self):
        w=waveform([0,.4,1.2,2.0,3.0,4],250,4)
        self.assertAlmostEqual(w[0],0);self.assertAlmostEqual(w[-1],0)
        self.assertGreater(w[1],0);self.assertGreater(w[2],0);self.assertLess(w[3],0);self.assertLess(w[4],0)
        self.assertAlmostEqual(np.trapezoid(waveform(np.linspace(0,4,10001),250,4),np.linspace(0,4,10001)),0,places=3)
        steady=normalized({})
        transient=normalized({'mode':'transient','direction':'cycle','periodS':4})
        self.assertNotEqual(request_key(steady,'scan'),request_key(transient,'scan'))
        self.assertNotEqual(request_key(transient,'scan'),request_key({**transient,'periodS':5},'scan'))
        left=normalized({'settings':{'position':'left'}})
        self.assertNotEqual(request_key(steady,'scan'),request_key(left,'scan'))
        for request in [{'mode':'transient','periodS':0},{'mode':'invented'},{'direction':'cycle'}]:
            with self.assertRaises(ValueError): normalized(request)

    def test_case_has_real_time_derivative_and_bidirectional_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder);write_transient_case(case)
            self.assertIn('default backward', (case/'system/fvSchemes').read_text())
            self.assertIn('application pimpleFoam', (case/'system/controlDict').read_text())
            velocity=(case/'0/U').read_text()
            self.assertIn('extrapolateProfile false',velocity)
            self.assertRegex(velocity,r'\(0 -?0\)')
            self.assertRegex(velocity,r'\(8 -?0\)')
            self.assertEqual(json.loads((case/'solver.json').read_text())['temporal'],'transient')

    def test_outer_convergence_cannot_borrow_inner_pressure_or_next_time(self):
        def block(t,initial):
            fields='\n'.join(f'Solving for {k}, Initial residual = {initial}, Final residual = 1e-10' for k in ['Ux','Uy','Uz','p'])
            return f'Time = {t}\nPIMPLE: iteration 12\n{fields}\nSolving for p, Initial residual = 1e-8, Final residual = 1e-10\n'
        log='Courant Number mean: 0.1 max: 0.4\n'+block(1,'0.001')+block(2,'1e-6')+'\nEnd\n'
        d=transient_diagnostics(log,2)
        self.assertTrue(d['completed']);self.assertFalse(d['outerResidualsPassed']);self.assertEqual(d['failedSteps'],1)
        self.assertFalse(transient_diagnostics(log,3)['completed'])
        self.assertFalse(transient_diagnostics(log.replace('\nEnd\n',''),2)['completed'])

if __name__=='__main__': unittest.main()
