import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_solver import iteration_diagnostics

def iteration(t,initial,final):
    return f'\nTime = {t}\n\n'+''.join(
        f'smoothSolver: Solving for {field}, Initial residual = {initial}, Final residual = {final}, No Iterations 3\n'
        for field in ['Ux','Uy','Uz','p'])

class ConvergenceTests(unittest.TestCase):
    def test_linear_convergence_cannot_pass_outer_convergence(self):
        d=iteration_diagnostics(iteration(300,2e-4,1e-8)+'\nEnd\n',300)
        self.assertFalse(d['residualsPassed'])
        self.assertTrue(d['completed'])

    def test_saved_field_does_not_borrow_later_residuals(self):
        log=iteration(300,2e-4,1e-8)+iteration(352,1e-6,1e-9)+'\nEnd\n'
        self.assertFalse(iteration_diagnostics(log,300)['residualsPassed'])
        self.assertTrue(iteration_diagnostics(log,352)['residualsPassed'])

    def test_interrupted_run_and_missing_fields_fail(self):
        self.assertFalse(iteration_diagnostics(iteration(300,1e-6,1e-9),300)['completed'])
        self.assertFalse(iteration_diagnostics(iteration(300,1e-6,1e-9)+'\nEnd\n',400)['residualsPassed'])

if __name__=='__main__': unittest.main()
