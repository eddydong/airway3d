import sys,unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_thermal import topology,ThermalSystem

class ThermalChecks(unittest.TestCase):
    def duct(self,settings=None):
        shape=(12,3,3)
        openings=[dict(normal=[-1,0,0],role='nostril',facesMm=[[0,j+.5,k+.5] for j in range(3) for k in range(3)]),
                  dict(normal=[1,0,0],role='outlet',facesMm=[[12,j+.5,k+.5] for j in range(3) for k in range(3)])]
        return ThermalSystem(topology(np.ones(shape,bool),np.zeros(3),1.,openings),1.,settings)

    def test_uniform_temperature_stays_uniform_including_nonzero_divergence(self):
        s=self.duct(dict(ambientC=30,bodyC=30,initialC=30,exhaledC=30))
        velocity=np.random.default_rng(3).normal(0,.2,(s.n,3))
        for _ in range(3):s.step(velocity,.02)
        np.testing.assert_allclose(s.air,30,atol=1e-7);np.testing.assert_allclose(s.wall_fields()[0],30,atol=1e-7)

    def test_inhaled_air_warms_and_wall_cools_with_finite_heat_capacity(self):
        s=self.duct(dict(initialC=37))
        for _ in range(50):s.step(np.tile([.1,0,0],(s.n,1)),.02)
        surface,flux=s.wall_fields()
        self.assertLess(surface.min(),36.9);self.assertGreater(surface.min(),22)
        self.assertGreater(s.air.reshape(12,3,3)[-1].mean(),s.air.reshape(12,3,3)[0].mean())
        self.assertGreater(flux.max(),0);self.assertLess(s.maxBalanceResidual,1e-7)

    def test_expiration_uses_warm_throat_air_and_can_heat_a_cool_wall(self):
        s=self.duct(dict(initialC=25,bodyC=25,exhaledC=37,ambientC=22))
        for _ in range(30):s.step(np.tile([-.1,0,0],(s.n,1)),.02)
        surface,flux=s.wall_fields();self.assertGreater(surface.max(),25);self.assertLess(flux.min(),0)
        self.assertGreater(s.air.reshape(12,3,3)[-1].mean(),s.air.reshape(12,3,3)[0].mean())

    def test_exposed_surface_uses_series_air_and_tissue_thermal_resistances(self):
        s=self.duct();s.air[:]=22;s.wall[:]=37;surface,flux=s.wall_fields()
        expected=15/(.001/(2*.026)+.001/(2*.5))
        np.testing.assert_allclose(flux,expected);np.testing.assert_allclose(surface,22+expected*.001/(2*.026))

    def test_backward_euler_refines_towards_analytic_coupled_slab_solution(self):
        top=topology(np.ones((1,1,1),bool),np.zeros(3),1,[])
        initial=np.array([22.,34.]);errors=[]
        for dt in [.02,.01]:
            s=ThermalSystem(top,1);s.air[:]=initial[0];s.wall[:]=initial[1]
            matrix=np.array([[-6*s.g/s.ca,6*s.g/s.ca],[s.g/s.cw,-(s.g+s.gb)/s.cw]])
            expected=np.full(2,37.)+expm(matrix*.4)@(initial-37.)
            for _ in range(round(.4/dt)):s.step(np.zeros((1,3)),dt)
            errors.append(float(np.max(abs(np.array([s.air[0],s.wall[0]])-expected))))
            self.assertLess(s.maxBalanceResidual,1e-8)
        self.assertLess(errors[1],errors[0]*.6);self.assertLess(errors[1],.003)

    def test_port_caps_are_not_mucosal_faces(self):
        s=self.duct();self.assertEqual(len(s.top['ports']),18)
        self.assertEqual(len(s.top['wall']),12*3*4)
        self.assertEqual(s.top['vertices'].shape,(144,4,3))

if __name__=='__main__':unittest.main()
