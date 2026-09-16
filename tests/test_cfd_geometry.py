import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_geometry import world_order
from mesh import idx_to_world

class WorldCoordinatesTests(unittest.TestCase):
    def test_source_voxels_match_existing_ct_mesh_world_coordinates(self):
        shape=(3,7,11);spacing=np.array([.625,.278,.278]);centre=np.array([93.,31.,104.])
        source=np.arange(np.prod(shape)).reshape(shape)
        world=world_order(source)
        origin=np.array([-centre[0],-centre[1]-spacing[0]/2,centre[2]-shape[1]*spacing[1]])
        for index in np.ndindex(shape):
            value=source[index]
            world_index=np.argwhere(world==value)[0]
            coordinate=origin+(world_index+.5)*spacing[[2,0,1]]
            expected=idx_to_world(np.array([index]),spacing,shape,centre)[0]
            np.testing.assert_allclose(coordinate,expected,atol=1e-12,rtol=0)

class ManifoldRepairs(unittest.TestCase):
    def test_connect_missing_ports_rejoins_a_pinched_nostril(self):
        from cfd_geometry import connect_missing_ports
        from scipy import ndimage as ndi
        domain=np.zeros((3,8,3),bool); domain[:,:3,:]=True
        ports=np.zeros(domain.shape,np.uint8); ports[1,7,1]=1
        walkable=np.zeros_like(domain); walkable[:,:3]=True; walkable[1,3:,1]=True
        restored=connect_missing_ports(domain,walkable,ports)
        self.assertGreater(restored,0)
        self.assertTrue((domain&(ports==1)).any())
        self.assertEqual(ndi.label(domain)[1],1)

    def test_keep_opening_components_does_not_drop_a_split_nostril(self):
        from cfd_geometry import keep_opening_components
        repaired=np.zeros((3,8,3),bool); repaired[:,:3]=True; repaired[:,6:]=True
        ports=np.zeros(repaired.shape,np.uint8); ports[:,7,:]=1; ports[:,0,:]=3
        domain,discarded=keep_opening_components(repaired,ports)
        self.assertEqual(int(domain.sum()),int(repaired.sum()))
        self.assertEqual(discarded,0)
        self.assertTrue((domain&(ports==1)).any())

if __name__=='__main__': unittest.main()
