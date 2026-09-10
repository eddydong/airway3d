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

if __name__=='__main__': unittest.main()
