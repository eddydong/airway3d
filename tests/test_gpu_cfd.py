"""GPU lattice backend checks that need no GPU: lattice encoding, opening flows, gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
import gpu_cfd
from cfd_geometry import mesh_arrays


def forked_domain(h=.7):
    """Two separate nostril tubes joining a common channel that ends at the outlet."""
    mask=np.zeros((12,28,6),bool)
    mask[2:10,1:12,1:5]=True     # common channel, outlet at y=1
    mask[2:5,12:24,1:5]=True     # left tube (positive x is the patient's left)
    mask[7:10,12:24,1:5]=True    # right tube
    ports=np.zeros(mask.shape,'u1')
    ports[2:5,23,1:5]=1;ports[7:10,23,1:5]=2;ports[:,1,:][mask[:,1,:]]=3
    exterior=np.zeros_like(mask);exterior[:,24:,:]=True
    return dict(mask=mask,ports=ports,exterior=exterior,origin=np.array([-3.,-9.,11.]),h=h,audit={})


def branch_field(domain,u=1.0):
    """Toward the outlet everywhere; continuity across the junction."""
    mask=domain['mask'];v=np.zeros((*mask.shape,3))
    v[...,1]=np.where(np.indices(mask.shape)[1]>=12,-u,-u*24/32)
    return v[mask]


class LatticeEncoding(unittest.TestCase):
    def test_ghost_flags_point_back_at_their_fluid_neighbour(self):
        d=forked_domain();flags,ids,faces=gpu_cfd.input_arrays(d)
        self.assertEqual(len(ids),int(d['mask'].sum()))
        # Fluid ids follow the viewer's C-order over the unpadded mask.
        xyz=np.array(np.unravel_index(ids,flags.shape,order='F')).T-2
        self.assertTrue(np.array_equal(xyz,np.argwhere(d['mask'])))
        flat=flags.ravel(order='F')
        self.assertTrue(np.all(flat[ids]==0))
        self.assertEqual(sorted(set(faces[:,4])),[1,2,3])
        for owner,ghost,axis,sign,side in faces:
            flag=int(flat[ghost]);self.assertEqual(flag&3,2)  # TYPE_E, not solid
            self.assertEqual(flag&(64|128),64 if side<3 else 128)
            direction=(flag>>2)&7;self.assertIn(direction,range(1,7))
            step=gpu_cfd.DIRECTIONS[direction-1]
            o=np.array(np.unravel_index(owner,flags.shape,order='F'));g=np.array(np.unravel_index(ghost,flags.shape,order='F'))
            self.assertTrue(np.array_equal(g+step,o),'ghost neighbour index must reach the owner')
            self.assertEqual(axis,np.nonzero(step)[0][0])
        self.assertEqual((flat==0).sum()+((flat&2)==2).sum()+(flat==1).sum(),flags.size,'every lattice cell is fluid, ghost or solid')

    def test_blocked_opening_is_rejected(self):
        d=forked_domain();d['ports'][d['ports']==2]=0
        with self.assertRaises(ValueError):gpu_cfd.input_arrays(d)

    def test_requested_pressure_beyond_density_limit_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):gpu_cfd.write_input(Path(tmp),forked_domain(),pressure=5000,period=4)


class OpeningFlows(unittest.TestCase):
    def test_planar_openings_recover_branch_and_total_flows(self):
        d=forked_domain();h=d['h']/1000;v=branch_field(d)
        owners,directions,sides=gpu_cfd.boundary_faces(d)
        index=np.full(d['mask'].shape,-1,np.int64);index[d['mask']]=np.arange(int(d['mask'].sum()))
        oi=index[tuple(owners.T)]
        area=12*h*h*1e6  # 3 x 4 cells per tube
        self.assertAlmostEqual(gpu_cfd.opening_flow_ml_s(v,oi[sides==1],directions[sides==1],h),area,places=9)
        self.assertAlmostEqual(gpu_cfd.opening_flow_ml_s(v,oi[sides==2],directions[sides==2],h),area,places=9)
        # Toward the outlet is negative at the outlet; frame_metrics flips it.
        self.assertAlmostEqual(-gpu_cfd.opening_flow_ml_s(v,oi[sides==3],directions[sides==3],h),2*area,places=9)

    def test_opening_summary_locates_each_planar_opening(self):
        d=forked_domain();h=d['h'];o=d['origin']
        by={s['name']:s for s in gpu_cfd.opening_summary(d)}
        left=by['left_nostril']
        self.assertEqual(left['normal'],[0,1,0]);self.assertEqual(left['faces'],12);self.assertEqual(left['side'],'L')
        # Faces sit on the plane y=24 (cell 23 + 1), centred on x cells 2..4 and z cells 1..4.
        self.assertAlmostEqual(left['centreMm'][1],24*h+o[1]);self.assertAlmostEqual(left['centreMm'][0],3.5*h+o[0]);self.assertAlmostEqual(left['centreMm'][2],3*h+o[2])
        self.assertAlmostEqual(left['radiusMm'],np.sqrt(12*h*h/np.pi))
        self.assertEqual(len(left['facesMm']),12);self.assertTrue(all(abs(f[1]-left['centreMm'][1])<1e-6 for f in left['facesMm']))
        self.assertEqual(by['outlet']['normal'],[0,-1,0]);self.assertEqual(by['outlet']['role'],'outlet')
        self.assertAlmostEqual(by['outlet']['centreMm'][1],1*h+o[1])

    def test_flat_nostrils_make_one_planar_opening_per_nostril(self):
        from cfd_geometry import flatten_nostrils
        d=forked_domain();mask,ports,ext=d['mask'],d['ports'],d['exterior']
        # Stair-step the left tube's end: one column protrudes by two cells into exterior air.
        mask[3,24:26,2:4]=True;ports[3,24:26,2:4]=1;ports[3,23,1:5]=1
        collar=flatten_nostrils(mask,ports,ext)
        owners,directions,sides=gpu_cfd.boundary_faces(dict(d,mask=mask,ports=ports,exterior=ext))
        self.assertEqual(set(directions[sides==1]),{2},'left nostril opens through +y faces only')
        self.assertEqual(len(set(owners[sides==1][:,1])),1,'all left faces lie on one plane')
        self.assertEqual(collar['left_nostril']['planeIndex'],26);self.assertGreater(collar['left_nostril']['cellsAdded'],0)
        self.assertEqual(collar['right_nostril']['cellsAdded'],0,'an already planar nostril is unchanged')
        # Lateral faces of the collar are walls: no other exterior-facing faces remain open.
        self.assertTrue(np.all(owners[sides==1][:,1]==25))


class Collection(unittest.TestCase):
    def synthetic_case(self,tmp,scale_left=1.0,frames=9,cycles=1,period=4):
        d=forked_domain();case=Path(tmp);n=int(d['mask'].sum())
        arrays=mesh_arrays(d);arrays['surface'].export(case/'airway.glb')
        d['audit'].update(geometryHash='g1',watertight=True,windingConsistent=True,cells=n,spacingMm=d['h'])
        (case/'geometry.json').write_text(json.dumps(d['audit']))
        meta=gpu_cfd.write_input(case,d,pressure=30,period=period,cycles=cycles,intervals=frames-1)
        meta['framesPerCycle']=(frames-1)//cycles;(case/'solver.json').write_text(json.dumps(meta))
        (case/'frames').mkdir()
        with (case/'history.jsonl').open('w') as history:
            for i in range(frames):
                t=i*meta['durationS']/(frames-1);v=branch_field(d,u=np.sin(2*np.pi*t/period)).astype('<f4')
                left=np.argwhere(d['mask'])[:,0]<6;v[left&(np.argwhere(d['mask'])[:,1]>=12)]*=scale_left
                v.tofile(case/'frames'/f'{i}.velocity.f32');np.zeros(n,'<f4').tofile(case/'frames'/f'{i}.pressure.f32')
                history.write(json.dumps(dict(frame=i,timeS=t,computeSeconds=i*2.,maxMach=.05,maxDensityDeviation=.001))+'\n')
        return case,d,json.loads((case/'geometry.json').read_text())

    def test_consistent_recording_passes_every_gate_in_viewer_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,d,geometry=self.synthetic_case(tmp)
            r=gpu_cfd.collect(case,d,geometry)
            self.assertEqual(r['status'],'converged',r['gates'])
            self.assertTrue(all(r['gates'].values()))
            self.assertEqual(r['field']['schemaVersion'],3);self.assertEqual(r['field']['storageOrder'],'fluid-C')
            self.assertEqual(r['field']['fluidCells'],int(d['mask'].sum()));self.assertEqual(len(r['field']['frames']),9)
            self.assertEqual(r['solver']['backend'],'gpu-lbm');self.assertEqual(r['field']['frames'][0]['timeS'],0)
            self.assertTrue((case/'occupancy.u8').exists())
            self.assertLess(r['massImbalanceFraction'],1e-6)
            self.assertGreater(r['field']['frames'][2]['flowMlS']['outlet'],0,'inspiration is positive throat flow')
            self.assertAlmostEqual(r['field']['frames'][2]['flowMlS']['L'],r['field']['frames'][2]['flowMlS']['R'])

    def test_lost_mass_fails_the_balance_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,d,geometry=self.synthetic_case(tmp,scale_left=.8)
            r=gpu_cfd.collect(case,d,geometry)
            self.assertFalse(r['gates']['massBalance']);self.assertEqual(r['status'],'unconverged')
            self.assertTrue(r['gates']['coordinateAlignment'])

    def test_geometry_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            case,d,geometry=self.synthetic_case(tmp)
            with self.assertRaises(ValueError):gpu_cfd.collect(case,d,dict(geometry,geometryHash='other'))


if __name__=='__main__':unittest.main()
