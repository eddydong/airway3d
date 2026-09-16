import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from cfd_library import control_point_requests
from cfd_service import normalized

class ControlPoints(unittest.TestCase):
    def test_discrete_independent_interventions_share_baseline_numerics(self):
        rows=control_point_requests(dict(backend='gpu-lbm',mode='transient',periodS=4,pressurePa=30,spacingMm=.7,refinement=1,cycles=2))
        self.assertEqual(len(rows),10)
        self.assertEqual({r['position'] for r in rows if r['baseline']},{'supine','left','right','upright'})
        seen=set()
        for row in rows:
            r=normalized(row['request']);self.assertEqual(r['periodS'],4);self.assertEqual(r['pressurePa'],30)
            values=[(side,region,v) for side,ops in r['settings']['operations'].items() for region,v in ops.items() if v]
            if not row['baseline']:
                self.assertEqual(len(values),1);self.assertEqual(values[0][2],1)
                self.assertEqual(r['settings']['position'],'supine');seen.add(values[0][:2])
        self.assertEqual(len(seen),6)

if __name__=='__main__':unittest.main()
