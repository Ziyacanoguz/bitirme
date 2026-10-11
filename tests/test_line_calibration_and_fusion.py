import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from halisaha.line_calibration import fit_line_calibration, project_line_calibration, unproject_line_calibration
from halisaha.multicamera import ConflictSafeComponents, associate, fuse_frames, valid_position


def calibration():
    return dict(image_width=1920,image_height=1080,division_center=[960,540],division_scale=1920,
                division_k=-.7,homography=[[1.1,.1,.5],[.02,1.8,.4],[.03,-.08,1]])


def row(frame,identity=1,x=.4,y=.5,height=100):
    return dict(frame=frame,track_id=identity,field_x=x,field_y=y,calibration_supported=1,
                confidence=.9,x1=100,x2=140,y1=100,y2=100+height)


class LineCalibrationTests(unittest.TestCase):
    def test_roundtrip(self):
        data=calibration()
        points=np.random.default_rng(1).uniform(0,1,(100,2))
        np.testing.assert_allclose(project_line_calibration(unproject_line_calibration(points,data),data),points,atol=1e-10)

    def test_resolution_scaling(self):
        data=calibration()
        points=np.array([[.1,.2],[.8,.9]])
        pixels=unproject_line_calibration(points,data)/2
        np.testing.assert_allclose(project_line_calibration(pixels,data,960,540),points,atol=1e-10)

    def test_recover_synthetic_lines_without_corners(self):
        data=calibration()
        lines=[]
        for axis,value in [(0,0),(0,1),(1,0),(1,1),(0,.5)]:
            points=np.full((9,2),value,dtype=float)
            points[:,1-axis]=np.linspace(.1,.9,9)
            lines.append(dict(axis=axis,value=value,points=unproject_line_calibration(points,data).tolist()))
        fitted=fit_line_calibration(lines,1920,1080,sample_parity=0)
        check=np.array([[.17,.23],[.7,.83],[.5,.5]])
        np.testing.assert_allclose(project_line_calibration(unproject_line_calibration(check,data),fitted),check,atol=1e-5)

    def test_missing_boundary_rejected(self):
        with self.assertRaises(ValueError):
            fit_line_calibration([],1920,1080)


class FusionTests(unittest.TestCase):
    def test_same_camera_overlap_blocks_transitive_merge(self):
        observations={(1,1):{1,2},(1,2):{2,3},(2,8):{1,2,3}}
        components=ConflictSafeComponents(observations)
        self.assertTrue(components.merge((1,1),(2,8)))
        self.assertFalse(components.merge((1,2),(2,8)))

    def test_bridge_nonoverlapping_fragments(self):
        components=ConflictSafeComponents({(1,1):{1,2},(1,2):{3,4},(2,8):{1,2,3,4}})
        self.assertTrue(components.merge((1,1),(2,8)))
        self.assertTrue(components.merge((1,2),(2,8)))

    def test_weak_evidence_not_shared(self):
        _,shared,candidates=associate([row(0)],[row(0,identity=8)],{},{},min_support=3)
        self.assertFalse(shared)
        self.assertFalse(candidates[0]['accepted'])

    def test_good_repeated_pair_is_shared(self):
        mapping,shared,_=associate([row(i) for i in range(10)],[row(i,identity=8,x=.41) for i in range(10)],{},{},min_support=3)
        self.assertEqual(mapping[(1,1)],mapping[(2,8)])
        self.assertEqual(len(shared),1)

    def test_appearance_gate_rejects(self):
        _,shared,_=associate([row(i) for i in range(10)],[row(i,identity=8) for i in range(10)],
                             {1:np.array([1.,0])},{8:np.array([-1.,0])},min_support=3)
        self.assertFalse(shared)

    def test_tie_is_not_forced(self):
        _,shared,_=associate([row(i) for i in range(10)],
                             [row(i,identity=j,x=.41) for i in range(10) for j in [8,9]],{},{},min_support=3)
        self.assertFalse(shared)

    def test_disagreement_stays_separate(self):
        result,totals=fuse_frames([row(0,x=.1)],[row(0,identity=8,x=.9)],{(1,1):1,(2,8):1},{1})
        self.assertEqual(len(result),2)
        self.assertTrue(all(r['status']=='conflict' and r['global_id']=='' for r in result))
        self.assertEqual(totals['disagreement_frames'],1)

    def test_unmatched_close_people_not_deduplicated(self):
        result,_=fuse_frames([row(0)],[row(0,identity=8)],{(1,1):1,(2,8):2},set())
        self.assertEqual(len(result),2)

    def test_fusion_prefers_more_resolved_observation(self):
        result,_=fuse_frames([row(0,x=.4,height=150)],[row(0,identity=8,x=.45,height=30)],
                             {(1,1):1,(2,8):1},{1})
        self.assertEqual(len(result),1)
        self.assertLess(result[0]['field_x'],.425)
        self.assertEqual(result[0]['observation_count'],2)

    def test_time_offset_uses_other_camera_correct_frame(self):
        result,_=fuse_frames([row(10)],[row(12,identity=8)],{(1,1):1,(2,8):1},{1},offset_frames=2)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['frame'],10)

    def test_outside_is_not_clamped(self):
        self.assertFalse(valid_position(row(0,x=-.01)))

    def test_valid_position_is_json_serializable_bool(self):
        self.assertIs(type(valid_position(row(0))), bool)

    def test_continuous_appearance_compatible_segments_reconnect(self):
        rows=[dict(row(f,identity=1 if f<5 else 2),local_track_id=7) for f in range(10)]
        mapping,_,_=associate(rows,[],{1:np.array([1.,0]),2:np.array([1.,0])},{})
        self.assertEqual(mapping[(1,1)],mapping[(1,2)])

    def test_continuity_does_not_reconnect_changed_appearance(self):
        rows=[dict(row(f,identity=1 if f<5 else 2),local_track_id=7) for f in range(10)]
        mapping,_,_=associate(rows,[],{1:np.array([1.,0]),2:np.array([0.,1])},{})
        self.assertNotEqual(mapping[(1,1)],mapping[(1,2)])

    def test_continuity_does_not_reconnect_position_jump(self):
        rows=[dict(row(f,identity=1 if f<5 else 2,x=.4 if f<5 else .7),local_track_id=7)
              for f in range(10)]
        mapping,_,_=associate(rows,[],{1:np.array([1.,0]),2:np.array([1.,0])},{})
        self.assertNotEqual(mapping[(1,1)],mapping[(1,2)])

    def test_same_local_id_without_appearance_is_not_enough(self):
        rows=[dict(row(f,identity=1 if f<5 else 2),local_track_id=7) for f in range(10)]
        mapping,_,_=associate(rows,[],{},{})
        self.assertNotEqual(mapping[(1,1)],mapping[(1,2)])


if __name__=='__main__':
    unittest.main()
