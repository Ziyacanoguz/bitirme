import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_reviewed_snapshot import choose_split
from stitch_tracklets import select_links, make_mapping


class SafetyTests(unittest.TestCase):
    def test_same_moment_both_cameras_have_same_split(self):
        for camera in ("camera1", "camera2"):
            self.assertEqual(choose_split(dict(time_seconds=950, camera=camera), 900, 1040, 35), "val")
            self.assertEqual(choose_split(dict(time_seconds=880, camera=camera), 900, 1040, 35), "excluded_guard")
            self.assertEqual(choose_split(dict(time_seconds=800, camera=camera), 900, 1040, 35), "train")

    def test_ambiguous_outgoing_not_merged(self):
        links = [(.1, 1, 2, 3, .02, .1), (.15, 1, 3, 3, .025, .1)]
        self.assertEqual(select_links(links, .15)[2], [])

    def test_ambiguous_incoming_not_merged(self):
        links = [(.1, 1, 3, 3, .02, .1), (.15, 2, 3, 3, .025, .1)]
        self.assertEqual(select_links(links, .15)[2], [])

    def test_clear_chain_keeps_first_id(self):
        links = [(.1, 1, 2, 3, .02, .1), (.2, 2, 3, 3, .025, .1)]
        pred, succ, selected = select_links(links, .15)
        self.assertEqual(len(selected), 2)
        self.assertEqual(make_mapping({1: {}, 2: {}, 3: {}}, pred, succ), {1: 1, 2: 1, 3: 1})


if __name__ == "__main__":
    unittest.main()
