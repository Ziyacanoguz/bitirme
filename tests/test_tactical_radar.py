import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halisaha.tactical_radar import (  # noqa: E402
    draw_pitch_diagram,
    render_tactical_radar_frame,
)


class TestTacticalRadar(unittest.TestCase):
    def test_draw_pitch_diagram(self):
        pitch = draw_pitch_diagram(width=800, height=500)
        self.assertEqual(pitch.shape, (500, 800, 3))
        self.assertEqual(pitch.dtype, np.uint8)

    def test_render_tactical_radar_frame(self):
        pitch = draw_pitch_diagram(width=800, height=500)
        frame_rows = [
            {"global_id": 1, "field_x": 0.25, "field_y": 0.50},
            {"global_id": 2, "field_x": 0.75, "field_y": 0.30},
        ]
        history = {}
        rendered = render_tactical_radar_frame(pitch, frame_rows, history, width=800, height=500)
        self.assertEqual(rendered.shape, (500, 800, 3))
        self.assertIn(1, history)
        self.assertIn(2, history)


if __name__ == "__main__":
    unittest.main()
