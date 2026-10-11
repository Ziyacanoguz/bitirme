import contextlib
import importlib.util
import io
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_global_id_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_global_id_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class GlobalIdPipelineArgumentsTests(unittest.TestCase):
    def arguments(self, method):
        args = []
        for name in ("video1", "video2", "tracks1", "tracks2", "calibration1",
                     "calibration2", "output-dir"):
            args.extend(("--" + name, name))
        return [*args, "--sync-method", method]

    def test_manual_event_rejects_missing_or_blank_evidence_note(self):
        for extra in ([], ["--sync-note", ""], ["--sync-note", " \t "]):
            with self.subTest(extra=extra):
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    pipeline.parse_args([*self.arguments("manual_event"), *extra])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--sync-note is required", error.getvalue())

    def test_manual_event_preserves_the_event_and_frame_evidence(self):
        note = "Kickoff: C1 frame 45000, C2 frame 45003"
        args = pipeline.parse_args([
            *self.arguments("manual_event"), "--offset-frames", "3", "--sync-note", f" {note} ",
        ])
        self.assertEqual(args.sync_note, note)
        self.assertEqual(args.offset_frames, 3)

    def test_non_manual_method_can_omit_note(self):
        args = pipeline.parse_args(self.arguments("hardware"))
        self.assertEqual(args.sync_note, "")

    def test_unverified_offset_can_be_explicitly_recorded(self):
        note = "Reused offset 0; common event has not been visually verified"
        args = pipeline.parse_args([
            *self.arguments("assumed_offset"), "--sync-note", note,
        ])
        self.assertEqual(args.sync_method, "assumed_offset")
        self.assertEqual(args.sync_note, note)

    def test_assumed_offset_requires_a_note(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(self.arguments("assumed_offset"))


if __name__ == "__main__":
    unittest.main()
