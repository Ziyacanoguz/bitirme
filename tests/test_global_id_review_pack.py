import sys
import copy
import csv
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from create_global_id_review_pack import (
    choose_frame,
    load_mapping,
    load_track_index,
    resolve_fusion_context,
    render_review_page,
    review_identity,
    risk_score,
    truthy,
)


def row(frame, height=50, confidence=.8, x=.5):
    return dict(frame=frame, track_id=1, confidence=confidence, x1=0, y1=0, x2=20, y2=height,
                field_x=x, field_y=.5)


def review_row(frame=10, one=1, two=2):
    return dict(review_id=1, frame_camera1=frame, frame_camera2=frame + 2,
                camera1_segment_id=one, camera2_segment_id=two,
                camera1_local_id=one, camera2_local_id=two, predicted_global_id=3,
                time_seconds="0.333", support_frame_count=12, median_field_distance=".02",
                appearance_distance=".1", camera1_image="images/one.jpg",
                camera2_image="images/two.jpg", camera1_image_sha256="image-one",
                camera2_image_sha256="image-two", decision="", person_label="", notes="")


def sources():
    return {
        "videos": [{"path": "/camera1.mp4", "bytes": 100, "modified_ns": 1},
                   {"path": "/camera2.mp4", "bytes": 200, "modified_ns": 2}],
        "tracks": [{"sha256": "track-one"}, {"sha256": "track-two"}],
        "fusion_run": {"sha256": "run"}, "mapping": {"sha256": "mapping"},
        "candidates": {"sha256": "candidates"},
    }


def run_review_script(rows, key, stored=None, action="", storage_error=False):
    """Run the real generated browser script with DOM/storage/download stand-ins."""
    page = render_review_page(rows, key)
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    payload = json.dumps({"rows": rows, "key": key, "stored": stored,
                          "storageError": storage_error})
    harness = r'''
const input = PAYLOAD;
let saved = input.stored;
let capturedBlob = null;
let clicked = false;
const statuses = {'progress': {textContent: ''}, 'save-status': {textContent: ''}};
const fakeCards = input.rows.map(() => {
    const radios = ['correct', 'wrong', 'unsure'].map(value => ({value, checked: false}));
    const person = {value: ''}, notes = {value: ''};
    return {radios, person, notes,
        querySelector(selector) {
            if (selector === '.person') return person;
            if (selector === '.notes') return notes;
            return radios.find(r => r.checked);
        },
        querySelectorAll() { return radios; },
        addEventListener() {}
    };
});
const document = {
    querySelectorAll() { return fakeCards; },
    getElementById(id) { return statuses[id]; },
    createElement() { return {click() {clicked = true;}}; }
};
const localStorage = {
    getItem(key) { if (input.storageError) throw new Error('blocked'); return saved; },
    setItem(key, value) { if (input.storageError) throw new Error('blocked'); saved = value; }
};
const URL = {createObjectURL(blob) {capturedBlob = blob; return 'blob:test';}, revokeObjectURL() {}};
const setTimeout = () => {};
'''.replace("PAYLOAD", payload)
    finish = r'''
(async () => {
    process.stdout.write(JSON.stringify({rows, saved, clicked,
        progress: statuses.progress.textContent, status: statuses['save-status'].textContent,
        csv: capturedBlob ? await capturedBlob.text() : null}));
})();
'''
    result = subprocess.run([shutil.which("node"), "-"], input=harness + script + action + finish,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


class ReviewPackTests(unittest.TestCase):
    def test_truthy_is_explicit(self):
        self.assertTrue(truthy("True"))
        self.assertFalse(truthy("False"))

    def test_choose_frame_accounts_for_offset_and_quality(self):
        first = {10: row(10, 30), 11: row(11, 80)}
        second = {12: row(12, 30), 13: row(13, 80)}
        self.assertEqual(choose_frame(first, second, 2), 11)

    def test_choose_frame_is_limited_to_candidate_evidence_span(self):
        first = {10: row(10, 120), 11: row(11, 40), 12: row(12, 200)}
        second = {10: row(10, 120), 11: row(11, 40), 12: row(12, 200)}
        self.assertEqual(choose_frame(first, second, 0, first_frame=11, last_frame=11), 11)
        self.assertIsNone(choose_frame(first, second, 0, first_frame=20, last_frame=30))

    def test_higher_distance_is_riskier(self):
        base = dict(median_distance=.01, appearance=.1, support_fraction=.9, support=20)
        risky = dict(base, median_distance=.08)
        self.assertGreater(risk_score(risky), risk_score(base))

    def test_exact_support_frames_exclude_non_supporting_cooccurrences(self):
        first = {10: row(10, 120), 11: row(11, 40), 12: row(12, 200)}
        second = {12: row(12, 120), 13: row(13, 40), 14: row(14, 200)}
        self.assertEqual(choose_frame(first, second, 2, support_frames=[10, 11]), 10)
        self.assertIsNone(choose_frame(first, second, 2, support_frames=[]))

    def test_non_finite_candidate_evidence_is_rejected(self):
        base = dict(median_distance=float("nan"), appearance=.1, support_fraction=.9, support=20)
        with self.assertRaisesRegex(ValueError, "Invalid candidate"):
            risk_score(base)

    def test_mapping_prefers_track_id_and_supports_legacy_local_id(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.csv"
            current.write_text("camera,track_id,local_id,global_id\n1,7,70,3\n", encoding="utf-8")
            mapping, column = load_mapping(current)
            self.assertEqual(column, "track_id")
            self.assertEqual(mapping[(1, 7)], 3)
            self.assertNotIn((1, 70), mapping)

            legacy = Path(directory) / "legacy.csv"
            legacy.write_text("camera,local_id,global_id\n2,8,4\n", encoding="utf-8")
            mapping, column = load_mapping(legacy)
            self.assertEqual(column, "local_id")
            self.assertEqual(mapping[(2, 8)], 4)

    def test_track_loader_rejects_duplicate_and_non_finite_observations(self):
        fields = ["frame", "track_id", "confidence", "x1", "y1", "x2", "y2", "field_x", "field_y"]
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.csv"
            with duplicate.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([row(1), row(1)])
            with self.assertRaisesRegex(ValueError, "duplicate frame/segment"):
                load_track_index(duplicate)

            invalid = Path(directory) / "invalid.csv"
            bad = row(1)
            bad["x1"] = np.nan
            with invalid.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(bad)
            with self.assertRaisesRegex(ValueError, "invalid observation"):
                load_track_index(invalid)

    def test_fusion_context_derives_offset_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tracks = [directory / "camera1.csv", directory / "camera2.csv"]
            for path in tracks:
                path.write_text("frame,track_id,confidence,x1,y1,x2,y2\n", encoding="utf-8")
            run_path = directory / "run.json"
            run_path.write_text(json.dumps({
                "status": "complete",
                "offset_frames": 3,
                "sources": [str(path) for path in tracks],
            }), encoding="utf-8")
            _run, offset = resolve_fusion_context(run_path, tracks)
            self.assertEqual(offset, 3)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                resolve_fusion_context(run_path, tracks, requested_offset=0)

    def test_fusion_context_rejects_wrong_tracking_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = [directory / "expected1.csv", directory / "expected2.csv"]
            actual = [directory / "actual1.csv", expected[1]]
            for path in set(expected + actual):
                path.write_text("x", encoding="utf-8")
            run_path = directory / "run.json"
            run_path.write_text(json.dumps({
                "status": "complete", "offset_frames": 0,
                "sources": [str(path) for path in expected],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_fusion_context(run_path, actual)

    def test_review_identity_invalidates_changed_frames_images_and_each_source(self):
        original = [review_row()]
        original_key = review_identity(original, sources())
        for field in ("frame_camera1", "frame_camera2", "camera1_segment_id", "camera2_segment_id",
                      "predicted_global_id", "camera1_image_sha256", "camera2_image_sha256"):
            with self.subTest(field=field):
                changed = [review_row()]
                changed[0][field] = str(changed[0][field]) + "-changed"
                self.assertNotEqual(original_key, review_identity(changed, sources()))
                self.assertNotEqual(original[0]["review_evidence_id"], changed[0]["review_evidence_id"])
        for source in sources():
            with self.subTest(source=source):
                changed_sources = sources()
                record = changed_sources[source]
                record = record[0] if isinstance(record, list) else record
                record["path"] = "/changed/source"
                self.assertNotEqual(original_key, review_identity([review_row()], changed_sources))

    def test_review_identity_survives_card_reordering_output_names_and_annotations(self):
        rows = [review_row(), review_row(frame=20, one=5, two=6)]
        original_key = review_identity(rows, sources())
        changed = copy.deepcopy(rows[::-1])
        for index, item in enumerate(changed, 1):
            item.update(review_id=index, camera1_image=f"images/renamed_{index}.jpg",
                        decision="correct", person_label="P01", notes="Test note")
        self.assertEqual(original_key, review_identity(changed, sources()))
        self.assertEqual(rows[0]["review_evidence_id"], changed[1]["review_evidence_id"])

    @unittest.skipUnless(shutil.which("node"), "Node.js required for generated browser-script checks")
    def test_browser_restores_exact_evidence_even_when_saved_order_differs(self):
        rows = [review_row(), review_row(frame=20, one=5, two=6)]
        key = review_identity(rows, sources())
        stored = copy.deepcopy(rows[::-1])
        stored[0].update(decision="wrong", notes="Different person")
        stored[1].update(decision="unsure", person_label="P01")
        result = run_review_script(rows, key, json.dumps(stored))
        self.assertEqual([item["decision"] for item in result["rows"]], ["unsure", "wrong"])
        self.assertEqual(result["rows"][0]["person_label"], "P01")
        self.assertEqual(result["rows"][1]["notes"], "Different person")
        self.assertEqual(result["progress"], "2 / 2 incelendi")

    @unittest.skipUnless(shutil.which("node"), "Node.js required for generated browser-script checks")
    def test_browser_ignores_stale_frame_legacy_pair_and_wrong_source_decisions(self):
        current = [review_row(frame=11)]
        key = review_identity(current, sources())
        old = [review_row(frame=10)]
        review_identity(old, sources())
        wrong_source = copy.deepcopy(current)
        wrong_source[0]["review_source_id"] = "other-source"
        for stored in (old, [review_row()], wrong_source):
            stored[0].update(decision="correct", person_label="P01", notes="Stale")
            result = run_review_script(current, key, json.dumps(stored))
            self.assertEqual(result["rows"][0]["decision"], "")
            self.assertEqual(result["rows"][0]["person_label"], "")
            self.assertEqual(result["rows"][0]["notes"], "")
            self.assertEqual(result["progress"], "0 / 1 incelendi")

    @unittest.skipUnless(shutil.which("node"), "Node.js required for generated browser-script checks")
    def test_browser_exports_quotes_newlines_and_provenance_when_storage_blocked(self):
        rows = [review_row()]
        key = review_identity(rows, sources())
        note = 'Açıklama, "iki"\nsatır </script>'
        action = ("fakeCards[0].radios[0].checked=true;"
                  "fakeCards[0].person.value='P01';"
                  f"fakeCards[0].notes.value={json.dumps(note)};download();")
        result = run_review_script(rows, key, storage_error=True, action=action)
        self.assertTrue(result["clicked"])
        self.assertIn("Otomatik kayıt kullanılamıyor", result["status"])
        exported = list(csv.DictReader(io.StringIO(result["csv"].lstrip("\ufeff"))))
        self.assertEqual(exported[0]["notes"], note)
        self.assertEqual(exported[0]["decision"], "correct")
        self.assertEqual(exported[0]["review_evidence_id"], rows[0]["review_evidence_id"])
        self.assertEqual(exported[0]["review_source_id"], rows[0]["review_source_id"])

    @unittest.skipUnless(shutil.which("node"), "Node.js required for generated browser-script checks")
    def test_browser_recovers_from_invalid_saved_data_and_reports_unreadable_storage(self):
        rows = [review_row()]
        key = review_identity(rows, sources())
        for stored in ("not-json", "{}", "null"):
            with self.subTest(stored=stored):
                result = run_review_script(rows, key, stored, action="save();")
                self.assertEqual(json.loads(result["saved"])[0]["decision"], "")
                self.assertEqual(result["progress"], "0 / 1 incelendi")
        result = run_review_script(rows, key, storage_error=True)
        self.assertIn("Önceki kayıt okunamadı", result["status"])


if __name__ == "__main__":
    unittest.main()
