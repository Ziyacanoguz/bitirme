"""Create a browser-friendly human review queue for cross-camera ID candidates."""
import argparse
import csv
import html
from hashlib import sha256
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def file_provenance(path, *, digest=False):
    """Return stable source facts without hashing multi-gigabyte videos by default."""
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "modified_ns": path.stat().st_mtime_ns,
    }
    if digest:
        value = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        result["sha256"] = value.hexdigest()
    return result


def review_identity(rows, provenance):
    """Bind saved decisions to the exact evidence and its source files.

    Output paths, card order, and human annotations do not identify evidence.
    Video metadata avoids hashing entire recordings; the displayed JPEG hashes
    also invalidate decisions if the actual visual evidence changes.
    """
    def digest(value):
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()

    source_id = digest({"schema": "halisaha_review_sources_v1", "provenance": provenance})
    evidence_fields = (
        "camera1_segment_id", "camera2_segment_id", "frame_camera1", "frame_camera2",
        "predicted_global_id", "camera1_image_sha256", "camera2_image_sha256",
    )
    for row in rows:
        row["review_source_id"] = source_id
        row["review_evidence_id"] = digest({
            "schema": "halisaha_review_evidence_v1", "source_id": source_id,
            "evidence": {field: row[field] for field in evidence_fields},
        })
    pack_id = digest(sorted(row["review_evidence_id"] for row in rows))
    return "halisaha-review-v3-" + pack_id


def load_mapping(path):
    """Load segment-to-global mapping, preferring the unambiguous track_id field."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = {"camera", "global_id"} - fields
        if missing:
            raise ValueError(f"Mapping missing columns: {', '.join(sorted(missing))}")
        identity_column = "track_id" if "track_id" in fields else "local_id" if "local_id" in fields else None
        if identity_column is None:
            raise ValueError("Mapping requires track_id (or legacy local_id)")
        result = {}
        for line, row in enumerate(reader, 2):
            try:
                camera = int(row["camera"])
                identity = int(row[identity_column])
                global_id = int(row["global_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Mapping line {line} is malformed") from error
            if camera not in (1, 2) or identity < 0 or global_id < 1:
                raise ValueError(f"Mapping line {line} has an invalid identity")
            key = (camera, identity)
            if key in result:
                raise ValueError(f"Duplicate mapping for camera {camera}, track {identity}")
            result[key] = global_id
    if not result:
        raise ValueError("Mapping is empty")
    return result, identity_column


def load_track_index(path):
    """Load one valid observation for each frame/segment; ambiguity is rejected."""
    required = {"frame", "track_id", "confidence", "x1", "y1", "x2", "y2"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        by_id = defaultdict(dict)
        for line, row in enumerate(reader, 2):
            try:
                frame = int(row["frame"])
                identity = int(row["track_id"])
                confidence = float(row["confidence"])
                x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: malformed observation on line {line}") from error
            values = (confidence, x1, y1, x2, y2)
            if (
                frame < 0
                or identity < 0
                or not all(math.isfinite(value) for value in values)
                or not 0 <= confidence <= 1
                or x2 <= x1
                or y2 <= y1
            ):
                raise ValueError(f"{path}: invalid observation on line {line}")
            for key in ("field_x", "field_y"):
                if key in row and row[key] != "":
                    try:
                        value = float(row[key])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path}: invalid {key} on line {line}") from error
                    if not math.isfinite(value):
                        raise ValueError(f"{path}: non-finite {key} on line {line}")
            if frame in by_id[identity]:
                raise ValueError(f"{path}: duplicate frame/segment ({frame}, {identity})")
            by_id[identity][frame] = row
    if not by_id:
        raise ValueError(f"{path}: tracking CSV is empty")
    return by_id


def resolve_fusion_context(fusion_run_path, tracks, requested_offset=None):
    """Bind review inputs and offset to the completed fusion run."""
    fusion_run_path = Path(fusion_run_path).resolve()
    try:
        run = json.loads(fusion_run_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid fusion run JSON: {fusion_run_path}") from error
    if run.get("status") != "complete":
        raise ValueError("Fusion run must have status=complete")
    try:
        offset = int(run["offset_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Fusion run has no valid offset_frames") from error
    if (isinstance(run.get("offset_frames"), bool)
            or float(run["offset_frames"]) != offset):
        raise ValueError("Fusion run has no valid offset_frames")
    if requested_offset is not None and int(requested_offset) != offset:
        raise ValueError(
            f"Requested offset {requested_offset} disagrees with fusion run offset {offset}"
        )

    records = run.get("inputs")
    if records is not None:
        if not isinstance(records, list) or len(records) < 2:
            raise ValueError("Fusion run inputs do not identify both tracking files")
        expected = records[:2]
    else:
        sources = run.get("sources")
        if not isinstance(sources, list) or len(sources) < 2:
            raise ValueError("Fusion run does not identify both tracking files")
        expected = [{"path": value} for value in sources[:2]]

    for camera, (actual, record) in enumerate(zip(tracks, expected), 1):
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"Fusion run camera {camera} input is malformed")
        expected_path = Path(record["path"]).expanduser()
        if not expected_path.is_absolute():
            expected_path = fusion_run_path.parent / expected_path
        if expected_path.resolve() != Path(actual).resolve():
            raise ValueError(f"Camera {camera} tracking file does not match fusion run")
        facts = file_provenance(actual, digest=bool(record.get("sha256")))
        if "bytes" in record and int(record["bytes"]) != facts["bytes"]:
            raise ValueError(f"Camera {camera} tracking file size changed after fusion")
        if record.get("sha256") and record["sha256"] != facts["sha256"]:
            raise ValueError(f"Camera {camera} tracking file changed after fusion")
    return run, offset


def risk_score(candidate):
    distance = float(candidate["median_distance"])
    appearance = candidate.get("appearance", "")
    measured_appearance = None if appearance == "" else float(appearance)
    fraction = float(candidate["support_fraction"])
    support = int(candidate["support"])
    values = (distance, fraction) + (() if measured_appearance is None else (measured_appearance,))
    if (not all(math.isfinite(value) for value in values) or distance < 0
            or not 0 <= fraction <= 1 or support < 1
            or measured_appearance is not None and not -1e-6 <= measured_appearance <= 2):
        raise ValueError("Invalid candidate evidence values")
    appearance = .5 if measured_appearance is None else max(0., measured_appearance) / .45
    return distance / .10 + .55 * appearance + .8 * (1 - fraction) + .4 / np.sqrt(support)


def choose_frame(first, second, offset_frames=0, first_frame=None, last_frame=None,
                 support_frames=None):
    common = sorted(set(first) & {frame - offset_frames for frame in second})
    if support_frames is not None:
        evidence = set(support_frames)
        common = [frame for frame in common if frame in evidence]
    if first_frame is not None:
        common = [frame for frame in common if frame >= int(first_frame)]
    if last_frame is not None:
        common = [frame for frame in common if frame <= int(last_frame)]
    if not common:
        return None
    def quality(frame):
        one, two = first[frame], second[frame + offset_frames]
        h1, h2 = (float(row["y2"]) - float(row["y1"]) for row in (one, two))
        confidence = np.sqrt(float(one["confidence"]) * float(two["confidence"]))
        try:
            distance = np.linalg.norm(
                np.asarray([float(one["field_x"]), float(one["field_y"])]) -
                np.asarray([float(two["field_x"]), float(two["field_y"])])
            )
        except (KeyError, TypeError, ValueError):
            distance = .1
        return min(h1, h2) * confidence - 150 * distance
    return max(common, key=quality)


def crop_context(frame, row, size=320):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = max(20, box_w * .9), max(20, box_h * .35)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Empty crop")
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 28, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    bx1, by1 = round((x1 - left) * scale) + x, round((y1 - top) * scale) + y
    bx2, by2 = round((x2 - left) * scale) + x, round((y2 - top) * scale) + y
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (65, 220, 180), 2)
    return canvas


def read_frame(capture, frame_number):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}")
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Global ID adaylari icin insan inceleme paketi")
    for name in (
        "video1", "video2", "tracks1", "tracks2", "candidates", "mapping",
        "fusion-run", "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--offset-frames",
        type=int,
        default=None,
        help="Varsa fusion run ile ayni olmalidir; verilmezse run.json'dan okunur",
    )
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--per-global", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    path_names = ("video1", "video2", "tracks1", "tracks2", "candidates", "mapping", "fusion_run")
    for name in path_names:
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    if args.max_pairs < 1 or args.per_global < 1:
        raise ValueError("max-pairs and per-global must be positive")
    expected_parent = args.fusion_run.parent
    if args.candidates != expected_parent / "candidates.csv" or args.mapping != expected_parent / "mapping.csv":
        raise ValueError("Candidates and mapping must come from the fusion-run directory")
    fusion_run, offset_frames = resolve_fusion_context(
        args.fusion_run, (args.tracks1, args.tracks2), args.offset_frames
    )
    provenance = {
        "fusion_run": file_provenance(args.fusion_run, digest=True),
        "videos": [file_provenance(path) for path in (args.video1, args.video2)],
        "tracks": [file_provenance(path, digest=True) for path in (args.tracks1, args.tracks2)],
        "candidates": file_provenance(args.candidates, digest=True),
        "mapping": file_provenance(args.mapping, digest=True),
    }
    mappings, mapping_identity_column = load_mapping(args.mapping)
    tracks = [load_track_index(path) for path in (args.tracks1, args.tracks2)]
    candidates = [row for row in load_csv(args.candidates) if truthy(row.get("accepted"))]
    candidates.sort(key=risk_score, reverse=True)
    chosen, per_global = [], Counter()
    for candidate in candidates:
        one, two = int(candidate["camera1_id"]), int(candidate["camera2_id"])
        global1, global2 = mappings.get((1, one)), mappings.get((2, two))
        if global1 is None or global1 != global2 or per_global[global1] >= args.per_global:
            continue
        first_frame = candidate.get("first_frame", "")
        last_frame = candidate.get("last_frame", "")
        first_frame = None if first_frame == "" else int(first_frame)
        last_frame = None if last_frame == "" else int(last_frame)
        if first_frame is not None and last_frame is not None and last_frame < first_frame:
            raise ValueError(f"Candidate ({one}, {two}) has an invalid evidence span")
        raw_support = candidate.get("support_frames", "")
        support_frames = json.loads(raw_support) if raw_support else None
        if support_frames is not None and (
            not isinstance(support_frames, list)
            or any(type(frame) is not int or frame < 0 for frame in support_frames)
        ):
            raise ValueError(f"Candidate ({one}, {two}) has invalid support frames")
        frame = choose_frame(
            tracks[0].get(one, {}),
            tracks[1].get(two, {}),
            offset_frames,
            first_frame,
            last_frame,
            support_frames,
        )
        if frame is None:
            continue
        chosen.append((candidate, one, two, global1, frame))
        per_global[global1] += 1
        if len(chosen) >= args.max_pairs:
            break
    if not chosen:
        raise RuntimeError("No reviewable accepted cross-camera pairs")
    captures = [cv2.VideoCapture(str(path)) for path in (args.video1, args.video2)]
    video_info = []
    rows = []
    try:
        if not all(capture.isOpened() for capture in captures):
            raise RuntimeError("Videos could not be opened")
        for camera, capture in enumerate(captures, 1):
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise ValueError(f"Camera {camera} has invalid video metadata")
            video_info.append({"fps": fps, "frames": frame_count})
        for camera, by_id in enumerate(tracks, 1):
            frame_count = video_info[camera - 1]["frames"]
            if any(frame < 0 or frame >= frame_count for values in by_id.values() for frame in values):
                raise ValueError(f"Camera {camera} tracking frame is outside the source video")
        for _candidate, _one, _two, _global_id, frame in chosen:
            if not 0 <= frame < video_info[0]["frames"]:
                raise ValueError(f"Camera 1 review frame is outside the source video: {frame}")
            second_frame = frame + offset_frames
            if not 0 <= second_frame < video_info[1]["frames"]:
                raise ValueError(f"Camera 2 review frame is outside the source video: {second_frame}")

        # No user-visible artifact is created until every source and frame bound
        # has been validated.
        args.output_dir.mkdir(parents=True)
        image_dir = args.output_dir / "images"
        image_dir.mkdir()
        for index, (candidate, one, two, global_id, frame) in enumerate(chosen, 1):
            observations = [tracks[0][one][frame], tracks[1][two][frame + offset_frames]]
            image_names = []
            image_digests = []
            for camera in (1, 2):
                source_frame = frame + (offset_frames if camera == 2 else 0)
                image = crop_context(read_frame(captures[camera - 1], source_frame), observations[camera - 1])
                name = f"pair_{index:03d}_camera{camera}.jpg"
                if not cv2.imwrite(str(image_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"Could not write review image: {image_dir / name}")
                image_names.append(f"images/{name}")
                image_digests.append(file_provenance(image_dir / name, digest=True)["sha256"])
            rows.append({
                "review_id": index,
                "frame_camera1": frame,
                "frame_camera2": frame + offset_frames,
                "time_seconds": f"{frame / video_info[0]['fps']:.3f}",
                "predicted_global_id": global_id,
                "camera1_segment_id": one,
                "camera1_local_id": observations[0].get("local_track_id", one),
                "camera2_segment_id": two,
                "camera2_local_id": observations[1].get("local_track_id", two),
                "support_frame_count": candidate["support"],
                "selection_evidence": "actual_support_frame" if candidate.get("support_frames") else
                    "evidence_span_only" if candidate.get("first_frame") else "legacy_cooccurrence",
                "support_fraction": candidate["support_fraction"],
                "median_field_distance": candidate["median_distance"],
                "appearance_distance": candidate.get("appearance", ""),
                "risk_score": f"{risk_score(candidate):.6f}",
                "camera1_image": image_names[0],
                "camera2_image": image_names[1],
                "camera1_image_sha256": image_digests[0],
                "camera2_image_sha256": image_digests[1],
                "decision": "",
                "person_label": "",
                "notes": "",
            })
    finally:
        for capture in captures:
            capture.release()
    storage_key = review_identity(rows, provenance)
    fields = list(rows[0])
    with (args.output_dir / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    page = render_review_page(rows, storage_key)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    report = {"schema": "halisaha_global_id_review_v3", "status": "complete",
              "review_pairs": len(rows), "accepted_candidates": len(candidates),
              "selection": "highest diagnostic risk first; capped per provisional global component",
              "identity_status": "requires_human_review", "offset_frames": offset_frames,
              "offset_source": "fusion_run_verified" if args.offset_frames is not None else "fusion_run_derived",
              "fusion_identity_status": fusion_run.get("identity_status", "unknown"),
              "mapping_identity_column": mapping_identity_column,
              "review_storage_key": storage_key,
              "review_source_id": rows[0]["review_source_id"],
              "provenance": provenance,
              "output": str((args.output_dir / "index.html").resolve())}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def render_review_page(rows, storage_key):
    cards = []
    for row in rows:
        i = row["review_id"]
        cards.append(f'''<section class="card" data-index="{i}">
<h2>#{i} · önerilen G{row['predicted_global_id']} · {row['time_seconds']} sn</h2>
<div class="images"><figure><img loading="lazy" alt="Kamera 1 oyuncu kutusu" src="{html.escape(row['camera1_image'])}"><figcaption>Kamera 1 · L{html.escape(str(row['camera1_local_id']))}</figcaption></figure>
<figure><img loading="lazy" alt="Kamera 2 oyuncu kutusu" src="{html.escape(row['camera2_image'])}"><figcaption>Kamera 2 · L{html.escape(str(row['camera2_local_id']))}</figcaption></figure></div>
<p>Destek: {row['support_frame_count']} kare · konum farkı: {float(row['median_field_distance']):.4f} · görünüş farkı: {html.escape(str(row['appearance_distance']))}</p>
<div class="choices"><label><input type="radio" name="d{i}" value="correct"> Aynı oyuncu</label>
<label><input type="radio" name="d{i}" value="wrong"> Farklı oyuncu</label>
<label><input type="radio" name="d{i}" value="unsure"> Kararsız</label></div>
<input class="person" placeholder="Kişi etiketi (örn. P01)"><input class="notes" placeholder="Not"></section>''')
    embedded = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    page = f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Global ID inceleme</title>
<style>body{{font:16px system-ui;background:#101820;color:#eef5ef;margin:24px}}.intro{{max-width:900px}}.toolbar{{position:sticky;top:0;z-index:2;background:#101820;padding:12px 0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}.card{{background:#192833;padding:18px;margin:18px 0;border-radius:12px;max-width:760px}}.images{{display:flex;gap:16px;flex-wrap:wrap}}figure{{margin:0;max-width:100%}}img{{width:320px;height:320px;max-width:100%;object-fit:contain;background:#111}}figcaption{{text-align:center}}label{{margin-right:18px;display:inline-block;padding:8px 0}}input.person,input.notes{{display:block;width:95%;margin-top:12px;padding:8px}}button{{padding:12px 18px;background:#50d9b1;border:0;border-radius:8px;font-weight:700;cursor:pointer}}#save-status{{font-size:13px;color:#a5bdca}}</style></head><body>
<div class="intro"><h1>Global ID insan kontrolü</h1><p>İki kutudaki kişinin aynı olup olmadığını işaretle. Bu kontrol seçilen an içindir; tek kare tüm izin kimliğini doğrulamaz. G numarası sistemin önerisidir. Net görünmeyen örneklerde “Kararsız” seçebilirsin.</p></div>
<div class="toolbar"><button onclick="download()">Sonuç CSV'sini indir</button><span id="progress"></span><span id="save-status"></span></div>{''.join(cards)}
<script>
const rows={embedded};
const storageKey={json.dumps(storage_key)};
const cards=[...document.querySelectorAll('.card')];
function q(s,e){{return e.querySelector(s)}}
function esc(v){{v=String(v??'');return ['"',',',String.fromCharCode(10),String.fromCharCode(13)].some(c=>v.includes(c))?'"'+v.replaceAll('"','""')+'"':v}}
function collect(){{cards.forEach((card,j)=>{{rows[j].decision=q('input[type=radio]:checked',card)?.value||'';rows[j].person_label=q('.person',card).value;rows[j].notes=q('.notes',card).value}});document.getElementById('progress').textContent=rows.filter(r=>r.decision).length+' / '+rows.length+' incelendi'}}
function save(){{collect();try{{localStorage.setItem(storageKey,JSON.stringify(rows));document.getElementById('save-status').textContent='Bu tarayıcıda otomatik kaydedildi. Paylaşmak için CSV indir.'}}catch(e){{document.getElementById('save-status').textContent='Otomatik kayıt kullanılamıyor. Ayrılmadan önce CSV indir.'}}}}
try{{const stored=JSON.parse(localStorage.getItem(storageKey)||'[]');if(!Array.isArray(stored))throw new Error('Invalid saved review');const byEvidence=new Map(stored.filter(r=>r&&typeof r.review_evidence_id==='string').map(r=>[r.review_evidence_id,r]));cards.forEach((card,j)=>{{const r=byEvidence.get(rows[j].review_evidence_id);if(!r||r.review_source_id!==rows[j].review_source_id)return;const radio=[...card.querySelectorAll('input[type=radio]')].find(i=>i.value===r.decision);if(radio)radio.checked=true;q('.person',card).value=typeof r.person_label==='string'?r.person_label:'';q('.notes',card).value=typeof r.notes==='string'?r.notes:''}})}}catch(e){{document.getElementById('save-status').textContent='Önceki kayıt okunamadı. Ayrılmadan önce CSV indir.'}}
cards.forEach(card=>card.addEventListener('input',save));collect();
function download(){{save();const keys=Object.keys(rows[0]);const csv=[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join(String.fromCharCode(13,10));const a=document.createElement('a');const url=URL.createObjectURL(new Blob([String.fromCharCode(0xFEFF),csv],{{type:'text/csv;charset=utf-8'}}));a.href=url;a.download='review_completed.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}}
</script></body></html>'''
    return page


if __name__ == "__main__":
    main()
