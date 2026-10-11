import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from halisaha.multicamera import (
    ASSOCIATION_SCHEMA_VERSION,
    associate,
    audit_time_offset,
    build_component_audit,
    fuse_frames,
)


# Even a successful conservative run may find no cross-camera candidates or no
# calibration-valid positions. Keep those empty outputs readable by consumers.
FUSED_FIELDS = [
    'frame', 'time', 'source_frame_camera1', 'source_frame_camera2',
    'offset_frames_applied', 'track_id', 'global_id', 'component_uid',
    'display_label', 'identity_evidence_score', 'identity_evidence_kind',
    'identity_evidence_semantics', 'field_x', 'field_y', 'position_disagreement',
    'confidence', 'detection_confidence_min', 'calibration_supported', 'status',
    'cameras', 'camera1_id', 'camera2_id', 'camera1_local_track_id',
    'camera2_local_track_id', 'observation_count',
]
CANDIDATE_FIELDS = [
    'camera1_id', 'camera2_id', 'first_frame', 'last_frame', 'support_frames',
    'support_span_frames', 'support', 'longest_consecutive_support', 'overlap',
    'support_fraction', 'median_distance', 'p95_distance', 'median_match_cost',
    'appearance', 'appearance_available', 'evidence_score', 'accepted',
    'component_link', 'reason', 'component_max_appearance',
]


def load_rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_features(path):
    data=np.load(path,allow_pickle=False)
    ids,embeddings=data['ids'],data['embeddings']
    if len(ids)!=len(embeddings) or embeddings.ndim!=2:
        raise ValueError(f'Invalid Re-ID archive shape: {path}')
    output={}
    for identity,embedding in zip(ids,embeddings):
        identity=int(identity)
        if identity in output:
            raise ValueError(f'Duplicate Re-ID identity {identity}: {path}')
        embedding=np.asarray(embedding,dtype=np.float64)
        norm=float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or norm<=1e-9:
            raise ValueError(f'Invalid Re-ID embedding for identity {identity}: {path}')
        output[identity]=embedding/norm
    return output


def write_csv(path,rows,fieldnames=None):
    if fieldnames is None:
        if not rows:
            raise ValueError(f'fieldnames required for empty CSV: {path}')
        fieldnames=list(rows[0])
        fieldnames.extend(sorted({key for row in rows for key in row}-set(fieldnames)))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def file_fingerprint(path):
    digest=sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def main():
    p=argparse.ArgumentParser(description='Temkinli iki kamera ortak takip ve global ID verisi')
    for arg in ('camera1','camera2','reid1','reid2','output-dir'):
        p.add_argument('--'+arg,required=True,type=Path)
    p.add_argument('--offset-frames',type=int,default=0,help='C1 f karesi C2 f+offset ile eslesir')
    p.add_argument('--fps',type=float,default=25)
    p.add_argument('--min-support',type=int,default=15)
    p.add_argument('--max-distance',type=float,default=.10)
    p.add_argument('--min-fraction',type=float,default=.35)
    p.add_argument('--min-consecutive-support',type=int,default=4)
    p.add_argument('--max-p95-distance',type=float,default=.08)
    p.add_argument('--appearance-gate',type=float,default=.45)
    p.add_argument('--ambiguity-margin',type=float,default=.12)
    p.add_argument('--missing-appearance-min-support',type=int,default=15)
    p.add_argument('--missing-appearance-min-fraction',type=float,default=.70)
    p.add_argument('--missing-appearance-max-p95',type=float,default=.065)
    p.add_argument('--component-appearance-gate',type=float,default=.50)
    p.add_argument('--max-disagreement',type=float,default=.10)
    args=p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows1,rows2=load_rows(args.camera1),load_rows(args.camera2)
    features1,features2=load_features(args.reid1),load_features(args.reid2)
    mapping,shared,candidates,diagnostics=associate(
        rows1,rows2,features1,features2,
        offset_frames=args.offset_frames,min_support=args.min_support,
        max_distance=args.max_distance,min_fraction=args.min_fraction,
        min_consecutive_support=args.min_consecutive_support,
        max_p95_distance=args.max_p95_distance,
        appearance_gate=args.appearance_gate,margin=args.ambiguity_margin,
        missing_appearance_min_support=args.missing_appearance_min_support,
        missing_appearance_min_fraction=args.missing_appearance_min_fraction,
        missing_appearance_max_p95=args.missing_appearance_max_p95,
        component_appearance_gate=args.component_appearance_gate,
        return_diagnostics=True,
    )
    components,component_evidence,component_uids,pair_evidence=build_component_audit(
        mapping,shared,candidates,diagnostics,rows1,rows2,offset_frames=args.offset_frames)
    fused,totals=fuse_frames(
        rows1,rows2,mapping,shared,offset_frames=args.offset_frames,fps=args.fps,
        max_disagreement=args.max_disagreement,pair_evidence=pair_evidence,
        component_evidence=component_evidence,component_uids=component_uids)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'fused.csv',fused, None if fused else FUSED_FIELDS)
    write_csv(args.output_dir/'candidates.csv',candidates, None if candidates else CANDIDATE_FIELDS)
    write_csv(args.output_dir/'continuity_evidence.csv',diagnostics['continuity'],[
        'camera','source_local_id','first_track_id','second_track_id','first_end_frame',
        'second_start_frame','gap_frames','position_distance','appearance','evidence_score',
        'accepted','component_link','reason','component_max_appearance'])
    write_csv(args.output_dir/'components.csv',components)
    source_local={}
    for camera,rows in ((1,rows1),(2,rows2)):
        for row in rows:
            source_local[(camera,int(row['track_id']))]=int(row.get('local_track_id',row['track_id']))
    write_csv(args.output_dir/'mapping.csv',[
        dict(camera=c,track_id=i,local_id=i,source_local_id=source_local[(c,i)],global_id=g,
             component_uid=component_uids[g],shared=int(g in shared),
             identity_evidence_score=(component_evidence[g] if component_evidence[g] is not None else ''),
             evidence_semantics='weakest_graph_link_heuristic_not_probability')
        for (c,i),g in sorted(mapping.items())])
    sync=audit_time_offset(rows1,rows2,candidates)
    (args.output_dir/'time_offset_audit.json').write_text(json.dumps(sync,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    accepted=[row for row in candidates if row['accepted']]
    report=dict(status='complete',schema_version=ASSOCIATION_SCHEMA_VERSION,
                 identity_status='provisional_not_human_verified',mode='offline_temporal_graph',
                 coordinate_units='normalized_not_meters',offset_frames=args.offset_frames,
                 inputs=[file_fingerprint(path) for path in (
                     args.camera1,args.camera2,args.reid1,args.reid2)],
                 shared_components=len(shared),total_components=len(set(mapping.values())),
                 accepted_candidate_pairs=len(accepted),candidate_pairs=len(candidates),
                 accepted_edges=len(accepted),candidate_edges=len(candidates),
                 cross_camera_component_links=sum(row['accepted'] and row['component_link'] for row in candidates),
                 cross_camera_corroborating_edges=sum(row['accepted'] and not row['component_link'] for row in candidates),
                 continuity_component_links=sum(row['accepted'] and row['component_link'] for row in diagnostics['continuity']),
                 candidate_decisions=dict(Counter(row['reason'] for row in candidates)),
                 continuity_decisions=dict(Counter(row['reason'] for row in diagnostics['continuity'])),
                 totals=totals,settings=dict(min_support=args.min_support,max_distance=args.max_distance,
                                            min_fraction=args.min_fraction,
                                            min_consecutive_support=args.min_consecutive_support,
                                            max_p95_distance=args.max_p95_distance,
                                            appearance_gate=args.appearance_gate,
                                            ambiguity_margin=args.ambiguity_margin,
                                            missing_appearance_min_support=args.missing_appearance_min_support,
                                            missing_appearance_min_fraction=args.missing_appearance_min_fraction,
                                            missing_appearance_max_p95=args.missing_appearance_max_p95,
                                            component_appearance_gate=args.component_appearance_gate,
                                            max_disagreement=args.max_disagreement),
                 evidence_score_semantics='heuristic_0_to_1_not_probability_or_measured_accuracy',
                 notes=['No team classification','No forced player count',
                        'Same-camera simultaneous tracks cannot enter one component',
                        'Local continuity is resolved before cross-camera edges',
                        'Missing Re-ID appearance requires stronger geometric evidence',
                        'Large cross-view disagreement is shown separately, not averaged'])
    (args.output_dir/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(json.dumps(sync,indent=2))


if __name__=='__main__':
    main()
