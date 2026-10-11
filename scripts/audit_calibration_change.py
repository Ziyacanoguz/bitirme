"""Small visual spot-check and all-row projection coverage; not a MOT benchmark."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from halisaha.multicamera import valid_position, xy


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    base=ROOT/'outputs/experiments/calibration_fusion_20260904'
    old=ROOT/'outputs/experiments/reviewed407_20260904'
    target=base/'calibration_change_audit.json'
    if target.exists():raise FileExistsError(target)
    pairs=[(0,11),(1,4),(2,3),(4,2),(5,6),(6,7),(8,9)]
    report=dict(note='Seven frame-45000 pairs visually spot-checked by assistant using crops and field context; not user-verified tracking ground truth.',
                frame=45000,pairs=pairs,coverage={},pair_distances={})
    for name,paths in [('old',[old/f'camera{c}/stitched.csv' for c in (1,2)]),
                       ('line_model',[base/f'camera{c}_field.csv' for c in (1,2)])]:
        rows=[read(p) for p in paths]
        report['coverage'][name]=[dict(rows=len(data),valid=sum(valid_position(r) for r in data),
                                      valid_fraction=sum(valid_position(r) for r in data)/len(data)) for data in rows]
        frames=[{int(r['track_id']):r for r in data if int(r['frame'])==45000} for data in rows]
        errors=[float(np.linalg.norm(xy(frames[0][a])-xy(frames[1][b]))) for a,b in pairs]
        report['pair_distances'][name]=dict(values=errors,median=float(np.median(errors)),mean=float(np.mean(errors)))
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
