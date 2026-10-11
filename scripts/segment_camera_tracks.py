"""Bound appearance drift by splitting long tracker IDs into short audit segments."""
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tracks',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--block-frames',type=int,default=50)
    p.add_argument('--max-gap',type=int,default=8)
    args=p.parse_args()
    if args.output.exists() or args.block_frames<1:
        raise ValueError('Choose new output and positive block-frames')
    with args.tracks.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        fields=list(dict.fromkeys(list(reader.fieldnames)+['local_track_id']))
        rows=list(reader)
    start=min(int(r['frame']) for r in rows)
    groups=defaultdict(list)
    for row in rows:
        groups[int(row['track_id'])].append(row)
    segment=-1
    for identity,values in sorted(groups.items()):
        last_frame=last_block=None
        last_point=None
        for row in sorted(values,key=lambda r:int(r['frame'])):
            frame=int(row['frame'])
            block=(frame-start)//args.block_frames
            point=np.array([float(row['field_x']),float(row['field_y'])])
            if (last_frame is None or block!=last_block or frame-last_frame>args.max_gap
                    or np.linalg.norm(point-last_point)>.12):
                segment+=1
            row.update(local_track_id=identity,track_id=segment)
            last_frame,last_block,last_point=frame,block,point
    rows.sort(key=lambda r:(int(r['frame']),int(r['track_id'])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Local IDs: {len(groups)}, short segments: {segment+1}')


if __name__=='__main__':
    main()
