"""Readable crop sheet for human cross-camera matching audits."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--frame", type=int, default=45000)
    p.add_argument("--label", default="Camera")
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.tracks.open(newline="", encoding="utf-8") as source:
        rows=[r for r in csv.DictReader(source) if int(r["frame"])==args.frame]
    cap=cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,args.frame)
    ok, image=cap.read()
    cap.release()
    if not ok or not rows:
        raise ValueError("No image or tracks")
    canvas=np.full((40+300*((len(rows)+5)//6),1200,3),(25,31,25),np.uint8)
    cv2.putText(canvas,f"{args.label} | frame {args.frame}",(15,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,240,230),1)
    for i,r in enumerate(sorted(rows,key=lambda r:int(r['track_id']))):
        x1,y1,x2,y2=[round(float(r[k])) for k in ('x1','y1','x2','y2')]
        crop=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
        if not crop.size:
            continue
        ratio=min(180/crop.shape[1],230/crop.shape[0])
        crop=cv2.resize(crop,(max(1,round(crop.shape[1]*ratio)),max(1,round(crop.shape[0]*ratio))))
        x,y=(i%6)*200,40+(i//6)*300
        canvas[y+36:y+36+crop.shape[0],x+(200-crop.shape[1])//2:x+(200-crop.shape[1])//2+crop.shape[1]]=crop
        cv2.putText(canvas,f"ID {r['track_id']}",(x+10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.65,(90,220,240),1)
        cv2.putText(canvas,f"{float(r['field_x']):.2f}, {float(r['field_y']):.2f}",(x+10,y+286),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,230,220),1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(args.output),canvas)


if __name__=='__main__':
    main()
