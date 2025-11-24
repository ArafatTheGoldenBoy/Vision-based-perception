#!/usr/bin/env python3
import json, csv
from pathlib import Path

ROOT = Path('dataset_carla10')
SPLIT = 'train'
COCO_IN = ROOT / 'annotations' / f'instances_{SPLIT}.json'   # your existing COCO from carla_to_coco
COCO_OUT = ROOT / 'annotations' / f'instances_{SPLIT}_with_depth.json'
INDEX_CSV = ROOT / 'meta' / f'index_{SPLIT}.csv'

# Build frame->depth_path map
frame2depth = {}
with open(INDEX_CSV, 'r') as f:
    r = csv.DictReader(f)
    for row in r:
        frame = int(row['frame'])
        frame2depth[frame] = row['depth_path']

with open(COCO_IN, 'r') as f:
    coco = json.load(f)

# Assume each image file name is frame-based like 000123.png
for img in coco['images']:
    name = Path(img['file_name']).stem
    try:
        frame = int(name)
    except:
        # if your naming is different, map via your own rule here
        frame = int(img.get('frame_id', -1))
    if frame in frame2depth:
        img['depth_file'] = frame2depth[frame]
    else:
        img['depth_file'] = None

with open(COCO_OUT, 'w') as f:
    json.dump(coco, f)
print('wrote', COCO_OUT)
