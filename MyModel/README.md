# YourName-MNv4-DepthSSD

Joint **object detection + monocular depth** for CARLA‑style driving scenes, built around:

- MobileNetV4 backbone (lightweight, real-time)
- HoPE‑ready Transformer neck on P3 (1/8 scale)
- Titans‑style external memory for temporal context
- SSDLite detection head
- Monocular depth head with metric supervision
- Robust per-object distance estimation from depth

Under the hood, the core joint architecture is implemented in [`model_defs.py`](model_defs.py). :contentReference[oaicite:0]{index=0}  
A legacy shim file, [`joint_det_depth_ho_pe_mem_det_depth_with_transformer_neck_memory.py`](joint_det_depth_ho_pe_mem_det_depth_with_transformer_neck_memory.py), just re‑exports everything from `model_defs` to keep old code working. :contentReference[oaicite:1]{index=1}  

---

## Architecture

```mermaid
graph TD
    RGB[RGB Image 640×360] --> B[MobileNetV4 Backbone]
    B --> P3[P3 (1/8 feature)]
    B --> P4[P4 (1/16 feature)]
    B --> P5[P5 (1/32 feature)]

    P3 --> NECK[TransformerNeck<br/>+ HoPE hook]
    NECK --> MEM[MemoryXAttn<br/>(Titans-style memory)]
    MEM --> P3m[Depth-aware P3]

    P3m --> DEPTH[DepthHead<br/>(inverse depth)]
    DEPTH --> DMap[Depth Map (m)]

    P3m --> SSD[SSDLite Head]
    SSD --> DETS[2D Detections<br/>(boxes, classes, scores)]

    DMap --> AGG[Robust per-object<br/>distance (median+MAD)]
    DETS --> AGG
    AGG --> Zest[Estimated distances<br/>(meters)]


---

## 2. `INSTRUCTIONS.md` (step‑by‑step how to run everything)

```markdown
# INSTRUCTIONS – Running YourName-MNv4-DepthSSD

This document walks you through:

1. Setting up the environment  
2. Capturing CARLA data (RGB + depth)  
3. Creating COCO + depth annotations  
4. Wiring your MobileNetV4 + SSDLite model  
5. Training joint detection + depth  
6. Running online inference in CARLA

---

## 1. Environment setup

### 1.1. Dependencies

- Python 3.9+  
- PyTorch + CUDA (optional but strongly recommended)  
- CARLA 0.10.x (or compatible)  
- Python packages:
  - `numpy`
  - `opencv-python`
  - `timm` (for MobileNetV4)
  - `pycocotools`
  - `torch`, `torchvision`
  - `carla` Python API

Example (adjust versions as needed):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python timm pycocotools
# CARLA Python egg should be added to PYTHONPATH per CARLA docs

2. Capture data from CARLA

Script: carla_capture_det_depth.py

This script connects to a CARLA server, spawns an ego vehicle and two cameras (RGB + depth), and records:

RGB images → dataset_carla10/images/{split}/xxxxxx.png

depth PNGs (16‑bit mm) → dataset_carla10/depth/{split}/xxxxxx.png

intrinsics → dataset_carla10/meta/intrinsics.json

per‑frame index CSV (pose, file paths) → dataset_carla10/meta/index_{split}.csv

2.1. Start CARLA server

Run CARLA (example command):

./CarlaUE4.sh -quality-level=Epic -world-port=2000

2.2. Run the capture script

In another terminal:

python carla_capture_det_depth.py


You can adjust at the top of the script:

OUT – output root (default: dataset_carla10)

SPLIT – 'train', 'val', or 'test'

N_FRAMES – number of frames to capture (default 5000).

When finished, you’ll have:

dataset_carla10/images/train/*.png

dataset_carla10/depth/train/*.png

dataset_carla10/meta/index_train.csv

Repeat for val/test splits if desired (change SPLIT and rerun).

3. Create COCO + depth annotations

You need a COCO JSON file with your detection annotations (e.g., cars, pedestrians, signs):

dataset_carla10/annotations/instances_train.json

dataset_carla10/annotations/instances_val.json

dataset_carla10/annotations/instances_test.json

(You can generate these via your own CARLA→COCO script.)

3.1. Attach depth paths to COCO

Use coco_pair_depth.py
 to merge depth file paths into the COCO images, based on the frame index CSV.

Example:

cd dataset_carla10
python ../coco_pair_depth.py   # uses ROOT = 'dataset_carla10' in the script


This will:

Read meta/index_train.csv

Read annotations/instances_train.json

Write annotations/instances_train_with_depth.json where each image has a "depth_file" field pointing to the matching PNG.

Repeat or adapt for val/test splits if you want depth there as well.

4. Dataset loader

Script: dataset_joint_det_depth.py

The CocoDetDepth dataset:

Loads RGB images from img_root / file_name

Loads depth from depth_file (uint16 mm) and converts to meters

Constructs detection targets from COCO annotations (bbox + category_id)

Minimal usage:

from dataset_joint_det_depth import CocoDetDepth

ds = CocoDetDepth(
    'dataset_carla10/annotations/instances_train_with_depth.json',
    'dataset_carla10'
)
sample = ds[0]
print(sample.keys())  # image, boxes, labels, depth, depth_valid, ...


You don’t need to modify this file unless your directory layout differs.

5. Wiring YourName-MNv4-DepthSSD

Core model lives in model_defs.py
:

JointDetDepthHoPEMem – wraps:

backbone (you will plug in MobileNetV4 here),

TransformerNeck (HoPE‑ready, operates on P3),

MemoryXAttn (Titans‑style external memory),

det_head (your SSDLite head),

DepthHead (inverse depth).

LossWeights and compute_multitask_loss – multi‑task loss that combines detection + depth.

Utilities: robust_object_distance, ground_depth_map, scale_from_ground.

There is also a legacy shim file:

joint_det_depth_ho_pe_mem_det_depth_with_transformer_neck_memory.py – simply re‑exports everything from model_defs, so older code can keep importing it.

5.1. Implement your backbone + SSDLite head

You will:

Implement a MobileNetV4-based backbone that returns at least a dict with 'P3', 'P4', 'P5' feature maps.

Implement a SSDLite head that consumes these features and outputs classification + box regression tensors (e.g. {'cls_logits': ..., 'bbox_reg': ..., 'fm_shapes': ...}).

Plug them into JointDetDepthHoPEMem.

You can put these classes directly into model_defs.py so they can access the shared utilities.

6. Training

Script: train_joint.py

This script wires together:

CocoDetDepth dataset

Your JointDetDepthHoPEMem model

A detection loss (SSD/SSDLite)

compute_multitask_loss for joint training

6.1. Implement build_model()

In train_joint.py you’ll see:

def build_model() -> JointDetDepthHoPEMem:
    """Construct JointDetDepthHoPEMem with your backbone + detection head."""
    raise NotImplementedError('Plug your backbone + detection head here')


Replace this with something like:

from model_defs import JointDetDepthHoPEMem
from model_defs import MNv4Backbone, MySSDLiteHead  # your implementations

def build_model() -> JointDetDepthHoPEMem:
    backbone = MNv4Backbone(pretrained=True, out_ch=256)
    det_head = MySSDLiteHead(num_classes=NUM_CLASSES, in_ch=256, num_anchors=6)
    model = JointDetDepthHoPEMem(
        backbone=backbone,
        det_head=det_head,
        p3_ch=256,
        use_neck=True,
        use_mem=True
    )
    return model

6.2. Implement detection loss

compute_multitask_loss expects a detection loss of the form:

L_det, det_logs = detection_loss_fn(dets_out, dets_targets)


You can implement an SSD/SSDLite loss (anchors + IoU matching + hard‑negative mining) and use it here.

6.3. Run training

Once build_model() and your detection loss are in place:

python train_joint.py


train_joint.py will:

Create a CocoDetDepth dataset using instances_train_with_depth.json.

Build your model and optimizer.

For each batch:

Load image, boxes, labels, depth, depth_valid.

Run the model to get detections and inverse depth.

Compute detection + depth losses via compute_multitask_loss.

Backprop and update weights.

You can monitor logs['L_total'], logs['L_det'], logs['L_silog'], etc. for training progress.

7. Running online inference in CARLA

Script: run_carla_inference.py

This file shows how to:

Grab an RGB frame from CARLA (get_rgb() placeholder).

Run the model: dets, invd, aux = model(x).

Convert invd to depth in meters.

Decode detections to boxes + IDs (via your tracker).

Use robust_object_distance(depth_m, box_xyxy, ...) to get per‑object distances.

Smooth distances over time with ZTracker.

Update memory with model.mem_write(aux['P3'].detach()).

You must implement these placeholders yourself:

get_rgb() – fetches the latest RGB frame from CARLA.

decode_boxes_and_ids(dets) – decodes SSDLite outputs into boxes + object IDs.

draw_box_and_text(rgb, box, text) – overlays detections and distances.

show(rgb) – displays or streams the frame.

Typical usage pattern (pseudo):

model = build_model().cuda().eval()
model.reset_memory()

while running:
    rgb = get_rgb()  # from CARLA
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    dets, invd, aux = model(x)
    depth_m = (1.0 / (invd[0, 0].clamp(min=1e-6))).detach().cpu()
    boxes, ids = decode_boxes_and_ids(dets)
    for b, obj_id in zip(boxes, ids):
        z = robust_object_distance(depth_m, b, erode_px=2, prev=ztrack.prev.get(obj_id))
        ...
    model.mem_write(aux['P3'].detach())

8. Where your custom fingerprint lives

To clearly identify this as your model:

Name the model YourName‑MNv4‑DepthSSD in the code (e.g. in docstrings and comments).

Keep your custom blocks and hooks in model_defs.py (e.g., Depth‑Aware Fusion, DevSignatureBlock, and your HoPE variant in AttnBlock.pos_encode).

Mention these components explicitly in your README and any paper/report.

That way, anyone looking at the repo can see the unique parts you designed: the HoPE‑ready neck, the Titans‑style memory, and the depth-aware integration with SSD.