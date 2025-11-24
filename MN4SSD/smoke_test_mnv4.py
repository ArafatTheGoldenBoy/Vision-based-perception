#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for MobileNetV4-SSD/SSDLite stack.

Checks:
  1) Anchor strides + coverage.
  2) Single-image inference with same preprocessing as infer_mnv4.py.
  3) Compare top predictions vs GT boxes from a COCO-style val.json.

Usage example:

  python smoke_test_mnv4.py \
    --weights ./runs/mnv4_ssdlite_320/best.ckpt \
    --classes D:/datasets/carla_ds10/classes.txt \
    --anns D:/datasets/carla_ds10/val.json \
    --data-root D:/datasets/carla_ds10 \
    --img-size 320 --head ssdlite \
    --image-idx 0 \
    --conf 0.3 --iou 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from mobilenetv4_ssd_full import MobileNetV4SSD            # model + anchors  :contentReference[oaicite:3]{index=3}
from infer_mnv4 import (                                  # reuse infer helpers  :contentReference[oaicite:4]{index=4}
    read_lines,
    cxcywh_to_xyxy,
    decode_ssd,
    nms_torch,
    letterbox_image,
)


def box_iou_np(box, boxes):
    """IoU between one box [4] and many boxes [N,4] in xyxy."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    inter = w * h
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-9
    return inter / union


def test_anchors(model: MobileNetV4SSD, img_size: int, device: str):
    print("\n=== Test 1: Anchor strides & coverage ===")
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        _, _, anchors_list = model(x)

    print("Guessed strides:", model.strides)
    for lvl, anc in enumerate(anchors_list):
        anc = anc.detach().cpu()  # [N,4] (cx,cy,w,h) in [0,1]
        cx = anc[:, 0]
        cy = anc[:, 1]
        print(
            f"  Level {lvl}: N={len(anc):6d}  "
            f"cx[min,max]={cx.min():.3f},{cx.max():.3f}  "
            f"cy[min,max]={cy.min():.3f},{cy.max():.3f}"
        )


def test_single_image(
    model: MobileNetV4SSD,
    anchors: torch.Tensor,
    classes_path: str,
    anns_path: str,
    data_root: str,
    img_size: int,
    head: str,
    image_idx: int,
    conf_thr: float,
    iou_thr: float,
    no_letterbox: bool,
    device: str,
):
    print("\n=== Test 2: Single-image inference vs GT ===")

    # Load classes and COCO val.json  
    names = read_lines(classes_path)
    js = json.loads(Path(anns_path).read_text(encoding="utf-8"))
    images = js["images"]
    anns = js["annotations"]

    if image_idx < 0 or image_idx >= len(images):
        raise IndexError(f"image_idx {image_idx} out of range (0..{len(images)-1})")

    img_info = images[image_idx]
    img_id = img_info["id"]
    img_rel = img_info["file_name"]
    img_path = Path(data_root) / img_rel

    print(f"Using image_idx={image_idx}, image_id={img_id}, path={img_path}")

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load original image
    im0 = Image.open(img_path).convert("RGB")
    W0, H0 = im0.size
    print(f"Original size: {W0} x {H0}")

    # Same preprocessing as infer_mnv4.py  :contentReference[oaicite:6]{index=6}
    if no_letterbox:
        proc = im0.resize((img_size, img_size), Image.BILINEAR)
        scale_x = img_size / float(W0)
        scale_y = img_size / float(H0)
        pad_x = 0.0
        pad_y = 0.0
        print("Preprocessing: WARP (no letterbox)")
    else:
        proc, scale_x, scale_y, pad_x, pad_y = letterbox_image(im0, img_size)
        print("Preprocessing: LETTERBOX")
    print(f"scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, pad_x={pad_x}, pad_y={pad_y}")

    x = torch.from_numpy(np.array(proc)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        cls_pred, box_pred, _ = model(x)  # [1,Na,C], [1,Na,4]
    cls_pred, box_pred = cls_pred[0], box_pred[0]

    # Decode and NMS (same as infer_mnv4.py)
    prob = torch.softmax(cls_pred, dim=-1)[:, 1:]  # drop background
    confs, lab = prob.max(dim=-1)
    keep0 = confs >= conf_thr
    print(f"Detections above conf={conf_thr}: {keep0.sum().item()} / {confs.numel()}")

    if keep0.sum() == 0:
        print("No detections above threshold; try lowering --conf.")
        return

    dec = decode_ssd(box_pred[keep0], anchors[keep0])
    xyxy = cxcywh_to_xyxy(dec).clamp(0, 1)
    xyxy_px = xyxy * img_size

    sx = max(scale_x, 1e-6)
    sy = max(scale_y, 1e-6)
    xyxy_px[:, [0, 2]] = (xyxy_px[:, [0, 2]] - pad_x) / sx
    xyxy_px[:, [1, 3]] = (xyxy_px[:, [1, 3]] - pad_y) / sy
    xyxy_px[:, [0, 2]] = xyxy_px[:, [0, 2]].clamp(0, W0)
    xyxy_px[:, [1, 3]] = xyxy_px[:, [1, 3]].clamp(0, H0)

    keep = nms_torch(xyxy_px, confs[keep0], iou_thr=iou_thr, topk=300)
    xyxy_px = xyxy_px[keep].cpu().numpy()
    scores = confs[keep0][keep].cpu().numpy()
    labels = (lab[keep0][keep] + 1).cpu().numpy()  # 1..C-1

    print(f"After NMS@{iou_thr}: {len(labels)} detections")

    # Collect GT boxes for this image (xyxy in original pixels)
    gt_boxes_xyxy = []
    gt_labels = []
    for a in anns:
        if a["image_id"] != img_id:
            continue
        x, y, w, h = a["bbox"]
        if w < 1 or h < 1:
            continue
        gt_boxes_xyxy.append([x, y, x + w, y + h])
        gt_labels.append(a["category_id"])
    gt_boxes_xyxy = np.array(gt_boxes_xyxy, dtype=np.float32) if gt_boxes_xyxy else np.zeros((0, 4), dtype=np.float32)

    print(f"GT boxes for this image: {gt_boxes_xyxy.shape[0]}")

    if gt_boxes_xyxy.shape[0] == 0:
        print("No GT boxes for this image in the COCO JSON.")
    else:
        print("First few GT boxes (xyxy):")
        print(gt_boxes_xyxy[:5])

    # Compare each prediction to best-matching GT box
    print("\nTop predictions vs best IoU to any GT:")
    for i in range(min(10, len(labels))):
        box = xyxy_px[i]
        score = scores[i]
        cls = labels[i]
        name = names[cls - 1] if 1 <= cls <= len(names) else f"cls{cls}"

        if gt_boxes_xyxy.shape[0] > 0:
            ious = box_iou_np(box, gt_boxes_xyxy)
            best_iou = float(ious.max())
        else:
            best_iou = float("nan")

        print(
            f"  pred[{i}]: {name:15s}  score={score:5.3f}  "
            f"box=[{box[0]:6.1f},{box[1]:6.1f},{box[2]:6.1f},{box[3]:6.1f}]  "
            f"best IoU to GT={best_iou:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--anns", required=True, help="COCO val.json")
    ap.add_argument("--data-root", required=True, help="Folder containing images/ etc.")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--head", choices=["ssdlite", "ssd"], default="ssdlite")
    ap.add_argument("--image-idx", type=int, default=0, help="Index into images[] in the COCO JSON")
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold for smoke test")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS in smoke test")
    ap.add_argument("--no-letterbox", action="store_true", help="use warp instead of letterbox")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = read_lines(args.classes)
    num_classes = len(names) + 1

    lite = (args.head == "ssdlite")
    model = MobileNetV4SSD(
        num_classes=num_classes,
        img_size=args.img_size,
        lite=lite,
    ).to(device)

    state = torch.load(args.weights, map_location=device)
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    print("[info] model weights loaded.")

    # 1) Anchor check
    test_anchors(model, args.img_size, device=device)

    # 2) Build anchors once (like infer_mnv4.py)
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        _, _, anchors_list = model(dummy)
    anchors = torch.cat(anchors_list, dim=0).to(device)
    print(f"[info] total anchors: {anchors.shape[0]}")

    # 3) Single-image test
    test_single_image(
        model=model,
        anchors=anchors,
        classes_path=args.classes,
        anns_path=args.anns,
        data_root=args.data_root,
        img_size=args.img_size,
        head=args.head,
        image_idx=args.image_idx,
        conf_thr=args.conf,
        iou_thr=args.iou,
        no_letterbox=args.no_letterbox,
        device=device,
    )


if __name__ == "__main__":
    main()
