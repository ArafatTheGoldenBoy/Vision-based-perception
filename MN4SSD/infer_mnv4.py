#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick MV4-SSD/SSDLite inference + visualization.
"""

import os, argparse, glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from mobilenetv4_ssd_full import MobileNetV4SSD
from ssd_utils import cxcywh_to_xyxy, decode_ssd, nms_torch, letterbox_image


def read_lines(p):
    return [
        x.strip()
        for x in Path(p).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


@torch.no_grad()
def main_infer():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="image file or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--head", choices=["ssdlite", "ssd"], default="ssdlite")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument(
        "--no-letterbox",
        action="store_true",
        help="disable aspect-ratio preserving resize",
    )
    args = ap.parse_args()

    names = read_lines(args.classes)
    num_classes = len(names) + 1  # +background
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lite = args.head == "ssdlite"
    m = MobileNetV4SSD(
        num_classes=num_classes, img_size=args.img_size, lite=lite
    ).to(device).eval()

    state = torch.load(args.weights, map_location=device)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    m.load_state_dict(state, strict=False)
    print("[info] model loaded.")

    # anchors once
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    cls_d, reg_d, anchors_list = m(dummy)
    anchors = torch.cat(anchors_list, dim=0)  # [Na,4] cxcywh
    print(f"[info] anchors: {anchors.shape[0]}")

    inp = Path(args.inp)
    paths = []
    if inp.is_dir():
        paths = sorted(Path(inp).glob("*.jpg")) + sorted(Path(inp).glob("*.png"))
    else:
        paths = [inp]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        im0 = Image.open(p).convert("RGB")
        W0, H0 = im0.size
        if args.no_letterbox:
            proc = im0.resize((args.img_size, args.img_size), Image.BILINEAR)
            scale_x = args.img_size / float(W0)
            scale_y = args.img_size / float(H0)
            pad_x = 0.0
            pad_y = 0.0
        else:
            proc, scale_x, scale_y, pad_x, pad_y = letterbox_image(im0, args.img_size)

        x = torch.from_numpy(np.array(proc)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)

        cls_pred, box_pred, _ = m(x)  # [1,Na,C], [1,Na,4]
        cls_pred, box_pred = cls_pred[0], box_pred[0]

        prob = torch.softmax(cls_pred, dim=-1)[:, 1:]  # drop background
        confs, lab = prob.max(dim=-1)
        keep0 = confs >= args.conf
        if keep0.sum() == 0:
            im0.save(out_dir / p.name)
            print(f"[ok] {p.name}: kept=0")
            continue

        dec = decode_ssd(box_pred[keep0], anchors[keep0])
        xyxy = cxcywh_to_xyxy(dec).clamp(0, 1)
        xyxy_px = xyxy * args.img_size

        sx = max(scale_x, 1e-6)
        sy = max(scale_y, 1e-6)
        xyxy_px[:, [0, 2]] = (xyxy_px[:, [0, 2]] - pad_x) / sx
        xyxy_px[:, [1, 3]] = (xyxy_px[:, [1, 3]] - pad_y) / sy
        xyxy_px[:, [0, 2]] = xyxy_px[:, [0, 2]].clamp(0, W0)
        xyxy_px[:, [1, 3]] = xyxy_px[:, [1, 3]].clamp(0, H0)

        keep = nms_torch(xyxy_px, confs[keep0], iou_thr=args.iou, topk=args.max_det)
        xyxy_px = xyxy_px[keep]
        scores = confs[keep0][keep]
        labels = lab[keep0][keep] + 1  # 1..C-1

        draw = ImageDraw.Draw(im0)
        for (x1, y1, x2, y2), sc, lb in zip(
            xyxy_px.tolist(), scores.tolist(), labels.tolist()
        ):
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            caption = f"{names[lb - 1]} {sc:.2f}"
            draw.text((x1 + 2, y1 + 2), caption, fill=(255, 255, 0))
        im0.save(out_dir / p.name)
        print(f"[ok] {p.name}: kept={len(labels)}")


if __name__ == "__main__":
    try:
        main_infer()
    except SystemExit:
        pass
