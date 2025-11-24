# ===============================
# File: infer_mnv4.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick MV4-SSD/SSDLite inference + visualization.
- Loads MobileNetV4SSD, decodes with model's anchors, applies class-agnostic NMS.
- Draws boxes on original images and writes to --out.

Usage:
  python infer_mnv4.py \
    --weights ./runs/mnv4_ssdlite_320/best.ckpt \
    --classes D:/datasets/carla_ds10/classes.txt \
    --in D:/datasets/carla_ds10/images \
    --out ./pred_vis \
    --img-size 320 --head ssdlite --conf 0.35 --iou 0.45
"""
import os, argparse, glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from mobilenetv4_ssd_full import MobileNetV4SSD

def read_lines(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5*w; y1 = cy - 0.5*h
    x2 = cx + 0.5*w; y2 = cy + 0.5*h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def decode_ssd(pred_t, anc_cxcywh, vxy=0.1, vwh=0.2):
    px, py, pw, ph = pred_t.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)
    cx = px * aw * vxy + acx
    cy = py * ah * vxy + acy
    w  = torch.exp(pw * vwh) * aw
    h  = torch.exp(ph * vwh) * ah
    return torch.stack([cx, cy, w, h], dim=-1)

def nms_torch(boxes, scores, iou_thr=0.45, topk=300):
    idxs = scores.argsort(descending=True)
    idxs = idxs[:min(topk, idxs.numel())]
    keep = []
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        cur = boxes[i].unsqueeze(0)
        rest = boxes[idxs[1:]]
        tl = torch.max(cur[:, :2], rest[:, :2])
        br = torch.min(cur[:, 2:], rest[:, 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[:, 0]*wh[:, 1]
        area_cur = (cur[:, 2]-cur[:, 0])*(cur[:, 3]-cur[:, 1])
        area_rest= (rest[:,2]-rest[:,0])*(rest[:,3]-rest[:,1])
        iou = inter / (area_cur + area_rest - inter + 1e-8)
        idxs = idxs[1:][iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def letterbox_image(im: Image.Image, size: int, fill=(114, 114, 114)):
    """
    Resize to a square canvas with preserved aspect ratio.
    Returns resized image plus scale/pad info used to undo the transform.
    """
    w0, h0 = im.size
    if w0 == 0 or h0 == 0:
        raise ValueError("Invalid image size for letterbox.")
    scale = min(size / float(w0), size / float(h0))
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    resized = im.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    scale_x = new_w / float(w0)
    scale_y = new_h / float(h0)
    return canvas, scale_x, scale_y, float(pad_x), float(pad_y)

@torch.no_grad()
def main_infer():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="image file or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--head", choices=["ssdlite","ssd"], default="ssdlite")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou",  type=float, default=0.45)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--no-letterbox", action="store_true", help="disable aspect-ratio preserving resize")
    args = ap.parse_args()

    names = read_lines(args.classes)
    num_classes = len(names) + 1  # +background
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lite = (args.head == "ssdlite")
    m = MobileNetV4SSD(num_classes=num_classes, img_size=args.img_size, lite=lite).to(device).eval()

    state = torch.load(args.weights, map_location=device)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    m.load_state_dict(state, strict=False)
    print("[info] model loaded.")

    # anchors once
    dummy = torch.zeros(1,3,args.img_size,args.img_size, device=device)
    cls_d, reg_d, anchors_list = m(dummy)
    anchors = torch.cat(anchors_list, dim=0)  # [Na,4] cxcywh
    print(f"[info] anchors: {anchors.shape[0]}")

    inp = Path(args.inp)
    paths = []
    if inp.is_dir():
        paths = sorted(Path(inp).glob("*.jpg")) + sorted(Path(inp).glob("*.png"))
    else:
        paths = [inp]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

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
        x = torch.from_numpy(np.array(proc)).float().permute(2,0,1).unsqueeze(0) / 255.0
        x = x.to(device)

        cls_pred, box_pred, _ = m(x)  # [1,Na,C], [1,Na,4]
        cls_pred, box_pred = cls_pred[0], box_pred[0]

        prob = torch.softmax(cls_pred, dim=-1)[:, 1:]  # drop background
        confs, lab = prob.max(dim=-1)
        keep0 = confs >= args.conf
        if keep0.sum() == 0:
            im0.save(out_dir / p.name)
            continue

        dec = decode_ssd(box_pred[keep0], anchors[keep0])
        xyxy = cxcywh_to_xyxy(dec).clamp(0,1)
        xyxy_px = xyxy * args.img_size
        sx = max(scale_x, 1e-6)
        sy = max(scale_y, 1e-6)
        xyxy_px[:, [0, 2]] = (xyxy_px[:, [0, 2]] - pad_x) / sx
        xyxy_px[:, [1, 3]] = (xyxy_px[:, [1, 3]] - pad_y) / sy
        xyxy_px[:, [0, 2]] = xyxy_px[:, [0, 2]].clamp(0, W0)
        xyxy_px[:, [1, 3]] = xyxy_px[:, [1, 3]].clamp(0, H0)

        keep = nms_torch(xyxy_px, confs[keep0], iou_thr=args.iou, topk=args.max_det)
        xyxy_px = xyxy_px[keep]
        scores  = confs[keep0][keep]
        labels  = lab[keep0][keep] + 1  # 1..C-1

        draw = ImageDraw.Draw(im0)
        for (x1,y1,x2,y2), sc, lb in zip(xyxy_px.tolist(), scores.tolist(), labels.tolist()):
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
            caption = f"{names[lb-1]} {sc:.2f}"
            draw.text((x1+2, y1+2), caption, fill=(255,255,0))
        im0.save(out_dir / p.name)
        print(f"[ok] {p.name}: kept={len(labels)}")

if __name__ == "__main__":
    try:
        main_infer()
    except SystemExit:
        pass

# ===============================
# File: export_mnv4_onnx.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export MobileNetV4SSD to ONNX + save anchors.
Outputs:
  - model.onnx: raw heads (cls logits, box deltas). No NMS inside.
  - anchors.npy: [Na,4] (cx,cy,w,h) relative for the chosen img_size.

Usage:
  python export_mnv4_onnx.py \
    --weights ./runs/mnv4_ssdlite_320/best.ckpt \
    --classes D:/datasets/carla_ds10/classes.txt \
    --img-size 320 --head ssdlite \
    --out ./export
"""
import argparse, numpy as np
from pathlib import Path
import torch
from mobilenetv4_ssd_full import MobileNetV4SSD

def main_export():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--head", choices=["ssdlite","ssd"], default="ssdlite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [x.strip() for x in Path(args.classes).read_text(encoding="utf-8").splitlines() if x.strip()]
    num_classes = len(names) + 1
    lite = (args.head == "ssdlite")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = MobileNetV4SSD(num_classes=num_classes, img_size=args.img_size, lite=lite).to(device).eval()

    state = torch.load(args.weights, map_location=device)
    state = state.get("model", state)
    m.load_state_dict(state, strict=False)

    x = torch.zeros(1,3,args.img_size,args.img_size, device=device)
    cls, reg, anchors_list = m(x)
    anchors = torch.cat(anchors_list, dim=0).detach().cpu().numpy()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "anchors.npy", anchors)
    print("[ok] anchors saved:", anchors.shape)

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        m, x, str(onnx_path),
        input_names=["images"], output_names=["cls", "reg", "anchors_placeholder"],
        opset_version=12, do_constant_folding=True,
        dynamic_axes={"images": {0: "batch"}}
    )
    print("[ok] exported:", onnx_path)

if __name__ == "__main__":
    try:
        main_export()
    except SystemExit:
        pass
