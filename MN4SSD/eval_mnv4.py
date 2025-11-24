#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantitative evaluation for MobileNetV4-SSD/SSDLite checkpoints against a COCO-style split.
- Runs inference over every image referenced in --anns.
- Matches detections against GT at IoU>=0.5, computes per-class AP/precision/recall, and
  writes a CSV table + matplotlib PR curve.

Usage:
  python eval_mnv4.py ^
    --weights ./runs/mnv4_ssdlite_320/best.ckpt ^
    --classes D:/datasets/carla_ds10/classes.txt ^
    --anns D:/datasets/carla_ds10/val.json ^
    --data-root D:/datasets/carla_ds10 ^
    --img-size 320 --head ssdlite --conf 0.05 --report-conf 0.35 ^
    --out ./runs/mnv4_eval
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch

from mobilenetv4_ssd_full import MobileNetV4SSD


def read_lines(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def decode_ssd(pred_t, anc_cxcywh, vxy=0.1, vwh=0.2):
    px, py, pw, ph = pred_t.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)
    cx = px * aw * vxy + acx
    cy = py * ah * vxy + acy
    w = torch.exp(pw * vwh) * aw
    h = torch.exp(ph * vwh) * ah
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
        inter = wh[:, 0] * wh[:, 1]
        area_cur = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_cur + area_rest - inter + 1e-8)
        idxs = idxs[1:][iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def box_iou_np(box, boxes):
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


def compute_ap(rec, prec):
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    recall_points = np.linspace(0, 1, 101)
    prec_interp = np.interp(recall_points, mrec, mpre)
    return float(prec_interp.mean())


def eval_single_class(cls_id, preds, gt_by_img, num_gt, iou_thr, report_conf):
    if num_gt == 0:
        return {
            "ap": float("nan"),
            "precision_curve": np.array([]),
            "recall_curve": np.array([]),
            "scores": np.array([]),
            "tp": np.array([]),
            "fp": np.array([]),
            "num_gt": 0,
            "num_pred": len(preds),
            "precision_at_conf": float("nan"),
            "recall_at_conf": float("nan"),
        }

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    n = len(preds_sorted)
    tp = np.zeros((n,), dtype=np.float32)
    fp = np.zeros((n,), dtype=np.float32)
    scores = np.zeros((n,), dtype=np.float32)

    gt_used = {img_id: np.zeros(len(per_cls.get(cls_id, [])), dtype=bool)
               for img_id, per_cls in gt_by_img.items() if cls_id in per_cls}

    for i, (img_id, score, box) in enumerate(preds_sorted):
        scores[i] = score
        gt_im = gt_by_img.get(img_id, {})
        gt_boxes = gt_im.get(cls_id)
        if gt_boxes is None or gt_boxes.size == 0:
            fp[i] = 1
            continue

        ious = box_iou_np(box, gt_boxes)
        best_idx = int(np.argmax(ious)) if ious.size else -1
        best_iou = float(ious[best_idx]) if ious.size else 0.0
        used_flags = gt_used.get(img_id)

        if best_iou >= iou_thr and used_flags is not None and not used_flags[best_idx]:
            tp[i] = 1
            used_flags[best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = compute_ap(recall, precision)

    k = int(np.sum(scores >= report_conf))
    if k > 0:
        tp_at_conf = tp[:k].sum()
        fp_at_conf = fp[:k].sum()
        prec_conf = tp_at_conf / max(1e-9, tp_at_conf + fp_at_conf)
        rec_conf = tp_at_conf / max(1, num_gt)
    else:
        prec_conf = 0.0
        rec_conf = 0.0

    return {
        "ap": ap,
        "precision_curve": precision,
        "recall_curve": recall,
        "scores": scores,
        "tp": tp,
        "fp": fp,
        "num_gt": num_gt,
        "num_pred": n,
        "precision_at_conf": prec_conf,
        "recall_at_conf": rec_conf,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--anns", required=True, help="COCO val.json to score against")
    ap.add_argument("--data-root", required=True, help="Folder containing the images/ directory")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--head", choices=["ssdlite", "ssd"], default="ssdlite")
    ap.add_argument("--conf", type=float, default=0.05, help="minimum confidence to keep detections before scoring")
    ap.add_argument("--report-conf", type=float, default=None, help="threshold used when reporting precision/recall (defaults to --conf)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for positives")
    ap.add_argument("--max-det", type=int, default=200)
    ap.add_argument("--out", type=str, default="./runs/mnv4_eval")
    args = ap.parse_args()

    report_conf = args.report_conf if args.report_conf is not None else args.conf
    names = read_lines(args.classes)
    num_classes = len(names) + 1

    js = json.loads(Path(args.anns).read_text(encoding="utf-8"))
    images = js["images"]
    anns = js["annotations"]
    cats = js["categories"]

    id2name = {c["id"]: c["name"] for c in cats}
    name2contig = {n: i + 1 for i, n in enumerate(names)}
    cat2contig = {cid: name2contig[id2name[cid]] for cid in id2name if id2name[cid] in name2contig}

    gt_by_img = {im["id"]: {} for im in images}
    gt_counts = defaultdict(int)
    for ann in anns:
        cls = cat2contig.get(ann["category_id"])
        if cls is None:
            continue
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            continue
        box = [x, y, x + w, y + h]
        gt_by_img[ann["image_id"]].setdefault(cls, []).append(box)
        gt_counts[cls] += 1

    # convert lists to numpy arrays for faster IoU math
    for per_img in gt_by_img.values():
        for cls, boxes in per_img.items():
            per_img[cls] = np.array(boxes, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lite = (args.head == "ssdlite")
    model = MobileNetV4SSD(num_classes=num_classes, img_size=args.img_size, lite=lite).to(device).eval()
    state = torch.load(args.weights, map_location=device)
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    _, _, anchors_list = model(dummy)
    anchors = torch.cat(anchors_list, dim=0)

    preds_per_class = {i: [] for i in range(1, num_classes)}
    data_root = Path(args.data_root)
    for idx, img_info in enumerate(images, 1):
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            print(f"[warn] missing image: {img_path}")
            continue
        im0 = Image.open(img_path).convert("RGB")
        W0, H0 = im0.size
        im = im0.resize((args.img_size, args.img_size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)

        cls_pred, box_pred, _ = model(x)
        cls_pred, box_pred = cls_pred[0], box_pred[0]

        prob = torch.softmax(cls_pred, dim=-1)[:, 1:]
        confs, lab = prob.max(dim=-1)
        mask = confs >= args.conf
        if mask.sum() == 0:
            continue

        dec = decode_ssd(box_pred[mask], anchors[mask])
        xyxy = cxcywh_to_xyxy(dec).clamp(0, 1)
        xyxy_px = xyxy.clone()
        xyxy_px[:, [0, 2]] *= W0
        xyxy_px[:, [1, 3]] *= H0

        keep = nms_torch(xyxy_px, confs[mask], iou_thr=args.iou, topk=args.max_det)
        xyxy_px = xyxy_px[keep].cpu().numpy()
        scores = confs[mask][keep].cpu().numpy()
        labels = (lab[mask][keep] + 1).cpu().numpy()

        for box, score, label in zip(xyxy_px, scores, labels):
            preds_per_class[label].append((img_info["id"], float(score), box))

        if idx % 20 == 0 or idx == len(images):
            print(f"[eval] processed {idx}/{len(images)} images")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_metrics = {}
    all_scores, all_tp, all_fp = [], [], []
    total_gt = sum(gt_counts.values())
    for cls_id in range(1, num_classes):
        metrics = eval_single_class(cls_id, preds_per_class.get(cls_id, []), gt_by_img,
                                    gt_counts.get(cls_id, 0), args.iou, report_conf)
        class_metrics[cls_id] = metrics
        if metrics["scores"].size > 0:
            all_scores.append(metrics["scores"])
            all_tp.append(metrics["tp"])
            all_fp.append(metrics["fp"])

    if all_scores:
        scores = np.concatenate(all_scores)
        tp = np.concatenate(all_tp)
        fp = np.concatenate(all_fp)
        order = np.argsort(-scores)
        tp_cum = np.cumsum(tp[order])
        fp_cum = np.cumsum(fp[order])
        recall = tp_cum / max(1, total_gt)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        micro_ap = compute_ap(recall, precision)
    else:
        recall = np.array([])
        precision = np.array([])
        micro_ap = float("nan")

    plot_path = out_dir / "precision_recall.png"
    plt.figure(figsize=(6, 5))
    if recall.size > 0:
        plt.plot(recall, precision, label=f"micro AP@{args.iou:.2f} = {micro_ap:.3f}")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.4)
    if recall.size > 0:
        plt.legend(loc="lower left")
    plt.title("Precision-Recall (micro average)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[ok] saved PR curve -> {plot_path}")

    header = ["class", f"AP@{args.iou}", f"P@{report_conf}", f"R@{report_conf}", "GT", "preds"]
    rows = []
    csv_lines = ["class,ap50,precision,recall,num_gt,num_pred"]
    ap_values = []
    for idx, name in enumerate(names, start=1):
        m = class_metrics.get(idx)
        if m is None:
            row = (name, float("nan"), float("nan"), float("nan"), 0, 0)
        else:
            row = (
                name,
                m["ap"],
                m["precision_at_conf"],
                m["recall_at_conf"],
                m["num_gt"],
                m["num_pred"],
            )
        ap_values.append(row[1])
        rows.append(row)
        csv_lines.append(f"{name},{row[1]:.4f},{row[2]:.4f},{row[3]:.4f},{row[4]},{row[5]}")

    macro_ap = np.nanmean(ap_values) if ap_values else float("nan")
    summary_path = out_dir / "metrics.csv"
    summary_path.write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"[ok] metrics table -> {summary_path}")

    print("-" * 72)
    print(f"{header[0]:15s} | {header[1]:>8s} | {header[2]:>8s} | {header[3]:>8s} | {header[4]:>6s} | {header[5]:>6s}")
    print("-" * 72)
    for row in rows:
        name, ap_v, prec_v, rec_v, gt_count, pred_count = row
        ap_s = "nan" if np.isnan(ap_v) else f"{ap_v:.3f}"
        p_s = "nan" if np.isnan(prec_v) else f"{prec_v:.3f}"
        r_s = "nan" if np.isnan(rec_v) else f"{rec_v:.3f}"
        print(f"{name:15s} | {ap_s:>8s} | {p_s:>8s} | {r_s:>8s} | {gt_count:6d} | {pred_count:6d}")
    print("-" * 72)
    print(f"[summary] micro AP={micro_ap:.3f}  macro AP={macro_ap:.3f}  GT boxes={total_gt}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
