# ssd_utils.py
# Common utilities for SSD-style models:
# - box format conversions
# - SSD encode/decode
# - letterbox resize
# - simple torch NMS

from typing import Tuple

import torch
from PIL import Image


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    b: [..., 4]
    """
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h).
    b: [..., 4]
    """
    x1, y1, x2, y2 = b.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU between two sets of boxes in (x1, y1, x2, y2) format.
    a: [Na, 4], b: [Nb, 4] -> [Na, Nb]
    """
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-8
    return inter / union


def encode_ssd(gt_cxcywh: torch.Tensor,
               anc_cxcywh: torch.Tensor,
               vxy: float = 0.1,
               vwh: float = 0.2) -> torch.Tensor:
    """
    Encode ground-truth boxes relative to anchors (cx,cy,w,h) for SSD.
    Both gt and anchors are in the same (relative) coordinate system.
    """
    gcx, gcy, gw, gh = gt_cxcywh.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)

    tx = (gcx - acx) / (aw * vxy + 1e-9)
    ty = (gcy - acy) / (ah * vxy + 1e-9)
    tw = torch.log((gw / (aw + 1e-9)).clamp(min=1e-9)) / vwh
    th = torch.log((gh / (ah + 1e-9)).clamp(min=1e-9)) / vwh
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode_ssd(pred_t: torch.Tensor,
               anc_cxcywh: torch.Tensor,
               vxy: float = 0.1,
               vwh: float = 0.2) -> torch.Tensor:
    """
    Decode SSD deltas back to (cx, cy, w, h).
    pred_t: [..., 4], anc_cxcywh: [..., 4]
    """
    px, py, pw, ph = pred_t.unbind(-1)
    acx, acy, aw, ah = anc_cxcywh.unbind(-1)

    cx = px * aw * vxy + acx
    cy = py * ah * vxy + acy
    w = torch.exp(pw * vwh) * aw
    h = torch.exp(ph * vwh) * ah
    return torch.stack([cx, cy, w, h], dim=-1)


def nms_torch(boxes: torch.Tensor,
              scores: torch.Tensor,
              iou_thr: float = 0.45,
              topk: int = 300) -> torch.Tensor:
    """
    Simple class-agnostic NMS in pure torch.
    boxes: [N,4] (x1,y1,x2,y2), scores: [N]
    Returns indices of kept boxes.
    """
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


def letterbox_image(im: Image.Image,
                    size: int,
                    fill=(114, 114, 114)) -> Tuple[Image.Image, float, float, float, float]:
    """
    Resize an image to a square canvas with preserved aspect ratio
    using padding. Returns:

      canvas, scale_x, scale_y, pad_x, pad_y

    such that you can map boxes like:

      x_px = (x_rel*size - pad_x) / scale_x
      y_px = (y_rel*size - pad_y) / scale_y
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


__all__ = [
    "cxcywh_to_xyxy",
    "xyxy_to_cxcywh",
    "box_iou_xyxy",
    "encode_ssd",
    "decode_ssd",
    "nms_torch",
    "letterbox_image",
]
