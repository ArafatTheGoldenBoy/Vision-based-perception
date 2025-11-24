# train_mnv4_ssd.py
# MobileNetV4-SSD / SSDLite training on your CARLA→COCO dataset

import os, json, math, argparse, time, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mobilenetv4_ssd_full import MobileNetV4SSD
from ssd_utils import box_iou_xyxy, letterbox_image, encode_ssd  # shared helpers


# -----------------------------
# Small utilities
# -----------------------------

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def read_lines(p: Path) -> List[str]:
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def coco_load(anns_path: Path):
    with open(anns_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Dataset (COCO minimal parser)
# -----------------------------

class CocoDet(Dataset):
    def __init__(
        self,
        data_root: str,
        anns_path: str,
        classes_txt: str,
        img_size: int = 320,
        augment: bool = False,
        letterbox: bool = True,
    ):
        self.root = Path(data_root)
        js = coco_load(Path(anns_path))
        self.imgs = js["images"]
        self.anns = js["annotations"]
        self.cats = js["categories"]
        self.img_size = int(img_size)
        self.augment = augment
        self.letterbox = letterbox

        # Mapping: cat_id -> cat_name
        id2name = {c["id"]: c["name"] for c in self.cats}
        # Order from classes.txt defines contiguous labels:
        names = read_lines(Path(classes_txt))
        self.name2contig = {n: i + 1 for i, n in enumerate(names)}  # 1..C (0 reserved for background)
        self.contig2name = {i + 1: n for i, n in enumerate(names)}
        self.num_classes = len(names) + 1  # + background

        # Build per-image gt lists
        self.img_by_id = {im["id"]: im for im in self.imgs}
        lists: Dict[int, Dict[str, list]] = {
            im["id"]: {"boxes": [], "labels": []} for im in self.imgs
        }
        for a in self.anns:
            im_id = a["image_id"]
            if im_id not in lists:
                continue
            x, y, w, h = a["bbox"]
            # Filter tiny / invalid boxes here as safety
            if w < 1 or h < 1:
                continue
            name = id2name.get(a["category_id"], None)
            if name is None or name not in self.name2contig:
                continue
            lists[im_id]["boxes"].append([x, y, w, h])
            lists[im_id]["labels"].append(self.name2contig[name])

        self.items = []
        for im in self.imgs:
            rel = im["file_name"]
            w, h = im["width"], im["height"]
            e = lists[im["id"]]
            self.items.append(
                {
                    "path": (self.root / rel).resolve(),
                    "size": (w, h),
                    "boxes": np.array(e["boxes"], dtype=np.float32),
                    "labels": np.array(e["labels"], dtype=np.int64),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        p = str(it["path"])
        # jpg/png forgiveness
        if not os.path.exists(p):
            base, ext = os.path.splitext(p)
            alt = base + (".png" if ext.lower() == ".jpg" else ".jpg")
            p = alt
        im = Image.open(p).convert("RGB")
        W0, H0 = im.size

        if self.letterbox:
            im, sx, sy, pad_x, pad_y = letterbox_image(im, self.img_size)
        else:
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            sx = self.img_size / float(W0)
            sy = self.img_size / float(H0)
            pad_x = pad_y = 0.0

        boxes = it["boxes"].copy()
        labels = it["labels"].copy()

        if boxes.size > 0:
            # map COCO xywh (in px) through resize+pad and then to relative cxcywh
            boxes[:, 0] = boxes[:, 0] * sx + pad_x
            boxes[:, 1] = boxes[:, 1] * sy + pad_y
            boxes[:, 2] = boxes[:, 2] * sx
            boxes[:, 3] = boxes[:, 3] * sy

            boxes[:, 0] = boxes[:, 0] / self.img_size
            boxes[:, 1] = boxes[:, 1] / self.img_size
            boxes[:, 2] = boxes[:, 2] / self.img_size
            boxes[:, 3] = boxes[:, 3] / self.img_size

            cxcywh = np.stack(
                [
                    boxes[:, 0] + 0.5 * boxes[:, 2],
                    boxes[:, 1] + 0.5 * boxes[:, 3],
                    boxes[:, 2],
                    boxes[:, 3],
                ],
                axis=-1,
            )
            boxes = np.clip(cxcywh, 0.0, 1.0)

        x = torch.from_numpy(np.array(im)).float() / 255.0
        x = x.permute(2, 0, 1)  # CHW

        y_boxes = (
            torch.from_numpy(boxes).float()
            if boxes.size > 0
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        y_labels = (
            torch.from_numpy(labels).long()
            if labels.size > 0
            else torch.zeros((0,), dtype=torch.long)
        )
        return x, y_boxes, y_labels


def collate(batch):
    xs, bs, ls = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return xs, list(bs), list(ls)


# -----------------------------
# Matcher + Loss (SSD style)
# -----------------------------

class SSDLoss(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        num_classes: int,
        pos_iou: float = 0.5,
        neg_iou: float = 0.4,
        neg_pos_ratio: int = 3,
        vxy: float = 0.1,
        vwh: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()
        self.anchors = anchors.to(device)  # [N,4] (cx,cy,w,h) relative
        self.num_classes = num_classes
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.neg_pos_ratio = neg_pos_ratio
        self.vxy, self.vwh = vxy, vwh

        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def match(self, gt_cxcywh: torch.Tensor, gt_labels: torch.Tensor):
        """
        Return:
          cls_tgt: [N] int64 in [0..C-1], 0=background, -1=ignore
          box_tgt: [N,4] encoded for positives, 0 otherwise
          pos_mask: [N] bool
        """
        N = self.anchors.shape[0]
        device = self.anchors.device

        cls_tgt = torch.zeros((N,), dtype=torch.long, device=device)  # 0=bg by default
        box_tgt = torch.zeros((N, 4), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((N,), dtype=torch.bool, device=device)

        if gt_cxcywh.numel() == 0:
            return cls_tgt, box_tgt, pos_mask

        # IoU between anchors and gt
        anc_xyxy = cxcywh_to_xyxy(self.anchors)
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
        iou = box_iou_xyxy(anc_xyxy, gt_xyxy)  # [Na, Ng]

        # best gt for each anchor, and best anchor for each gt (force match)
        iou_max, gt_idx = iou.max(dim=1)           # per-anchor
        _, best_anchor_per_gt = iou.max(dim=0)     # per-gt

        # positives: IoU >= pos_iou
        pos = iou_max >= self.pos_iou
        # force match: ensure each gt has at least one anchor
        pos[best_anchor_per_gt] = True

        # ignore: between neg_iou and pos_iou → set cls_tgt = -1
        ign = (iou_max > self.neg_iou) & (~pos)

        # fill cls targets
        cls_tgt[ign] = -1  # ignore
        cls_tgt[pos] = gt_labels[gt_idx[pos]]  # gt labels already 1..C-1 (background=0)

        # encode boxes for positives
        if pos.any():
            box_tgt[pos] = encode_ssd(
                gt_cxcywh[gt_idx[pos]], self.anchors[pos], self.vxy, self.vwh
            )

        pos_mask[:] = pos
        return cls_tgt, box_tgt, pos_mask

    def forward(self,
                cls_pred: torch.Tensor,
                box_pred: torch.Tensor,
                gt_boxes_list,
                gt_labels_list):
        """
        cls_pred: [B, Na, C]
        box_pred: [B, Na, 4]
        gt_boxes_list: list of [Ng,4]  (cx,cy,w,h) relative
        gt_labels_list: list of [Ng]   (1..C-1)
        """
        device = cls_pred.device
        B, Na, C = cls_pred.shape
        total_cls, total_box = 0.0, 0.0
        total_pos = 0

        for b in range(B):
            gt_b = gt_boxes_list[b].to(device)
            gt_l = gt_labels_list[b].to(device)

            cls_t, box_t, pos_mask = self.match(gt_b, gt_l)

            # classification loss with hard-negative mining
            ce = self.ce(cls_pred[b], cls_t.clamp(min=0))  # CE uses targets >=0
            pos_ce = ce[pos_mask]
            neg_mask = (cls_t == 0)

            num_pos = int(pos_mask.sum().item())
            total_pos += num_pos

            if num_pos > 0:
                # select hardest negatives
                num_neg = min(
                    int(self.neg_pos_ratio * num_pos),
                    int(neg_mask.sum().item()),
                )
                if num_neg > 0:
                    neg_losses = ce[neg_mask]
                    topk = torch.topk(neg_losses, k=num_neg, largest=True).values
                    cls_loss = (pos_ce.sum() + topk.sum()) / (num_pos + num_neg)
                else:
                    cls_loss = pos_ce.mean()
                # bbox loss on positives
                reg = self.smooth_l1(box_pred[b][pos_mask], box_t[pos_mask]).mean()
            else:
                # no positives: take top-k negatives only (fallback)
                num_neg = min(100, int(neg_mask.sum().item()))
                if num_neg > 0:
                    topk = torch.topk(ce[neg_mask], k=num_neg, largest=True).values
                    cls_loss = topk.mean()
                else:
                    cls_loss = torch.tensor(0.0, device=device)
                reg = torch.tensor(0.0, device=device)

            total_cls += cls_loss
            total_box += reg

        # average over batch
        total_cls = total_cls / B
        total_box = total_box / B
        return total_cls, total_box, total_pos / max(1, B)


# -----------------------------
# Training plumbing
# -----------------------------

def build_model(
    num_classes: int,
    img_size: int,
    head: str,
    backbone: str = "mobilenetv4_conv_small",
    tag: str = "e500_r256_in1k",
):
    lite = head.lower() == "ssdlite"
    model = MobileNetV4SSD(
        num_classes=num_classes,
        img_size=img_size,
        lite=lite,
        backbone_name=backbone,
        pretrained_tag=tag,
    )
    return model


@torch.no_grad()
def make_anchors(model: MobileNetV4SSD, img_size: int, device: str = "cpu"):
    # Trigger on a fake image to build per-level anchors
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    cls_pred, reg_pred, anchors_list = model(x)
    anchors = torch.cat(anchors_list, dim=0).detach()
    return anchors


def train_one_epoch(
    model,
    loss_fn,
    loader,
    optimizer,
    scaler,
    device: str = "cuda",
    log_every: int = 50,
):
    model.train()
    total, n = 0.0, 0

    for it, (imgs, boxes_list, labels_list) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True)

        # forward
        with torch.autocast("cuda", enabled=(device == "cuda")):
            cls_pred, box_pred, _ = model(imgs)   # [B,Na,C], [B,Na,4]
            cls_loss, box_loss, _ = loss_fn(
                cls_pred, box_pred, boxes_list, labels_list
            )
            loss = cls_loss + box_loss

        optimizer.zero_grad(set_to_none=True)

        # AMP backward + step
        scaler.scale(loss).backward()
        # optional: unscale + clip if needed
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        n += 1
        if it % log_every == 0:
            print(
                f"[train] it={it:04d}/{len(loader)}  "
                f"cls={cls_loss.item():.4f}  box={box_loss.item():.4f}  total={loss.item():.4f}"
            )

    return total / max(1, n)


@torch.no_grad()
def validate(model, loss_fn, loader, device: str = "cuda"):
    model.eval()
    total, n = 0.0, 0
    for imgs, boxes_list, labels_list in loader:
        imgs = imgs.to(device, non_blocking=True)
        cls_pred, box_pred, _ = model(imgs)
        cls_loss, box_loss, _ = loss_fn(cls_pred, box_pred, boxes_list, labels_list)
        total += (cls_loss + box_loss).item()
        n += 1
    return total / max(1, n)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-anns", required=True)
    ap.add_argument("--val-anns", required=True)
    ap.add_argument("--classes", required=True, help="path to classes.txt")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--head", type=str, default="ssdlite", choices=["ssdlite", "ssd"])
    ap.add_argument("--backbone", type=str, default="mobilenetv4_conv_small")
    ap.add_argument("--tag", type=str, default="e500_r256_in1k")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="./runs/mnv4_ssd")
    ap.add_argument(
        "--no-letterbox",
        action="store_true",
        help="disable aspect-ratio preserving resize (legacy warp)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Datasets
    use_letterbox = not args.no_letterbox
    train_ds = CocoDet(
        args.data_root,
        args.train_anns,
        args.classes,
        img_size=args.img_size,
        augment=False,
        letterbox=use_letterbox,
    )
    val_ds = CocoDet(
        args.data_root,
        args.val_anns,
        args.classes,
        img_size=args.img_size,
        augment=False,
        letterbox=use_letterbox,
    )
    num_classes = train_ds.num_classes
    print(f"[info] num_classes (incl. background) = {num_classes}")

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        num_classes, args.img_size, head=args.head, backbone=args.backbone, tag=args.tag
    ).to(device)

    # Anchors (once) → Loss
    anchors = make_anchors(model, args.img_size, device=device)  # [N,4] relative
    loss_fn = SSDLoss(anchors=anchors, num_classes=num_classes, device=device)

    # Optimizer / AMP
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, loss_fn, train_ld, optimizer, scaler, device=device
        )
        va = validate(model, loss_fn, val_ld, device=device)
        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={tr:.4f}  val_loss={va:.4f}  ({dt:.1f}s)"
        )

        # save last
        last_path = Path(args.out) / "last.ckpt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "img_size": args.img_size,
                "classes": read_lines(Path(args.classes)),
                "head": args.head,
            },
            last_path,
        )

        if va < best_val:
            best_val = va
            best_path = Path(args.out) / "best.ckpt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "img_size": args.img_size,
                    "classes": read_lines(Path(args.classes)),
                    "head": args.head,
                },
                best_path,
            )
            print(f"[save] best -> {best_path} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
