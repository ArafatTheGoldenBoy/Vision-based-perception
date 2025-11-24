# mobilenetv4_ssd_full.py
# MobileNetV4-SSD / SSDLite (no HF auth; uses timm local registry + pretrained tag)
# Author: You + ChatGPT

import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import timm


# -----------------------------
# Small utils / building blocks
# -----------------------------

def conv_bn(in_ch, out_ch, k=3, s=1, p=None, groups=1):
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DWSeparableConv(nn.Module):
    """Depthwise-separable conv used in SSDLite heads."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


# -----------------------------
# Anchor generator (SSD-style)
# -----------------------------

class AnchorGeneratorSSD(nn.Module):
    """
    Simple SSD anchor generator:
    - per-level sizes given as (min_size, max_size) in relative scale to input (0..1)
    - per-level aspect ratios list (e.g., [1.0, 2.0, 0.5])
    Returns per-level [N, H*W*anchors, 4] (cx, cy, w, h) in relative coordinates.
    """
    def __init__(self,
                 strides: List[int],
                 img_size: int,
                 scales: List[Tuple[float, float]],
                 aspect_ratios: List[List[float]]):
        super().__init__()
        assert len(scales) == len(aspect_ratios) == len(strides)
        self.strides = strides
        self.img_size = img_size
        self.scales = scales
        self.aspect_ratios = aspect_ratios

    @torch.no_grad()
    def grid_anchors(self, feat: torch.Tensor, level: int) -> torch.Tensor:
        b, c, h, w = feat.shape

        # center locations in relative coords
        stride_rel = self.strides[level] / float(self.img_size)
        cy = (torch.arange(h, device=feat.device) + 0.5) * stride_rel
        cx = (torch.arange(w, device=feat.device) + 0.5) * stride_rel
        yy, xx = torch.meshgrid(cy, cx, indexing="ij")
        centers = torch.stack([xx, yy], dim=-1)  # [h, w, 2] = (cx, cy)

        # per-level sizes (relative)
        min_s, max_s = self.scales[level]
        ars = self.aspect_ratios[level]
        sizes = []
        # SSD tradition: two ratio=1 anchors: min_s and sqrt(min_s*max_s)
        sizes.append((min_s, min_s))
        sizes.append((math.sqrt(min_s * max_s), math.sqrt(min_s * max_s)))
        for ar in ars:
            if abs(ar - 1.0) < 1e-6:
                continue
            w_ = min_s * math.sqrt(ar)
            h_ = min_s / math.sqrt(ar)
            sizes.append((w_, h_))

        sizes = torch.tensor(sizes, device=feat.device, dtype=feat.dtype)  # [A, 2]
        A = sizes.shape[0]

        # Explicitly expand to [h, w, A, 2] (no implicit broadcasting with cat)
        ctr = centers.view(h, w, 1, 2).expand(h, w, A, 2)   # [h, w, A, 2]
        wh  = sizes.view(1, 1, A, 2).expand(h, w, A, 2)     # [h, w, A, 2]

        anchors = torch.cat([ctr, wh], dim=-1)              # [h, w, A, 4]  (cx, cy, w, h)
        anchors = anchors.reshape(-1, 4)                    # [h*w*A, 4]
        return anchors


    @torch.no_grad()
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.grid_anchors(f, i) for i, f in enumerate(feats)]


# -----------------------------
# SSD / SSDLite heads per level
# -----------------------------

# --- in LevelHead.__init__ ---
class LevelHead(nn.Module):
    def __init__(self, in_ch, num_anchors, num_classes, lite: bool):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        mid = max(64, in_ch // 2)
        block = DWSeparableConv if lite else conv_bn
        self.cls = nn.Sequential(
            block(in_ch, mid, k=3, s=1),
            nn.Conv2d(mid, num_anchors * num_classes, 3, 1, 1),
        )
        self.reg = nn.Sequential(
            block(in_ch, mid, k=3, s=1),
            nn.Conv2d(mid, num_anchors * 4, 3, 1, 1),
        )

    def forward(self, x):
        cls = self.cls(x)  # [B, A*C, H, W]
        reg = self.reg(x)  # [B, A*4, H, W]

        b, ac, h, w = cls.shape
        # -> [B, C, A, H, W] -> [B, H, W, A, C] -> [B, H*W*A, C]
        cls = (cls
               .view(b, self.num_classes, self.num_anchors, h, w)
               .permute(0, 3, 4, 2, 1)
               .contiguous()
               .view(b, -1, self.num_classes))

        b, ar, h, w = reg.shape
        # -> [B, 4, A, H, W] -> [B, H, W, A, 4] -> [B, H*W*A, 4]
        reg = (reg
               .view(b, 4, self.num_anchors, h, w)
               .permute(0, 3, 4, 2, 1)
               .contiguous()
               .view(b, -1, 4))
        return cls, reg



# -----------------------------
# Full Model
# -----------------------------

class MobileNetV4SSD(nn.Module):
    """
    MobileNetV4 backbone (features_only) + extra downsampling layers + SSD/SSDLite heads.
    Loads weights from timm local registry with the correct pretrained tag overlay.
    """
    def __init__(
        self,
        num_classes: int,
        lite: bool = False,
        img_size: int = 320,
        backbone_name: str = 'mobilenetv4_conv_small',
        pretrained_tag: str = 'e500_r256_in1k',
        feat_levels_from_backbone: int = 3,
        num_levels: int = 5,
        ratios: List[List[float]] = None,
    ):
        super().__init__()
        assert num_levels >= feat_levels_from_backbone >= 1

        # 1) Backbone from local timm registry (no HF auth), with tag overlay
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            pretrained_cfg_overlay={"tag": pretrained_tag},
        )

        self.img_size = img_size
        chs = self.backbone.feature_info.channels()
        self.backbone_indices = list(range(len(chs) - feat_levels_from_backbone, len(chs)))
        backbone_channels = [chs[i] for i in self.backbone_indices]

        # 2) Extra downsampling layers to reach `num_levels`
        self.extra = nn.ModuleList()
        in_ch = backbone_channels[-1]
        while len(backbone_channels) + len(self.extra) < num_levels:
            self.extra.append(nn.Sequential(
                nn.Conv2d(in_ch, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ))
            in_ch = 256
        out_channels = backbone_channels + [256] * len(self.extra)

        # 3) Default aspect ratios per level (SSD-style)
        if ratios is None:
            ratios = [
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1/3.0],
                [1.0, 2.0, 0.5, 3.0, 1/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5],
            ][:num_levels]

        self.ratios = ratios
        self.num_classes = num_classes
        self.lite = lite

        # 4) Heads per level
        self.heads = nn.ModuleList()
        for c, r in zip(out_channels, self.ratios):
            num_anchors = 2 + (len([a for a in r if abs(a - 1.0) > 1e-6]))
            self.heads.append(LevelHead(c, num_anchors=num_anchors, num_classes=num_classes, lite=lite))

        # 5) Anchor generator config (SSD scales)
        self.strides = self._guess_strides()  # based on feature sizes
        self.scales = self._make_scales(num_levels)
        self.anchor_gen = AnchorGeneratorSSD(self.strides, img_size, self.scales, self.ratios)

    # --- Helpers for strides & scales
    def _guess_strides(self) -> List[int]:
        with torch.no_grad():
            x = torch.zeros(1, 3, self.img_size, self.img_size)
            feats = self._get_feats(x)
        strides = []
        h_prev = self.img_size
        for f in feats:
            h = f.shape[-2]
            stride = h_prev // h
            strides.append(stride)
            h_prev = h
        return strides

    def _make_scales(self, L: int) -> List[Tuple[float, float]]:
        # typical SSD schedule across L levels
        s_min, s_max = 0.1, 0.9
        scales = []
        for k in range(1, L + 1):
            sk = s_min + (s_max - s_min) * (k - 1) / max(1, (L - 1))
            sk1 = s_min + (s_max - s_min) * (k) / max(1, (L - 1))
            scales.append((sk, sk1))
        return scales

    # --- Feature extraction (backbone + extras)
    def _get_feats(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats_all = self.backbone(x)  # list
        feats = [feats_all[i] for i in self.backbone_indices]
        y = feats[-1]
        for extra in self.extra:
            y = extra(y)
            feats.append(y)
        return feats

    def forward(self, x: torch.Tensor):
        feats = self._get_feats(x)
        cls_all, reg_all = [], []
        for f, head in zip(feats, self.heads):
            cls, reg = head(f)
            cls_all.append(cls)
            reg_all.append(reg)
        cls_out = torch.cat(cls_all, dim=1)  # [B, sum(AHW), C]
        reg_out = torch.cat(reg_all, dim=1)  # [B, sum(AHW), 4]
        anchors = self.anchor_gen(feats)     # list of [AHW, 4] per level (relative)
        return cls_out, reg_out, anchors


# -----------------------------
# Demo / CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="MobileNetV4-SSD/SSDLite (timm local registry)")
    parser.add_argument("--classes", type=int, default=21, help="number of classes")
    parser.add_argument("--img", type=int, default=320, help="square input size")
    parser.add_argument("--lite", action="store_true", help="use SSDLite depthwise-separable heads")
    parser.add_argument("--backbone", type=str, default="mobilenetv4_conv_small", help="timm backbone name")
    parser.add_argument("--tag", type=str, default="e500_r256_in1k", help="pretrained tag to overlay")
    args = parser.parse_args()

    model = MobileNetV4SSD(
        num_classes=args.classes,
        lite=args.lite,
        img_size=args.img,
        backbone_name=args.backbone,
        pretrained_tag=args.tag,
    )
    model.eval()

    x = torch.randn(1, 3, args.img, args.img)
    with torch.no_grad():
        cls, reg, anchors = model(x)

    num_anchors = sum(a.shape[0] for a in anchors)
    print("\n=== Forward pass OK ===")
    print(f"Input: {tuple(x.shape)}")
    print(f"classification logits: {tuple(cls.shape)}  [B, N_anchors, C]")
    print(f"bbox regression:       {tuple(reg.shape)}  [B, N_anchors, 4]")
    print(f"levels: {len(anchors)}; per-level anchors: {[a.shape[0] for a in anchors]}; total={num_anchors}")
    print(f"strides (guessed): {model.strides}")
    print(f"scales: {[(round(a,3), round(b,3)) for a,b in model.scales]}")


if __name__ == "__main__":
    main()
