from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import math
class MySSDLiteHead(nn.Module):
    """
    Lightweight SSD-style head over P3, P4, P5.
    - Depthwise separable conv blocks (SSDLite).
    - Outputs raw cls/reg maps; loss computed separately.

    dets_out structure:
      {
        'cls': [B, A_tot, num_classes],
        'reg': [B, A_tot, 4]
      }
    """
    def __init__(self, num_classes: int, in_ch: int = 256, num_anchors: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def dw_sep_block(c_in):
            return nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_in, 1, bias=False),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
            )

        # one block + predictors per scale
        self.blocks = nn.ModuleList([dw_sep_block(in_ch) for _ in range(3)])  # P3,P4,P5
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(in_ch, num_anchors * num_classes, 3, padding=1)
            for _ in range(3)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(in_ch, num_anchors * 4, 3, padding=1)
            for _ in range(3)
        ])

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scales = ['P3', 'P4', 'P5']
        cls_all, reg_all = [], []
        for i, name in enumerate(scales):
            f = feats[name]
            b = self.blocks[i](f)
            cls = self.cls_heads[i](b)   # [B, A*C, H, W]
            reg = self.reg_heads[i](b)   # [B, A*4, H, W]

            B, _, H, W = cls.shape
            cls = cls.view(B, self.num_anchors, self.num_classes, H, W)
            reg = reg.view(B, self.num_anchors, 4, H, W)

            cls_all.append(cls.permute(0, 1, 3, 4, 2).reshape(B, -1, self.num_classes))
            reg_all.append(reg.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        cls_all = torch.cat(cls_all, dim=1)   # [B, A_tot, num_classes]
        reg_all = torch.cat(reg_all, dim=1)   # [B, A_tot, 4]
        return {'cls': cls_all, 'reg': reg_all}
