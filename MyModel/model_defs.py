# JointDetDepth-HoPE-Mem
# Single-model object detection + monocular depth with:
#  - Shared backbone (plug your MobileNet/FPN or ViT-FPN)
#  - TransformerNeck on P3 (1/8 scale) with 2D RoPE (drop-in point for HoPE)
#  - Titans-style external memory read (global, light)
#  - Detection head (SSD/YOLO/DETR) unchanged API
#  - Lightweight depth head
#  - Robust per-object distance aggregation (median + MAD + optional alpha-beta)
#  - Optional ground-plane scaling and teacher distillation hooks
#
# Notes:
#  - Replace `apply_rope_2d` with your actual HoPE variant. The hook is in AttnBlock.pos_encode().
#  - Memory is implemented as stateful buffers with EMA writes; call `.mem_write()` each frame.
#  - Keep everything at 1/8 scale in the neck/memory to stay real-time.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= Positional Encoding (HoPE hook) ========================= #

def apply_rope_2d(q: torch.Tensor, k: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2D Rotary Positional Embedding (RoPE) for attention. Acts as a safe baseline.
    Replace this with your HoPE variant if you have one. Shapes:
      q, k: [B, Heads, N, Dh], with N = H*W (flattened spatial tokens)
    """
    B, Heads, N, Dh = q.shape
    assert N == H * W, "Token count must match H*W"
    # Build 2D angles for x (W) and y (H)
    device = q.device
    # half-dim for rotary (even Dh assumed)
    half = Dh // 2
    if half == 0:
        return q, k
    # Frequencies (Theta) â€” base 10k like standard RoPE
    base = 10000.0
    idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (idx / half))  # [half]

    # Token grid positions
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    pos_y = yy.reshape(-1).float()  # [N]
    pos_x = xx.reshape(-1).float()  # [N]

    # Angles for y and x
    ang_y = torch.einsum('n,d->nd', pos_y, inv_freq)  # [N,half]
    ang_x = torch.einsum('n,d->nd', pos_x, inv_freq)  # [N,half]

    # Expand to [B,Heads,N,half]
    cos_y = ang_y.cos()[None, None, :, :]
    sin_y = ang_y.sin()[None, None, :, :]
    cos_x = ang_x.cos()[None, None, :, :]
    sin_x = ang_x.sin()[None, None, :, :]

    def rotary(t):
        t1, t2 = t[..., :half], t[..., half:2*half]
        # Apply y-rotation on first half, x-rotation on second half
        # y component
        ty = (t1 * cos_y) + (rotate_half(t1) * sin_y)
        # x component
        tx = (t2 * cos_x) + (rotate_half(t2) * sin_x)
        out = torch.cat([ty, tx, t[..., 2*half:]], dim=-1)
        return out

    def rotate_half(x):
        # [B,H,N,half] -> rotate pairs
        x_odd = x[..., 1::2]
        x_even = x[..., ::2]
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = -x_odd
        x_rot[..., 1::2] = x_even
        return x_rot

    q = rotary(q)
    k = rotary(k)
    return q, k

# ========================= Transformer Neck (with HoPE hook) ========================= #

class AttnBlock(nn.Module):
    def __init__(self, dim: int, nhead: int = 4, attn_drop: float = 0.0, ffn_drop: float = 0.0):
        super().__init__()
        assert dim % nhead == 0, "dim must be divisible by nhead"
        self.dim = dim
        self.nhead = nhead
        self.dh = dim // nhead
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop_attn = nn.Dropout(attn_drop)
        self.drop_ffn = nn.Dropout(ffn_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(ffn_drop),
            nn.Linear(dim*4, dim)
        )

    def pos_encode(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # >>> Replace apply_rope_2d with your HoPE implementation if available.
        return apply_rope_2d(q, k, H, W)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, N, C], N=H*W
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.nhead, self.dh).permute(2, 0, 3, 1, 4)  # [3,B,Heads,N,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.pos_encode(q, k, H, W)
        attn = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        attn = self.drop_attn(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(out)
        x = self.norm1(x)
        x = x + self.drop_ffn(self.ffn(x))
        x = self.norm2(x)
        return x

class TransformerNeck(nn.Module):
    def __init__(self, c_in: int = 256, blocks: int = 1, nhead: int = 4):
        super().__init__()
        self.align = nn.Conv2d(c_in, c_in, 1, bias=False)
        self.blocks = nn.ModuleList([AttnBlock(c_in, nhead=nhead) for _ in range(blocks)])

    def forward(self, f_p3: torch.Tensor) -> torch.Tensor:
        # f_p3: [B,C,H,W]
        x = self.align(f_p3)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B,N,C]
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

# ========================= Titans-style Memory (global, light) ========================= #

class MemoryXAttn(nn.Module):
    def __init__(self, c: int = 256, slots: int = 128, ema: float = 0.9):
        super().__init__()
        self.c = c
        self.slots = slots
        self.ema = ema
        # Memory as buffers (stateful, not trained)
        self.register_buffer('mem_k', torch.zeros(slots, c))
        self.register_buffer('mem_v', torch.zeros(slots, c))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        # Projections
        self.qproj = nn.Conv2d(c, c, 1)
        self.kproj = nn.Linear(c, c)
        self.vproj = nn.Linear(c, c)
        self.fuse  = nn.Conv2d(2*c, c, 1)

    @torch.no_grad()
    def reset_memory(self):
        self.mem_k.zero_(); self.mem_v.zero_(); self.initialized.zero_()

    @torch.no_grad()
    def mem_write(self, f_t: torch.Tensor, gate: float = 0.1):
        """EMA write using global pooled features.
        f_t: [B,C,H,W]
        """
        g = f_t.mean(dim=[0, 2, 3])  # [C]
        if self.initialized.item() == 0:
            # initialize keys as random proj of first feature, values as same
            self.mem_k.copy_(g.unsqueeze(0).repeat(self.slots, 1))
            self.mem_v.copy_(g.unsqueeze(0).repeat(self.slots, 1))
            self.initialized.fill_(1)
        else:
            self.mem_v.mul_(self.ema).add_((1 - self.ema) * g)
            # simple key refresh: drift slightly toward new g
            self.mem_k.mul_(self.ema).add_((1 - self.ema) * g)

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f_t.shape
        q = self.qproj(f_t).flatten(2).transpose(1, 2)  # [B,N,C]
        # If uninitialized, pass-through
        if self.initialized.item() == 0:
            return f_t
        k = self.kproj(self.mem_k).unsqueeze(0)        # [1,S,C]
        v = self.vproj(self.mem_v).unsqueeze(0)        # [1,S,C]
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (C ** 0.5), dim=-1)  # [B,N,S]
        m = (attn @ v).transpose(1, 2).view(B, C, H, W)
        return self.fuse(torch.cat([f_t, m], dim=1))

# ========================= Heads ========================= #

class DepthHead(nn.Module):
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1),    nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1)       # inverse depth logits
        )
    def forward(self, f_p3: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        invd_low = F.softplus(self.net(f_p3)) + 1e-4        # [B,1,H/8,W/8]
        invd = F.interpolate(invd_low, size=out_hw, mode='bilinear', align_corners=False)
        return invd  # 1/m (up to scale if not supervised metrically)

# ========================= Joint Model ========================= #

class JointDetDepthHoPEMem(nn.Module):
    def __init__(self, backbone: nn.Module, det_head: nn.Module, p3_ch: int = 256,
                 use_neck: bool = True, use_mem: bool = True):
        super().__init__()
        self.backbone = backbone          # must return dict with 'P3'
        self.use_neck = use_neck
        self.use_mem = use_mem
        self.neck = TransformerNeck(p3_ch, blocks=1) if use_neck else nn.Identity()
        self.mem  = MemoryXAttn(p3_ch, slots=128) if use_mem else nn.Identity()
        self.det_head = det_head
        self.depth_head = DepthHead(p3_ch)

    @torch.no_grad()
    def reset_memory(self):
        if isinstance(self.mem, MemoryXAttn):
            self.mem.reset_memory()

    @torch.no_grad()
    def mem_write(self, p3: torch.Tensor):
        if isinstance(self.mem, MemoryXAttn):
            self.mem.mem_write(p3)

    def forward(self, x: torch.Tensor) -> Tuple[Dict, torch.Tensor, Dict[str, torch.Tensor]]:
        feats: Dict[str, torch.Tensor] = self.backbone(x)  # should include 'P3'
        p3 = feats['P3']
        p3 = self.neck(p3)            # TransformerNeck w/ HoPE hook
        p3 = self.mem(p3)             # Memory cross-attention fusion
        feats_mod = dict(feats)
        feats_mod['P3'] = p3
        dets = self.det_head(feats_mod)                 # unchanged API
        invd = self.depth_head(p3, x.shape[-2:])        # [B,1,H,W]
        aux = {'P3': p3}
        return dets, invd, aux

# ========================= Losses & Utilities ========================= #

def silog(pred_z: torch.Tensor, gt_z: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    p = torch.log(pred_z[mask] + eps)
    g = torch.log(gt_z[mask] + eps)
    d = p - g
    return torch.sqrt((d**2).mean() - 0.85*(d.mean()**2) + eps)

def smooth_edge_loss(z: torch.Tensor):
    dx = z[:, :, 1:, :] - z[:, :, :-1, :]
    dy = z[:, :, :, 1:] - z[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

def depth_losses(inv_d: torch.Tensor, gt_depth_m: torch.Tensor, valid_mask: torch.Tensor):
    pred_z = 1.0 / (inv_d + 1e-6)
    l_silog = silog(pred_z, gt_depth_m, valid_mask)
    l_smooth = smooth_edge_loss(pred_z)
    l_l1 = (pred_z - gt_depth_m).abs()[valid_mask].mean()
    return l_silog, l_smooth, l_l1

@torch.no_grad()
def robust_object_distance(depth_m: torch.Tensor, box_xyxy: Tuple[int,int,int,int], erode_px: int = 2,
                           prev: Optional[float] = None, percentile: Optional[float] = None) -> Optional[float]:
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 += erode_px; y1 += erode_px; x2 -= erode_px; y2 -= erode_px
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_m[y1:y2, x1:x2]
    vals = roi.reshape(-1)
    vals = vals[torch.isfinite(vals) & (vals > 0)]
    if vals.numel() < 50:
        return None
    if percentile is not None:
        z = torch.quantile(vals, percentile)
    else:
        m = vals.median()
        mad = (vals - m).abs().median() + 1e-6
        keep = (vals - m).abs() < 3.0 * 1.4826 * mad
        z = vals[keep].median()
    if prev is None:
        return float(z)
    alpha = 0.6
    return float(prev + alpha * (z - prev))

# ========================= Ground-plane scaling (optional at runtime) ========================= #

@torch.no_grad()
def ground_depth_map(H: int, W: int, fx: float, fy: float, cx: float, cy: float,
                     cam_h_m: float, pitch_rad: float, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    rx, ry, rz = x, y, torch.ones_like(x)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    ryp = cp * ry - sp * rz
    rzp = sp * ry + cp * rz
    t = cam_h_m / (ryp.clamp(min=1e-6))
    Z_plane = (rzp * t).clamp(min=1e-3, max=1e4)
    return Z_plane

@torch.no_grad()
def scale_from_ground(unscaled_Z: torch.Tensor, Z_plane: torch.Tensor, ground_mask: torch.Tensor) -> float:
    vals_pred = unscaled_Z[ground_mask]
    vals_plane = Z_plane[ground_mask]
    if vals_pred.numel() < 100:
        return 1.0
    ratio = (vals_plane / (vals_pred.clamp(min=1e-3))).float()
    m = ratio.median()
    mad = (ratio - m).abs().median() + 1e-6
    keep = (ratio - m).abs() < 3.0 * 1.4826 * mad
    if keep.sum() < 100:
        return float(m)
    return float(ratio[keep].median())

# ========================= Training sketch ========================= #

@dataclass
class LossWeights:
    depth: float = 0.7
    silog: float = 1.0
    smooth: float = 0.1
    l1: float = 0.2
    distill: float = 0.0  # >0 if you use a teacher


def compute_multitask_loss(dets_out: Dict, dets_targets: Dict,
                           invd: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor,
                           invd_teacher: Optional[torch.Tensor], w: LossWeights,
                           detection_loss_fn) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Detection loss via your existing head/loss
    L_det, det_logs = detection_loss_fn(dets_out, dets_targets)
    # Depth losses
    L_silog, L_smooth, L_l1 = depth_losses(invd, depth_gt, valid_mask)
    L_depth = w.silog * L_silog + w.smooth * L_smooth + w.l1 * L_l1
    # Teacher distillation (optional)
    if invd_teacher is not None and w.distill > 0.0:
        conf_mask = (invd_teacher > 0).float()
        L_distill = ((invd - invd_teacher).abs() * conf_mask).sum() / (conf_mask.sum() + 1e-6)
    else:
        L_distill = torch.zeros((), device=invd.device)
    L = L_det + w.depth * L_depth + w.distill * L_distill
    logs = {
        'L_det': float(L_det.item()),
        'L_silog': float(L_silog.item()),
        'L_smooth': float(L_smooth.item()),
        'L_l1': float(L_l1.item()),
        'L_distill': float(L_distill.item()),
        'L_total': float(L.item())
    }
    return L, logs

# ========================= CARLA runtime sketch ========================= #

"""
Pseudocode for your synchronous tick loop:

model.reset_memory()
for tick in sim:
    rgb = get_rgb()
    H, W = rgb.shape[-2:]
    x = preprocess(rgb)
    dets, invd, aux = model(x)
    # Optional ground-plane scaling if you didn't lock scale in training
    if use_ground_scale:
        Z_unscaled = 1.0 / (invd + 1e-6)
        Zg = ground_depth_map(H, W, fx, fy, cx, cy, cam_h, pitch, device=Z_unscaled.device)
        gmask = get_ground_mask(H, W)  # bottom rows or semantic "road"
        s = scale_from_ground(Z_unscaled[0,0], Zg, gmask)
        depth_m = s * Z_unscaled[0,0]
    else:
        depth_m = (1.0 / (invd[0,0] + 1e-6))

    # Per-object distances
    boxes = decode_boxes(dets)  # [N,4] in xyxy, with scores/classes
    distances = []
    for b in boxes:
        z = robust_object_distance(depth_m, b, erode_px=2, prev=None)
        distances.append(z)

    # After a successful forward, update memory
    model.mem_write(aux['P3'].detach())

    render_or_log(rgb, boxes, distances, fps, etc.)
"""
