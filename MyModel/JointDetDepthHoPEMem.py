"""Legacy convenience shim.

The original `joint_det_depth_ho_pe_mem_det_depth_with_transformer_neck_memory.py` file has
been split into dedicated modules:
- model_defs.py: core model, HoPE/Transformer neck, memory, depth utilities, multitask loss.
- carla_capture_det_depth.py: CARLA 0.10 data capture loop.
- coco_pair_depth.py: COCO annotation merger adding depth file references.
- dataset_joint_det_depth.py: PyTorch dataset loader for joint detection + depth.
- train_joint.py: training skeleton wiring the dataset, model, and losses.
- run_carla_inference.py: inference loop sketch for synchronous CARLA runs.

Import from `model_defs` directly, or keep using this file which simply re-exports all
symbols for backward compatibility.
"""

from model_defs import *  # noqa: F401,F403

class JointDetDepthHoPEMem(nn.Module):
    def __init__(self, backbone: nn.Module, det_head: nn.Module, p3_ch: int = 256,
                 use_neck: bool = True, use_mem: bool = True, use_depth_fusion: bool = True):
        super().__init__()
        self.backbone = backbone          # MNv4Backbone now
        self.use_neck = use_neck
        self.use_mem = use_mem
        self.neck = TransformerNeck(p3_ch, blocks=1) if use_neck else nn.Identity()
        self.mem  = MemoryXAttn(p3_ch, slots=128) if use_mem else nn.Identity()
        self.det_head = det_head
        self.depth_head = DepthHead(p3_ch)
        self.use_depth_fusion = use_depth_fusion
        if use_depth_fusion:
            self.depth_fusion = DepthAwareFusion(c=p3_ch)

    ...
    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)  # {'P3','P4','P5', ...}
        p3 = feats['P3']

        p3 = self.neck(p3)    # HoPE / TransformerNeck
        p3 = self.mem(p3)     # Titans-style memory

        # Depth prediction (full-res)
        invd = self.depth_head(p3, x.shape[-2:])  # [B,1,H,W]

        # >>> Your custom piece:
        if self.use_depth_fusion:
            # downsample inverse depth to P3's resolution and detach to keep
            # detection gradients from messing with depth branch (your choice).
            invd_low = F.interpolate(invd.detach(), size=p3.shape[-2:],
                                     mode='bilinear', align_corners=False)
            p3 = self.depth_fusion(p3, invd_low)

        feats_mod = dict(feats)
        feats_mod['P3'] = p3     # depth-aware features

        dets = self.det_head(feats_mod)   # SSD/SSDLite-style head
        aux = {'P3': p3}
        return dets, invd, aux

