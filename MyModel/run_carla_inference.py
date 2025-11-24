import cv2
import numpy as np
import torch

from model_defs import JointDetDepthHoPEMem, robust_object_distance


class ZTracker:
    def __init__(self):
        self.prev = {}

    def update(self, obj_id, z):
        prev = self.prev.get(obj_id)
        if prev is None:
            self.prev[obj_id] = z
            return z
        alpha = 0.6
        zhat = prev + alpha * (z - prev)
        self.prev[obj_id] = zhat
        return zhat


def loop(model: JointDetDepthHoPEMem):
    """Example synchronous CARLA loop wiring detections + per-object distance."""
    ztrack = ZTracker()
    while True:
        rgb = get_rgb()
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        dets, invd, aux = model(x)
        depth_m = (1.0 / (invd[0, 0].clamp(min=1e-6))).detach().cpu()
        boxes, ids = decode_boxes_and_ids(dets)  # implement with your tracker
        for b, obj_id in zip(boxes, ids):
            z = robust_object_distance(depth_m, b, erode_px=2, prev=ztrack.prev.get(obj_id))
            if z is None:
                continue
            zhat = ztrack.update(obj_id, z)
            draw_box_and_text(rgb, b, f"{zhat:.1f} m")
        show(rgb)
        model.mem_write(aux['P3'].detach())
