import json, cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class CocoDetDepth(Dataset):
    def __init__(self, coco_json, img_root, transforms=None, depth_scale_mm=True):
        self.transforms = transforms
        self.depth_scale_mm = depth_scale_mm
        self.img_root = Path(img_root)
        with open(coco_json, 'r') as f:
            self.coco = json.load(f)
        self.id2img = {im['id']: im for im in self.coco['images']}
        self.imgs = list(self.id2img.values())
        self.ann_by_img = {}
        for ann in self.coco['annotations']:
            self.ann_by_img.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        iminfo = self.imgs[idx]
        img_p = self.img_root / iminfo['file_name']
        img = cv2.imread(str(img_p))[:, :, ::-1]
        H, W = img.shape[:2]
        # depth
        depth_file = iminfo.get('depth_file', None)
        if depth_file is None:
            depth_m = None
            valid_mask = None
        else:
            d = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)  # uint16 mm
            if d.shape != (H, W):
                raise ValueError(f"Depth map size {d.shape} does not match RGB {(H, W)}; expected 16:9 @ 640-wide capture.")
            depth_m = (d.astype(np.float32) / 1000.0)  # convert millimeters → meters (absolute depth)
            valid_mask = (d > 0)
        # det targets
        anns = self.ann_by_img.get(iminfo['id'], [])
        boxes = []
        labels = []
        for a in anns:
            x,y,w,h = a['bbox']
            boxes.append([x,y,x+w,y+h])
            labels.append(a['category_id'])
        sample = {
            'image': img,
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0,4), np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), np.int64),
            'depth': depth_m,
            'depth_valid': valid_mask,
            'id': iminfo['id'],
            'file_name': iminfo['file_name']
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
