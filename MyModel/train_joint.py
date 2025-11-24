import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_joint_det_depth import CocoDetDepth
from model_defs import JointDetDepthHoPEMem, LossWeights, compute_multitask_loss


class DummyDetLoss(nn.Module):
    def forward(self, out, tgt):
        # Replace with your detector loss (cls+box+obj)
        return out['loss'], {'det_stub': 0.0}


def collate_fn(batch):
    return batch


def build_model() -> JointDetDepthHoPEMem:
    """Construct JointDetDepthHoPEMem with your backbone + detection head."""
    raise NotImplementedError('Plug your backbone + detection head here')


def train():
    dataset = CocoDetDepth('dataset_carla10/annotations/instances_train_with_depth.json',
                           'dataset_carla10', transforms=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = build_model().cuda()
    det_loss_fn = DummyDetLoss()
    w = LossWeights(depth=0.7, silog=1.0, smooth=0.1, l1=0.2, distill=0.0)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(20):
        model.train()
        for batch in loader:
            # prepare tensors
            # NOTE: adapt this to your detector's expected targets
            imgs = torch.stack([torch.from_numpy(x['image']).permute(2, 0, 1) for x in batch]).float().cuda() / 255.0
            depth_gt, valid = [], []
            for x in batch:
                if x['depth'] is None:
                    depth_gt.append(torch.zeros(1, imgs.shape[2], imgs.shape[3]))
                    valid.append(torch.zeros(1, imgs.shape[2], imgs.shape[3]).bool())
                else:
                    d = torch.from_numpy(x['depth']).unsqueeze(0)
                    v = torch.from_numpy(x['depth_valid']).unsqueeze(0).bool()
                    depth_gt.append(d)
                    valid.append(v)
            depth_gt = torch.stack(depth_gt).cuda()
            valid = torch.stack(valid).cuda()

            dets_out, invd, _ = model(imgs)
            L, logs = compute_multitask_loss(dets_out, tgt=None,  # supply real targets
                                             invd=invd, depth_gt=depth_gt, valid_mask=valid,
                                             invd_teacher=None, w=w,
                                             detection_loss_fn=det_loss_fn)
            opt.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print('epoch', epoch)


if __name__ == '__main__':
    train()
