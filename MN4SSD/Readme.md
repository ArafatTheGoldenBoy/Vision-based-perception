High‑level summary

Anchors were broken on higher feature levels → fixed stride computation in MobileNetV4SSD so anchors cover the whole image, not just a corner.

Evaluation geometry was inconsistent and off → made eval_mnv4.py use the same letterbox / warp logic as train & infer, and correctly map boxes back to the original image.

Training crashed with GradScaler → fixed GradScaler initialization so AMP works.

Box / NMS / letterbox helpers were duplicated → created ssd_utils.py and reused the same implementations in train, eval, and infer to avoid divergence.