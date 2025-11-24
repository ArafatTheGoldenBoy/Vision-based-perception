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
