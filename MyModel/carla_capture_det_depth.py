#!/usr/bin/env python3
import os, csv, json, math, time
from pathlib import Path
import numpy as np
import carla

OUT = Path('dataset_carla10')  # change as needed
SPLIT = 'train'                # 'train'|'val'|'test'
# 16:9 frames at 640 px width → 360 px height to stay lightweight and match training
IMG_W = 640
IMG_H = int(round(IMG_W * 9 / 16))
FOV = 90.0
MAX_DEPTH_M = 1000.0           # CARLA depth normalization far plane
TICK = 0.05                    # 20 Hz
N_FRAMES = 5000

# ---------------- helpers ----------------
def intrinsics_from_fov(w, h, fov_deg):
    fx = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy

def carla_depth_to_meters(image):
    # CARLA encodes depth in 24 bits across RGB; normalized to [0,1]
    # Z_meters = normalized * MAX_DEPTH_M
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3].astype(np.uint32)
    # normalized in [0,1]
    norm = (rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 256 * 256) / float(256**3 - 1)
    depth_m = (norm * MAX_DEPTH_M).astype(np.float32)
    return depth_m

# ---------------- main ----------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'images' / SPLIT).mkdir(parents=True, exist_ok=True)
    (OUT / 'depth' / SPLIT).mkdir(parents=True, exist_ok=True)
    meta_dir = OUT / 'meta'; meta_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client('localhost', 2000); client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK
    world.apply_settings(settings)

    # ego vehicle
    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.filter('vehicle.*model3*')[0]
    spawn = np.random.choice(world.get_map().get_spawn_points())
    ego = world.spawn_actor(veh_bp, spawn)

    # camera blueprints
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_W))
    cam_bp.set_attribute('image_size_y', str(IMG_H))
    cam_bp.set_attribute('fov', str(FOV))

    dep_bp = bp_lib.find('sensor.camera.depth')
    dep_bp.set_attribute('image_size_x', str(IMG_W))
    dep_bp.set_attribute('image_size_y', str(IMG_H))
    dep_bp.set_attribute('fov', str(FOV))

    cam_tf = carla.Transform(carla.Location(x=1.4, z=1.6))  # hood/roof mount
    rgb_cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
    dpt_cam = world.spawn_actor(dep_bp, cam_tf, attach_to=ego)

    fx, fy, cx, cy = intrinsics_from_fov(IMG_W, IMG_H, FOV)

    # queues
    from queue import Queue
    q_rgb, q_dpt = Queue(), Queue()
    rgb_cam.listen(q_rgb.put); dpt_cam.listen(q_dpt.put)

    # write intrinsics & stream meta
    with open(meta_dir / 'intrinsics.json', 'w') as f:
        json.dump({'w': IMG_W, 'h': IMG_H, 'fov': FOV, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}, f, indent=2)

    with open(meta_dir / f'index_{SPLIT}.csv', 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['frame', 'image_path', 'depth_path', 'timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

        for i in range(N_FRAMES):
            world.tick()
            irgb = q_rgb.get(True, 5.0)
            idpt = q_dpt.get(True, 5.0)
            assert irgb.frame == idpt.frame, 'desync: rgb/depth frames differ'
            frame = irgb.frame

            # save RGB
            rgb_arr = np.frombuffer(irgb.raw_data, dtype=np.uint8).reshape((IMG_H, IMG_W, 4))[:, :, :3][:, :, ::-1]
            img_path = OUT / 'images' / SPLIT / f'{frame:06d}.png'
            import cv2
            cv2.imwrite(str(img_path), rgb_arr)

            # save depth as 16-bit PNG (millimeters)
            depth_m = carla_depth_to_meters(idpt)
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            dpt_path = OUT / 'depth' / SPLIT / f'{frame:06d}.png'
            cv2.imwrite(str(dpt_path), depth_mm)

            # pose
            tf = rgb_cam.get_transform(); loc = tf.location; rot = tf.rotation
            writer.writerow([frame, str(img_path), str(dpt_path), irgb.timestamp,
                             loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw])

    # cleanup
    rgb_cam.stop(); dpt_cam.stop()
    rgb_cam.destroy(); dpt_cam.destroy(); ego.destroy()
    settings.synchronous_mode = False; world.apply_settings(settings)

if __name__ == '__main__':
    main()
