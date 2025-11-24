from __future__ import annotations
import os, sys, math, time, argparse, random, json, queue, traceback, atexit, shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# ---------- small IO helpers ----------
def _flush_print(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------- COCO writing (final + periodic, atomic) ----------
def write_coco_jsons(images: List[Dict], anns: List[Dict],
                     categories: List[Dict], out_dir: Path, split: float,
                     prefix: str = ""):
    """
    Writes {prefix}train.json and {prefix}val.json to out_dir atomically.
    Splits by enumeration index of images (0..N-1), then remaps IDs contiguously.
    """
    if not images:
        _flush_print("[WARN] No images captured; skipping JSON write.")
        return

    N = len(images)
    idx = list(range(N))
    random.shuffle(idx)
    k = int(N * split)
    idx_train = set(idx[:k])

    def dump_json(path_json: Path, use_val: bool):
        js = {
            'images': [],
            'annotations': [],
            'categories': categories,
            'info': {'description': 'CARLA COCO export'}
        }
        id_map = {}
        new_img_id = 1

        # Use enumeration index to decide train/val; map raw img['id'] -> new_img_id
        for j, img in enumerate(images):
            is_val = (j not in idx_train)
            if is_val != use_val:
                continue
            id_map[img['id']] = new_img_id
            img_copy = dict(img)
            img_copy['id'] = new_img_id
            js['images'].append(img_copy)
            new_img_id += 1

        new_ann_id = 1
        valid_img_ids = set(id_map.keys())
        for ann in anns:
            if ann['image_id'] not in valid_img_ids:
                continue
            a = dict(ann)
            a['id'] = new_ann_id
            a['image_id'] = id_map[ann['image_id']]
            js['annotations'].append(a)
            new_ann_id += 1

        _ensure_dir(path_json.parent)
        tmp_path = Path(str(path_json) + ".tmp")
        with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(js, f, indent=2)
        try:
            os.replace(tmp_path, path_json)  # atomic on Windows too
        except Exception:
            shutil.move(tmp_path, path_json)
        _flush_print(f"[OK] Wrote {path_json} | images={len(js['images'])} anns={len(js['annotations'])}")

    train_path = out_dir / f"{prefix}train.json"
    val_path   = out_dir / f"{prefix}val.json"
    _flush_print(f"[INFO] Writing COCO JSONs -> {train_path.name}, {val_path.name} ...")
    dump_json(train_path, use_val=False)
    dump_json(val_path,   use_val=True)
    _flush_print("[DONE] images dir:", out_dir / 'images')

# ---------- runtime state for atexit fallback ----------
class _State:
    images: List[Dict] = []
    anns: List[Dict] = []
    categories: List[Dict] = []
    out_dir: Path | None = None
    split: float = 0.9
    wrote_final: bool = False
STATE = _State()

def _atexit_writer():
    try:
        if (not STATE.wrote_final) and STATE.images and STATE.out_dir:
            _flush_print("[INFO] atexit: writing final COCO JSONs as last resort...")
            write_coco_jsons(STATE.images, STATE.anns, STATE.categories, STATE.out_dir, STATE.split, prefix="")
            STATE.wrote_final = True
    except Exception:
        traceback.print_exc()
atexit.register(_atexit_writer)

# ---------- CARLA helpers ----------
try:
    import carla
    from carla import TrafficLightState
except Exception as e:
    _flush_print("[ERROR] Could not import 'carla' Python API. Ensure CARLA egg is on sys.path.")
    _flush_print(e); sys.exit(1)

def build_K(w: int, h: int, fov_deg: float) -> np.ndarray:
    focal = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(world_loc: carla.Location, K: np.ndarray, world2cam: np.ndarray) -> Tuple[float, float, float]:
    p = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    pc = world2cam @ p
    # UE → conventional camera coordinates: (x,y,z) -> (y, -z, x)
    pc = np.array([pc[1], -pc[2], pc[0]], dtype=np.float64)
    z = pc[2]
    uvw = K @ pc
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    return float(u), float(v), float(z)

def bbox2d_from_actor_bb(actor: carla.Actor, K: np.ndarray, world2cam: np.ndarray,
                         w: int, h: int) -> Tuple[int, int, int, int] | None:
    bb: carla.BoundingBox = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())
    us, vs, zs = [], [], []
    for v in verts:
        u, v2, z = get_image_point(v, K, world2cam)
        us.append(u); vs.append(v2); zs.append(z)
    zs = np.array(zs)
    if not np.any(zs > 0):
        return None
    umin = max(0, int(min(us))); vmin = max(0, int(min(vs)))
    umax = min(w - 1, int(max(us))); vmax = min(h - 1, int(max(vs)))
    bw = umax - umin; bh = vmax - vmin
    if bw < 2 or bh < 2:
        return None
    return (umin, vmin, bw, bh)

def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union

def is_actor_stop_sign(a: carla.Actor) -> bool:
    tid = (getattr(a, 'type_id', '') or '').lower()
    if 'stop' in tid:
        return True
    try:
        for k in ('name', 'type', 'sign_type'):
            val = a.attributes.get(k, '') if hasattr(a, 'attributes') else ''
            if isinstance(val, str) and 'stop' in val.lower():
                return True
    except Exception:
        pass
    return False

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town10HD')
    ap.add_argument('--w', type=int, default=960)
    ap.add_argument('--h', type=int, default=540)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--sensor-tick', type=float, default=0.05, help='Seconds between captures (e.g., 0.05=20Hz)')
    ap.add_argument('--frames', type=int, default=2000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--spawn-npcs', type=int, default=60, help='extra traffic vehicles')
    ap.add_argument('--max-dist', type=float, default=120.0, help='meters; ignore actors beyond this')
    ap.add_argument('--min-area', type=int, default=50, help='min bbox area in pixels to keep')
    ap.add_argument('--split', type=float, default=0.9, help='train fraction')
    ap.add_argument('--flush-every', type=int, default=50, help='Write tmp JSONs every N frames (0=disable)')
    ap.add_argument('--save-every', type=int, default=1, help='Save/annotate every Nth frame (default: 1 = every frame)')
    # Extended labeling options
    ap.add_argument('--car-sides', action='store_true', help='Split car into car_front/back/left/right based on orientation to camera')
    ap.add_argument('--side-thresh-deg', type=float, default=45.0, help='Angle threshold in degrees for front/back vs side classification')
    args = ap.parse_args()

    if args.save_every < 1:
        _flush_print("[ERROR] --save-every must be >= 1")
        sys.exit(2)

    random.seed(args.seed)
    out_dir = Path(args.out)
    img_dir = out_dir / 'images'
    _ensure_dir(out_dir); _ensure_dir(img_dir)

    # Quick write test to fail fast if path is wrong/locked
    try:
        test_path = out_dir / "_write_test.tmp"
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        _flush_print("[ERROR] Cannot write to output folder:", out_dir)
        _flush_print("        Reason:", e)
        sys.exit(2)

    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "truck"},
        {"id": 4, "name": "bus"},
        {"id": 5, "name": "bicycle"},
        {"id": 6, "name": "motorcycle"},
        {"id": 7, "name": "traffic_light_red"},
        {"id": 8, "name": "traffic_light_yellow"},
        {"id": 9, "name": "traffic_light_green"},
        {"id": 10, "name": "stop sign"},
    ]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    tl_state_to_id = {
        TrafficLightState.Red: cat_name_to_id["traffic_light_red"],
        TrafficLightState.Yellow: cat_name_to_id["traffic_light_yellow"],
        TrafficLightState.Green: cat_name_to_id["traffic_light_green"],
    }
    stop_sign_id = cat_name_to_id["stop sign"]

    # Optional extended categories
    car_side_ids = {}
    next_id = max(c["id"] for c in categories) + 1
    if args.car_sides:
        for nm in ("car_front", "car_back", "car_left", "car_right"):
            car_side_ids[nm] = next_id
            categories.append({"id": next_id, "name": nm})
            next_id += 1
    try:
        with open(out_dir / 'classes.txt', 'w', encoding='utf-8') as f:
            for c in categories: f.write(c['name'] + '\n')
    except Exception as e:
        _flush_print("[WARN] Failed to write classes.txt:", e)

    # Expose to atexit
    STATE.images = []
    STATE.anns = []
    STATE.categories = categories
    STATE.out_dir = out_dir
    STATE.split = args.split

    client = carla.Client(args.host, args.port); client.set_timeout(20.0)
    world = client.get_world()
    if args.town and args.town not in world.get_map().name:
        world = client.load_world(args.town); time.sleep(1.0)

    # Weather diversity
    weather_presets = [getattr(carla.WeatherParameters, x) for x in dir(carla.WeatherParameters) if x[0].isupper()]
    world.set_weather(random.choice(weather_presets))

    blueprints = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spawn = carla.Transform(carla.Location(x=95.70, y=66.00, z=2.00), carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0))
    # Ego blueprint fallback
    try:
        ego_bp = blueprints.find('vehicle.taxi.ford')
    except Exception:
        vlist = blueprints.filter('vehicle.*')
        if not vlist: raise RuntimeError('No vehicle blueprints found.')
        ego_bp = random.choice(vlist)
        
    ego = world.try_spawn_actor(ego_bp, spawn)
    if ego is None:
        raise RuntimeError('Failed to spawn ego vehicle')

    # RGB camera
    cam_bp = blueprints.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.w))
    cam_bp.set_attribute('image_size_y', str(args.h))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(args.sensor_tick))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.6))  # hood mount
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # Sync mode
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.sensor_tick
    world.apply_settings(settings)

    tm = None
    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        ego.set_autopilot(True)
        try:
            tm.set_global_distance_to_leading_vehicle(2.5)
            tm.set_random_device_seed(args.seed)
            tm.global_percentage_speed_difference(random.uniform(-10, 10))
        except Exception:
            pass
    except Exception:
        tm = None

    # NPCs
    npcs = []
    for _ in range(int(args.spawn_npcs)):
        vbp = random.choice(blueprints.filter('vehicle.*'))
        npc = world.try_spawn_actor(vbp, random.choice(spawn_points))
        if npc:
            try: npc.set_autopilot(True)
            except Exception: pass
            npcs.append(npc)

    # Bounded sensor queue (keep freshest frame)
    q_img: queue.Queue = queue.Queue(maxsize=1)
    def _on_img(img):
        try: q_img.get_nowait()
        except queue.Empty: pass
        try: q_img.put_nowait(img)
        except queue.Full: pass
    camera.listen(_on_img)

    K = build_K(args.w, args.h, args.fov)
    images: List[Dict] = STATE.images
    anns: List[Dict]   = STATE.anns
    ann_id = 1

    def cat_id_for_actor(a: carla.Actor) -> int | None:
        tid = a.type_id
        if tid.startswith('walker.pedestrian'): return 1
        if tid.startswith('vehicle.'):
            if 'truck' in tid: return 3
            if 'bus' in tid:   return 4
            if 'bicycle' in tid: return 5
            if 'motorcycle' in tid or 'motorbike' in tid: return 6
            # Car and others default to 2 unless side-splitting enabled
            return 2
        return None

    # Warm-up (avoid first-frame stalls)
    first_img = None
    for _ in range(20):  # ~1s at 20Hz
        world.tick()
        try:
            first_img = q_img.get(timeout=1.0); break
        except queue.Empty:
            pass
    if first_img is None:
        raise RuntimeError("Camera didn't produce frames during warmup.")
    last_frame = first_img.frame

    _flush_print('[INFO] Recording...')
    try:
        for i in range(1, args.frames + 1):
            world.tick()
            try:
                image: carla.Image = q_img.get(timeout=2.0)
            except queue.Empty:
                _flush_print(f"[WARN] Camera frame timeout at loop index {i}; continuing.")
                continue

            # drop stale frames
            if image.frame <= last_frame:
                while True:
                    try:
                        nxt = q_img.get_nowait()
                        if nxt.frame > last_frame:
                            image = nxt; break
                    except queue.Empty:
                        break
                if image.frame <= last_frame:
                    continue
            last_frame = image.frame

            # Only save/annotate every Nth sim step (but keep ticking every step)
            if (i % args.save_every) != 0:
                if i % 50 == 0:
                    _flush_print(f"[Frame {i}] (skipped) images={len(images)} anns={len(anns)}")
                if args.flush_every > 0 and (i % args.flush_every == 0):
                    try:
                        write_coco_jsons(images, anns, categories, out_dir, args.split, prefix="tmp.")
                    except Exception:
                        _flush_print("[ERROR] Periodic JSON write failed:"); traceback.print_exc()
                continue

            # Compute world->camera
            world_2_cam = np.array(camera.get_transform().get_inverse_matrix())

            # Save image & COCO image entry
            img_path = img_dir / f"{i:08d}.jpg"  # use sim step index; JSON remaps IDs later
            image.save_to_disk(str(img_path))
            images.append({
                'id': i,
                'file_name': f"images/{img_path.name}",
                'width': args.w,
                'height': args.h
            })

            dyn_boxes = []

            ego_tf = ego.get_transform()
            cam_forward = camera.get_transform().get_forward_vector()

            # Vehicles
            for a in world.get_actors().filter('*vehicle*'):
                if a.id == ego.id: continue
                if a.get_transform().location.distance(ego_tf.location) > args.max_dist: continue
                ray = a.get_transform().location - camera.get_transform().location
                if cam_forward.dot(ray) <= 0: continue
                cat = cat_id_for_actor(a)
                if cat is None: continue
                # Car side-classification (optional)
                if args.car_sides and cat == 2:
                    try:
                        vf = a.get_transform().get_forward_vector()
                        cf = camera.get_transform().get_forward_vector()
                        cr = camera.get_transform().get_right_vector()
                        dot = float(cf.dot(vf))
                        # thresholds
                        t = math.cos(math.radians(max(0.0, min(89.0, args.side_thresh_deg))))
                        if dot <= -t:
                            cat_name = 'car_front'
                        elif dot >= t:
                            cat_name = 'car_back'
                        else:
                            # side: decide by projection on camera right vector
                            side_dot = float(cr.dot(vf))
                            cat_name = 'car_right' if side_dot > 0 else 'car_left'
                        cat = car_side_ids.get(cat_name, cat)
                    except Exception:
                        pass
                bb2d = bbox2d_from_actor_bb(a, K, world_2_cam, args.w, args.h)
                if bb2d is None or bb2d[2]*bb2d[3] < args.min_area: continue
                anns.append({
                    'id': ann_id, 'image_id': i, 'category_id': cat,
                    'bbox': [float(bb2d[0]), float(bb2d[1]), float(bb2d[2]), float(bb2d[3])],
                    'area': float(bb2d[2]*bb2d[3]), 'iscrowd': 0
                }); ann_id += 1
                dyn_boxes.append(bb2d)

            # Pedestrians
            for a in world.get_actors().filter('walker.pedestrian.*'):
                if a.get_transform().location.distance(ego_tf.location) > args.max_dist: continue
                ray = a.get_transform().location - camera.get_transform().location
                if cam_forward.dot(ray) <= 0: continue
                bb2d = bbox2d_from_actor_bb(a, K, world_2_cam, args.w, args.h)
                if bb2d is None or bb2d[2]*bb2d[3] < args.min_area: continue
                anns.append({
                    'id': ann_id, 'image_id': i, 'category_id': 1,
                    'bbox': [float(bb2d[0]), float(bb2d[1]), float(bb2d[2]), float(bb2d[3])],
                    'area': float(bb2d[2]*bb2d[3]), 'iscrowd': 0
                }); ann_id += 1
                dyn_boxes.append(bb2d)

            # Stop sign actors (when available)
            try:
                for a in world.get_actors().filter('traffic.*'):
                    if 'light' in (a.type_id or '').lower(): continue
                    if not is_actor_stop_sign(a): continue
                    if a.get_transform().location.distance(ego_tf.location) > args.max_dist: continue
                    ray = a.get_transform().location - camera.get_transform().location
                    if cam_forward.dot(ray) <= 0: continue
                    bb2d = bbox2d_from_actor_bb(a, K, world_2_cam, args.w, args.h)
                    if bb2d is None or bb2d[2]*bb2d[3] < args.min_area: continue
                    if any(iou_xywh(bb2d, db) > 0.3 for db in dyn_boxes): continue
                    anns.append({
                        'id': ann_id, 'image_id': i, 'category_id': stop_sign_id,
                        'bbox': [float(bb2d[0]), float(bb2d[1]), float(bb2d[2]), float(bb2d[3])],
                        'area': float(bb2d[2]*bb2d[3]), 'iscrowd': 0
                    }); ann_id += 1
            except Exception:
                pass

            # Traffic lights with explicit color labels
            try:
                for tl in world.get_actors().filter('traffic.traffic_light*'):
                    state = tl.get_state() if hasattr(tl, "get_state") else None
                    cat = tl_state_to_id.get(state)
                    if cat is None:
                        continue
                    if tl.get_transform().location.distance(ego_tf.location) > args.max_dist: continue
                    ray = tl.get_transform().location - camera.get_transform().location
                    if cam_forward.dot(ray) <= 0: continue
                    bb2d = bbox2d_from_actor_bb(tl, K, world_2_cam, args.w, args.h)
                    if bb2d is None or bb2d[2] * bb2d[3] < args.min_area:
                        continue
                    if any(iou_xywh(bb2d, db) > 0.3 for db in dyn_boxes):
                        continue
                    anns.append({
                        'id': ann_id, 'image_id': i, 'category_id': cat,
                        'bbox': [float(bb2d[0]), float(bb2d[1]), float(bb2d[2]), float(bb2d[3])],
                        'area': float(bb2d[2]*bb2d[3]), 'iscrowd': 0
                    }); ann_id += 1
            except Exception:
                pass

            # Removed heuristic crosswalk detection (class not exported)

            if i % 50 == 0:
                _flush_print(f"[Frame {i}] images={len(images)} anns={len(anns)}")

            # periodic tmp flush by loop index
            if args.flush_every > 0 and (i % args.flush_every == 0):
                try:
                    write_coco_jsons(images, anns, categories, out_dir, args.split, prefix="tmp.")
                except Exception:
                    _flush_print("[ERROR] Periodic JSON write failed:"); traceback.print_exc()

        _flush_print('[INFO] Finished recording frames.')

        # ---- WRITE JSONS NOW (before cleanup) ----
        try:
            _flush_print("[INFO] Writing COCO JSONs to:", str(out_dir.resolve()))
            write_coco_jsons(images, anns, categories, out_dir, args.split, prefix="")
            STATE.wrote_final = True
        except Exception:
            _flush_print("[ERROR] Early JSON write failed:")
            traceback.print_exc()
        # ------------------------------------------

    except Exception:
        _flush_print("[ERROR] Exception during capture loop:"); traceback.print_exc()

    finally:
        # cleanup must not block final writing
        try: camera.stop()
        except Exception: pass
        try: ego.set_autopilot(False)
        except Exception: pass
        try:
            tm and tm.set_synchronous_mode(False)
        except Exception:
            pass
        for a in [camera, ego] + npcs:
            try: a.destroy()
            except Exception: pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    # final write (only if early write didn't happen)
    try:
        if not STATE.wrote_final:
            write_coco_jsons(images, anns, categories, out_dir, args.split, prefix="")
            STATE.wrote_final = True
    except Exception:
        _flush_print("[ERROR] Writing final COCO JSONs failed:"); traceback.print_exc()

if __name__ == '__main__':
    main()
