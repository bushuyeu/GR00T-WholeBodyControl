#!/usr/bin/env python3
"""
Publish upper-body joint targets from results.pkl to GR00T-WholeBodyControl via ROS.

This script is designed to match the control loop you printed:

upper_body dim: 28
upper_body names (order):
  0..6   = left arm (7)
  7..13  = left hand (7)
  14..20 = right arm (7)
  21..27 = right hand (7)

The run_g1_control_loop expects a waypoint vector of length:
  28 (upper_body) + 3 (nav) + 1 (base_height) = 32

Therefore: we publish ONLY:
  - target_upper_body_pose (len 28)
  - navigate_cmd (len 3)
  - base_height_command (scalar)
  - timestamp, target_time

We DO NOT publish wrist_pose (it can change expected concatenated dims).

results.pkl format:
  {"fps": float, "dof_pos": np.ndarray (T, ndof), optional "dof_names": list[str]}

Usage:
  python gr00t_wbc/control/main/teleop/publish_upper_body_from_results.py \
    --results resources/poses/results.pkl \
    --loop \
    --teleop-frequency 30
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Path to g1.xml for MuJoCo collision checking
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_G1_XML_PATH = _PROJECT_ROOT / "gr00t_wbc" / "sim2mujoco" / "resources" / "robots" / "g1" / "g1.xml"

from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
)
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher


# The exact ordering expected by run_g1_control_loop.py (from your printout)
UPPER_BODY_NAMES_28 = [
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # left hand (7)
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # right hand (7)
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]


# Your MuJoCo results default ordering (29 DOF: legs + waist + arms, no hands)
DEFAULT_DOF_NAMES_29 = [
    # left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# Joint position limits from Unitree G1 29-DOF URDF (rad)
# Source: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf
JOINT_POSITION_LIMITS = {
    # left leg
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    # right leg
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    # waist
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    # left arm
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    # right arm
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.972222054, 1.972222054),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}

# 5% inward margin (matches POSITION_CRITICAL_MARGIN in joint_safety.py)
POSITION_CLAMP_MARGIN = 0.05

# Tighter limits to prevent arm-torso self-collision.
# URDF limits define the full mechanical range, but certain shoulder poses
# cause the arm to contact the torso. These overrides restrict adduction
# (arm moving inward toward body) to a safe range.
# Left roll negative = adduction; right roll positive = adduction (mirrored).
SELF_COLLISION_SAFE_LIMITS = {
    "left_shoulder_roll_joint": (-0.75, 2.2515),   # restrict adduction from -1.59 to -0.75 rad
    "right_shoulder_roll_joint": (-2.2515, 0.75),   # restrict adduction from  1.59 to  0.75 rad
}


def clamp_to_joint_limits(dof_pos: np.ndarray, dof_names: list, margin: float = POSITION_CLAMP_MARGIN) -> int:
    """Clamp joint positions to safe range (URDF limits with inward margin). Returns count of clamped values."""
    clamped = 0
    for i, name in enumerate(dof_names):
        if name in JOINT_POSITION_LIMITS:
            # Use self-collision-safe limits when available, else URDF limits
            lo, hi = SELF_COLLISION_SAFE_LIMITS.get(name, JOINT_POSITION_LIMITS[name])
            rng = hi - lo
            safe_lo = lo + rng * margin
            safe_hi = hi - rng * margin
            out_of_range = np.sum((dof_pos[:, i] < safe_lo) | (dof_pos[:, i] > safe_hi))
            clamped += out_of_range
            dof_pos[:, i] = np.clip(dof_pos[:, i], safe_lo, safe_hi)
    return clamped


def remove_self_collisions(dof_pos: np.ndarray, dof_names: list,
                           model_xml: str = str(_G1_XML_PATH)) -> int:
    """Remove self-collisions by blending frames toward neutral pose.

    For each frame that causes self-collision in MuJoCo, binary-search
    the minimum blend toward neutral (zero) that eliminates all contacts.
    Returns count of adjusted frames.
    """
    import mujoco
    import re
    import tempfile

    # Load XML and strip terrain/floor/groundplane (may reference missing mesh files).
    # Pass the model directory so MuJoCo can resolve relative mesh paths.
    model_dir = str(Path(model_xml).parent)
    xml_text = Path(model_xml).read_text()
    # Remove all non-robot elements (terrain, floor, groundplane) that may
    # reference missing mesh files. We only need the robot for collision checking.
    xml_text = '\n'.join(
        line for line in xml_text.split('\n')
        if not any(kw in line for kw in ['terrain_mesh', 'groundplane', 'name="floor"'])
    )
    xml_text = re.sub(r'<body\s+name="terrain_body"[^>]*>.*?</body>', '', xml_text, flags=re.DOTALL)

    # Rewrite meshdir to absolute path so from_xml_string can find STLs.
    xml_text = xml_text.replace('meshdir="meshes"', f'meshdir="{model_dir}/meshes"')
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)

    # Map dof names to qpos addresses
    joint_map = {}  # dof_index -> qpos_address
    for i, name in enumerate(dof_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joint_map[i] = model.jnt_qposadr[jid]

    # Robot body IDs (descendants of pelvis) for self-collision filtering
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_body_ids = set()
    for bid in range(model.nbody):
        parent = bid
        while parent > 0:
            if parent == pelvis_id:
                robot_body_ids.add(bid)
                break
            parent = model.body_parentid[parent]
    robot_body_ids.add(pelvis_id)

    neutral = np.zeros(len(dof_names))

    def _get_self_collisions() -> list:
        """Return list of colliding body-name pairs, or empty list if clean."""
        mujoco.mj_forward(model, data)
        pairs = []
        for ci in range(data.ncon):
            c = data.contact[ci]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if b1 in robot_body_ids and b2 in robot_body_ids:
                n1 = model.body(b1).name
                n2 = model.body(b2).name
                pairs.append((n1, n2))
        return pairs

    def _has_self_collision() -> bool:
        return len(_get_self_collisions()) > 0

    def _set_pose(pose: np.ndarray):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.793  # pelvis standing height
        for dof_i, qpos_addr in joint_map.items():
            data.qpos[qpos_addr] = pose[dof_i]

    T = len(dof_pos)
    collision_pairs_seen: dict = {}  # pair -> count

    def _run_pass(pass_name: str) -> int:
        nonlocal collision_pairs_seen
        last_safe = neutral.copy()
        fixed = 0
        for frame_idx in range(T):
            if frame_idx % 2000 == 0:
                print(f"  [{pass_name}] checking frame {frame_idx}/{T}...", flush=True)
            _set_pose(dof_pos[frame_idx])
            collisions = _get_self_collisions()
            if not collisions:
                last_safe = dof_pos[frame_idx].copy()
                continue

            for pair in collisions:
                key = tuple(sorted(pair))
                collision_pairs_seen[key] = collision_pairs_seen.get(key, 0) + 1

            lo, hi = 0.0, 1.0
            for _ in range(12):
                alpha = (lo + hi) / 2
                blended = dof_pos[frame_idx] * (1 - alpha) + last_safe * alpha
                _set_pose(blended)
                if _has_self_collision():
                    lo = alpha
                else:
                    hi = alpha

            dof_pos[frame_idx] = dof_pos[frame_idx] * (1 - hi) + last_safe * hi
            fixed += 1
        return fixed

    # Pass 1: fix collision frames by blending toward last safe frame
    adjusted = _run_pass("pass 1")
    print(f"  [pass 1] fixed {adjusted}/{T} frames")

    # Pass 2: smooth then re-check
    if adjusted > 0:
        dof_pos[:] = gaussian_filter1d(dof_pos, sigma=2.0, axis=0)
        print(f"  [pass 2] applied smoothing (sigma=2.0), re-checking...")
        refix = _run_pass("pass 2")
        print(f"  [pass 2] re-fixed {refix}/{T} frames after smoothing")

    # Report which body pairs had collisions
    if collision_pairs_seen:
        print(f"  collision pairs found:")
        for pair, count in sorted(collision_pairs_seen.items(), key=lambda x: -x[1]):
            print(f"    {pair[0]} ↔ {pair[1]}: {count} frames")

    return adjusted


COLLISION_LOG_PATH = "/tmp/gr00t_collision_log.jsonl"


def fix_from_collision_log(dof_pos: np.ndarray, fps: float, speed: float,
                           t0_wall: float, initial_pose_seconds: float,
                           log_path: str = COLLISION_LOG_PATH,
                           window: int = 15) -> int:
    """Fix trajectory frames that caused collisions during a sim run.

    Reads collision timestamps from the sim's log, maps them to trajectory
    frame indices, and smoothly blends those frames (± window) toward the
    nearest safe neighbors.

    Returns count of frames fixed.
    """
    import json

    if not Path(log_path).exists():
        print(f"[info] no collision log at {log_path} — no collisions occurred in pass 1")
        return 0

    # Read collision events
    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print("[info] collision log is empty — no collisions detected in pass 1")
        return 0

    # Map wall-clock times to frame indices
    # Motion starts at t0_wall + initial_pose_seconds
    t_motion_start = t0_wall + initial_pose_seconds
    T = len(dof_pos)

    collision_frames = set()
    for ev in events:
        t_wall = ev["t"]
        t_motion = (t_wall - t_motion_start) * speed
        if t_motion < 0:
            continue  # collision during initial settle — skip
        frame_idx = int(round(t_motion * fps))
        # Add frame ± window
        for f in range(max(0, frame_idx - window), min(T, frame_idx + window + 1)):
            collision_frames.add(f)

    if not collision_frames:
        print("[info] no collision frames mapped from log")
        return 0

    # Report collision pairs from log
    pair_counts: dict = {}
    for ev in events:
        for pair in ev.get("pairs", []):
            key = tuple(sorted(pair))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    print(f"[info] collision log: {len(events)} events, {len(collision_frames)} frames to fix (±{window} window)")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"    {pair[0]} ↔ {pair[1]}: {count} events")

    # Fix collision frames by blending toward nearest safe neighbor
    sorted_frames = sorted(collision_frames)
    fixed = 0
    for frame_idx in sorted_frames:
        # Find nearest non-collision frame before and after
        before = frame_idx - 1
        while before >= 0 and before in collision_frames:
            before -= 1
        after = frame_idx + 1
        while after < T and after in collision_frames:
            after += 1

        if before >= 0 and after < T:
            # Blend between before and after
            span = after - before
            alpha = (frame_idx - before) / span
            dof_pos[frame_idx] = (1 - alpha) * dof_pos[before] + alpha * dof_pos[after]
        elif before >= 0:
            dof_pos[frame_idx] = dof_pos[before].copy()
        elif after < T:
            dof_pos[frame_idx] = dof_pos[after].copy()
        fixed += 1

    # Smooth the fixed region to avoid discontinuities
    dof_pos[:] = gaussian_filter1d(dof_pos, sigma=3.0, axis=0)
    print(f"[info] fixed {fixed} frames, applied smoothing (sigma=3.0)")

    return fixed


def load_results(path: str) -> Dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "dof_pos" not in obj or "fps" not in obj:
        raise ValueError("results.pkl must be a dict with at least keys: 'fps' and 'dof_pos'")
    dof_pos = obj["dof_pos"]
    if not isinstance(dof_pos, np.ndarray) or dof_pos.ndim != 2:
        raise ValueError("results.pkl['dof_pos'] must be a 2D numpy array (T, ndof)")
    return obj


def load_dof_names_override(path: Optional[str]) -> Optional[List[str]]:
    if path is None:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "dof_names" in data:
        data = data["dof_names"]
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("dof names json must be list[str] or {'dof_names': list[str]}")
    return data


def lerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * a + w * b


def interp_dof_pos(dof_pos: np.ndarray, fps: float, t: float) -> np.ndarray:
    """Linear interpolation of dof_pos (T, ndof) sampled at fps to time t (seconds)."""
    T, _ = dof_pos.shape
    x = t * fps
    if x <= 0:
        return dof_pos[0].copy()
    if x >= (T - 1):
        return dof_pos[-1].copy()
    i0 = int(np.floor(x))
    i1 = i0 + 1
    w = float(x - i0)
    return lerp(dof_pos[i0], dof_pos[i1], w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.pkl containing fps and dof_pos")
    ap.add_argument("--dof-names-json", default=None, help="Optional JSON: list[str] or {'dof_names': list[str]}")
    ap.add_argument("--teleop-frequency", type=float, default=30.0, help="Publish frequency (Hz)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--loop", action="store_true", help="Loop the motion")
    ap.add_argument("--initial-pose-seconds", type=float, default=1.0, help="Longer target_time on first publish")
    ap.add_argument("--hand-mode", choices=["zero", "hold"], default="zero",
                    help="What to do with hand joints (not present in results). "
                         "'zero' sets them to 0.0, 'hold' keeps last commanded value.")
    ap.add_argument("--upper-body-only", action="store_true",
                    help="Only publish upper body poses, do NOT publish navigate_cmd or "
                         "base_height_command. This allows the control loop's RL policy "
                         "to handle leg locomotion independently.")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian smoothing sigma (in frames). Removes jitter from noisy "
                         "pose estimates. Try 2.0-5.0. 0 = no smoothing (default).")
    ap.add_argument("--collision-free", action="store_true",
                    help="Post-process trajectory to remove ALL self-collisions using "
                         "MuJoCo collision detection. Frames with self-contact are blended "
                         "toward neutral pose until collision-free.")
    ap.add_argument("--two-pass", action="store_true",
                    help="Two-pass collision removal. Pass 1: play trajectory once (no loop) "
                         "while the sim logs collisions to /tmp/gr00t_collision_log.jsonl. "
                         "Pass 2: fix the logged collision frames, then loop.")
    args = ap.parse_args()

    results = load_results(args.results)
    fps = float(results["fps"])
    dof_pos = results["dof_pos"]

    if args.smooth > 0:
        dof_pos = gaussian_filter1d(dof_pos, sigma=args.smooth, axis=0)
        print(f"[info] applied Gaussian smoothing (sigma={args.smooth:.1f} frames)")

    # Need dof_names before clamping
    src_dof_names_early = None
    if isinstance(results.get("dof_names", None), list) and len(results["dof_names"]) == dof_pos.shape[1]:
        src_dof_names_early = list(results["dof_names"])
    elif dof_pos.shape[1] == len(DEFAULT_DOF_NAMES_29):
        src_dof_names_early = DEFAULT_DOF_NAMES_29

    if src_dof_names_early:
        clamped = clamp_to_joint_limits(dof_pos, src_dof_names_early)
        if clamped > 0:
            print(f"[info] clamped {clamped} out-of-range joint values to 95% of URDF limits")
    else:
        print("[warn] could not resolve dof_names for clamping — skipping position clamp")

    if args.collision_free and src_dof_names_early:
        # Upsample 4x before collision check so interpolated midpoints are also validated.
        # Without this, linear interpolation between safe keyframes can pass through
        # collision space (e.g., hands crossing paths between two safe poses).
        from scipy.interpolate import interp1d
        T_orig = len(dof_pos)
        upsample = 8
        x_orig = np.linspace(0, 1, T_orig)
        x_up = np.linspace(0, 1, T_orig * upsample)
        dof_pos = interp1d(x_orig, dof_pos, axis=0, kind='linear')(x_up)
        fps = fps * upsample
        print(f"[info] upsampled {T_orig} → {len(dof_pos)} frames for collision checking")

        print("[info] running collision-free post-processing (MuJoCo)...")
        adjusted = remove_self_collisions(dof_pos, src_dof_names_early)
        if adjusted > 0:
            print(f"[info] adjusted {adjusted}/{len(dof_pos)} frames to remove self-collisions")
        else:
            print("[info] no self-collisions found — trajectory is clean")

    T, ndof = dof_pos.shape

    src_dof_names = load_dof_names_override(args.dof_names_json)
    if src_dof_names is None:
        if isinstance(results.get("dof_names", None), list) and len(results["dof_names"]) == ndof:
            src_dof_names = list(results["dof_names"])
        else:
            if ndof != len(DEFAULT_DOF_NAMES_29):
                raise ValueError(
                    f"ndof={ndof} but no dof_names provided and DEFAULT_DOF_NAMES_29 is length 29. "
                    "Provide --dof-names-json or store dof_names in results.pkl."
                )
            src_dof_names = DEFAULT_DOF_NAMES_29

    name_to_col = {n: i for i, n in enumerate(src_dof_names)}

    duration = (T - 1) / fps
    print(f"[info] loaded results: T={T}, ndof={ndof}, fps={fps}")
    print(f"[info] motion duration ~ {duration:.2f}s @ {fps}fps, speed={args.speed}x, loop={args.loop}")
    print(f"[info] publishing to {CONTROL_GOAL_TOPIC} @ {args.teleop_frequency} Hz")
    print("[info] target_upper_body_pose dim = 28 (matches control loop)")

    # Track missing joints (hands) once
    missing = [jn for jn in UPPER_BODY_NAMES_28 if jn not in name_to_col]
    if missing:
        print(f"[warn] {len(missing)} upper_body joints missing from results (expected for hands). "
              f"Example: {missing[:10]}")
        if args.hand_mode == "zero":
            print("[info] hand-mode=zero -> missing joints set to 0.0")
        else:
            print("[info] hand-mode=hold -> missing joints keep last commanded value")

    ros_manager = ROSManager(node_name="VideoImitationPublisher")
    node = ros_manager.node
    pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    rate = node.create_rate(args.teleop_frequency)

    def _play_once(label: str, do_loop: bool) -> float:
        """Play through the trajectory. Returns wall-clock start time (time.time())."""
        last_upper = np.zeros((28,), dtype=float)
        iteration = 0
        loop_count = 0
        last_progress = -1
        # time.time() for cross-process correlation with collision log
        t0_wall_clock = time.time()
        # time.monotonic() for ROS message timestamps (control loop expects monotonic)
        t0_mono = time.monotonic()

        try:
            while ros_manager.ok():
                t_mono = time.monotonic()
                t_motion = (t_mono - t0_mono) * float(args.speed)

                if t_motion >= duration:
                    if do_loop:
                        t0_mono = t_mono
                        t_motion = 0.0
                        iteration = 0
                        loop_count += 1
                        print(f"[{label}] loop {loop_count} complete, restarting...", flush=True)
                    else:
                        print(f"[{label}] reached end of motion")
                        break

                # Progress every 10%
                pct = int(t_motion / duration * 100) // 10 * 10
                if pct != last_progress and pct > 0:
                    last_progress = pct
                    frame = int(t_motion * fps)
                    from datetime import datetime
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] [{label}] {pct}% — {t_motion:.1f}s / {duration:.1f}s — frame {frame}/{T}", flush=True)

                q_src = interp_dof_pos(dof_pos, fps=fps, t=t_motion)

                q_upper = np.empty((28,), dtype=float)
                for i, jn in enumerate(UPPER_BODY_NAMES_28):
                    if jn in name_to_col:
                        q_upper[i] = float(q_src[name_to_col[jn]])
                    else:
                        q_upper[i] = 0.0 if args.hand_mode == "zero" else float(last_upper[i])

                last_upper = q_upper.copy()

                if iteration == 0:
                    target_time = t_mono + float(args.initial_pose_seconds)
                else:
                    target_time = t_mono + (1.0 / float(args.teleop_frequency))

                msg = {
                    "timestamp": t_mono,
                    "target_time": target_time,
                    "target_upper_body_pose": q_upper,
                }

                if not args.upper_body_only:
                    msg["navigate_cmd"] = np.asarray(DEFAULT_NAV_CMD, dtype=float)
                    msg["base_height_command"] = float(DEFAULT_BASE_HEIGHT)

                pub.publish(msg)

                if iteration == 0:
                    mode_str = "upper-body-only" if args.upper_body_only else "full"
                    print(f"[info] {label}: sent initial waypoint ({mode_str}). "
                          f"settle {args.initial_pose_seconds:.2f}s")
                    time.sleep(float(args.initial_pose_seconds))

                iteration += 1
                rate.sleep()

        except ros_manager.exceptions() as e:
            print(f"[info] {label}: interrupted: {e}")

        return t0_wall_clock

    try:
        if args.two_pass:
            log_path = COLLISION_LOG_PATH
            # Don't delete or truncate — sim may have the file open.
            # We filter by timestamp in fix_from_collision_log instead.

            # Pass 1: play once, let sim record collisions
            print("=" * 50)
            print("[PASS 1] Playing trajectory once — sim is recording collisions...")
            print(f"[PASS 1] Press ] in Terminal 1 to activate, wait 5s, press 9")
            print("=" * 50)
            t0 = _play_once("pass 1", do_loop=False)

            # Read collision log and fix
            print("=" * 50)
            print("[FIX] Reading collision log and fixing trajectory...")
            print("=" * 50)
            fixed = fix_from_collision_log(
                dof_pos, fps=fps, speed=float(args.speed),
                t0_wall=t0, initial_pose_seconds=float(args.initial_pose_seconds),
            )
            if fixed > 0:
                # Re-clamp after smoothing
                if src_dof_names_early:
                    clamp_to_joint_limits(dof_pos, src_dof_names_early)

            # Pass 2: loop with fixed trajectory
            print("=" * 50)
            print(f"[PASS 2] Playing fixed trajectory (loop={args.loop})...")
            print("=" * 50)
            _play_once("pass 2", do_loop=args.loop)
        else:
            _play_once("play", do_loop=args.loop)

    except ros_manager.exceptions() as e:
        print(f"[info] ROSManager interrupted: {e}")
    finally:
        print("[info] shutting down")
        ros_manager.shutdown()


if __name__ == "__main__":
    main()