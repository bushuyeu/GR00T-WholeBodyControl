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
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

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
    args = ap.parse_args()

    results = load_results(args.results)
    fps = float(results["fps"])
    dof_pos = results["dof_pos"]

    if args.smooth > 0:
        dof_pos = gaussian_filter1d(dof_pos, sigma=args.smooth, axis=0)
        print(f"[info] applied Gaussian smoothing (sigma={args.smooth:.1f} frames)")

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

    # State for hold mode
    last_upper = np.zeros((28,), dtype=float)

    iteration = 0
    t0_wall = time.monotonic()

    try:
        while ros_manager.ok():
            t_now = time.monotonic()
            t_motion = (t_now - t0_wall) * float(args.speed)

            if t_motion >= duration:
                if args.loop:
                    t0_wall = t_now
                    t_motion = 0.0
                    iteration = 0
                else:
                    print("[info] reached end of motion; stopping")
                    break

            q_src = interp_dof_pos(dof_pos, fps=fps, t=t_motion)

            # Build q_upper in EXACT expected order
            q_upper = np.empty((28,), dtype=float)
            for i, jn in enumerate(UPPER_BODY_NAMES_28):
                if jn in name_to_col:
                    q_upper[i] = float(q_src[name_to_col[jn]])
                else:
                    q_upper[i] = 0.0 if args.hand_mode == "zero" else float(last_upper[i])

            last_upper = q_upper.copy()

            # target_time: longer on first send so interpolation_policy has time to move safely
            if iteration == 0:
                target_time = t_now + float(args.initial_pose_seconds)
            else:
                target_time = t_now + (1.0 / float(args.teleop_frequency))

            msg = {
                "timestamp": t_now,
                "target_time": target_time,
                "target_upper_body_pose": q_upper,                         # (28,)
            }

            # Only include lower body commands if not in upper-body-only mode
            if not args.upper_body_only:
                msg["navigate_cmd"] = np.asarray(DEFAULT_NAV_CMD, dtype=float)  # (3,)
                msg["base_height_command"] = float(DEFAULT_BASE_HEIGHT)         # scalar

            pub.publish(msg)

            if iteration == 0:
                mode_str = "upper-body-only" if args.upper_body_only else "full (upper=28, nav=3, base=1)"
                print(f"[info] sent initial waypoint ({mode_str}). "
                      f"settle {args.initial_pose_seconds:.2f}s")
                time.sleep(float(args.initial_pose_seconds))

            iteration += 1
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"[info] ROSManager interrupted: {e}")
    finally:
        print("[info] shutting down")
        ros_manager.shutdown()


if __name__ == "__main__":
    main()