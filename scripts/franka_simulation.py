"""
PyBullet simulation for the confined-workspace Franka Panda experiment.

The module exposes `run_simulation()`, which executes one parameterized
pick-and-place trial, records robot and contact states, computes the
AABB-based grid-invasion indicator (`invaded_grid_count`), and returns a
trial-level summary. The batch generator imports this module to reproduce the
layout-by-trial dataset used in the paper.
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import csv
import time
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in {"scripts", "src"} else SCRIPT_DIR
OUTPUT_DIR = Path(
    os.environ.get(
        "FRANKA_SIMULATION_OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs" / "single_trial"),
    )
).expanduser().resolve()


def output_path(*parts):
    """Return an output path under FRANKA_SIMULATION_OUTPUT_DIR."""
    return str(OUTPUT_DIR.joinpath(*parts))


def ensure_parent_dir(path):
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ============================================================
# Runtime configuration
# ============================================================

HEADLESS = False

# Risk-grid settings
MIN_CLEARANCE_MM  = 120.0
GRID_SPACING_MM   = 50.0

# AABB invasion check settings
AABB_INFLATION_M  = 0.005
USE_PRECISE_PASS  = True

# Reachability filter
USE_REACHABLE_FILTER = True
ROBOT_BASE_POSITION  = [0.0, 0.0, 0.2]
REACHABLE_RADIUS_M   = 0.85
REACHABLE_Z_MIN      = 0.05
REACHABLE_Z_MAX      = 1.4

# Step-level lookahead label
WILL_INVADE_LOOKAHEAD_STEPS = 60

# Any robot-structure contact is recorded as a trial-level collision.

# Layout image settings
LAYOUT_IMAGE_SIZE   = 128
LAYOUT_IMAGE_DIR    = output_path("layout_images")
LAYOUT_WORLD_X_MIN  = -0.7
LAYOUT_WORLD_X_MAX  =  1.2
LAYOUT_WORLD_Y_MIN  = -1.0
LAYOUT_WORLD_Y_MAX  =  1.0

COLOR_FENCE         = [255,   0,   0]
COLOR_PILLAR        = [  0,   0, 255]
COLOR_WORKSPACE     = [  0, 200,   0]
COLOR_TRAY          = [200, 200,   0]
COLOR_ROBOT_BASE    = [128, 128, 128]
COLOR_PICK          = [255, 128,   0]
COLOR_PLACE         = [128,   0, 255]

# Output paths for a single standalone run.
INVASION_LOG_PATH    = output_path("grid_invasion_log.csv")
TIMESERIES_LOG_PATH  = output_path("grid_invasion_timeseries.csv")
SUMMARY_TXT_PATH     = output_path("grid_invasion_summary.txt")
STEP_DATA_PATH       = output_path("step_data.csv")
TRIAL_SUMMARY_PATH   = output_path("trial_summary.csv")


# ============================================================
# Scene geometry
# ============================================================
DEFAULT_FENCE_X         = 0.5
FENCE_HEIGHT            = 1.2
FENCE_OPENING_HALF      = 0.27
FENCE_TOTAL_LENGTH      = 1.7
FENCE_THICKNESS         = 0.03
FENCE_COLOR             = [1.0, 0.3, 0.3, 1.0]

DEFAULT_PILLAR_X        = -0.5
DEFAULT_PILLAR_Y        = -0.3
CYL_RADIUS              = 0.1
CYL_HEIGHT              = 1.2
CYL_COLOR               = [0.6, 0.3, 0.8, 1.0]

WS1_X                   = 0.55
WS1_Y                   = 0.0
WS1_SIZE                = [0.5, 0.5, 0.3]
WS1_COLOR               = [0.3, 0.7, 0.3, 1.0]

WS2_POSITION            = [0.05, -0.6, 0.0]
WS2_SIZE                = [0.65, 0.50, 0.30]
WS2_COLOR               = [0.7, 0.5, 0.2, 1.0]

CUBE_SIZE               = 0.02
CUBE_COLOR              = [0.2, 0.2, 0.9, 1.0]

TRAY1_INNER_SIZE        = [0.25, 0.4]
TRAY1_WALL_HEIGHT       = 0.05
TRAY1_WALL_THICKNESS    = 0.015
TRAY1_COLOR             = [0.4, 0.4, 0.45, 1.0]

TRAY2_INNER_SIZE        = [0.4, 0.40]
TRAY2_WALL_HEIGHT       = 0.05
TRAY2_WALL_THICKNESS    = 0.015
TRAY2_COLOR             = [0.4, 0.4, 0.45, 1.0]

BASE_HEIGHT             = 0.2
BASE_SIZE               = [0.3, 0.3, BASE_HEIGHT]
BASE_COLOR              = [0.5, 0.5, 0.5, 1.0]

SAFE_Z                  = 0.65
PLACE_Z_LOWER           = 0.42


CYL_COLOR   = [0.50, 0.50, 0.48, 1.0]
WS1_COLOR   = [0.42, 0.44, 0.46, 1.0]
WS2_COLOR   = [0.42, 0.44, 0.46, 1.0]
TRAY1_COLOR = [0.18, 0.20, 0.22, 1.0]
TRAY2_COLOR = [0.18, 0.20, 0.22, 1.0]
BASE_COLOR  = [0.28, 0.28, 0.28, 1.0]


# ============================================================
# Minimum-jerk trajectory
# ============================================================
class MinimumJerkTrajectory:
    def __init__(self, dt=1.0/240.0):
        self.dt = dt

    def generate(self, q_start, q_end, n_steps):
        T   = n_steps * self.dt
        tau = np.linspace(0.0, 1.0, n_steps)
        p_c =  10*tau**3 - 15*tau**4 +   6*tau**5
        v_c = (30*tau**2 - 60*tau**3 +  30*tau**4) / T
        a_c = (60*tau    - 180*tau**2 + 120*tau**3) / (T**2)
        dq  = q_end - q_start
        return (q_start + np.outer(p_c, dq),
                np.outer(v_c, dq),
                np.outer(a_c, dq))


# ============================================================
# Piecewise minimum-jerk trajectory
# ============================================================
class PiecewiseMinJerk:
    def __init__(self, dt=1.0/240.0):
        self.dt = dt

    def _quintic_coeff(self, q0, q1, v0, v1, T):
        a0 = q0
        a1 = v0
        a2 = np.zeros_like(q0)
        a3 = (20*(q1-q0) - (8*v1 + 12*v0)*T) / (2*T**3)
        a4 = (30*(q0-q1) + (14*v1 + 16*v0)*T) / (2*T**4)
        a5 = (12*(q1-q0) - 6*(v1+v0)*T) / (2*T**5)
        return a0, a1, a2, a3, a4, a5

    def generate(self, waypoints, durations):
        assert len(waypoints) >= 2
        assert len(durations) == len(waypoints) - 1
        n_seg = len(durations)
        via_vels = [np.zeros(7)]
        for i in range(1, n_seg):
            T_prev = durations[i-1] * self.dt
            T_next = durations[i]   * self.dt
            v_est  = 0.5 * ((waypoints[i]   - waypoints[i-1]) / T_prev +
                            (waypoints[i+1] - waypoints[i])   / T_next)
            via_vels.append(v_est)
        via_vels.append(np.zeros(7))

        all_pos, all_vel, all_acc = [], [], []
        for seg in range(n_seg):
            q0 = waypoints[seg]
            q1 = waypoints[seg+1]
            v0 = via_vels[seg]
            v1 = via_vels[seg+1]
            T  = durations[seg] * self.dt
            n  = durations[seg]
            a0, a1, a2, a3, a4, a5 = self._quintic_coeff(q0, q1, v0, v1, T)
            ts = np.linspace(0, T, n, endpoint=(seg == n_seg-1))
            for t in ts:
                pos = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
                vel = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
                acc = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
                all_pos.append(pos)
                all_vel.append(vel)
                all_acc.append(acc)
        return np.array(all_pos), np.array(all_vel), np.array(all_acc)


# ============================================================
# Step-level logging
# ============================================================
class StepDataLogger:
    JOINT_LL = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    JOINT_UL = np.array([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973])

    def __init__(self, robot_id, ee_idx, n_joints, structure_ids, dt=1.0/240.0):
        self.robot_id = robot_id
        self.ee_idx = ee_idx
        self.n_joints = n_joints
        self.structure_ids = structure_ids
        self.dt = dt
        self.records = []
        self._step = 0
        self._current_phase = "init"

    def set_phase(self, phase_name):
        self._current_phase = phase_name

    def _calc_manipulability(self, joint_angles):
        zero_vec = [0.0] * self.n_joints
        full_q = list(joint_angles) + [0.04, 0.04]
        full_zero = zero_vec + [0.0, 0.0]
        try:
            jac_t, jac_r = p.calculateJacobian(
                self.robot_id, self.ee_idx,
                [0, 0, 0], full_q, full_zero, full_zero)
            jac_t = np.array(jac_t)[:, :self.n_joints]
            jac_r = np.array(jac_r)[:, :self.n_joints]
            J = np.vstack([jac_t, jac_r])
            JJT = J @ J.T
            det_val = np.linalg.det(JJT)
            return float(np.sqrt(max(det_val, 0.0)))
        except Exception:
            return 0.0

    def _calc_joint_margins(self, joint_angles):
        margins = []
        for i, q in enumerate(joint_angles):
            ll = self.JOINT_LL[i]
            ul = self.JOINT_UL[i]
            dist_to_limit = min(q - ll, ul - q)
            half_range = (ul - ll) / 2.0
            margin = dist_to_limit / half_range
            margins.append(float(np.clip(margin, 0.0, 1.0)))
        return margins

    def _calc_distances_to_structures(self):
        distances = {}
        global_min = float('inf')
        global_min_obs = "none"
        global_min_link = -1

        for name, ids in self.structure_ids.items():
            id_list = ids if isinstance(ids, list) else [ids]
            min_dist = float('inf')
            for obs_id in id_list:
                pts = p.getClosestPoints(
                    bodyA=self.robot_id, bodyB=obs_id, distance=2.0)
                for pt in pts:
                    d = pt[8]
                    if d < min_dist:
                        min_dist = d
                    if d < global_min:
                        global_min = d
                        global_min_obs = name
                        global_min_link = pt[3]
            distances[name] = min_dist if min_dist < float('inf') else 2.0
        return distances, global_min, global_min_obs, global_min_link

    def _calc_collisions(self):
        """Check robot contact with all static structures at the current step."""
        total_force = 0.0
        total_contacts = 0
        colliding_set = set()

        for name, ids in self.structure_ids.items():
            id_list = ids if isinstance(ids, list) else [ids]
            for obs_id in id_list:
                contacts = p.getContactPoints(
                    bodyA=self.robot_id, bodyB=obs_id)
                if contacts:
                    for c in contacts:
                        # PyBullet contact tuple index 9 is normal force.
                        total_force += c[9]
                        total_contacts += 1
                    colliding_set.add(name)

        collision_occurred = 1 if total_contacts > 0 else 0
        colliding_str = ",".join(sorted(colliding_set)) if colliding_set else "none"

        return collision_occurred, total_force, total_contacts, colliding_str

    def _calc_joint_torques(self):
        """Return the maximum absolute applied motor torque and all joint torques."""
        states = p.getJointStates(self.robot_id, range(self.n_joints))
        # state[3] = applied joint motor torque
        torques = np.array([abs(s[3]) for s in states])
        return float(np.max(torques)), torques.tolist()

    def record(self):
        states = p.getJointStates(self.robot_id, range(self.n_joints))
        joint_angles = np.array([s[0] for s in states])
        joint_vels   = np.array([s[1] for s in states])

        ee_state = p.getLinkState(self.robot_id, self.ee_idx)
        ee_pos = ee_state[4]
        ee_orn = ee_state[5]

        manip = self._calc_manipulability(joint_angles)
        margins = self._calc_joint_margins(joint_angles)
        distances, min_d, min_obs, min_link = self._calc_distances_to_structures()

        coll_occurred, coll_force, coll_count, coll_structures = \
            self._calc_collisions()
        max_torque, all_torques = self._calc_joint_torques()

        in_hazard = 1 if min_d <= MIN_CLEARANCE_MM / 1000.0 else 0
        intrusion_depth_mm = max(0.0, MIN_CLEARANCE_MM - min_d * 1000.0)

        record = {
            'step': self._step,
            'time_s': self._step * self.dt,
            'phase': self._current_phase,
            **{f'joint_{i+1}': float(joint_angles[i]) for i in range(self.n_joints)},
            **{f'joint_vel_{i+1}': float(joint_vels[i]) for i in range(self.n_joints)},
            'ee_x': float(ee_pos[0]),
            'ee_y': float(ee_pos[1]),
            'ee_z': float(ee_pos[2]),
            'ee_qx': float(ee_orn[0]),
            'ee_qy': float(ee_orn[1]),
            'ee_qz': float(ee_orn[2]),
            'ee_qw': float(ee_orn[3]),
            'manipulability': manip,
            **{f'joint_margin_{i+1}': margins[i] for i in range(self.n_joints)},
            'min_joint_margin': float(min(margins)),
            **{f'dist_{name}': float(distances[name]) for name in distances},
            'min_distance_overall': float(min_d) if min_d < float('inf') else 2.0,
            'closest_obstacle': min_obs,
            'closest_link_id': int(min_link),
            'in_hazard_now': in_hazard,
            'intrusion_depth_mm': intrusion_depth_mm,
            'will_invade_in_next_N_steps': 0,
            'collision_occurred':     coll_occurred,
            'collision_force_total':  coll_force,
            'collision_count':        coll_count,
            'colliding_structures':   coll_structures,
            'max_joint_torque':       max_torque,
        }
        self.records.append(record)
        self._step += 1

    def post_process_will_invade(self, lookahead_steps=WILL_INVADE_LOOKAHEAD_STEPS):
        n = len(self.records)
        in_hazard_arr = np.array([r['in_hazard_now'] for r in self.records])
        for i in range(n):
            end = min(i + lookahead_steps, n)
            future_hazard = in_hazard_arr[i+1:end].max() if end > i+1 else 0
            self.records[i]['will_invade_in_next_N_steps'] = int(future_hazard)

    def save_csv(self, path, layout_id=0, trial_id=0, append=False):
        if not self.records:
            return
        for r in self.records:
            r['layout_id'] = layout_id
            r['trial_id'] = trial_id
        keys = list(self.records[0].keys())
        for k in ['trial_id', 'layout_id']:
            if k in keys:
                keys.remove(k)
                keys.insert(0, k)
        mode = 'a' if append and os.path.exists(path) else 'w'
        ensure_parent_dir(path)
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if mode == 'w':
                writer.writeheader()
            writer.writerows(self.records)


# ============================================================
# Layout image generation
# ============================================================
class LayoutImageGenerator:
    def __init__(self, image_size=LAYOUT_IMAGE_SIZE,
                 x_min=LAYOUT_WORLD_X_MIN, x_max=LAYOUT_WORLD_X_MAX,
                 y_min=LAYOUT_WORLD_Y_MIN, y_max=LAYOUT_WORLD_Y_MAX):
        self.size = image_size
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

    def world_to_pixel(self, x, y):
        px = int((x - self.x_min) / (self.x_max - self.x_min) * (self.size - 1))
        py = int((y - self.y_min) / (self.y_max - self.y_min) * (self.size - 1))
        py = self.size - 1 - py
        return px, py

    def draw_box_top_view(self, center_x, center_y, size_x, size_y, color):
        x_min = center_x - size_x / 2
        x_max = center_x + size_x / 2
        y_min = center_y - size_y / 2
        y_max = center_y + size_y / 2
        px_min, py_max = self.world_to_pixel(x_min, y_min)
        px_max, py_min = self.world_to_pixel(x_max, y_max)
        px_min = max(0, min(self.size - 1, px_min))
        px_max = max(0, min(self.size - 1, px_max))
        py_min = max(0, min(self.size - 1, py_min))
        py_max = max(0, min(self.size - 1, py_max))
        self.image[py_min:py_max+1, px_min:px_max+1] = color

    def draw_circle_top_view(self, center_x, center_y, radius, color):
        cx_px, cy_px = self.world_to_pixel(center_x, center_y)
        scale_x = (self.size - 1) / (self.x_max - self.x_min)
        radius_px = int(radius * scale_x)
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx*dx + dy*dy <= radius_px*radius_px:
                    px = cx_px + dx
                    py = cy_px + dy
                    if 0 <= px < self.size and 0 <= py < self.size:
                        self.image[py, px] = color

    def draw_marker(self, x, y, color, size=2):
        cx_px, cy_px = self.world_to_pixel(x, y)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                px = cx_px + dx
                py = cy_px + dy
                if 0 <= px < self.size and 0 <= py < self.size:
                    self.image[py, px] = color

    def generate(self, layout_params, trial_params=None):
        fence_x  = layout_params.get('fence_x',  DEFAULT_FENCE_X)
        pillar_x = layout_params.get('pillar_x', DEFAULT_PILLAR_X)
        pillar_y = layout_params.get('pillar_y', DEFAULT_PILLAR_Y)

        fence_segment_length = (FENCE_TOTAL_LENGTH - 2 * FENCE_OPENING_HALF) / 2
        fence_left_y = -FENCE_OPENING_HALF - fence_segment_length / 2
        fence_right_y = FENCE_OPENING_HALF + fence_segment_length / 2

        self.draw_box_top_view(fence_x, fence_left_y,
                                FENCE_THICKNESS, fence_segment_length,
                                COLOR_FENCE)
        self.draw_box_top_view(fence_x, fence_right_y,
                                FENCE_THICKNESS, fence_segment_length,
                                COLOR_FENCE)
        self.draw_box_top_view(WS1_X, WS1_Y, WS1_SIZE[0], WS1_SIZE[1],
                                COLOR_WORKSPACE)
        self.draw_box_top_view(WS2_POSITION[0], WS2_POSITION[1],
                                WS2_SIZE[0], WS2_SIZE[1], COLOR_WORKSPACE)
        self.draw_box_top_view(WS1_X, WS1_Y,
                                TRAY1_INNER_SIZE[0], TRAY1_INNER_SIZE[1],
                                COLOR_TRAY)
        self.draw_box_top_view(WS2_POSITION[0], WS2_POSITION[1],
                                TRAY2_INNER_SIZE[0], TRAY2_INNER_SIZE[1],
                                COLOR_TRAY)
        self.draw_circle_top_view(pillar_x, pillar_y, CYL_RADIUS, COLOR_PILLAR)
        self.draw_box_top_view(ROBOT_BASE_POSITION[0], ROBOT_BASE_POSITION[1],
                                0.1, 0.1, COLOR_ROBOT_BASE)

        if trial_params is not None:
            pick_x = WS1_X + trial_params.get('pick_x_offset', 0.0)
            pick_y = WS1_Y + trial_params.get('pick_y_offset', 0.0)
            place_x = WS2_POSITION[0] + trial_params.get('place_x_offset', 0.0)
            place_y = WS2_POSITION[1] + trial_params.get('place_y_offset', 0.0)
            self.draw_marker(pick_x, pick_y, COLOR_PICK, size=2)
            self.draw_marker(place_x, place_y, COLOR_PLACE, size=2)
        return self.image

    def save(self, path):
        path = os.fspath(path)
        ensure_parent_dir(path)
        if Path(path).suffix.lower() == ".npy":
            np.save(path, self.image)
            return True
        try:
            from PIL import Image
            Image.fromarray(self.image).save(path)
            return True
        except ImportError:
            np.save(path.replace('.png', '.npy'), self.image)
            return False


def save_xy_safety_margin_figure(layout_params, trial_params, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Patch
    from matplotlib.lines import Line2D

    c = MIN_CLEARANCE_MM / 1000.0

    fence_x = layout_params.get('fence_x', DEFAULT_FENCE_X)
    pillar_x = layout_params.get('pillar_x', DEFAULT_PILLAR_X)
    pillar_y = layout_params.get('pillar_y', DEFAULT_PILLAR_Y)

    pick_x = WS1_X + trial_params.get('pick_x_offset', 0.0)
    pick_y = WS1_Y + trial_params.get('pick_y_offset', 0.0)
    place_x = WS2_POSITION[0] + trial_params.get('place_x_offset', 0.0)
    place_y = WS2_POSITION[1] + trial_params.get('place_y_offset', 0.0)

    margin_color = "#F2D58A"  # muted safety yellow
    fence_color  = "#D96A4A"  # muted orange-red
    pillar_color = "#6B72D6"  # softened blue
    table_color  = "#8BCB88"  # soft green
    tray_color   = "#D8C95A"  # muted yellow
    base_color   = "#8F9298"  # neutral gray

    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=300)

    def draw_box(cx, cy, sx, sy, color, label=None, z=3, alpha=0.95, lw=0.8):
        ax.add_patch(Rectangle(
            (cx - sx / 2, cy - sy / 2), sx, sy,
            facecolor=color,
            edgecolor='black',
            linewidth=lw,
            alpha=alpha,
            zorder=z,
            label=label
        ))

    def draw_box_margin(cx, cy, sx, sy):
        ax.add_patch(Rectangle(
            (cx - sx / 2 - c, cy - sy / 2 - c),
            sx + 2 * c, sy + 2 * c,
            facecolor=margin_color, edgecolor='#B8A600',
            linewidth=0.8, alpha=0.28, zorder=1
        ))

    def draw_circle(cx, cy, r, color, label=None, z=3):
        ax.add_patch(Circle(
            (cx, cy), r,
            facecolor=color, edgecolor='black', linewidth=0.8,
            alpha=0.95, zorder=z, label=label
        ))

    def draw_circle_margin(cx, cy, r):
        ax.add_patch(Circle(
            (cx, cy), r + c,
            facecolor=margin_color, edgecolor='#B8A600',
            linewidth=0.8, alpha=0.28, zorder=1
        ))

    fence_segment_length = (FENCE_TOTAL_LENGTH - 2 * FENCE_OPENING_HALF) / 2
    fence_left_y = -FENCE_OPENING_HALF - fence_segment_length / 2
    fence_right_y = FENCE_OPENING_HALF + fence_segment_length / 2

    # Safety-margin drawing helpers are kept for optional explanatory figures.
    # for cy in [fence_left_y, fence_right_y]:
    #     draw_box_margin(fence_x, cy, FENCE_THICKNESS, fence_segment_length)

    # draw_box_margin(WS1_X, WS1_Y, WS1_SIZE[0], WS1_SIZE[1])
    # draw_box_margin(WS2_POSITION[0], WS2_POSITION[1], WS2_SIZE[0], WS2_SIZE[1])

    tray1_outer = [TRAY1_INNER_SIZE[0] + 2 * TRAY1_WALL_THICKNESS,
                   TRAY1_INNER_SIZE[1] + 2 * TRAY1_WALL_THICKNESS]
    tray2_outer = [TRAY2_INNER_SIZE[0] + 2 * TRAY2_WALL_THICKNESS,
                   TRAY2_INNER_SIZE[1] + 2 * TRAY2_WALL_THICKNESS]
    # draw_box_margin(WS1_X, WS1_Y, tray1_outer[0], tray1_outer[1])
    # draw_box_margin(WS2_POSITION[0], WS2_POSITION[1], tray2_outer[0], tray2_outer[1])
    # draw_circle_margin(pillar_x, pillar_y, CYL_RADIUS)

    # Actual structures
    draw_box(fence_x, fence_left_y, FENCE_THICKNESS, fence_segment_length, fence_color)
    draw_box(fence_x, fence_right_y, FENCE_THICKNESS, fence_segment_length, fence_color)
    draw_box(WS1_X, WS1_Y, WS1_SIZE[0], WS1_SIZE[1], table_color)
    draw_box(WS2_POSITION[0], WS2_POSITION[1], WS2_SIZE[0], WS2_SIZE[1], table_color)
    draw_box(WS1_X, WS1_Y, tray1_outer[0], tray1_outer[1], tray_color)
    draw_box(WS2_POSITION[0], WS2_POSITION[1], tray2_outer[0], tray2_outer[1], tray_color)
    draw_circle(pillar_x, pillar_y, CYL_RADIUS, pillar_color)
    draw_box(ROBOT_BASE_POSITION[0], ROBOT_BASE_POSITION[1], 0.10, 0.10, base_color)

    ax.scatter(pick_x, pick_y, s=70, c="#E89B2E", marker='o', edgecolors='black',
               linewidths=0.7, zorder=5, label='Pick')
    ax.scatter(place_x, place_y, s=80, c="#A76AD8", marker='X', edgecolors='black',
               linewidths=0.7, zorder=5, label='Place')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(LAYOUT_WORLD_X_MIN, LAYOUT_WORLD_X_MAX)
    ax.set_ylim(LAYOUT_WORLD_Y_MIN, LAYOUT_WORLD_Y_MAX)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, color='0.86', linewidth=0.7)
    ax.set_title('Top-view safety-margin regions')

    legend_items = [
        Patch(facecolor=fence_color, edgecolor='black', label='Safety fence'),
        Patch(facecolor=table_color, edgecolor='black', label='Workspace'),
        Patch(facecolor=tray_color, edgecolor='black', label='Tray'),
        Patch(facecolor=pillar_color, edgecolor='black', label='Pillar'),
        Patch(facecolor=base_color, edgecolor='black', label='Robot base'),
        Line2D([0], [0], marker='o', color='w', label='Pick',
               markerfacecolor='#FCA607', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Place',
               markerfacecolor='#C55EFD', markeredgecolor='black', markersize=8),
    ]
    ax.legend(
        handles=legend_items,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8,
    )

    ensure_parent_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Grid-invasion tracking
# ============================================================
class HazardGridTracker:
    def __init__(self, robot_id, clearance_mm, spacing_mm, dt=1.0/240.0,
                 verbose=True):
        self.robot_id = robot_id
        self.clearance_m = clearance_mm / 1000.0
        self.spacing_m   = spacing_mm / 1000.0
        self.dt = dt
        self.verbose = verbose
        self.grid_points = []
        self._grid_xyz   = None
        self.aabb_history = []
        self._step = 0
        self._current_phase = "init"
        self.invasion_records = []
        self.invaded_flags = None

    def set_phase(self, phase_name):
        self._current_phase = phase_name

    def _generate_box_grid(self, size, position, name):
        sx, sy, sz = size
        cx, cy, cz = position
        c = self.clearance_m
        s = self.spacing_m
        x_min, x_max = cx - sx/2 - c, cx + sx/2 + c
        y_min, y_max = cy - sy/2 - c, cy + sy/2 + c
        z_min, z_max = cz - sz/2 - c, cz + sz/2 + c
        z_min = max(z_min, 0.005)
        ix_min, ix_max = cx - sx/2, cx + sx/2
        iy_min, iy_max = cy - sy/2, cy + sy/2
        iz_min, iz_max = cz - sz/2, cz + sz/2
        xs = np.arange(x_min, x_max + s/2, s)
        ys = np.arange(y_min, y_max + s/2, s)
        zs = np.arange(z_min, z_max + s/2, s)
        added = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    inside = (ix_min <= x <= ix_max and
                              iy_min <= y <= iy_max and
                              iz_min <= z <= iz_max)
                    if not inside:
                        self.grid_points.append({
                            'x': float(x), 'y': float(y), 'z': float(z),
                            'structure': name})
                        added += 1
        return added

    def _generate_cylinder_grid(self, radius, height, position, name):
        cx, cy, cz = position
        c = self.clearance_m
        s = self.spacing_m
        outer_r = radius + c
        z_min = max(cz - height/2 - c, 0.005)
        z_max = cz + height/2 + c
        xs = np.arange(cx - outer_r, cx + outer_r + s/2, s)
        ys = np.arange(cy - outer_r, cy + outer_r + s/2, s)
        zs = np.arange(z_min, z_max + s/2, s)
        added = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    dr = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if z < cz - height/2:
                        dz = (cz - height/2) - z
                    elif z > cz + height/2:
                        dz = z - (cz + height/2)
                    else:
                        dz = 0.0
                    if dz == 0.0:
                        dist = max(0.0, dr - radius)
                    elif dr <= radius:
                        dist = dz
                    else:
                        dist = math.sqrt((dr - radius)**2 + dz**2)
                    if 0 < dist <= c:
                        self.grid_points.append({
                            'x': float(x), 'y': float(y), 'z': float(z),
                            'structure': name})
                        added += 1
        return added

    def _filter_by_reachability(self):
        if not USE_REACHABLE_FILTER:
            return
        if not self.grid_points:
            return
        n_before = len(self.grid_points)
        all_xyz = np.array(
            [[g['x'], g['y'], g['z']] for g in self.grid_points],
            dtype=np.float64)
        base_xy = np.array([ROBOT_BASE_POSITION[0], ROBOT_BASE_POSITION[1]])
        horizontal_dist = np.linalg.norm(all_xyz[:, :2] - base_xy, axis=1)
        mask = (
            (horizontal_dist <= REACHABLE_RADIUS_M) &
            (all_xyz[:, 2] >= REACHABLE_Z_MIN) &
            (all_xyz[:, 2] <= REACHABLE_Z_MAX))
        kept_indices = np.where(mask)[0]
        self.grid_points = [self.grid_points[i] for i in kept_indices]
        n_after = len(self.grid_points)
        if self.verbose:
            ratio = n_after / n_before * 100 if n_before > 0 else 0
            print(f"  Reachability filter: {n_before} -> {n_after} ({ratio:.1f}%)")

    def build_grid(self, env_info):
        if self.verbose:
            print(f"\n  Building risk grid (clearance={self.clearance_m*1000:.0f}mm, "
                  f"spacing={self.spacing_m*1000:.0f}mm)")
        for item in env_info:
            shape = item['shape']
            name = item['name']
            if shape == 'box':
                self._generate_box_grid(item['size'], item['position'], name)
            elif shape == 'cylinder':
                self._generate_cylinder_grid(
                    item['radius'], item['height'], item['position'], name)
        if self.verbose:
            print(f"  Grid points before filtering: {len(self.grid_points)}")
        self._filter_by_reachability()
        self._grid_xyz = np.array(
            [[g['x'], g['y'], g['z']] for g in self.grid_points],
            dtype=np.float64)
        self.invaded_flags = np.zeros(len(self.grid_points), dtype=bool)
        if self.verbose:
            print(f"  Final grid points: {len(self.grid_points)}")

    def record_aabb(self):
        link_aabbs = []
        for link_idx in range(-1, p.getNumJoints(self.robot_id)):
            aabb_min, aabb_max = p.getAABB(self.robot_id, link_idx)
            link_aabbs.append((link_idx, np.array(aabb_min), np.array(aabb_max)))
        self.aabb_history.append({
            'step': self._step,
            'time_s': self._step * self.dt,
            'phase': self._current_phase,
            'aabbs': link_aabbs})
        self._step += 1

    def post_process(self):
        if self._grid_xyz is None or len(self.aabb_history) == 0:
            return
        n_grid  = len(self._grid_xyz)
        n_steps = len(self.aabb_history)
        if self.verbose:
            print(f"\n  Post-processing grid invasion ({n_grid} points x {n_steps} steps)")
        t0 = time.time()
        first_step  = np.full(n_grid, -1, dtype=np.int64)
        first_link  = np.full(n_grid, -1, dtype=np.int64)
        first_phase = ['' for _ in range(n_grid)]
        gx, gy, gz = self._grid_xyz[:, 0], self._grid_xyz[:, 1], self._grid_xyz[:, 2]
        margin = AABB_INFLATION_M
        for step_data in self.aabb_history:
            step = step_data['step']
            phase = step_data['phase']
            for link_idx, amin, amax in step_data['aabbs']:
                not_yet = first_step == -1
                if not np.any(not_yet):
                    break
                inside = (
                    (gx >= amin[0] - margin) & (gx <= amax[0] + margin) &
                    (gy >= amin[1] - margin) & (gy <= amax[1] + margin) &
                    (gz >= amin[2] - margin) & (gz <= amax[2] + margin))
                new_inv = inside & not_yet
                if np.any(new_inv):
                    first_step[new_inv] = step
                    first_link[new_inv] = link_idx
                    for idx in np.where(new_inv)[0]:
                        first_phase[idx] = phase
        t1 = time.time() - t0
        n_after_1st = int(np.sum(first_step >= 0))
        if self.verbose:
            print(f"    First pass: {n_after_1st} points ({t1:.2f}s)")
        if USE_PRECISE_PASS and n_after_1st > 0:
            t0 = time.time()
            cand_mask = first_step >= 0
            new_first_step  = np.full(n_grid, -1, dtype=np.int64)
            new_first_link  = np.full(n_grid, -1, dtype=np.int64)
            new_first_phase = ['' for _ in range(n_grid)]
            for step_data in self.aabb_history:
                step = step_data['step']
                phase = step_data['phase']
                for link_idx, amin, amax in step_data['aabbs']:
                    not_yet = new_first_step == -1
                    if not np.any(not_yet):
                        break
                    inside = (
                        (gx >= amin[0]) & (gx <= amax[0]) &
                        (gy >= amin[1]) & (gy <= amax[1]) &
                        (gz >= amin[2]) & (gz <= amax[2]))
                    new_inv = inside & not_yet & cand_mask
                    if np.any(new_inv):
                        new_first_step[new_inv] = step
                        new_first_link[new_inv] = link_idx
                        for idx in np.where(new_inv)[0]:
                            new_first_phase[idx] = phase
            first_step = new_first_step
            first_link = new_first_link
            first_phase = new_first_phase
            t2 = time.time() - t0
            if self.verbose:
                print(f"    Precise pass: {int(np.sum(first_step>=0))} points ({t2:.2f}s)")
        self.invaded_flags = first_step >= 0
        for idx in np.where(self.invaded_flags)[0]:
            g = self.grid_points[idx]
            self.invasion_records.append({
                'time_s': float(first_step[idx] * self.dt),
                'step': int(first_step[idx]),
                'phase': first_phase[idx],
                'grid_id': int(idx),
                'x': g['x'], 'y': g['y'], 'z': g['z'],
                'structure': g['structure'],
                'invaded_link': int(first_link[idx])})
        self.invasion_records.sort(key=lambda r: r['step'])
        if self.verbose:
            print(f"  Invaded grid points: {int(np.sum(self.invaded_flags))}")

    def save_invasion_csv(self, path=INVASION_LOG_PATH):
        if not self.invasion_records:
            return
        ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.invasion_records[0].keys())
            writer.writeheader()
            writer.writerows(self.invasion_records)
        if self.verbose:
            print(f"  Saved {path}: {len(self.invasion_records)} rows")

    def save_timeseries_csv(self, path=TIMESERIES_LOG_PATH):
        if not self.invasion_records:
            return
        records_by_step = {}
        for r in self.invasion_records:
            records_by_step.setdefault(r['step'], []).append(r)
        rows = []
        invaded_set = set()
        for step_data in self.aabb_history:
            step = step_data['step']
            new_count = 0
            if step in records_by_step:
                for r in records_by_step[step]:
                    if r['grid_id'] not in invaded_set:
                        invaded_set.add(r['grid_id'])
                        new_count += 1
            rows.append({
                'time_s': step * self.dt, 'step': step,
                'phase': step_data['phase'],
                'cumulative_invaded': len(invaded_set),
                'new_invasions_this_step': new_count})
        ensure_parent_dir(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# PyBullet geometry helpers
# ============================================================
def create_box(size, position, color, mass=0):
    half_extents = [s / 2 for s in size]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position)

def create_cylinder(radius, height, position, color, mass=0):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position)

def create_tray(center_position, inner_size, wall_height, wall_thickness, color, name_prefix):
    cx, cy, cz = center_position
    ix, iy = inner_size
    outer_x = ix + 2 * wall_thickness
    outer_y = iy + 2 * wall_thickness
    ids = []
    parts_info = []
    sz = [outer_x, outer_y, wall_thickness]
    pos = [cx, cy, cz + wall_thickness/2]
    ids.append(create_box(sz, pos, color))
    parts_info.append({'shape': 'box', 'size': sz, 'position': pos,
                       'name': f"{name_prefix}_bottom"})
    sz = [wall_thickness, outer_y, wall_height]
    pos = [cx + ix/2 + wall_thickness/2, cy, cz + wall_thickness + wall_height/2]
    ids.append(create_box(sz, pos, color))
    parts_info.append({'shape': 'box', 'size': sz, 'position': pos,
                       'name': f"{name_prefix}_wall_xp"})
    pos = [cx - ix/2 - wall_thickness/2, cy, cz + wall_thickness + wall_height/2]
    ids.append(create_box(sz, pos, color))
    parts_info.append({'shape': 'box', 'size': sz, 'position': pos,
                       'name': f"{name_prefix}_wall_xn"})
    sz = [ix, wall_thickness, wall_height]
    pos = [cx, cy + iy/2 + wall_thickness/2, cz + wall_thickness + wall_height/2]
    ids.append(create_box(sz, pos, color))
    parts_info.append({'shape': 'box', 'size': sz, 'position': pos,
                       'name': f"{name_prefix}_wall_yp"})
    pos = [cx, cy - iy/2 - wall_thickness/2, cz + wall_thickness + wall_height/2]
    ids.append(create_box(sz, pos, color))
    parts_info.append({'shape': 'box', 'size': sz, 'position': pos,
                       'name': f"{name_prefix}_wall_yn"})
    return ids, parts_info

# ============================================================
# Simulation entry point
# ============================================================
def run_simulation(layout_params=None, trial_params=None,
                   headless=None, verbose=None,
                   save_step_data=True, save_layout_image=True,
                   save_xy_figure=False):
    """Run one pick-and-place simulation trial and return a summary dict."""
    if layout_params is None:
        layout_params = {}
    if trial_params is None:
        trial_params = {}
    if headless is None:
        headless = HEADLESS
    if verbose is None:
        verbose = not headless

    fence_x  = layout_params.get('fence_x',  DEFAULT_FENCE_X)
    pillar_x = layout_params.get('pillar_x', DEFAULT_PILLAR_X)
    pillar_y = layout_params.get('pillar_y', DEFAULT_PILLAR_Y)
    pick_x_offset  = trial_params.get('pick_x_offset',  0.0)
    pick_y_offset  = trial_params.get('pick_y_offset',  0.0)
    place_x_offset = trial_params.get('place_x_offset', 0.0)
    place_y_offset = trial_params.get('place_y_offset', 0.0)
    layout_id = layout_params.get('layout_id', 0)
    trial_id  = trial_params.get('trial_id',   0)

    DT = 1.0 / 240.0
    if headless:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # Keep the interactive PyBullet camera free during simulation.
        # Uncomment this block if a fixed GUI view is needed for manual review.
        # p.resetDebugVisualizerCamera(
        #     cameraDistance=2.5,
        #     cameraYaw=45,
        #     cameraPitch=-30,
        #     cameraTargetPosition=[0.3, -0.2, 0.3],
        # )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    # floor
    floor_color = [0.55, 0.55, 0.55, 1.0]  # matte concrete

    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(
        plane_id,
        -1,
        textureUniqueId=-1,
        rgbaColor=floor_color
    )
    


    franka_base_id = create_box(
        BASE_SIZE,
        [0.0, 0.0, BASE_HEIGHT/2],
        BASE_COLOR
    )
    
    pandaId = p.loadURDF("franka_panda/panda.urdf",
                         basePosition=ROBOT_BASE_POSITION, useFixedBase=True)

    fence_segment_length = (FENCE_TOTAL_LENGTH - 2 * FENCE_OPENING_HALF) / 2
    fence_left_y = -FENCE_OPENING_HALF - fence_segment_length / 2
    fence_left_size = [FENCE_THICKNESS, fence_segment_length, FENCE_HEIGHT]
    fence_left_pos  = [fence_x, fence_left_y, FENCE_HEIGHT/2]
    fence_left_id   = create_box(fence_left_size, fence_left_pos, FENCE_COLOR)

    fence_right_y = FENCE_OPENING_HALF + fence_segment_length / 2
    fence_right_size = [FENCE_THICKNESS, fence_segment_length, FENCE_HEIGHT]
    fence_right_pos  = [fence_x, fence_right_y, FENCE_HEIGHT/2]
    fence_right_id   = create_box(fence_right_size, fence_right_pos, FENCE_COLOR)

    ws1_pos = [WS1_X, WS1_Y, WS1_SIZE[2]/2]
    ws1_id  = create_box(WS1_SIZE, ws1_pos, WS1_COLOR)
    ws2_pos = [WS2_POSITION[0], WS2_POSITION[1], WS2_SIZE[2]/2]
    ws2_id  = create_box(WS2_SIZE, ws2_pos, WS2_COLOR)
    cyl_pos = [pillar_x, pillar_y, CYL_HEIGHT/2]
    cyl_id  = create_cylinder(CYL_RADIUS, CYL_HEIGHT, cyl_pos, CYL_COLOR)

    tray1_ids, tray1_parts = create_tray(
        [WS1_X, WS1_Y, WS1_SIZE[2]],
        TRAY1_INNER_SIZE, TRAY1_WALL_HEIGHT, TRAY1_WALL_THICKNESS,
        TRAY1_COLOR, "tray1")
    tray2_ids, tray2_parts = create_tray(
        [WS2_POSITION[0], WS2_POSITION[1], WS2_SIZE[2]],
        TRAY2_INNER_SIZE, TRAY2_WALL_HEIGHT, TRAY2_WALL_THICKNESS,
        TRAY2_COLOR, "tray2")

    cube_x = WS1_X + pick_x_offset
    cube_y = WS1_Y + pick_y_offset
    cube_z = WS1_SIZE[2] + TRAY1_WALL_THICKNESS + CUBE_SIZE
    cube_vis = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[CUBE_SIZE]*3, rgbaColor=CUBE_COLOR)
    cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_SIZE]*3)
    cubeId = p.createMultiBody(0.05, cube_col, cube_vis,
                               [cube_x, cube_y, cube_z])

    place_x = WS2_POSITION[0] + place_x_offset
    place_y = WS2_POSITION[1] + place_y_offset

    p.changeDynamics(cubeId, -1,
                     lateralFriction=5.0, spinningFriction=1.0,
                     rollingFriction=0.1, linearDamping=0.9, angularDamping=0.9)
    for j in range(p.getNumJoints(pandaId)):
        p.changeDynamics(pandaId, j, lateralFriction=5.0)

    if verbose:
        print(f"\nStarting simulation: layout_id={layout_id}, trial_id={trial_id}")
        print(f"   fence_x={fence_x:.3f}, pillar=({pillar_x:.3f}, {pillar_y:.3f})")
        print(f"   pick=({cube_x:.3f}, {cube_y:.3f}), place=({place_x:.3f}, {place_y:.3f})")

    hazard_grid = HazardGridTracker(
        robot_id=pandaId, clearance_mm=MIN_CLEARANCE_MM,
        spacing_mm=GRID_SPACING_MM, dt=DT, verbose=verbose)
    env_info = [
        {'shape': 'box', 'size': fence_left_size,
         'position': fence_left_pos,  'name': 'fence_left'},
        {'shape': 'box', 'size': fence_right_size,
         'position': fence_right_pos, 'name': 'fence_right'},
        {'shape': 'box', 'size': WS1_SIZE,
         'position': ws1_pos, 'name': 'ws1_base'},
        {'shape': 'box', 'size': WS2_SIZE,
         'position': ws2_pos, 'name': 'ws2_base'},
        {'shape': 'cylinder', 'radius': CYL_RADIUS, 'height': CYL_HEIGHT,
         'position': cyl_pos, 'name': 'cylinder'}]
    env_info.extend(tray1_parts)
    env_info.extend(tray2_parts)

    hazard_grid.build_grid(env_info)


    structure_ids = {
        'fence_left':  fence_left_id,
        'fence_right': fence_right_id,
        'ws1_base':    ws1_id,
        'ws2_base':    ws2_id,
        'cylinder':    cyl_id,
        'tray1':       tray1_ids,
        'tray2':       tray2_ids}
    step_logger = StepDataLogger(
        robot_id=pandaId, ee_idx=11, n_joints=7,
        structure_ids=structure_ids, dt=DT)
    

    EE_IDX   = 11
    N_JOINTS = 7
    FJ1, FJ2 = 9, 10
    HOME_Q   = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    ll = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    ul = np.array([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973])
    jr = ul - ll

    DOWN_ORN = p.getQuaternionFromEuler([math.pi, 0, 0])
    JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
    VEL_SAFETY_MARGIN = 0.80
    JOINT_VEL_SAFE    = JOINT_VEL_LIMITS * VEL_SAFETY_MARGIN
    PER_JOINT_KP = np.array([0.15, 0.20, 0.15, 0.25, 0.15, 0.20, 0.15])
    PER_JOINT_KD = np.array([1.0,  1.5,  1.0,  1.5,  1.0,  1.5,  1.0])
    PER_JOINT_MAX_VEL = JOINT_VEL_SAFE.copy()
    STEPS_PER_RAD = 120
    STEP_MIN = 120
    STEP_MAX = 400
    JOINT_STEP_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6])

    mj_traj = MinimumJerkTrajectory(dt=DT)
    pw_traj = PiecewiseMinJerk(dt=DT)

    def get_current_q():
        return np.array([p.getJointState(pandaId, j)[0] for j in range(N_JOINTS)])

    def reset_to_home():
        for i, q in enumerate(HOME_Q):
            p.resetJointState(pandaId, i, q)
        p.resetJointState(pandaId, FJ1, 0.04)
        p.resetJointState(pandaId, FJ2, 0.04)

    def ik_solve(target_pos, target_orn=None):
        if target_orn is None:
            target_orn = DOWN_ORN
        current_q = get_current_q().tolist()
        result = p.calculateInverseKinematics(
            pandaId, EE_IDX, target_pos, target_orn,
            lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
            jointRanges=jr.tolist(), restPoses=current_q,
            maxNumIterations=200, residualThreshold=1e-5)
        return np.array(result[:N_JOINTS])

    def calc_steps(q_from, q_to):
        dq = np.abs(q_to - q_from)
        weighted_dq = dq * JOINT_STEP_WEIGHT
        n_from_weight = np.max(weighted_dq) * STEPS_PER_RAD
        min_time = np.max(dq / JOINT_VEL_SAFE)
        PEAK_FACTOR = 1.875
        n_from_vel_limit = (min_time * PEAK_FACTOR) / DT
        n = int(np.clip(max(n_from_weight, n_from_vel_limit), STEP_MIN, STEP_MAX))
        return n, np.max(dq)

    def check_velocity_limits(vel_traj):
        max_vels = np.max(np.abs(vel_traj), axis=0)
        ratios = max_vels / JOINT_VEL_SAFE
        worst_ratio = np.max(ratios)
        if worst_ratio > 1.0:
            return False, worst_ratio
        return True, 1.0

    def execute_trajectory(pos_traj, vel_traj):
        for i in range(len(pos_traj)):
            q_ref = pos_traj[i]
            v_ref = vel_traj[i]
            for j in range(N_JOINTS):
                p.setJointMotorControl2(
                    pandaId, j, p.POSITION_CONTROL,
                    targetPosition=q_ref[j], targetVelocity=v_ref[j],
                    positionGain=float(PER_JOINT_KP[j]),
                    velocityGain=float(PER_JOINT_KD[j]),
                    force=240, maxVelocity=float(PER_JOINT_MAX_VEL[j]))
            p.stepSimulation()
            if not headless:
                time.sleep(DT)
            hazard_grid.record_aabb()
            step_logger.record()

    def move_through_waypoints(ee_positions, step_counts=None, ee_orn=None):
        q_waypoints = [get_current_q()]
        for pos in ee_positions:
            q_waypoints.append(ik_solve(pos, ee_orn))
        if step_counts is None:
            step_counts = []
            for i in range(len(q_waypoints) - 1):
                n, _ = calc_steps(q_waypoints[i], q_waypoints[i+1])
                step_counts.append(n)
        for attempt in range(3):
            pos_t, vel_t, _ = pw_traj.generate(q_waypoints, step_counts)
            is_safe, scale = check_velocity_limits(vel_t)
            if is_safe:
                break
            step_counts = [int(np.clip(s * scale * 1.05, STEP_MIN, STEP_MAX * 2))
                           for s in step_counts]
        execute_trajectory(pos_t, vel_t)

    def hold_position(steps=60):
        q_hold = get_current_q()
        for i in range(steps):
            for j in range(N_JOINTS):
                p.setJointMotorControl2(
                    pandaId, j, p.POSITION_CONTROL,
                    targetPosition=q_hold[j], targetVelocity=0.0,
                    positionGain=float(PER_JOINT_KP[j]),
                    velocityGain=float(PER_JOINT_KD[j]), force=240)
            p.stepSimulation()
            if not headless:
                time.sleep(DT)
            hazard_grid.record_aabb()
            step_logger.record()

    def control_gripper(target_pos, force=20, steps=80):
        q_hold = get_current_q()
        for i in range(steps):
            for j in range(N_JOINTS):
                p.setJointMotorControl2(
                    pandaId, j, p.POSITION_CONTROL,
                    targetPosition=q_hold[j], targetVelocity=0.0,
                    positionGain=float(PER_JOINT_KP[j]),
                    velocityGain=float(PER_JOINT_KD[j]), force=240)
            for fj in [FJ1, FJ2]:
                p.setJointMotorControl2(
                    pandaId, fj, p.POSITION_CONTROL,
                    targetPosition=target_pos, force=force, maxVelocity=0.5)
            p.stepSimulation()
            if not headless:
                time.sleep(DT)
            hazard_grid.record_aabb()
            step_logger.record()

    def wait_steps(n=60):
        hold_position(n)

    GATE_X = fence_x
    GATE_Y = 0.0

    reset_to_home()
    control_gripper(0.04)
    wait_steps(80)

    cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
    cx, cy, cz  = cube_pos

    if verbose: print(f"  [1/7] Approach...")
    hazard_grid.set_phase("approach")
    step_logger.set_phase("approach")
    move_through_waypoints(ee_positions=[
        [cx, cy, SAFE_Z],
        [cx, cy, cz + 0.18],
        [cx, cy, cz + 0.02]])
    wait_steps(20)

    if verbose: print(f"  [2/7] Grasp")
    hazard_grid.set_phase("grasp")
    step_logger.set_phase("grasp")
    control_gripper(0.015, force=25, steps=80)
    wait_steps(30)

    if verbose: print(f"  [3/7] Lift...")
    hazard_grid.set_phase("lift")
    step_logger.set_phase("lift")
    move_through_waypoints(ee_positions=[[cx, cy, SAFE_Z]])
    wait_steps(20)

    if verbose: print(f"  [4/7] Transit...")
    hazard_grid.set_phase("transit")
    step_logger.set_phase("transit")
    move_through_waypoints(ee_positions=[
        [GATE_X,  GATE_Y, SAFE_Z],
        [0.15,    GATE_Y, SAFE_Z],
        [place_x, place_y, SAFE_Z]])
    wait_steps(20)

    if verbose: print(f"  [5/7] Lower...")
    hazard_grid.set_phase("lower")
    step_logger.set_phase("lower")
    move_through_waypoints(ee_positions=[[place_x, place_y, PLACE_Z_LOWER]])
    wait_steps(20)

    if verbose: print(f"  [6/7] Release")
    hazard_grid.set_phase("release")
    step_logger.set_phase("release")
    control_gripper(0.04, force=20, steps=80)
    wait_steps(30)

    if verbose: print(f"  [7/7] Retract...")
    hazard_grid.set_phase("retract")
    step_logger.set_phase("retract")
    move_through_waypoints(ee_positions=[
        [place_x, place_y, SAFE_Z],
        [0.3, 0.0, 0.55]])
    wait_steps(30)

    final_cube_pos, _ = p.getBasePositionAndOrientation(cubeId)

    hazard_grid.set_phase("home")
    step_logger.set_phase("home")
    reset_to_home()
    control_gripper(0.04)
    wait_steps(80)

    if verbose:
        print("  Simulation complete")

    hazard_grid.post_process()
    step_logger.post_process_will_invade()

    # Optional layout images for model input or explanatory figures.
    layout_image_path = None
    if save_xy_figure:
        xy_figure_path = os.path.join(
            LAYOUT_IMAGE_DIR,
            f"layout_{layout_id:03d}_trial_{trial_id:03d}_xy_safety_margin.png",
        )
        save_xy_safety_margin_figure(layout_params, trial_params, xy_figure_path)

    if save_layout_image:
        layout_image_path = os.path.join(
            LAYOUT_IMAGE_DIR,
            f"layout_{layout_id:03d}_trial_{trial_id:03d}.npy")
        img_gen = LayoutImageGenerator()
        img_gen.generate(layout_params, trial_params)
        img_gen.save(layout_image_path)
        if verbose:
            print(f"  Saved layout image: {layout_image_path}")

    if save_step_data:
        step_logger.save_csv(STEP_DATA_PATH,
                             layout_id=layout_id, trial_id=trial_id,
                             append=False)
        if verbose:
            print(f"  Saved step data: {STEP_DATA_PATH} ({len(step_logger.records)} rows)")

    n_invaded = int(np.sum(hazard_grid.invaded_flags)) \
                if hazard_grid.invaded_flags is not None else 0
    voxel_volume_cm3 = (GRID_SPACING_MM / 10.0) ** 3
    invaded_volume_cm3 = n_invaded * voxel_volume_cm3

    place_dist = math.sqrt((final_cube_pos[0] - place_x)**2 +
                           (final_cube_pos[1] - place_y)**2)
    task_success = 1 if place_dist < 0.15 else 0

    # Trial-level summaries derived from step-level records.
    if step_logger.records:
        all_manip = [r['manipulability'] for r in step_logger.records]
        all_min_d = [r['min_distance_overall'] for r in step_logger.records]
        all_min_jm = [r['min_joint_margin'] for r in step_logger.records]
        all_in_haz = [r['in_hazard_now'] for r in step_logger.records]
        all_intr_depth = [r['intrusion_depth_mm'] for r in step_logger.records]
        n_steps_total = len(step_logger.records)

        mean_manip = float(np.mean(all_manip))
        min_manip  = float(np.min(all_manip))
        mean_min_d = float(np.mean(all_min_d))
        min_min_d  = float(np.min(all_min_d))
        mean_min_jm = float(np.mean(all_min_jm))
        min_min_jm  = float(np.min(all_min_jm))
        hazard_time_ratio = float(np.mean(all_in_haz))
        max_intrusion = float(np.max(all_intr_depth))
        task_time_s = n_steps_total * DT

        all_coll       = [r['collision_occurred']    for r in step_logger.records]
        all_coll_force = [r['collision_force_total'] for r in step_logger.records]
        all_torques    = [r['max_joint_torque']      for r in step_logger.records]
        all_coll_structs = [r['colliding_structures'] for r in step_logger.records]

        any_collision_occurred = 1 if max(all_coll) > 0 else 0
        total_collision_steps  = sum(all_coll)
        max_collision_force    = float(max(all_coll_force))
        mean_collision_force   = float(np.mean([f for f in all_coll_force if f > 0])) \
                                 if max_collision_force > 0 else 0.0
        max_joint_torque_overall  = float(max(all_torques))
        mean_joint_torque_overall = float(np.mean(all_torques))

        coll_struct_set = set()
        for s in all_coll_structs:
            if s and s != "none":
                for name in s.split(","):
                    coll_struct_set.add(name)
        collided_structures = ",".join(sorted(coll_struct_set)) \
                              if coll_struct_set else "none"
    else:
        mean_manip = min_manip = 0.0
        mean_min_d = min_min_d = 0.0
        mean_min_jm = min_min_jm = 0.0
        hazard_time_ratio = 0.0
        max_intrusion = 0.0
        task_time_s = 0.0
        any_collision_occurred = 0
        total_collision_steps = 0
        max_collision_force = 0.0
        mean_collision_force = 0.0
        max_joint_torque_overall = 0.0
        mean_joint_torque_overall = 0.0
        collided_structures = "none"

    clean_success = 1 if (task_success == 1 and any_collision_occurred == 0) else 0

    result = {
        'layout_id': layout_id,
        'trial_id':  trial_id,
        'fence_x':   fence_x,
        'pillar_x':  pillar_x,
        'pillar_y':  pillar_y,
        'pick_x':    cube_x,
        'pick_y':    cube_y,
        'place_x':   place_x,
        'place_y':   place_y,
        'pick_x_offset':  pick_x_offset,
        'pick_y_offset':  pick_y_offset,
        'place_x_offset': place_x_offset,
        'place_y_offset': place_y_offset,
        'layout_image_path': layout_image_path,
        'total_grid_points':  len(hazard_grid.grid_points),
        'invaded_grid_count': n_invaded,
        'invaded_volume_cm3': invaded_volume_cm3,
        'invasion_ratio':     n_invaded / len(hazard_grid.grid_points)
                              if len(hazard_grid.grid_points) > 0 else 0,
        'mean_manipulability':  mean_manip,
        'min_manipulability':   min_manip,
        'mean_min_distance':    mean_min_d,
        'min_min_distance':     min_min_d,
        'mean_min_joint_margin': mean_min_jm,
        'min_min_joint_margin':  min_min_jm,
        'hazard_time_ratio':    hazard_time_ratio,
        'max_intrusion_depth_mm': max_intrusion,
        'task_success':         task_success,
        'any_collision_occurred':    any_collision_occurred,
        'total_collision_steps':     total_collision_steps,
        'max_collision_force':       max_collision_force,
        'mean_collision_force':      mean_collision_force,
        'collided_structures':       collided_structures,
        'max_joint_torque_overall':  max_joint_torque_overall,
        'mean_joint_torque_overall': mean_joint_torque_overall,
        'clean_success':             clean_success,
        'task_time_s':          task_time_s,
        'final_cube_x':         final_cube_pos[0],
        'final_cube_y':         final_cube_pos[1],
        'final_cube_z':         final_cube_pos[2],
        '_hazard_grid':  hazard_grid,
        '_step_logger':  step_logger,
    }

    p.disconnect()
    return result


def save_trial_summary(result_dict, path=TRIAL_SUMMARY_PATH, append=False):
    row = {k: v for k, v in result_dict.items() if not k.startswith('_')}
    keys = list(row.keys())
    mode = 'a' if append and os.path.exists(path) else 'w'
    ensure_parent_dir(path)
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if mode == 'w':
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Standalone demo
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Franka Panda confined-workspace simulation")
    print(f" HEADLESS = {HEADLESS}")
    print("="*60)

    layout = {
        'layout_id': 0,
        'fence_x':   DEFAULT_FENCE_X,
        'pillar_x':  DEFAULT_PILLAR_X,
        'pillar_y':  DEFAULT_PILLAR_Y,
    }
    trial = {
        'trial_id':       0,
        'pick_x_offset':  0.0,
        'pick_y_offset':  0.0,
        'place_x_offset': 0.0,
        'place_y_offset': 0.0,
    }

    result = run_simulation(layout, trial)

    print("\n" + "="*60)
    print("Simulation summary")
    print("="*60)
    for k, v in result.items():
        if k.startswith('_'):
            continue
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")

    print("\n" + "="*60)
    print("Collision and torque summary")
    print("="*60)
    if result['any_collision_occurred']:
        print("  Collision occurred")
        print(f"     collision steps : {result['total_collision_steps']}")
        print(f"     max force       : {result['max_collision_force']:.2f} N")
        print(f"     mean force      : {result['mean_collision_force']:.2f} N")
        print(f"     structures      : {result['collided_structures']}")
    else:
        print("  No collision")
    print(f"  max joint torque  : {result['max_joint_torque_overall']:.2f} N m")
    print(f"  mean joint torque : {result['mean_joint_torque_overall']:.2f} N m")
    print()
    print("  Trial labels:")
    print(f"     task_success         : {result['task_success']}")
    print(f"     any_collision        : {result['any_collision_occurred']}")
    print(f"     clean_success        : {result['clean_success']}")
    print("="*60)

    hg = result['_hazard_grid']
    hg.save_invasion_csv()
    hg.save_timeseries_csv()
    save_trial_summary(result, append=False)
    print(f"\n  trial_summary: {TRIAL_SUMMARY_PATH}")

    print("\nDone")
