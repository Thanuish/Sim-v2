"""Clutch-based relative mapping from a VR controller to an EE target.

Hold the grip button to engage: from that moment, the controller's motion
(translation and rotation, both in the world frame) is applied to the EE pose
the arm had when you engaged. Release to move your hand freely without moving
the robot ("ratcheting"), exactly like a mouse lifted off the desk.
"""
from __future__ import annotations

import mujoco
import numpy as np

# Keep targets inside a safe box above the table (MuJoCo world frame)
WORKSPACE_LO = np.array([-0.15, -0.75, 0.77])
WORKSPACE_HI = np.array([0.95, 0.75, 1.55])


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_conj(a):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, a)
    return out


def quat_normalize(q):
    q = np.asarray(q, float)
    return q / (np.linalg.norm(q) + 1e-12)


class ArmTeleop:
    ENGAGE_ON = 0.6
    ENGAGE_OFF = 0.4

    STICK_DEADZONE = 0.15
    GRIPPER_RATE = 1.6      # full open->close in ~0.6 s at full stick deflection

    def __init__(self, pos_scale: float = 1.0, z_min: float | None = None):
        self.pos_scale = pos_scale
        # lowest allowed EE target height (task dependent: cloth needs the fingertips on the table)
        self.ws_lo = WORKSPACE_LO.copy()
        if z_min is not None:
            self.ws_lo[2] = z_min
        self.engaged = False
        self.target_pos = None
        self.target_quat = None
        self._c0_pos = self._c0_quat = self._e0_pos = self._e0_quat = None
        self.gripper = 0.0      # 0 = open, 1 = closed
        self._epoch = None      # client view-alignment epoch (changes on recenter)

    def reset(self, ee_pos, ee_quat):
        self.engaged = False
        self.target_pos = np.array(ee_pos, float)
        self.target_quat = np.array(ee_quat, float)
        self.gripper = 0.0

    def update(self, ctrl: dict | None, dt: float = 0.01):
        """ctrl: {'pose':[x,y,z,qw,qx,qy,qz] (MuJoCo world), 'grip':0..1, 'stick':[x,y]}

        Gripper: positive stick y closes, negative opens. The web client sends thumbstick UP as
        positive (it flips WebXR's y), so on the headset UP closes and DOWN opens.
        The gripper stays where you leave it.
        """
        if ctrl is None or ctrl.get("pose") is None:
            self.engaged = False
            return
        pose = np.asarray(ctrl["pose"], float)
        c_pos, c_quat = pose[:3], quat_normalize(pose[3:7])
        grip = float(ctrl.get("grip", 0.0))
        stick = ctrl.get("stick") or [0.0, 0.0]
        sy = float(stick[1]) if len(stick) > 1 else 0.0
        if abs(sy) > self.STICK_DEADZONE:
            self.gripper = float(np.clip(self.gripper + sy * self.GRIPPER_RATE * dt, 0.0, 1.0))

        # The client re-aligned the VR world (recenter): re-anchor so the robot doesn't jump.
        epoch = ctrl.get("epoch")
        if self.engaged and epoch != self._epoch:
            self._c0_pos, self._c0_quat = c_pos.copy(), c_quat.copy()
            self._e0_pos, self._e0_quat = self.target_pos.copy(), self.target_quat.copy()
        self._epoch = epoch

        if not self.engaged and grip > self.ENGAGE_ON:
            self.engaged = True
            self._c0_pos, self._c0_quat = c_pos.copy(), c_quat.copy()
            self._e0_pos, self._e0_quat = self.target_pos.copy(), self.target_quat.copy()
        elif self.engaged and grip < self.ENGAGE_OFF:
            self.engaged = False

        if self.engaged:
            dpos = (c_pos - self._c0_pos) * self.pos_scale
            self.target_pos = np.clip(self._e0_pos + dpos, self.ws_lo, WORKSPACE_HI)
            drot = quat_mul(c_quat, quat_conj(self._c0_quat))   # world-frame delta
            self.target_quat = quat_normalize(quat_mul(drot, self._e0_quat))

    def disengage(self):
        self.engaged = False
