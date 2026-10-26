"""The two teleoperated UR5e arms: controller input -> clutch mapping -> differential IK ->
joint and gripper commands. Knows nothing about cloth, recording or the network."""
from __future__ import annotations

import mujoco
import numpy as np

from . import scene as S
from .ik import DiffIK
from .teleop import ArmTeleop


class DualArms:
    # The joint servos are PD position actuators: force = kp (ctrl - q) - kv qdot. Alone they trail
    # a moving command by kv / kp (0.1 s): the arm visibly lags the operator's hand. Commanding
    # ctrl = q_cmd + (kv / kp) qdot_cmd turns them into PD with velocity feedforward,
    # force = kp (q_cmd - q) + kv (qdot_cmd - qdot), which follows a moving command without lag.
    FF_SMOOTH = 0.02        # [s] low-pass of qdot_cmd: the headset sends at 72-90 Hz, the control
                            # runs at 100 Hz; even with the hand extrapolated between packets
                            # (teleop.py) the per-tick difference is a little uneven

    def __init__(self, info: S.SceneInfo, task: str, pos_scale: float = 1.0, feedforward: bool = True):
        self.info = info
        m = info.model
        self.ik = {s: DiffIK(m, info.arm_qpos_adr[s], info.arm_dof_adr[s], info.ee_site[s], S.HOME_Q[s])
                   for s in S.SIDES}
        self.teleop = {s: ArmTeleop(pos_scale=pos_scale, z_min=S.PINCH_Z_MIN[task]) for s in S.SIDES}
        self.q_cmd = {s: S.HOME_Q[s].copy() for s in S.SIDES}
        act = {s: np.asarray(info.arm_act[s]) for s in S.SIDES}
        self.ff_tau = {s: (-m.actuator_biasprm[a, 2] / m.actuator_gainprm[a, 0]) * feedforward
                       for s, a in act.items()}
        self.ctrl_lo = {s: m.actuator_ctrlrange[a, 0] for s, a in act.items()}
        self.ctrl_hi = {s: m.actuator_ctrlrange[a, 1] for s, a in act.items()}
        self.qd_cmd = {s: np.zeros(6) for s in S.SIDES}

    def reset(self, d: mujoco.MjData):
        for s in S.SIDES:
            self.q_cmd[s] = S.HOME_Q[s].copy()
            self.qd_cmd[s] = np.zeros(6)
            pos, quat = self.ik[s].fk(self.q_cmd[s], d.qpos)
            self.teleop[s].reset(pos, quat)

    def control(self, d: mujoco.MjData, inputs: dict | None, dt: float):
        """One control tick. inputs: the headset message ({'left': {...}, 'right': {...}}), or
        None when it is stale (the arms disengage and hold still)."""
        for s in S.SIDES:
            tel = self.teleop[s]
            if not tel.engaged:
                # keep the virtual target glued to the commanded pose while idle
                tel.target_pos, tel.target_quat = self.ik[s].fk(self.q_cmd[s])
            if inputs is not None:
                c = inputs.get(s)
                tel.update(dict(c, t=inputs.get("t")) if c else c, dt)
            else:
                tel.disengage()
            q_prev = self.q_cmd[s]
            if tel.engaged:
                self.q_cmd[s] = self.ik[s].step(self.q_cmd[s], tel.target_pos, tel.target_quat, dt)
            qd = (self.q_cmd[s] - q_prev) / dt
            self.qd_cmd[s] += (qd - self.qd_cmd[s]) * min(1.0, dt / self.FF_SMOOTH)
            d.ctrl[self.info.arm_act[s]] = np.clip(self.q_cmd[s] + self.ff_tau[s] * self.qd_cmd[s],
                                                   self.ctrl_lo[s], self.ctrl_hi[s])
            d.ctrl[self.info.grip_act[s]] = 255.0 * tel.gripper

    def gripper(self, side: str) -> float:
        """Commanded closure, 0 open .. 1 closed."""
        return self.teleop[side].gripper

    def engaged(self, side: str) -> bool:
        return self.teleop[side].engaged

    def ee_pose(self, d: mujoco.MjData, side: str):
        sid = self.info.ee_site[side]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[sid])
        return d.site_xpos[sid].copy(), q
