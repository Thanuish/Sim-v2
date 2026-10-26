"""Cloth engines: what the rest of the simulator needs from a cloth simulation.

The server, recorder and web export only use this interface, so the cloth simulation can be
swapped (MuJoCo's flex cloth today, the GPU cloth of vrteleop/gpu_cloth.py next) without
touching them.

Per control tick the server calls update(d, grippers, dt) (grasping), after every physics step
step(d) (an engine that simulates outside MuJoCo advances here), and reads verts(d) for
streaming, recording and fold metrics.
"""
from __future__ import annotations

import abc

import mujoco
import numpy as np

from . import cloth as C
from . import scene as S


class ClothEngine(abc.ABC):
    name: str = "base"
    info: C.ClothInfo          # the garment pattern: faces, rest_uv, texcoord (render, record)
    max_held: int              # vertices one gripper can hold (recording layout)
    material_id: int = -1      # scene material for the headset (-1: from the MuJoCo flex)
    render_subdiv: int = 2     # smoothing levels the headset applies to the mesh

    @abc.abstractmethod
    def reset(self, d: mujoco.MjData):
        """After the scene was reset: drop all grasps."""

    @abc.abstractmethod
    def update(self, d: mujoco.MjData, grippers: dict[str, float], dt: float):
        """One control tick; grippers: commanded closure per side (0 open .. 1 closed)."""

    def step(self, d: mujoco.MjData):
        """After each physics step (engines simulating outside MuJoCo advance here)."""

    @abc.abstractmethod
    def verts(self, d: mujoco.MjData) -> np.ndarray:
        """(N, 3) vertex positions in the MuJoCo world frame."""

    @abc.abstractmethod
    def held(self, side: str) -> list[int]:
        """Vertices the gripper on `side` is pinching."""

    def holding(self, side: str) -> bool:
        return bool(self.held(side))

    def metrics(self, d: mujoco.MjData) -> dict:
        """Fold progress: coverage (1 flat .. ~0.3 folded), height, lifted."""
        return self.metrics_of(self.verts(d))

    def metrics_of(self, verts: np.ndarray) -> dict:
        """metrics() for a snapshot of verts(d) (thread safe: the server runs it off the sim loop)."""
        return C.fold_metrics(self.info, verts, S.TABLE_Z)

    def sync_render(self, renderer: mujoco.Renderer, d: mujoco.MjData):
        """Before a MuJoCo render: make it show the current cloth (engines outside MuJoCo)."""

    def version(self, d: mujoco.MjData):
        """Changes whenever verts(d) changes (the server streams the cloth only then)."""
        return d.time


class MujocoCloth(ClothEngine):
    """The jeans as a MuJoCo flex in the scene model, with the modelled pinch grasp."""
    name = "mujoco"
    max_held = C.PinchGrasp.K

    def __init__(self, info: S.SceneInfo):
        self.info = info.cloth
        self.model = info.model
        cc = info.cloth_config
        self.grasp = {s: C.PinchGrasp(info.model, info.cloth, s, info.grasp_eq[s],
                                      slip_force=cc.slip_force, release_gap=cc.release_gap)
                      for s in S.SIDES}

    def reset(self, d):
        for g in self.grasp.values():
            g.reset(self.model, d)

    def update(self, d, grippers, dt):
        for s, g in self.grasp.items():
            g.update(self.model, d, grippers[s], dt)

    def verts(self, d):
        return C.verts(self.info, d)

    def held(self, side):
        return self.grasp[side].held


class GPUCloth(ClothEngine):
    """The jeans simulated on the GPU (vrteleop/gpu_cloth.py, XPBD, ~1 cm resolution) next to
    MuJoCo, which only simulates the robots. Every 1/60 s of sim time the robot's collision geoms
    (as oriented boxes) and gripper frames are handed to the cloth, which then runs one frame.
    The pinch follows the rules of cloth.PinchGrasp: it triggers while the jaws close with
    fabric between the pads and takes every point of every layer inside the jaw volume; those
    points ride with the gripper until the pads open by `release_gap`."""
    name = "gpu"
    max_held = 96
    render_subdiv = 0                   # 1 cm is smooth already; each level is 4x the headset's work
    P = C.PinchGrasp                    # jaw geometry and trigger thresholds

    def __init__(self, info: S.SceneInfo, arch: str = "gpu"):
        from . import gpu_cloth as G    # Taichi is only needed when this engine is used
        G.init(arch)
        m = self.model = info.model
        cc = info.cloth_config
        self.cfg = cc
        self.sim = G.GPUJeans(cc.garment, cc.gpu, (*S.JEANS_POS, 0.0), S.JEANS_YAW, S.TABLE_Z,
                              max_boxes=256)
        J = self.sim.J
        nF = len(self.sim.faces) // 2
        self.info = C.ClothInfo(name="jeans", rest_uv=J["uv"], faces=self.sim.faces,
                                face_layer=np.repeat([0, 1], nF), texcoord=J["tc"],
                                face_tc=J["face_tc"], garment=self.sim.garment)
        self.info.nvert = self.sim.N
        self.material_id = m.material("denim").id
        # colliders: every collision geom of the arms and grippers as oriented boxes. A curved
        # mesh (the finger links) is cut into slices along its length, one box each: a single
        # bounding box of a finger reaches across the jaw and would trap pinched fabric
        geoms = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                 and (m.body(m.geom_bodyid[g]).name or "").startswith(("left_", "right_"))]
        gid, off, half = [], [], []
        for g in geoms:
            for c, h in _geom_boxes(m, g):
                gid.append(g); off.append(c); half.append(h)
        self.geoms, self.box_off, self.box_half = np.array(gid), np.array(off), np.array(half)
        self.sim.set_box_sides([S.SIDES.index(m.body(m.geom_bodyid[g]).name.split("_")[0])
                                for g in self.geoms])
        self.grip_body = [m.body(f"{s}_gripper_base").id for s in S.SIDES]
        self.site = {s: m.site(f"{s}_gripper_pinch").id for s in S.SIDES}
        self.driver_q = {s: m.jnt_qposadr[m.joint(f"{s}_gripper_right_driver_joint").id] for s in S.SIDES}
        self.frame_dt = 1.0 / cc.gpu.fps
        self._held = {s: [] for s in S.SIDES}
        self._grasp = {s: {"armed": True, "stall": 0.0, "prev": 0.0, "gap0": 0.0} for s in S.SIDES}
        self._verts = None
        self._t_next = 0.0
        self.render_mesh = C.ClothRenderMesh(m, self.sim.faces)
        self._frame = 0                                   # cloth frames simulated
        self._synced: dict[int, int] = {}                # renderer -> frame it shows

    def sync_render(self, renderer, d):
        if self._synced.get(id(renderer)) != self._frame:
            self.render_mesh.update(self.model, d, self.verts(d), renderer)
            self._synced[id(renderer)] = self._frame

    def version(self, d):
        return self._frame

    # ------------------------------------------------------------ stepping
    def _poses(self, d):
        R = d.geom_xmat[self.geoms].reshape(-1, 3, 3)
        c = d.geom_xpos[self.geoms] + np.einsum("nij,nj->ni", R, self.box_off)
        gc = d.xpos[self.grip_body]
        gR = d.xmat[self.grip_body].reshape(-1, 3, 3)
        return (c, R, self.box_half), (gc, gR)

    def reset(self, d):
        self.sim.reset()
        boxes, grips = self._poses(d)
        self.sim.set_frame_poses(boxes, grips)          # start and end of the first frame
        self._held = {s: [] for s in S.SIDES}
        self._grasp = {s: {"armed": True, "stall": 0.0, "prev": 0.0, "gap0": 0.0} for s in S.SIDES}
        self._verts = None
        self._frame += 1
        self._t_next = d.time + self.frame_dt

    def step(self, d):
        if d.time < self._t_next - 1e-9:
            return
        self._t_next += self.frame_dt
        if self._t_next < d.time:                        # fell far behind (e.g. after a pause)
            self._t_next = d.time + self.frame_dt
        boxes, grips = self._poses(d)
        self.sim.set_frame_poses(boxes, grips)
        self.sim.frame()
        self._verts = None
        self._frame += 1

    def verts(self, d):
        if self._verts is None:
            self._verts = self.sim.positions()
        return self._verts

    # ------------------------------------------------------------ grasp
    def held(self, side):
        return self._held[side]

    def update(self, d, grippers, dt):
        P = self.P
        for k, s in enumerate(S.SIDES):
            st = self._grasp[s]
            jaw = float(np.clip(d.qpos[self.driver_q[s]] / 0.8, 0, 1))
            closing = grippers[s] >= P.CMD_CLOSE
            if not closing or jaw < P.JAW_RELEASE:
                if self._held[s]:
                    self._release(d, k, s)
                if not closing:
                    st["armed"] = True
                st["stall"] = 0.0
            elif self._held[s]:
                gap = P.pad_gap(jaw)
                st["gap0"] = min(st["gap0"], gap)          # the jaws keep closing after the pinch
                if gap > st["gap0"] + self.cfg.release_gap:
                    self._release(d, k, s)                    # jaws opened: nothing squeezes the fabric
            elif st["armed"]:
                stalled = jaw > P.JAW_STALL and abs(jaw - st["prev"]) < 0.02 * dt / 0.01
                st["stall"] = st["stall"] + dt if stalled else 0.0
                if jaw >= P.JAW_TRIGGER or st["stall"] > 0.12:
                    st["armed"] = False                    # one attempt per closing motion
                    self._try_grasp(d, k, s, jaw)
            st["prev"] = jaw

    def _try_grasp(self, d, k, s, jaw):
        P = self.P
        V = self.verts(d)
        R = d.site_xmat[self.site[s]].reshape(3, 3)
        p = d.site_xpos[self.site[s]]
        loc = (V - p) @ R                                  # pinch-site frame
        # only fabric between the pads' inner faces is squeezed: a wider volume would also take
        # fabric draped around the outside of the fingers, which then hangs on them after release
        half_gap = P.pad_gap(jaw) / 2 + 0.001
        ok = ((np.abs(loc[:, 0]) <= P.HALF_WIDTH) & (np.abs(loc[:, 1]) <= half_gap)
              & (loc[:, 2] >= P.Z_RANGE[0]) & (loc[:, 2] <= P.Z_RANGE[1]))
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return
        centre = np.array([0.0, 0.0, 0.006])
        idx = idx[np.argsort(np.linalg.norm(loc[idx] - centre, axis=1))][:self.max_held]
        b = self.grip_body[k]
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        self.sim.pin_to_gripper(k, idx, (V[idx] - pb) @ Rb)
        self.sim.set_arm_friction(k, self.cfg.gpu.gripper_friction)   # squeezing: it grips
        self._held[s] = [int(i) for i in idx]
        self._grasp[s]["gap0"] = P.pad_gap(jaw)

    def _release(self, d, k, s):
        self.sim.release_gripper(k)
        self._held[s] = []
        # friction needs a normal force: open fingers don't squeeze the fabric bunched around
        # them, so it slides off (with friction a wad of it stayed wedged in the open jaws)
        self.sim.set_arm_friction(k, 0.0)


def _geom_boxes(m: mujoco.MjModel, g: int, slices: int = 5, min_len: float = 0.03):
    """(centre, half size) boxes in the geom frame covering geom g: its bounding box, or for a
    mesh longer than min_len the bounding boxes of `slices` cuts along its longest axis."""
    a = m.geom_aabb[g]
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or 2 * a[3:].max() < min_len:
        return [(a[:3].copy(), a[3:].copy())]
    mid = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    ax = int(np.argmax(np.ptp(V, axis=0)))
    edges = np.linspace(V[:, ax].min(), V[:, ax].max(), slices + 1)
    out = []
    for k in range(slices):
        sel = (V[:, ax] >= edges[k] - 1e-9) & (V[:, ax] <= edges[k + 1] + 1e-9)
        if sel.sum() >= 3:
            lo, hi = V[sel].min(axis=0), V[sel].max(axis=0)
            out.append(((lo + hi) / 2, (hi - lo) / 2))
    return out


ENGINES = {"mujoco": MujocoCloth, "gpu": GPUCloth}


def make_engine(info: S.SceneInfo, name: str | None = None) -> ClothEngine | None:
    """The cloth engine for a built scene (None for tasks without cloth); the scene must have
    been built for it (scene.build_scene(..., cloth_engine=name))."""
    if info.cloth_config is None:
        return None
    name = name or info.cloth_engine or "mujoco"
    if name not in ENGINES:
        raise ValueError(f"unknown cloth engine '{name}' (available: {', '.join(ENGINES)})")
    if name != (info.cloth_engine or "mujoco"):
        raise ValueError(f"the scene was built for the '{info.cloth_engine}' cloth engine, not '{name}'")
    return ENGINES[name](info)
