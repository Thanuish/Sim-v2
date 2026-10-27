"""Jeans on Newton's VBD cloth solver (Vertex Block Descent), next to MuJoCo, which only simulates
the robots.

    python -m vrteleop.server --cloth-engine newton          # the default where CUDA is available

The garment is the same two-panel pair of jeans as the other engines (cloth.jeans_garment: front
and back panel sewn along the out- and inseams, open at the waist and the hems), at the [newton]
section's spacing. Every 1/60 s of sim time:

  * the robot: each collision geom of the arms and grippers is a kinematic Newton body with the
    geom's real shape (capsules, cylinders, boxes; meshes as their convex hulls, as MuJoCo itself
    collides them). Their poses at the start and the end of the frame come from MuJoCo and are
    interpolated across the substeps; the solver gets the start of each substep as the previous
    pose and the end as the current one, so its friction sees the robot move (one-way coupling,
    robot -> cloth, as RoboSimStudio's one-way coupler does);
  * the cloth: SolverVBD with membrane stretch and bending, full self-contact (vertex-triangle
    and edge-edge, detected every substep) and contact against the robot and the table on the
    whole surface (edges and faces, not only the points: with points ~8 mm apart and a contact
    radius of 1 mm, point-only contact lets a finger slip between them);
  * the whole frame runs as one CUDA graph.

Grasping, chosen by [grasp] model:
  * "pinch": the points between the pads when the jaws close ride with the gripper (zero mass,
    put at the gripper's pose before every substep), released when the pads open by
    release_gap: the rules of cloth.PinchGrasp, which the GPU engine follows too;
  * "friction": nothing is attached. The pads squeeze the fabric and friction holds it, as with
    a real gripper.

Patterns from RoboSimStudio (github.com/robosimstudio/robosimstudio, Apache-2.0, see NOTICE):
held particles weigh nothing so VBD leaves them to the pin (EntityHeldCloth); per-shape friction
for the pads, written as the shape coefficient that Newton's sqrt(cloth mu x shape mu) combination
turns into the wanted pair value (write_gripper_friction); the solver's previous body pose taken
from the start of the substep (the "held" atom's calibration note); a settle before the episode.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import warp as wp

import newton

from . import cloth as C
from . import scene as S
from .cloth_engine import ClothEngine, initial_cloth, rest_threads

wp.config.quiet = True

FREE_SIDE = -1


@wp.func
def _lerp_tf(t0: wp.transform, t1: wp.transform, a: float):
    p = wp.transform_get_translation(t0) * (1.0 - a) + wp.transform_get_translation(t1) * a
    q = wp.quat_slerp(wp.transform_get_rotation(t0), wp.transform_get_rotation(t1), a)
    return wp.transform(p, q)


@wp.kernel
def _set_bodies(pose0: wp.array[wp.transform], pose1: wp.array[wp.transform], a0: float, a1: float,
                inv_dt: float, q_prev: wp.array[wp.transform], q_now: wp.array[wp.transform],
                qd_now: wp.array[wp.spatial_vector]):
    """The robot's colliders at the start (q_prev) and the end (q_now) of a substep."""
    i = wp.tid()
    t0 = _lerp_tf(pose0[i], pose1[i], a0)
    t1 = _lerp_tf(pose0[i], pose1[i], a1)
    q_prev[i] = t0
    q_now[i] = t1
    v = (wp.transform_get_translation(t1) - wp.transform_get_translation(t0)) * inv_dt
    dq = wp.transform_get_rotation(t1) * wp.quat_inverse(wp.transform_get_rotation(t0))
    axis = wp.vec3()
    ang = float(0.0)
    wp.quat_to_axis_angle(dq, axis, ang)
    if ang > 3.14159265:
        ang = ang - 6.2831853
    qd_now[i] = wp.spatial_vector(v, axis * ang * inv_dt)


@wp.kernel
def _hold(grip0: wp.array[wp.transform], grip1: wp.array[wp.transform], a1: float,
          held: wp.array2d[wp.int32], local: wp.array2d[wp.vec3], count: wp.array[wp.int32],
          q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    """Pinched points: where the gripper carries them at the end of the substep (they weigh
    nothing, so the solve leaves them there)."""
    k, j = wp.tid()
    if j >= count[k]:
        return
    i = held[k, j]
    q[i] = wp.transform_point(_lerp_tf(grip0[k], grip1[k], a1), local[k, j])
    qd[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _tethers(teth_j: wp.array2d[wp.int32], teth_L: wp.array2d[wp.float32], inv_mass: wp.array[wp.float32],
             inv_dt: float, q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    """Long-range attachments (Kim et al. 2012), after each substep: a point is never further from
    its nearest held point than the fabric between them. VBD's few iterations per substep can't
    carry a lift across a garment of 14000 light points (a corner lifted 20 cm stretched 1 % of the
    threads by 25 %); denim hardly stretches. The velocity follows the correction."""
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    for k in range(2):
        j = teth_j[i, k]
        if j >= 0:
            dv = q[i] - q[j]
            l = wp.length(dv)
            if l > teth_L[i, k]:
                corr = dv * ((l - teth_L[i, k]) / l)
                q[i] = q[i] - corr
                qd[i] = qd[i] - corr * inv_dt


@wp.kernel
def _sum_forces(n: wp.array[wp.int32], shape: wp.array[wp.int32], f: wp.array[wp.spatial_vector],
                offset: int, side_of_shape: wp.array[wp.int32], out: wp.array[wp.vec3]):
    """Contact force of the cloth on each arm, summed over its shapes [N, world]."""
    i = wp.tid()
    if i >= n[0]:
        return
    s = shape[i]
    if s < 0:
        return
    k = side_of_shape[s]
    if k < 0:
        return
    w = f[offset + i]
    wp.atomic_add(out, k, wp.vec3(w[0], w[1], w[2]))


@wp.kernel
def _zero3(a: wp.array[wp.vec3]):
    a[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation matrices -> (n, 4) quaternions x y z w."""
    out = np.zeros((len(R), 4))
    q = np.zeros(4)
    for i, M in enumerate(R):
        mujoco.mju_mat2Quat(q, M.ravel())
        out[i] = (q[1], q[2], q[3], q[0])
    return out


def _hull(m: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull MuJoCo collides a mesh geom with: (vertices in the geom frame, faces)."""
    adr = m.mesh_graphadr[mesh_id]
    va = m.mesh_vertadr[mesh_id]
    if adr < 0:                                   # no hull stored: the mesh as it is
        F = m.mesh_face[m.mesh_faceadr[mesh_id]:m.mesh_faceadr[mesh_id] + m.mesh_facenum[mesh_id]]
        return m.mesh_vert[va:va + m.mesh_vertnum[mesh_id]].copy(), F.copy()
    g = m.mesh_graph[adr:]
    nv, nf = int(g[0]), int(g[1])
    vid = g[2 + nv:2 + 2 * nv]                    # vert_globalid
    faces = g[2 + 2 * nv + nv + 3 * nf:2 + 2 * nv + nv + 3 * nf + 3 * nf].reshape(nf, 3)   # face_globalid
    used, inv = np.unique(faces, return_inverse=True)
    del vid
    return m.mesh_vert[va + used].copy(), inv.reshape(nf, 3)


class NewtonVBDCloth(ClothEngine):
    """The jeans on Newton's VBD solver; see the module docstring."""
    name = "newton"
    max_held = 96
    render_subdiv = 0
    P = C.PinchGrasp

    def __init__(self, info: S.SceneInfo):
        m = self.model = info.model
        cc = self.cfg = info.cloth_config
        p = self.p = cc.newton
        self.device = wp.get_device("cuda:0" if p.device == "cuda" else "cpu")
        if p.device == "cuda" and not wp.is_cuda_available():
            raise RuntimeError("no CUDA GPU for the Newton cloth ([newton] device = \"cuda\")")
        self.grasp_model = cc.grasp_model
        self.full_surface = p.full_surface_contact and self.device.is_cuda   # SDFs need CUDA
        g, J, self.X0 = S.external_garment(cc, "newton")
        self.J, self.garment = J, g
        self.faces = J["faces"].astype(np.int32)
        N = self.N = len(self.X0)
        nF = len(self.faces) // 2
        self.info = C.ClothInfo(name="jeans", rest_uv=J["uv"], faces=self.faces,
                                face_layer=np.repeat([0, 1], nF), texcoord=J["tc"],
                                face_tc=J["face_tc"], garment=g)
        self.info.nvert = N
        self.material_id = m.material("denim").id

        # ---- the robot's colliders: one kinematic body per MuJoCo collision geom
        self.geoms = np.array([gi for gi in range(m.ngeom)
                               if (m.geom_contype[gi] or m.geom_conaffinity[gi])
                               and (m.body(m.geom_bodyid[gi]).name or "").startswith(("left_", "right_"))])
        side = [S.SIDES.index(m.body(m.geom_bodyid[gi]).name.split("_")[0]) for gi in self.geoms]
        pads = [("_pad" in (m.geom(gi).name or "") or "pad" in m.body(m.geom_bodyid[gi]).name) for gi in self.geoms]
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        mu_shape = lambda mu: mu * mu / p.cloth_friction      # pair mu = sqrt(cloth mu x shape mu)
        cfg_robot = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu_shape(p.gripper_friction),
                                                    ke=p.contact_ke, kd=p.contact_kd)
        cfg_pad = cfg_robot.copy()
        cfg_pad.mu = mu_shape(p.pad_friction)
        shape_side = []
        # the fingers (every link below the gripper base): while they pinch, they stop touching the
        # fabric (RoboSimStudio's EntityHeldCloth: the held patch rides the hand, and fingers left
        # colliding slice and drag the free fabric next to it: 1.5 kN contact forces and 80 %
        # stretch here before this)
        finger = [any(t in m.body(m.geom_bodyid[gi]).name for t in
                      ("driver", "coupler", "spring_link", "follower", "pad")) for gi in self.geoms]
        self._finger_shapes = {s: [] for s in S.SIDES}
        for gi, k, pad, fg in zip(self.geoms, side, pads, finger):
            if fg:
                self._finger_shapes[S.SIDES[k]].append(len(shape_side))
            body = b.add_body(mass=0.0, is_kinematic=True, label=f"mj_geom_{gi}")
            cfg = cfg_pad if pad else cfg_robot
            t, sz = int(m.geom_type[gi]), m.geom_size[gi]
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                b.add_shape_box(body, hx=sz[0], hy=sz[1], hz=sz[2], cfg=cfg)
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                b.add_shape_capsule(body, radius=sz[0], half_height=sz[1], cfg=cfg)
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                b.add_shape_cylinder(body, radius=sz[0], half_height=sz[1], cfg=cfg)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                b.add_shape_sphere(body, radius=sz[0], cfg=cfg)
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                V, F = _hull(m, m.geom_dataid[gi])
                mesh = newton.Mesh(V.astype(np.float32), F.ravel().astype(np.int32), compute_inertia=False)
                if self.full_surface:               # edge / face contact samples the shape's SDF
                    mesh.build_sdf(device=self.device, target_voxel_size=0.001)
                b.add_shape_mesh(body, mesh=mesh, cfg=cfg)
            else:
                raise ValueError(f"collider geom type {t} not supported")
            shape_side.append(k)
        self.n_bodies = len(self.geoms)
        # the table (static), with its own friction
        cfg_table = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu_shape(p.table_friction),
                                                    ke=p.contact_ke, kd=p.contact_kd)
        tg = m.geom("table_top").id if _has_geom(m, "table_top") else None
        if tg is not None:
            hx, hy, hz = m.geom_size[tg]
            tpos = m.body_pos[m.geom_bodyid[tg]] + m.geom_pos[tg]     # the table body is not rotated
        else:
            hx, hy, hz, tpos = 0.8, 0.9, 0.02, np.array([0.45, 0.0, S.TABLE_Z - 0.02])
        b.add_shape_box(-1, xform=wp.transform(wp.vec3(*map(float, tpos)), wp.quat_identity()),
                        hx=float(hx), hy=float(hy), hz=float(hz), cfg=cfg_table)
        shape_side.append(FREE_SIDE)
        # the garment: mass lumped by area, so the jeans weigh garment.mass
        F = self.faces
        X = self.X0
        area = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1).sum()
        self.areal_density = g.mass / area
        b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                         vertices=[wp.vec3(*map(float, v)) for v in X], indices=F.ravel().tolist(),
                         density=float(self.areal_density), tri_ke=p.tri_ke, tri_ka=p.tri_ka, tri_kd=p.tri_kd,
                         edge_ke=p.edge_ke, edge_kd=p.edge_kd, particle_radius=p.particle_radius,
                         label="jeans")
        b.color()
        nm = self.nm = b.finalize(device=self.device)
        nm.soft_contact_ke, nm.soft_contact_kd, nm.soft_contact_mu = p.contact_ke, p.contact_kd, p.cloth_friction
        self.p0 = nm.particle_count - N                      # first garment particle (0: only cloth)
        SB = newton.solvers.SolverBase
        self.solver = newton.solvers.SolverVBD(
            nm, iterations=p.iterations, integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=p.self_contact,
            particle_self_contact_margin=p.self_contact_margin, particle_self_contact_gap=p.self_contact_gap,
            collision_frequency_type={SB.CollisionSlot.SOFT_SELF_CONTACT: SB.CollisionFrequencyType.PRE_INIT})
        self.pipeline = newton.CollisionPipeline(nm, soft_contact_gap=p.body_contact_gap,
                                                 enable_rigid_soft_full_surface_contact=self.full_surface)
        self.contacts = self.pipeline.contacts()
        self.obs = None
        try:
            self.obs = self.solver.observables([self.solver.ObservableKind.CONTACT_F], contacts=self.contacts)
        except Exception:                                    # no contact forces (e.g. on the CPU)
            self.obs = None
        self.s0, self.s1 = nm.state(), nm.state()
        self.control = nm.control()
        # per-frame inputs (device buffers the captured graph reads)
        self._pose0 = wp.zeros(self.n_bodies, dtype=wp.transform, device=self.device)
        self._pose1 = wp.zeros(self.n_bodies, dtype=wp.transform, device=self.device)
        self._grip0 = wp.zeros(2, dtype=wp.transform, device=self.device)
        self._grip1 = wp.zeros(2, dtype=wp.transform, device=self.device)
        self._held = wp.zeros((2, self.max_held), dtype=wp.int32, device=self.device)
        self._local = wp.zeros((2, self.max_held), dtype=wp.vec3, device=self.device)
        self._count = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._force = wp.zeros(2, dtype=wp.vec3, device=self.device)
        self._teth_j = wp.full((nm.particle_count, 2), -1, dtype=wp.int32, device=self.device)
        self._teth_L = wp.zeros((nm.particle_count, 2), dtype=wp.float32, device=self.device)
        self._teth_np = (np.full((nm.particle_count, 2), -1, np.int32), np.zeros((nm.particle_count, 2), np.float32))
        self._shape_side = wp.array(np.array(shape_side + [FREE_SIDE] * (nm.shape_count - len(shape_side)),
                                             np.int32), dtype=wp.int32, device=self.device)
        self._mass0 = nm.particle_mass.numpy().copy()
        self._inv0 = nm.particle_inv_mass.numpy().copy()
        self._graph = None
        self.frame_dt = 1.0 / p.fps
        self.dt = self.frame_dt / p.substeps

        self.grip_body = [m.body(f"{s}_gripper_base").id for s in S.SIDES]
        self.site = {s: m.site(f"{s}_gripper_pinch").id for s in S.SIDES}
        self.driver_q = {s: m.jnt_qposadr[m.joint(f"{s}_gripper_right_driver_joint").id] for s in S.SIDES}
        self._held_idx = {s: [] for s in S.SIDES}
        self._grasp = {s: {"armed": True, "stall": 0.0, "prev": 0.0, "gap0": 0.0} for s in S.SIDES}
        self._verts = None
        self._t_next = 0.0
        self.render_mesh = C.ClothRenderMesh(m, self.faces)
        self.rest_threads = rest_threads(J, self.X0)
        self._frame = 0
        self._synced: dict[int, int] = {}
        self.contact_force = np.zeros((2, 3))
        self.versions = {"newton": newton.__version__, "warp": wp.__version__}

    # ------------------------------------------------------------ simulation
    def _poses(self, d: mujoco.MjData):
        """Collider and gripper poses now, as (n, 7) transforms (x y z qx qy qz qw)."""
        R = d.geom_xmat[self.geoms].reshape(-1, 3, 3)
        geoms = np.c_[d.geom_xpos[self.geoms], _quat_xyzw(R)].astype(np.float32)
        gR = d.xmat[self.grip_body].reshape(-1, 3, 3)
        grips = np.c_[d.xpos[self.grip_body], _quat_xyzw(gR)].astype(np.float32)
        return geoms, grips

    def _simulate_frame(self):
        p = self.p
        n = p.substeps
        s0, s1 = self.s0, self.s1
        wp.launch(_zero3, dim=2, inputs=[self._force], device=self.device)
        if p.self_contact:
            self.solver.rebuild_bvh(s0)
        for k in range(n):
            wp.launch(_set_bodies, dim=self.n_bodies, device=self.device,
                      inputs=[self._pose0, self._pose1, k / n, (k + 1) / n, 1.0 / self.dt],
                      outputs=[s0.body_q, s1.body_q, s1.body_qd])
            wp.launch(_hold, dim=(2, self.max_held), device=self.device,
                      inputs=[self._grip0, self._grip1, (k + 1) / n, self._held, self._local, self._count],
                      outputs=[s0.particle_q, s0.particle_qd])
            s0.clear_forces()
            self.pipeline.collide(s0, self.contacts)
            if self.obs is not None:
                self.solver.step(s0, s1, self.control, self.contacts, self.dt, observables=self.obs)
            else:
                self.solver.step(s0, s1, self.control, self.contacts, self.dt)
            wp.launch(_tethers, dim=self.nm.particle_count, device=self.device,
                      inputs=[self._teth_j, self._teth_L, self.nm.particle_inv_mass, 1.0 / self.dt],
                      outputs=[s1.particle_q, s1.particle_qd])
            s0, s1 = s1, s0
        if self.obs is not None:
            c = self.contacts
            wp.launch(_sum_forces, dim=c.soft_contact_max, device=self.device,
                      inputs=[c.soft_contact_count, c.soft_contact_shape, self.obs.contact_f,
                              c.rigid_contact_max, self._shape_side], outputs=[self._force])
        if n % 2:                                       # the next frame starts from self.s0 again
            wp.copy(self.s0.particle_q, s0.particle_q)
            wp.copy(self.s0.particle_qd, s0.particle_qd)
            wp.copy(self.s0.body_q, s0.body_q)

    def _run_frame(self):
        if self.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self._simulate_frame()
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        else:
            self._simulate_frame()

    def _set_frame_poses(self, d, first=False):
        geoms, grips = self._poses(d)
        if first:
            self._pose1.assign(geoms)
            self._grip1.assign(grips)
        wp.copy(self._pose0, self._pose1)
        wp.copy(self._grip0, self._grip1)
        self._pose1.assign(geoms)
        self._grip1.assign(grips)

    def step(self, d):
        if d.time < self._t_next - 1e-9:
            return
        self._t_next += self.frame_dt
        if self._t_next < d.time:                        # fell far behind (e.g. after a pause)
            self._t_next = d.time + self.frame_dt
        self._set_frame_poses(d)
        self._run_frame()
        self._verts = None
        self._force_np = None
        self._frame += 1

    def settle(self, d, frames: int):
        """Simulate `frames` frames with the robot standing still (scene construction, not time)."""
        self._set_frame_poses(d, first=True)
        for _ in range(frames):
            self._set_frame_poses(d)
            self._run_frame()
        self._verts = None

    # ------------------------------------------------------------ ClothEngine
    def reset(self, d, rng=None, randomize: float = 0.0, cloth_init: str = "flat"):
        self._release_all()
        X, drop = initial_cloth(self.X0, rng, randomize, cloth_init)
        self._load_state(X)
        self.settle(d, int(round(1.6 * self.p.fps)) if drop else self.p.settle_frames)
        self._grasp = {s: {"armed": True, "stall": 0.0, "prev": 0.0, "gap0": 0.0} for s in S.SIDES}
        self._frame += 1
        self._t_next = d.time + self.frame_dt

    def _load_state(self, X):
        q = self.s0.particle_q.numpy()
        q[self.p0:self.p0 + self.N] = X
        for s in (self.s0, self.s1):
            s.particle_q.assign(q)
            s.particle_qd.zero_()
        self._verts = None

    def verts(self, d):
        if self._verts is None:
            self._verts = self.s0.particle_q.numpy()[self.p0:self.p0 + self.N]
        return self._verts

    def version(self, d):
        return self._frame

    def sync_render(self, renderer, d):
        if self._synced.get(id(renderer)) != self._frame:
            self.render_mesh.update(self.model, d, self.verts(d), renderer)
            self._synced[id(renderer)] = self._frame

    def held(self, side):
        return self._held_idx[side]

    def metrics(self, d):
        out = super().metrics(d)
        f = self.forces()
        out.update({f"force_{s}": float(np.linalg.norm(f[k])) for k, s in enumerate(S.SIDES)})
        return out

    def forces(self) -> np.ndarray:
        """(2, 3) contact force of the cloth on the left / right arm [N, world], last frame."""
        if getattr(self, "_force_np", None) is None:
            self._force_np = self._force.numpy().astype(np.float64)
        return self._force_np

    def settings(self) -> dict:
        return {"engine": self.name, **self.versions, "device": str(self.device),
                "areal_density": float(self.areal_density), "particles": self.N,
                "newton": dataclasses.asdict(self.p), "grasp_model": self.grasp_model}

    # ------------------------------------------------------------ grasp
    def update(self, d, grippers, dt):
        P = self.P
        for k, s in enumerate(S.SIDES):
            st = self._grasp[s]
            jaw = float(np.clip(d.qpos[self.driver_q[s]] / 0.8, 0, 1))
            closing = grippers[s] >= P.CMD_CLOSE
            if self.grasp_model == "friction":
                # nothing is attached; report what lies between closed pads (HUD, recording)
                # (anchors of the tethers, which only stop the fabric overstretching: they hold nothing)
                if closing and jaw > P.JAW_STALL and self._frame - st.get("checked", -99) >= 6:
                    st["checked"] = self._frame
                    idx = [int(i) for i in self._between_pads(d, s, jaw)]
                    if idx != self._held_idx[s]:
                        self._set_tethers(k, idx)
                    self._held_idx[s] = idx
                elif (not closing or jaw <= P.JAW_STALL) and self._held_idx[s]:
                    self._held_idx[s] = []
                    self._set_tethers(k, [])
                continue
            if not closing or jaw < P.JAW_RELEASE:
                if self._held_idx[s]:
                    self._release(k, s)
                if not closing:
                    st["armed"] = True
                st["stall"] = 0.0
            elif self._held_idx[s]:
                gap = P.pad_gap(jaw)
                st["gap0"] = min(st["gap0"], gap)
                if gap > st["gap0"] + self.cfg.release_gap:
                    self._release(k, s)
            elif st["armed"]:
                stalled = jaw > P.JAW_STALL and abs(jaw - st["prev"]) < 0.02 * dt / 0.01
                st["stall"] = st["stall"] + dt if stalled else 0.0
                if jaw >= P.JAW_TRIGGER or st["stall"] > 0.12:
                    st["armed"] = False
                    self._try_grasp(d, k, s, jaw)
            st["prev"] = jaw

    def _between_pads(self, d, s, jaw):
        P = self.P
        V = self.verts(d)
        R = d.site_xmat[self.site[s]].reshape(3, 3)
        loc = (V - d.site_xpos[self.site[s]]) @ R
        half_gap = P.pad_gap(jaw) / 2 + 0.001
        ok = ((np.abs(loc[:, 0]) <= P.HALF_WIDTH) & (np.abs(loc[:, 1]) <= half_gap)
              & (loc[:, 2] >= P.Z_RANGE[0]) & (loc[:, 2] <= P.Z_RANGE[1]))
        idx = np.where(ok)[0]
        centre = np.array([0.0, 0.0, 0.006])
        return idx[np.argsort(np.linalg.norm(loc[idx] - centre, axis=1))][:self.max_held]

    def _try_grasp(self, d, k, s, jaw):
        idx = self._between_pads(d, s, jaw)
        if len(idx) == 0:
            return
        V = self.verts(d)
        b = self.grip_body[k]
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        held = self._held.numpy()
        local = self._local.numpy()
        held[k, :len(idx)] = idx + self.p0
        local[k, :len(idx)] = (V[idx] - pb) @ Rb
        self._held.assign(held)
        self._local.assign(local)
        cnt = self._count.numpy()
        cnt[k] = len(idx)
        self._count.assign(cnt)
        self._set_kinematic(idx, True)
        self._fingers_collide(s, False)
        self._set_tethers(k, idx)
        self._held_idx[s] = [int(i) for i in idx]
        self._grasp[s]["gap0"] = self.P.pad_gap(jaw)

    def _set_kinematic(self, idx, on: bool):
        gi = np.asarray(idx, int) + self.p0
        mass = self.nm.particle_mass.numpy()
        inv = self.nm.particle_inv_mass.numpy()
        mass[gi] = 0.0 if on else self._mass0[gi]
        inv[gi] = 0.0 if on else self._inv0[gi]
        self.nm.particle_mass.assign(mass)
        self.nm.particle_inv_mass.assign(inv)

    def _fingers_collide(self, s, on: bool):
        bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags = self.nm.shape_flags.numpy()
        idx = self._finger_shapes[s]
        flags[idx] = (flags[idx] | bit) if on else (flags[idx] & ~bit)
        self.nm.shape_flags.assign(flags)

    TETHER_SLACK = 0.02             # a tether is 2 % (+ 1 mm) longer than the fabric it spans

    def _set_tethers(self, k: int, idx):
        """Tether every point to its nearest anchor (`idx`, garment indices) of the same panel, at
        the pattern distance between them (as gpu_cloth.GPUJeans._set_tethers): a pinch of the top
        layer only must not lift the layer below through a shortcut."""
        uv, panel = self.J["uv"], self.J["layer"]
        j = np.full(self.N, -1, np.int32)
        L = np.zeros(self.N, np.float32)
        idx = np.asarray(idx, int)
        if len(idx):
            for pnl in (0, 1):
                tgt = idx[(panel[idx] == pnl) | (panel[idx] == 2)]
                pts = np.where((panel == pnl) | (panel == 2))[0]
                if len(tgt) == 0:
                    continue
                D = np.linalg.norm(uv[pts, None, :] - uv[None, tgt, :], axis=2)
                a = np.argmin(D, axis=1)
                dist = D[np.arange(len(pts)), a]
                better = (j[pts] < 0) | (dist < L[pts])
                j[pts[better]] = tgt[a[better]]
                L[pts[better]] = dist[better]
            L = L * (1 + self.TETHER_SLACK) + 0.001
        tj, tl = self._teth_np
        tj[:, k], tl[:, k] = -1, 0.0
        sel = slice(self.p0, self.p0 + self.N)
        tj[sel, k] = np.where(j >= 0, j + self.p0, -1)
        tl[sel, k] = L
        self._teth_j.assign(tj)
        self._teth_L.assign(tl)

    def _release(self, k, s):
        if self._held_idx[s]:
            self._set_kinematic(self._held_idx[s], False)
            self._fingers_collide(s, True)
            self._set_tethers(k, [])
        cnt = self._count.numpy()
        cnt[k] = 0
        self._count.assign(cnt)
        self._held_idx[s] = []

    def _release_all(self):
        for k, s in enumerate(S.SIDES):
            self._release(k, s)


def _has_geom(m, name):
    try:
        m.geom(name)
        return True
    except KeyError:
        return False
