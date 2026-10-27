"""Builds the bimanual UR5e + Robotiq 2F-85 MuJoCo scene with MjSpec.

World frame (MuJoCo, Z-up):
    +x  forward (away from the operator, across the table)
    +y  operator's left
    +z  up
The two arms are mounted on the back edge of the table at y = +/-ARM_Y.

Tasks (select with `--task`):
    jeans   fold a pair of denim jeans (deformable flex cloth)          [default]
    blocks  pick three cubes and a cylinder and drop them into a bin

The surroundings (room, wood floor, laminate lab table, lights with shadows) are the
same for every task; textures are procedural (see textures.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import cloth as C
from . import cloth_config as CC
from . import textures as T

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
UR5E_XML = ASSETS / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = ASSETS / "robotiq_2f85" / "2f85.xml"

TABLE_Z = 0.75          # table top height [m]
TABLE_CENTER = (0.4, 0.0)
TABLE_HALF = (0.55, 0.75)
ARM_Y = 0.36            # lateral offset of each arm base [m]
ARM_X = 0.0             # arm bases sit on the back edge of the table
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
# Joint-space home poses: elbow up, gripper pointing down, pinch point at
# (0.45, +/-0.22, 0.95), i.e. 20 cm above the table. Chosen so that, seen from the
# first-person eye point (HEAD_POS), the elbows stay out of the centre of view.
# Pan differs per side because the UR5e shoulder is offset laterally; wrist_3 = pan
# keeps both grippers' fingers aligned with world x.
HOME_Q = {
    "left":  np.array([-0.59, -1.612, 1.954, -1.912, -1.5708, -0.59]),
    "right": np.array([0.013, -1.612, 1.954, -1.912, -1.5708, 0.013]),
}
SIDES = ("left", "right")
# First-person eye point: centred between the arm bases, slightly behind and above them.
# Keep in sync with OPERATOR_EYE_MJ in web/main.js.
HEAD_POS = [-0.30, 0.0, TABLE_Z + 0.77]

TASKS = {
    "jeans": "Fold the jeans",
    "blocks": "Put the blocks in the bin",
}
DEFAULT_TASK = "jeans"

# --- blocks task: manipulable objects: name -> (type, size, rgba, nominal xy)
OBJECTS = {
    "cube_red":   ("box", [0.025, 0.025, 0.025], [0.85, 0.2, 0.2, 1], (0.50, 0.18)),
    "cube_green": ("box", [0.025, 0.025, 0.025], [0.2, 0.75, 0.3, 1], (0.45, 0.00)),
    "cube_blue":  ("box", [0.025, 0.025, 0.025], [0.2, 0.4, 0.9, 1], (0.50, -0.18)),
    "cylinder":   ("cylinder", [0.022, 0.05, 0], [0.95, 0.75, 0.2, 1], (0.62, 0.08)),
}
BIN_POS = (0.66, -0.12)

# --- jeans task: waistband centre (x, y) and direction of the legs (yaw; -pi/2 = toward -y).
# The jeans lie front side up across the table, waistband on the operator's left, hems on
# the right; the left arm reaches the waistband, the right arm the hems.
JEANS_POS = (0.50, 0.50)
JEANS_YAW = -np.pi / 2
# Lowest allowed pinch-point height per task (teleop workspace floor). For cloth the
# fingertips must be able to reach the table: the 2F-85 tips swing ~18 mm below the
# pinch point when the jaws close.
PINCH_Z_MIN = {"jeans": TABLE_Z + 0.016, "blocks": TABLE_Z + 0.02}

ROOM_HALF = (2.6, 2.6)
ROOM_HEIGHT = 2.8


@dataclass
class SceneInfo:
    model: mujoco.MjModel
    xml: str
    task: str = DEFAULT_TASK
    objects: dict = field(default_factory=dict)         # rigid task objects (name -> spec tuple)
    cloth: C.ClothInfo | None = None
    cloth_config: CC.ClothConfig | None = None           # garment + cloth simulation settings
    cloth_engine: str | None = None                      # "mujoco" (flex in this model), "gpu" or "newton"
    grasp_eq: dict = field(default_factory=dict)        # side -> [eq ids] (cloth pinch)
    web_textures: dict = field(default_factory=dict)    # extra textures for the web client
    arm_qpos_adr: dict = field(default_factory=dict)   # side -> (6,) qpos indices
    arm_dof_adr: dict = field(default_factory=dict)    # side -> (6,) dof indices
    arm_act: dict = field(default_factory=dict)        # side -> (6,) actuator ids
    grip_act: dict = field(default_factory=dict)       # side -> actuator id
    grip_qpos_adr: dict = field(default_factory=dict)  # side -> driver joint qpos index
    ee_site: dict = field(default_factory=dict)        # side -> site id (gripper pinch)
    obj_qpos_adr: dict = field(default_factory=dict)   # object -> qpos start (7)
    cameras: list = field(default_factory=list)


def _add_box(body, name, pos, size, rgba=(1, 1, 1, 1), collide=True, material=None, group=0):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = pos
    g.size = size
    g.rgba = rgba
    g.group = group
    if material:
        g.material = material
    if not collide:
        g.contype = 0
        g.conaffinity = 0
    return g


def _add_environment(spec: mujoco.MjSpec):
    """Room, floor, lab table and lights. Identical for every task."""
    wb = spec.worldbody
    T.add_texture(spec, "wood_floor", T.wood_floor())
    T.add_texture(spec, "laminate", T.table_laminate())
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.86, 0.88, 0.9]
    sky.rgb2 = [0.55, 0.58, 0.62]
    sky.width = sky.height = 256
    # 0.8 m wide plank tile, repeated per metre (texuniform)
    T.add_material(spec, "floor_wood", "wood_floor", texrepeat=(1.25, 1.25), texuniform=True,
                   specular=0.25, shininess=0.4, reflectance=0.04, roughness=0.55)
    T.add_material(spec, "table_laminate", "laminate", texrepeat=(1.5, 1.5), texuniform=True,
                   specular=0.35, shininess=0.5, reflectance=0.03, roughness=0.45)
    T.add_material(spec, "aluminium", rgba=(0.72, 0.74, 0.76, 1), specular=0.7, shininess=0.7,
                   roughness=0.35, metallic=0.9)
    T.add_material(spec, "wall_paint", rgba=(0.86, 0.85, 0.82, 1), specular=0.05, shininess=0.05,
                   roughness=0.9)
    T.add_material(spec, "skirting", rgba=(0.93, 0.93, 0.92, 1), specular=0.2, roughness=0.6)
    T.add_material(spec, "light_panel", rgba=(1, 1, 0.97, 1), specular=0, roughness=1)
    spec.material("light_panel").emission = 1.0

    # Lights: a large overhead fixture (soft shadows) + two fill lights
    key = wb.add_light()
    key.name = "ceiling"
    key.pos = [0.45, 0.1, 2.6]
    key.dir = [0, 0, -1]
    key.diffuse = [0.72, 0.71, 0.68]
    key.specular = [0.25, 0.25, 0.25]
    key.castshadow = True
    key.cutoff = 70
    key.exponent = 2
    for pos in ([-0.8, 1.2, 2.2], [1.6, -1.0, 2.2]):
        light = wb.add_light()
        light.pos = pos
        light.dir = (np.array([0.45, 0, TABLE_Z]) - pos) / np.linalg.norm(np.array([0.45, 0, TABLE_Z]) - pos)
        light.diffuse = [0.22, 0.22, 0.24]
        light.specular = [0.05, 0.05, 0.05]
        light.castshadow = False
    spec.visual.headlight.ambient = [0.22, 0.22, 0.23]
    spec.visual.headlight.diffuse = [0.18, 0.18, 0.18]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8

    # Floor + room
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [ROOM_HALF[0], ROOM_HALF[1], 0.05]
    floor.material = "floor_wood"
    room = wb.add_body()
    room.name = "room"
    rx, ry, rh = ROOM_HALF[0], ROOM_HALF[1], ROOM_HEIGHT
    for name, pos, size in (("wall_front", [rx, 0, rh / 2], [0.02, ry, rh / 2]),
                            ("wall_back", [-rx, 0, rh / 2], [0.02, ry, rh / 2]),
                            ("wall_left", [0, ry, rh / 2], [rx, 0.02, rh / 2]),
                            ("wall_right", [0, -ry, rh / 2], [rx, 0.02, rh / 2])):
        _add_box(room, name, pos, size, collide=False, material="wall_paint", group=1)
    for name, pos, size in (("skirt_front", [rx - 0.03, 0, 0.05], [0.01, ry, 0.05]),
                            ("skirt_back", [-rx + 0.03, 0, 0.05], [0.01, ry, 0.05]),
                            ("skirt_left", [0, ry - 0.03, 0.05], [rx, 0.01, 0.05]),
                            ("skirt_right", [0, -ry + 0.03, 0.05], [rx, 0.01, 0.05])):
        _add_box(room, name, pos, size, collide=False, material="skirting", group=1)
    _add_box(room, "ceiling_light", [0.45, 0.1, rh - 0.02], [0.6, 0.3, 0.01], collide=False,
             material="light_panel", group=1)

    # Lab table: laminate top on an aluminium frame
    table = wb.add_body()
    table.name = "table"
    tx, ty = TABLE_HALF
    table.pos = [TABLE_CENTER[0], TABLE_CENTER[1], TABLE_Z]
    top = _add_box(table, "table_top", [0, 0, -0.02], [tx, ty, 0.02], material="table_laminate")
    top.friction = [0.4, 0.005, 0.0001]          # laminate: low friction for fabric
    _add_box(table, "table_edge", [0, 0, -0.045], [tx - 0.01, ty - 0.01, 0.005], collide=False,
             material="aluminium", group=1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            _add_box(table, f"table_leg_{sx}{sy}", [sx * (tx - 0.05), sy * (ty - 0.05), -TABLE_Z / 2],
                     [0.022, 0.022, TABLE_Z / 2 - 0.02], collide=False, material="aluminium")
    for sy in (-1, 1):      # stretchers between the legs
        _add_box(table, f"table_rail_{sy}", [0, sy * (ty - 0.05), -TABLE_Z + 0.15],
                 [tx - 0.05, 0.015, 0.015], collide=False, material="aluminium")


def _add_blocks(spec: mujoco.MjSpec):
    wb = spec.worldbody
    bx, by = BIN_POS
    tray = wb.add_body()
    tray.name = "bin"
    tray.pos = [bx, by, TABLE_Z]
    col = [0.25, 0.25, 0.28, 1]
    _add_box(tray, "bin_floor", [0, 0, 0.005], [0.09, 0.09, 0.005], col)
    for i, (px, py, sx, sy) in enumerate([(0.09, 0, 0.005, 0.09), (-0.09, 0, 0.005, 0.09),
                                          (0, 0.09, 0.09, 0.005), (0, -0.09, 0.09, 0.005)]):
        _add_box(tray, f"bin_wall{i}", [px, py, 0.035], [sx, sy, 0.035], col)
    for name, (gtype, size, rgba, (x, y)) in OBJECTS.items():
        b = wb.add_body()
        b.name = name
        b.pos = [x, y, TABLE_Z + size[1 if gtype == "cylinder" else 2] + 0.002]
        j = b.add_freejoint()
        j.name = f"{name}_joint"
        g = b.add_geom()
        g.name = f"{name}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX if gtype == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = size
        g.rgba = rgba
        g.mass = 0.08
        g.friction = [1.2, 0.01, 0.001]
        g.condim = 4


def _add_denim_material(spec: mujoco.MjSpec, g: C.GarmentSpec):
    front, back = C.denim_textures(g)
    # texture atlas: front panel on the left half, back panel on the right half
    atlas = np.concatenate([T.decode_png(front), T.decode_png(back)], axis=1)
    T.add_texture(spec, "denim_atlas", atlas)
    T.add_material(spec, "denim", "denim_atlas", specular=0.08, shininess=0.1, roughness=0.95)


def _add_jeans(spec: mujoco.MjSpec, cfg: CC.ClothConfig):
    g = cfg.garment
    _add_denim_material(spec, g)
    x, y = JEANS_POS
    info = C.add_garment(spec, "jeans", [x, y, TABLE_Z + g.sphere_r + 0.0003], JEANS_YAW, g,
                         material="denim")
    # Every collision geom of the grippers (pads, finger links, palm) touches the continuous cloth
    # surface, not only the vertex spheres: those are 5 cm apart, and a finger moved sideways
    # would slip between them, through the fabric, and the jeans would hang on it.
    # Silicone finger pads: mu ~ 1 on fabric (grasp.pad_friction).
    for b in spec.bodies:
        if "_gripper_" not in b.name:
            continue
        for gm in b.geoms:
            if gm.contype or gm.conaffinity:
                gm.conaffinity = gm.conaffinity | C.FLEX_CONTYPE
            if gm.name.endswith(("_pad1", "_pad2")):
                gm.friction = [cfg.pad_friction, 0.01, 0.001]
    for side in SIDES:
        C.add_grasp_constraints(spec, info, side)
    return info, {}


def build_spec(task: str = DEFAULT_TASK, cloth: CC.ClothConfig | None = None,
               cloth_engine: str = "mujoco"):
    """Returns (spec, cloth_info or None, extra web textures). `cloth`: garment and cloth
    simulation settings (default: config/cloth.toml, see cloth_config). With a cloth engine other
    than "mujoco" the jeans are simulated outside the model: it only gets their material."""
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}', choose from {list(TASKS)}")
    spec = mujoco.MjSpec()
    spec.modelname = f"dual_ur5e_{task}"
    spec.option.timestep = 0.002
    if task == "jeans":
        cloth = cloth or CC.load()
        # flex bending elasticity is integrated implicitly only by the discrete integrator;
        # pyramidal cones keep the ~300 cloth contacts cheap enough for real time
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        # Timestep (config: simulation.timestep, 4 ms): pays for the full cloth collision model
        # at the quality of 3 ms, also for a crumpled heap; 5 ms degrades a heap, 6 ms is unstable.
        spec.option.timestep = cloth.timestep
        # Conjugate-gradient solver with a fixed iteration budget: the two-layer garment makes
        # Newton's matrix factorisation ~1.6x slower; CG keeps a flat per-step cost (no
        # frame-time spikes in VR). The budget (simulation.solver_iterations, 30) matters: at 24
        # CG stops short while a leg hangs from the grippers and the threads stretch up to ~17 %
        # (rubbery jeans); 30 keeps them at < 1.2 % for 99 % of the edges.
        # (scripts/cloth_bench.py measures this.)
        spec.option.solver = mujoco.mjtSolver.mjSOL_CG
        spec.option.iterations = cloth.solver_iterations
        spec.option.tolerance = 1e-6
    else:
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    wb = spec.worldbody

    _add_environment(spec)

    # Scene cameras
    cam = wb.add_camera()
    cam.name = "overhead"
    cam.pos = [0.45, 0, TABLE_Z + 1.1]
    cam.quat = _look_quat([0.45, 0, TABLE_Z + 1.1], [0.45, 0, TABLE_Z], up=[1, 0, 0])
    cam.fovy = 60
    # Egocentric "robot head" camera = where the operator's eyes are in first-person mode
    cam = wb.add_camera()
    cam.name = "head_cam"
    cam.pos = HEAD_POS
    cam.quat = _look_quat(HEAD_POS, [0.45, 0, TABLE_Z])
    cam.fovy = 70
    cam = wb.add_camera()
    cam.name = "front"
    cam.pos = [1.45, 0, TABLE_Z + 0.55]
    cam.quat = _look_quat(cam.pos, [0.35, 0, TABLE_Z + 0.1])
    cam.fovy = 55

    # Arms + grippers
    for side, sign in (("left", 1), ("right", -1)):
        arm = mujoco.MjSpec.from_file(str(UR5E_XML))
        grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
        # Drop lights and keyframes from the child models
        for l in list(arm.lights):
            arm.delete(l)
        for k in list(arm.keys):
            arm.delete(k)
        # Wrist camera on the gripper, looking along the tool axis
        gbase = grip.body("base")
        wc = gbase.add_camera()
        wc.name = "wrist_cam"
        wc.pos = [0.09, 0, 0.02]
        wc.quat = _look_quat([0.09, 0, 0.02], [0, 0, 0.22], up=[1, 0, 0])
        wc.fovy = 75
        arm.option.cone = spec.option.cone           # avoid attach-conflict warnings
        arm.option.impratio = spec.option.impratio
        arm.option.integrator = spec.option.integrator
        grip.option.cone = spec.option.cone
        grip.option.impratio = spec.option.impratio
        grip.option.integrator = spec.option.integrator
        arm.site("attachment_site").attach_body(grip.body("base_mount"), "gripper_", "")
        frame = wb.add_frame()
        frame.pos = [ARM_X, sign * ARM_Y, TABLE_Z]
        frame.quat = [0, 0, 0, 1]   # yaw 180deg so the home pose reaches toward +x
        spec.attach(arm, prefix=f"{side}_", frame=frame)

    # Gravity compensation on all robot bodies so the position servos track precisely
    for b in spec.bodies:
        if b.name.startswith(("left_", "right_")):
            b.gravcomp = 1.0

    # Halve the servo damping of the arm joints: the Menagerie values (kv = kp/5)
    # give ~0.2 s lag, which feels sluggish in VR. kv = kp/10 is still overdamped.
    for a in spec.actuators:
        if any(a.name.endswith(n) for n in ARM_ACTS):
            bp = np.array(a.biasprm)
            bp[2] *= 0.5
            a.biasprm = bp

    cloth_info, web_tex = None, {}
    if task == "blocks":
        _add_blocks(spec)
    elif task == "jeans" and cloth_engine == "mujoco":
        cloth_info, web_tex = _add_jeans(spec, cloth)
    elif task == "jeans":
        # simulated outside the model: only its material, and a render-only mesh the engine
        # updates before MuJoCo renders (camera screens, recorded images, videos)
        _add_denim_material(spec, cloth.garment)
        _, J, X = external_garment(cloth, cloth_engine)
        C.ClothRenderMesh.add(spec, X, J, "denim")
    return spec, cloth_info, web_tex


def external_garment(cloth: CC.ClothConfig, cloth_engine: str):
    """The jeans as an engine outside MuJoCo simulates them: (garment, jeans_garment dict, rest
    positions lying flat on the table)."""
    if cloth_engine == "newton":
        p = cloth.newton
        g, J = C.layered_garment(cloth.garment, p.spacing, p.self_contact_margin + 2 * p.particle_radius)
        lift = p.particle_radius + 0.0005
    else:
        g, J, thick = C.fine_garment(cloth.garment, cloth.gpu.spacing, cloth.gpu.thickness)
        lift = 0.5 * thick
    return g, J, C.place_flat(J, JEANS_POS, JEANS_YAW, TABLE_Z, lift)


def _look_quat(pos, target, up=(0, 0, 1)):
    """Quaternion (w,x,y,z) for a MuJoCo camera at pos looking at target.
    MuJoCo cameras look along -z with +y up."""
    pos, target, up = map(lambda v: np.asarray(v, float), (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def build_scene(task: str = DEFAULT_TASK, cloth: CC.ClothConfig | None = None,
                cloth_engine: str = "mujoco") -> SceneInfo:
    if task == "jeans":
        cloth = cloth or CC.load()
    spec, cloth_info, web_tex = build_spec(task, cloth, cloth_engine)
    model = spec.compile()
    info = SceneInfo(model=model, xml=spec.to_xml(), task=task, web_textures=web_tex,
                     cloth_config=cloth, cloth_engine=cloth_engine if task == "jeans" else None)
    for side in SIDES:
        jids = [model.joint(f"{side}_{j}").id for j in ARM_JOINTS]
        info.arm_qpos_adr[side] = np.array([model.jnt_qposadr[j] for j in jids])
        info.arm_dof_adr[side] = np.array([model.jnt_dofadr[j] for j in jids])
        info.arm_act[side] = np.array([model.actuator(f"{side}_{a}").id for a in ARM_ACTS])
        info.grip_act[side] = model.actuator(f"{side}_gripper_fingers_actuator").id
        info.grip_qpos_adr[side] = model.jnt_qposadr[model.joint(f"{side}_gripper_right_driver_joint").id]
        info.ee_site[side] = model.site(f"{side}_gripper_pinch").id
    if task == "blocks":
        info.objects = dict(OBJECTS)
        for name in OBJECTS:
            info.obj_qpos_adr[name] = model.jnt_qposadr[model.joint(f"{name}_joint").id]
    if cloth_info is not None:
        C.bind(cloth_info, model)
        info.cloth = cloth_info
        for side in SIDES:
            info.grasp_eq[side] = [model.equality(f"{side}_pinch_{k}").id for k in range(C.PinchGrasp.K)]
    info.cameras = [model.camera(i).name for i in range(model.ncam)]
    return info


def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def reset(info: SceneInfo, data: mujoco.MjData, rng: np.random.Generator | None = None,
          randomize: float = 0.0, cloth_init: str = "flat"):
    """Reset to home pose; optionally jitter object positions by +/- randomize metres.

    Jeans: `randomize` jitters the position (+/- randomize m) and the heading
    (+/- 3*randomize rad); cloth_init="crumpled" drops the jeans from 30 cm in a random
    orientation and lets them settle into a realistic heap first."""
    m = info.model
    mujoco.mj_resetData(m, data)
    for side in SIDES:
        data.qpos[info.arm_qpos_adr[side]] = HOME_Q[side]
        data.ctrl[info.arm_act[side]] = HOME_Q[side]
        data.ctrl[info.grip_act[side]] = 0.0
    rng = rng or np.random.default_rng()
    for name, (gtype, size, _, (x, y)) in info.objects.items():
        a = info.obj_qpos_adr[name]
        dx, dy = rng.uniform(-randomize, randomize, 2) if randomize > 0 else (0, 0)
        z = TABLE_Z + (size[1] if gtype == "cylinder" else size[2]) + 0.002
        yaw = rng.uniform(-np.pi, np.pi) if randomize > 0 else 0.0
        data.qpos[a:a + 3] = [x + dx, y + dy, z]
        data.qpos[a + 3:a + 7] = _yaw_quat(yaw)
    if info.cloth is not None:
        _reset_cloth(info, data, rng, randomize, cloth_init)
    mujoco.mj_forward(m, data)


def _reset_cloth(info: SceneInfo, data, rng, randomize, cloth_init):
    m, ci = info.model, info.cloth
    for e in info.grasp_eq.values():
        data.eq_active[e] = 0
    rest = C.rest_world(ci, m, data)
    centre = rest.mean(axis=0)
    if cloth_init == "crumpled":
        # hold the jeans up in a random orientation, 30 cm above the table, then drop them
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = rng.uniform(0.6, 1.4)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, ax, ang)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        xyz = (rest - centre) @ R.T * 0.9 + centre + [0, 0, 0.30]
        lo = xyz[:, 2].min()
        if lo < TABLE_Z + 0.02:
            xyz[:, 2] += TABLE_Z + 0.02 - lo
        C.place(ci, m, data, xyz)
        mujoco.mj_forward(m, data)
        for _ in range(int(1.6 / m.opt.timestep)):
            mujoco.mj_step(m, data)
        data.time = 0.0
        data.qvel[:] = 0
        return
    yaw = rng.uniform(-3 * randomize, 3 * randomize) if randomize > 0 else 0.0
    dxy = rng.uniform(-randomize, randomize, 2) if randomize > 0 else np.zeros(2)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    xyz = (rest - centre) @ Rz.T + centre + [dxy[0], dxy[1], 0]
    C.place(ci, m, data, xyz)


if __name__ == "__main__":
    import sys
    import time
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    t0 = time.time()
    info = build_scene(task)
    m = info.model
    d = mujoco.MjData(m)
    reset(info, d)
    print(f"task={task} built in {time.time() - t0:.2f}s  nq={m.nq} nv={m.nv} nu={m.nu} "
          f"nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh} nflex={m.nflex} neq={m.neq}")
    print("cameras:", info.cameras)
    for s in SIDES:
        print(s, "ee", d.site_xpos[info.ee_site[s]].round(3),
              "z-axis", d.site_xmat[info.ee_site[s]].reshape(3, 3)[:, 2].round(2))
    t0 = time.time()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    print(f"2 s of sim in {time.time() - t0:.2f}s wall")
    for s in SIDES:
        print(s, "after 2s ee", d.site_xpos[info.ee_site[s]].round(3),
              "q err", (d.qpos[info.arm_qpos_adr[s]] - HOME_Q[s]).round(4))
    for n in info.objects:
        a = info.obj_qpos_adr[n]
        print(n, d.qpos[a:a + 3].round(3))
    if info.cloth is not None:
        V = C.verts(info.cloth, d)
        print("jeans verts", len(V), "bbox", V.min(0).round(3), V.max(0).round(3))
        print("fold metrics", C.fold_metrics(info.cloth, V, TABLE_Z))
