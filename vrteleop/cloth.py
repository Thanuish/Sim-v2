"""Deformable garments (jeans) for the folding task.

The garment is a MuJoCo flex (dim=2, one body with 3 slide joints per vertex):
  * the flex provides bending stiffness (elastic2d="bend") and the rendered surface;
  * like woven fabric, the threads (warp along the leg, weft across) are inextensible
    (tendon equality constraints) while the bias diagonals are soft springs, so the
    cloth shears and drapes; air drag slows falling fabric;
  * contacts are handled by one small sphere geom per vertex. MuJoCo's native
    flex-geom collisions are capped at 50 contacts per (flex, body) pair and
    produce ~4 contacts per vertex, which is both wrong (a table only holds up
    50 vertices) and far too slow for real-time teleoperation. Vertex spheres give
    one contact per touching vertex, grip well between the 2F-85 pads, and also
    make the fabric collide with itself so folded layers stack instead of merging.

The flat pattern is a pair of jeans laid out front-side up. Coordinates in the
pattern are (u, v): u runs from the waistband (u=0) to the hems (u=length), v runs
across the garment (v<0 = the garment's right leg as it lies on the table).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO

import mujoco
import numpy as np

CLOTH_CONTYPE = 2        # vertex spheres: collide with the world (1) and each other (2)
CLOTH_CONAFFINITY = 3
FLEX_FRONT = 8           # continuous cloth surface of the front panel ...
FLEX_BACK = 16           # ... and of the back panel
FLEX_CONTYPE = FLEX_FRONT | FLEX_BACK   # geoms with these conaffinity bits touch the continuous
                                        # cloth surface (the grippers)
FLEX_SELF = 32           # the whole garment's surface against itself (triangle-triangle)


@dataclass
class GarmentSpec:
    """Size and fabric of a garment (jeans pattern). The values used come from config/cloth.toml
    (see cloth_config); these defaults are a pair of adult jeans laid flat (W32 L32 regular)."""
    length: float = 1.00     # waistband to hem [m]
    rise: float = 0.27       # waistband to crotch
    waist: float = 0.39      # flat width at the waistband
    hip: float = 0.45        # flat width at the crotch line
    thigh: float = 0.225     # width of one leg at the crotch line (<= hip / 2)
    hem: float = 0.19        # width of one leg at the hem
    spacing: float = 0.055   # simulation mesh resolution [m] (two panels -> ~280 vertices; real time on this Mac)
    mass: float = 0.65       # a pair of 12 oz denim jeans weighs ~0.6-0.8 kg
    friction: float = 0.4    # denim on laminate / denim on denim (pads use their own, higher value)
    # bending stiffness: D = E t^3 / 12(1-nu^2) ~ 3e-5 N m, measured values for doubled 12 oz denim
    bend_young: float = 6.0e3  # Young's modulus [Pa] used with `thickness` for the bending energy
    thickness: float = 0.004   # two layers of denim
    sphere_r: float = 0.005    # collision sphere radius per vertex (fabric rests 5 mm above the table)
    # cloth-on-cloth contacts: 1 = frictionless (layers slide on each other). Friction with the
    # table and the finger pads is unaffected. 3 adds denim-on-denim friction at ~20 % more cost.
    self_condim: int = 1
    # Layers must never pass through each other (the pale inside would show). Each vertex sphere
    # touches the continuous surface of both panels (the other panel, and its own panel where it
    # is folded onto itself), and the surface also collides with itself triangle against
    # triangle, which catches edges slicing between the spheres. ~60 % more CPU than the older,
    # faster model (both_way/own_panel/self_collide False, sphere_pairs True), which lets
    # folds and crumples cut through themselves (`--cloth-fast`).
    both_way: bool = True      # both panels' spheres touch the other panel's surface
    own_panel: bool = True     # spheres also touch their own panel's surface (folds onto itself)
    sphere_pairs: bool = False  # vertex spheres also touch each other
    self_collide: bool = True  # triangle-triangle collisions of the whole surface with itself
    # Contacts of the cloth surface start 6 mm before layers touch and push from 3 mm: layers
    # squeezed together fast (a leg compressed between the grippers buckles into sharp folds)
    # are caught before they slice through each other; without it they tunnel, interlock and
    # the contact count explodes (the sim then crawls). Layers rest 3 mm further apart.
    surface_margin: float = 0.006
    surface_gap: float = 0.003
    layer_gap: float = 0.012   # front panel rests this far above the back panel (> 2 * sphere_r)
    damping: float = 0.004
    shear_stiffness: float = 8.0   # bias springs [N/m]: woven denim shears easily
    air_drag: float = 0.012    # per-vertex linear drag [N s/m]: falling fabric floats down (~2.5 m/s terminal)
    # inextensibility: stiff, well-damped edge constraints (denim stretches < 2-3 %)
    edge_solref: tuple = (0.004, 1.0)
    edge_solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2)


# The older, cheaper collision model (--cloth-fast): folds and crumples can cut through themselves.
FAST_COLLISIONS = dict(both_way=False, own_panel=False, sphere_pairs=True, self_collide=False)


@dataclass
class ClothInfo:
    name: str
    rest_uv: np.ndarray                      # (N, 2) flat pattern coordinates [m]
    faces: np.ndarray                        # (F, 3) triangle vertex indices
    face_layer: np.ndarray                   # (F,) panel of each face: 0 = front, 1 = back
    texcoord: np.ndarray                     # (T, 2) texture coordinates in [0, 1] (texture atlas)
    face_tc: np.ndarray                      # (F, 3) index into texcoord for each face corner
    garment: GarmentSpec
    vert_bodies: list = field(default_factory=list)   # body names per vertex
    # filled after compile:
    flex_id: int = -1                        # front-panel flex (material, radius)
    flex_ids: list = field(default_factory=list)
    nvert: int = 0
    body_ids: np.ndarray | None = None      # (N,) vertex body ids (vertex i = body_ids[i])
    flex_names: list = field(default_factory=list)
    qpos_adr: np.ndarray | None = None      # (N, 3) qpos indices of each vertex's slide joints
    dof_adr: np.ndarray | None = None


# ----------------------------------------------------------------- pattern
def _leg(g: GarmentSpec, t: float):
    """(leg width, outer seam |v|) at fraction t from crotch (0) to hem (1)."""
    thigh = min(g.thigh, g.hip / 2)
    # the leg narrows quickly over the thigh, then runs almost straight to the hem
    k = 1 - (1 - t) ** 1.6
    return thigh + (g.hem - thigh) * k, g.hip / 2 + 0.008 * t


def jeans_pattern(g: GarmentSpec, with_edges: bool = False):
    """Structured triangle mesh of flat jeans: a hip panel that splits into two legs.

    Returns (uv (N,2), faces (F,3)). The crotch vertex is shared by both legs.
    """
    h = g.spacing
    m = max(2, int(round(g.hip / 2 / h)))          # columns per half width
    n_up = max(2, int(round(g.rise / h)))
    n_leg = max(3, int(round((g.length - g.rise) / h)))
    uv, rows = [], []
    for i in range(n_up + 1):
        t = i / n_up
        u = g.rise * t
        w = g.waist + (g.hip - g.waist) * t ** 0.6       # hips flare below the waistband
        rows.append(list(range(len(uv), len(uv) + 2 * m + 1)))
        uv += [(u, -w / 2 + w * j / (2 * m)) for j in range(2 * m + 1)]
    crotch = rows[-1]
    legs = {"R": [crotch[:m + 1]], "L": [crotch[m:]]}   # v<0: right leg, v>0: left leg
    for i in range(1, n_leg + 1):
        t = i / n_leg
        u = g.rise + (g.length - g.rise) * t
        wl_eff, outer = _leg(g, t)
        ids_r, ids_l = [], []
        for j in range(m + 1):
            a = j / m
            ids_r.append(len(uv)); uv.append((u, -outer + wl_eff * a))
        for j in range(m + 1):
            a = j / m
            ids_l.append(len(uv)); uv.append((u, outer - wl_eff + wl_eff * a))
        legs["R"].append(ids_r)
        legs["L"].append(ids_l)
    faces = []
    threads = set()     # warp (along the leg) and weft (across) edges
    diagonals = set()   # bias edges

    def strip(r0, r1):
        for j in range(len(r0) - 1):
            a, b, c, d = r0[j], r0[j + 1], r1[j], r1[j + 1]
            threads.update({tuple(sorted(e)) for e in ((a, b), (c, d), (a, c), (b, d))})
            # alternate the diagonal so the mesh has no directional bias
            if j % 2 == 0:
                faces.extend([(a, c, b), (b, c, d)])
                diagonals.add(tuple(sorted((b, c))))
            else:
                faces.extend([(a, c, d), (a, d, b)])
                diagonals.add(tuple(sorted((a, d))))

    for i in range(n_up):
        strip(rows[i], rows[i + 1])
    for s in "RL":
        for i in range(n_leg):
            strip(legs[s][i], legs[s][i + 1])
    uv = np.array(uv, dtype=float)
    faces = np.array(faces, dtype=np.int64)
    # consistent winding: normals of the flat pattern point up (+z in the table frame)
    p = np.c_[uv, np.zeros(len(uv))]
    n = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
    flip = n[:, 2] < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    if with_edges:
        return uv, faces, np.array(sorted(threads)), np.array(sorted(diagonals))
    return uv, faces


def texcoords(uv: np.ndarray, g: GarmentSpec) -> np.ndarray:
    """Pattern (u, v) -> texture (s, t) in [0, 1]. Texture x = across, y = along the leg."""
    half = g.hip / 2 + 0.02
    s = (uv[:, 1] + half) / (2 * half)
    t = uv[:, 0] / g.length
    return np.c_[s, 1.0 - t]


def jeans_garment(g: GarmentSpec):
    """Two-layer jeans: a front and a back panel sewn together along the outseams and
    the inseams (through the crotch), open at the waist and at both hems, so the legs are
    tubes. Rest shape: lying flat, back panel on the bottom, front panel `layer_gap` above
    it, seams halfway between.

    Returns dict with
      uv (N,2)    pattern coordinates of every vertex [m] (front and back share the pattern)
      z (N,)      rest height above the back panel [m]
      layer (N,)  0 = front panel, 1 = back panel, 2 = seam (shared by both)
      faces (F,3) outward-facing triangles (front: up, back: down)
      threads, diagonals (E,2)  warp/weft edges and bias edges
      tc (T,2), face_tc (F,3)   texture atlas coordinates: front texture | back texture
    """
    uv, faces, threads, diags = jeans_pattern(g, with_edges=True)
    n = len(uv)
    edges = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (min(a, b), max(a, b))
            edges[k] = edges.get(k, 0) + 1
    boundary = [k for k, c in edges.items() if c == 1]
    eps = 1e-6
    opening = (np.abs(uv[:, 0]) < eps) | (np.abs(uv[:, 0] - g.length) < eps)   # waist + hem rows
    seam = set()
    for a, b in boundary:
        if opening[a] and opening[b]:
            continue                      # an edge of the waist or hem opening: not sewn
        seam.update((a, b))               # side seams, inseams, crotch (+ ends of the openings)
    back_id = np.arange(n)
    k = n
    for v in range(n):
        if v not in seam:
            back_id[v] = k
            k += 1
    N = k
    uv_all = np.zeros((N, 2))
    uv_all[:n] = uv
    uv_all[back_id] = uv
    gap = g.layer_gap
    z = np.full(N, 0.0)
    layer = np.full(N, 1)
    z[:n] = gap
    layer[:n] = 0
    seam_idx = np.array(sorted(seam))
    z[seam_idx] = gap / 2
    layer[seam_idx] = 2
    back_faces = back_id[faces][:, [0, 2, 1]]             # flipped: outward normal points down
    all_faces = np.vstack([faces, back_faces])

    def both(e):
        s_ = {tuple(sorted(x)) for x in np.asarray(e).tolist()}
        s_ |= {tuple(sorted(x)) for x in back_id[np.asarray(e)].tolist()}
        return np.array(sorted(s_))

    tc1 = texcoords(uv, g)
    tc = np.vstack([tc1 * [0.5, 1.0], tc1 * [0.5, 1.0] + [0.5, 0.0]])
    face_tc = np.vstack([faces, (faces + n)[:, [0, 2, 1]]])
    # Interleave the numbering (front vertex, then its back twin) so that coupled vertices
    # have nearby dof indices: MuJoCo's sparse solver factorises in dof order, and a
    # front-block/back-block order makes it ~20x slower through fill-in.
    order = []
    for v in range(n):
        order.append(v)
        if back_id[v] != v:
            order.append(int(back_id[v]))
    new = np.empty(N, int)
    new[np.array(order)] = np.arange(N)
    twins = np.array([(v, int(back_id[v])) for v in range(n) if back_id[v] != v])
    return {"uv": uv_all[order], "z": z[order], "layer": layer[order], "faces": new[all_faces],
            "threads": new[both(threads)], "diagonals": new[both(diags)], "twins": new[twins],
            "tc": tc, "face_tc": face_tc}


# ------------------------------------------------- cloth simulated outside MuJoCo
def fine_garment(g: GarmentSpec, spacing: float, thickness: float):
    """The garment at the GPU engine's resolution: (garment, jeans_garment dict, collision
    thickness [m]). The two panels start a little more than one thickness apart (seams
    halfway) and settle into contact."""
    thick = thickness * spacing
    g, J = layered_garment(g, spacing, 1.2 * thick)
    return g, J, thick


def layered_garment(g: GarmentSpec, spacing: float, layer_gap: float):
    """The garment at `spacing` with its front panel `layer_gap` [m] above the back one:
    (garment, jeans_garment dict)."""
    g = dataclasses.replace(g, spacing=spacing, layer_gap=layer_gap)
    return g, jeans_garment(g)


# Construction of a pair of five-pocket jeans, for an engine that gives each part of the garment its
# own weight and stiffness (the Newton engine): (where, extra layers' weight, bending, stretch) as
# multiples of the single layer of denim. The weights count the layers sewn together there (folded
# waistband with interfacing, a hem folded twice, flat-felled seams, pocket patches, cotton pocket
# bags, the fly's zip and facing); stitched layers bend stiffer than the sum of their layers, so
# the bending factors sit between n and n^3 for n layers. Estimates from the construction, checked
# by the total: ~0.65-0.75 kg for adult 12 oz jeans. The regions match denim_textures' drawing.
WAISTBAND = 0.038        # [m] from the top
HEM = 0.025              # [m] from the bottom of the leg
SEAM_BAND = 0.010        # [m] either side of the out- and inseams
GARMENT_PARTS = {
    #               mass  bend  stretch
    "waistband":   (3.0, 10.0, 3.0),
    "fly":         (4.0, 8.0, 2.0),
    "hem":         (3.0, 8.0, 2.0),
    "seam":        (2.5, 6.0, 2.0),
    "back_pocket": (2.0, 3.0, 1.5),
    "pocket_bag":  (1.7, 1.3, 1.0),   # front: two layers of cotton pocketing (~0.14 kg/m^2 each) behind the denim
}


def garment_parts(J: dict, g: GarmentSpec) -> dict[str, np.ndarray]:
    """Boolean mask per vertex of each part of GARMENT_PARTS, from the pattern coordinates (u from
    the waist, v across; v < 0 = the right leg) and the panel (J["layer"]: 0 front, 1 back, 2 the
    seams both share)."""
    u, v, layer = J["uv"][:, 0], J["uv"][:, 1], J["layer"]
    front, back = layer != 1, layer != 0
    seam_pts = J["uv"][layer == 2]
    seam_pts = seam_pts[(seam_pts[:, 0] > 1e-6) & (seam_pts[:, 0] < g.length - 1e-6)]   # not the openings
    d = np.full(len(u), np.inf)
    for k in range(0, len(seam_pts), 512):
        d = np.minimum(d, np.linalg.norm(J["uv"][:, None] - seam_pts[None, k:k + 512], axis=2).min(axis=1))
    centre = (np.abs(v) < 0.008) & (u < g.rise)                     # the fly / centre back seam
    bp_u0 = WAISTBAND + 0.085                                        # back pockets as drawn
    return {
        "waistband": u < WAISTBAND,
        "fly": front & (np.abs(v) < 0.035) & (u < 0.16),
        "hem": u > g.length - HEM,
        "seam": (d < SEAM_BAND) | (back & centre),
        "back_pocket": back & (u > bp_u0) & (u < bp_u0 + 0.105) & (np.abs(np.abs(v) - 0.095) < 0.075),
        "pocket_bag": front & (u > WAISTBAND) & (u < 0.22) & (np.abs(v) > 0.09),
    }


def garment_structure(J: dict, g: GarmentSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex (mass, bending, stretch) multiples of plain denim. Where parts overlap (a seam
    through the waistband) the larger factor wins."""
    out = np.ones((3, len(J["uv"])))
    for name, mask in garment_parts(J, g).items():
        out[:, mask] = np.maximum(out[:, mask], np.array(GARMENT_PARTS[name])[:, None])
    return out[0], out[1], out[2]


def place_flat(J: dict, pos, yaw: float, table_z: float, lift: float) -> np.ndarray:
    """World positions of a jeans_garment lying flat: waistband centre at pos, legs along yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (np.c_[J["uv"], J["z"]] @ R.T + [pos[0], pos[1], table_z + lift]).astype(np.float32)


class ClothRenderMesh:
    """A render-only mesh geom in the MuJoCo model that shows a cloth simulated elsewhere (the
    GPU engine): MuJoCo renders (camera screens, recorded images, videos) draw it after its
    vertices are uploaded with update()."""
    NAME = "jeans_render"

    @classmethod
    def add(cls, spec: mujoco.MjSpec, X: np.ndarray, J: dict, material: str):
        mesh = spec.add_mesh()
        mesh.name = cls.NAME
        mesh.uservert = np.asarray(X, float).ravel().tolist()
        mesh.userface = J["faces"].ravel().tolist()
        mesh.usertexcoord = J["tc"].ravel().tolist()
        mesh.userfacetexcoord = J["face_tc"].ravel().tolist()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # a thin sheet, no volume
        gm = spec.worldbody.add_geom()
        gm.name = cls.NAME
        gm.type = mujoco.mjtGeom.mjGEOM_MESH
        gm.meshname = cls.NAME
        gm.material = material
        gm.contype = gm.conaffinity = 0                            # drawn only, never collides
        gm.group = 1

    def __init__(self, m: mujoco.MjModel, faces: np.ndarray):
        self.mid = m.mesh(self.NAME).id
        self.gid = m.geom(self.NAME).id
        self.va, self.n = m.mesh_vertadr[self.mid], m.mesh_vertnum[self.mid]
        self.na = m.mesh_normaladr[self.mid]
        if m.mesh_normalnum[self.mid] != self.n:
            raise ValueError("render mesh: MuJoCo changed the vertex layout")
        self.faces = faces

    def update(self, m: mujoco.MjModel, d: mujoco.MjData, V: np.ndarray, renderer: mujoco.Renderer):
        """Show vertex positions V (world frame) in `renderer`."""
        p, R = d.geom_xpos[self.gid], d.geom_xmat[self.gid].reshape(3, 3)
        local = (np.asarray(V, np.float64) - p) @ R
        F = self.faces
        fn = np.cross(local[F[:, 1]] - local[F[:, 0]], local[F[:, 2]] - local[F[:, 0]])
        # area-weighted vertex normals: the face normals summed per corner (bincount: ~10x
        # faster than np.add.at)
        idx = F.ravel()
        vn = np.stack([np.bincount(idx, weights=np.repeat(fn[:, c], 3), minlength=len(local))
                       for c in range(3)], axis=1)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        m.mesh_vert[self.va:self.va + self.n] = local
        m.mesh_normal[self.na:self.na + self.n] = vn
        mujoco.mjr_uploadMesh(m, renderer._mjr_context, self.mid)


# ------------------------------------------------------------------ MuJoCo
def add_garment(spec: mujoco.MjSpec, name: str, pos, yaw: float, g: GarmentSpec | None = None,
                material: str | None = None, rgba=None) -> ClothInfo:
    """Adds a pair of jeans to `spec`, lying flat at `pos` (the waistband centre), with the
    legs pointing along the world direction given by `yaw` (0 = +x)."""
    g = g or GarmentSpec()
    J = jeans_garment(g)
    uv, faces, threads, diagonals = J["uv"], J["faces"], J["threads"], J["diagonals"]
    pts = " ".join(f"{u:.5f} {v:.5f} {z:.5f}" for (u, v), z in zip(uv, J["z"]))
    el = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    mat = f'material="{material}"' if material else ""
    if rgba is None:     # the material's texture carries the colour; without one use plain denim blue
        rgba = (1, 1, 1, 1) if material else (0.23, 0.33, 0.52, 1)
    child = mujoco.MjSpec.from_string(f"""
<mujoco><worldbody><body name="{name}">
  <flexcomp name="{name}" type="direct" dim="2" radius="0.0025" mass="{g.mass}"
            point="{pts}" element="{el}" rgba="{' '.join(map(str, rgba))}" {mat}>
    <contact contype="{FLEX_FRONT}" conaffinity="0" condim="{g.self_condim}" friction="{g.friction} 0.005 0.0001"
             solref="0.004 1" selfcollide="none" internal="false"/>
    <edge equality="false" damping="{g.damping}"/>
    <elasticity young="{g.bend_young}" poisson="0.2" thickness="{g.thickness}"
                damping="{g.damping * 0.5}" elastic2d="bend"/>
  </flexcomp>
</body></worldbody></mujoco>""")
    # One flex for the whole garment carries the bending elasticity and is what gets drawn
    # (texture coordinates per face corner: the seams join two differently textured panels).
    # Two extra, hidden, collision-only flexes share its vertices: the front panel and the
    # back panel. Each panel's vertex spheres collide with the *other* panel's continuous
    # surface, which keeps the layers apart without every sphere touching its own triangles.
    whole = child.flexes[0]
    whole.texcoord = J["tc"].astype(float).ravel().tolist()
    whole.elemtexcoord = J["face_tc"].astype(int).ravel().tolist()
    whole.contype, whole.conaffinity = 0, 0
    names = list(whole.vertbody)
    nF = len(faces) // 2
    flex_names = [whole.name]
    layer_of = dict(zip(names, J["layer"]))
    if g.self_collide:
        # Safety net: the garment's surface also collides with itself, triangle against
        # triangle (MuJoCo flex self-collision). The vertex spheres keep stacked layers apart,
        # but they sit 5 cm apart, and in folds and crumples the triangles between them
        # would otherwise slice through each other (the pale inside then shows).
        whole.contype = whole.conaffinity = FLEX_SELF
        whole.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_AUTO
        whole.margin, whole.gap = g.surface_margin, g.surface_gap
    for part, (lay, fc, bit) in {"front": ((0, 2), faces[:nF], FLEX_FRONT),
                                "back": ((1, 2), faces[nF:], FLEX_BACK)}.items():
        vs = np.where(np.isin(J["layer"], lay))[0]
        local = -np.ones(len(uv), int)
        local[vs] = np.arange(len(vs))
        f = child.add_flex()
        f.name = f"{name}_{part}_col"
        f.dim = 2
        f.radius = whole.radius
        f.group = 3                               # hidden
        f.rgba = [0.2, 0.3, 0.5, 0.0]
        f.vertbody = [names[i] for i in vs]
        f.elem = local[fc].ravel().tolist()
        f.contype, f.conaffinity = bit, 0
        f.condim, f.friction, f.solref = whole.condim, whole.friction, whole.solref
        f.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE   # self contact: the whole flex
        f.internal = whole.internal
        f.margin, f.gap = g.surface_margin, g.surface_gap
        f.young = 0.0                             # no elasticity: contact only
    # one collision sphere per vertex body (hidden: group 3)
    vbodies = []
    for b in child.bodies:
        if b.name.startswith(f"{name}_") and b.name != name:
            for j in b.joints:
                j.damping = [g.air_drag, 0.0, 0.0]
            gm = b.add_geom()
            gm.name = f"{b.name}_col"
            gm.type = mujoco.mjtGeom.mjGEOM_SPHERE
            gm.size = [g.sphere_r, 0, 0]
            gm.contype = CLOTH_CONTYPE
            lay = layer_of[b.name]
            # world + other spheres; front-panel spheres also touch the back panel's surface.
            # One direction is enough to stop the panels passing through each other (from
            # either side) and halves the number of contacts.
            gm.conaffinity = 1 | CLOTH_CONTYPE | ({0: FLEX_BACK, 1: FLEX_FRONT}.get(int(lay), 0) if g.both_way else (FLEX_BACK if int(lay) == 0 else 0))
            if g.own_panel:
                gm.conaffinity |= {0: FLEX_FRONT, 1: FLEX_BACK}.get(int(lay), FLEX_FRONT | FLEX_BACK)
            if not g.sphere_pairs:
                gm.conaffinity &= ~CLOTH_CONTYPE
            # condim is combined with max(): contacts with the table and the finger pads keep
            # friction (condim 3), cloth-on-cloth contacts use g.self_condim
            gm.condim = g.self_condim
            gm.friction = [g.friction, 0.005, 0.0001]
            gm.solref = [0.005, 1]
            gm.group = 3
            gm.mass = 0         # mass comes from the flexcomp
            gm.rgba = [0.2, 0.3, 0.5, 0.5]
            st = b.add_site()
            st.name = f"{b.name}_s"
            st.group = 5
            vbodies.append(b.name)
    assert vbodies == names, "vertex body order must match the flex vertex order"
    frame = spec.worldbody.add_frame()
    frame.pos = list(pos)
    frame.quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    frame.attach_body(child.body(name), "", "")
    if g.sphere_pairs:
        # Neighbouring vertices are far apart (spacing >> 2r), except across the crotch where
        # the two inseams start at the same point: exclude those few sphere pairs. (The front
        # and back panel lie layer_gap > 2r apart, so they do collide: jeans can't fold through
        # themselves.)
        p3 = np.c_[uv, J["z"]]
        close = np.linalg.norm(p3[:, None] - p3[None], axis=2) < 2.2 * g.sphere_r
        pairs = {tuple(p) for p in zip(*np.nonzero(np.triu(close, 1)))}
        # front/back twins: kept apart by the panel surfaces, their spheres would only add contacts
        pairs |= {tuple(sorted(map(int, t))) for t in J["twins"]}
        for i, j in sorted(pairs):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = vbodies[i], vbodies[j]
    # Woven fabric: the threads (warp along the leg, weft across) barely stretch, but the cloth
    # shears easily on the bias. Constraining every triangle edge (flex edge equality) also
    # locks shear, and on a coarse mesh that makes the jeans move like cardboard. So: thread
    # edges are inextensible (tendon equality), bias edges are soft springs.
    for k, (i, j) in enumerate(threads):
        t = spec.add_tendon()
        t.name = f"{name}_thread{k}"
        t.group = 5                    # hidden: never drawn by the viewer or recorded cameras
        t.rgba = [0, 0, 0, 0]
        t.wrap_site(f"{vbodies[i]}_s"); t.wrap_site(f"{vbodies[j]}_s")
        t.damping = [g.damping, 0.0, 0.0]
        e = spec.add_equality()
        e.type = mujoco.mjtEq.mjEQ_TENDON
        e.name1 = t.name
        e.solref = list(g.edge_solref)
        e.solimp = list(g.edge_solimp)
    # The two panels must not pass through each other. Sphere-sphere contacts alone can't
    # guarantee it (with 5 cm spacing a sphere slides off its neighbour and drops between the
    # others), so the vertex spheres also collide with the continuous cloth surface (flex
    # triangles): layers can't interpenetrate, both between the two panels and between
    # folded layers.
    for k, (i, j) in enumerate(diagonals):
        t = spec.add_tendon()
        t.name = f"{name}_bias{k}"
        t.group = 5                    # hidden: never drawn by the viewer or recorded cameras
        t.rgba = [0, 0, 0, 0]
        t.wrap_site(f"{vbodies[i]}_s"); t.wrap_site(f"{vbodies[j]}_s")
        t.stiffness = [g.shear_stiffness, 0.0, 0.0]
        t.damping = [g.damping, 0.0, 0.0]
    info = ClothInfo(name=name, rest_uv=uv, faces=faces, face_layer=np.repeat([0, 1], nF),
                     texcoord=J["tc"], face_tc=J["face_tc"], garment=g, vert_bodies=names)
    info.flex_names = flex_names
    return info


def bind(info: ClothInfo, m: mujoco.MjModel):
    info.flex_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, n) for n in info.flex_names]
    if min(info.flex_ids) < 0:
        raise KeyError(f"flexes {info.flex_names} not found")
    info.flex_id = info.flex_ids[0]
    info.body_ids = np.array([m.body(n).id for n in info.vert_bodies])
    info.nvert = len(info.body_ids)
    qa, da = [], []
    for b in info.body_ids:
        j0, nj = m.body_jntadr[b], m.body_jntnum[b]
        qa.append([m.jnt_qposadr[j0 + k] for k in range(nj)])
        da.append([m.jnt_dofadr[j0 + k] for k in range(nj)])
    info.qpos_adr = np.array(qa)
    info.dof_adr = np.array(da)


def verts(info: ClothInfo, d: mujoco.MjData) -> np.ndarray:
    return d.xpos[info.body_ids]


def _slide_axes(m: mujoco.MjModel, d: mujoco.MjData, b: int) -> np.ndarray:
    """World directions (3x3, columns) of the slide joints of vertex body b."""
    R = d.xmat[b].reshape(3, 3)
    j0 = m.body_jntadr[b]
    return np.stack([R @ m.jnt_axis[j0 + k] for k in range(m.body_jntnum[b])], axis=1)


def place(info: ClothInfo, m: mujoco.MjModel, d: mujoco.MjData, xyz: np.ndarray):
    """Set every vertex to world position xyz (N, 3) with zero velocity."""
    mujoco.mj_kinematics(m, d)
    for i, b in enumerate(info.body_ids):
        A = _slide_axes(m, d, b)
        rest = d.xpos[b] - A @ d.qpos[info.qpos_adr[i]]
        d.qpos[info.qpos_adr[i]] = np.linalg.lstsq(A, xyz[i] - rest, rcond=None)[0]
        d.qvel[info.dof_adr[i]] = 0
    mujoco.mj_kinematics(m, d)


def rest_world(info: ClothInfo, m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """World positions of the rest shape (all vertex slide joints at zero)."""
    mujoco.mj_kinematics(m, d)
    out = np.zeros((info.nvert, 3))
    for i, b in enumerate(info.body_ids):
        out[i] = d.xpos[b] - _slide_axes(m, d, b) @ d.qpos[info.qpos_adr[i]]
    return out


# ----------------------------------------------------------------- metrics
def footprint_area(xy: np.ndarray, faces: np.ndarray, res: float = 0.005) -> float:
    """Area [m^2] of the garment's projection onto the table (triangles rasterised at `res`).

    Vectorised: a pixel counts if its centre lies within half a pixel of a triangle (filled
    triangle plus outline, like PIL's polygon fill). The server computes this on the physics
    thread, and a per-triangle Python loop stalled it for ~50 ms (VR hitches every second)."""
    xy = np.asarray(xy, np.float64)
    faces = np.asarray(faces)
    lo = xy.min(0) - 2 * res
    size = np.ceil((xy.max(0) + 2 * res - lo) / res).astype(int) + 1
    img = np.zeros((int(size[1]), int(size[0])), bool)
    P = ((xy - lo) / res).astype(np.float32)[faces]               # (F, 3, 2) in pixels
    p0 = np.ceil(P.min(1) - 0.5).astype(np.int32)                 # first candidate pixel
    ext = (np.floor(P.max(1) + 0.5).astype(np.int32) - p0).max(1) + 1   # candidates per side
    # orient every triangle counter-clockwise; edge normals (unit, pointing inwards)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    cw = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) < 0
    b, c = np.where(cw[:, None], c, b), np.where(cw[:, None], b, c)
    V = np.stack([a, b, c], 1)
    E = np.roll(V, -1, axis=1) - V                                # edges a->b, b->c, c->a
    N = np.stack([-E[..., 1], E[..., 0]], -1)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    # triangles in batches by bounding-box size: K x K pixel candidates per triangle
    K = 3
    todo = np.ones(len(faces), bool)
    while todo.any():
        sel = np.where(todo & (ext <= K))[0]
        todo[sel] = False
        for chunk in np.array_split(sel, max(1, len(sel) * K * K // 2_000_000 + 1)):
            if len(chunk) == 0:
                continue
            g = np.arange(K, dtype=np.int32)
            px = p0[chunk, None, None, :] + np.stack(np.meshgrid(g, g, indexing="xy"), -1)  # (n,K,K,2)
            # inwards distance of the pixel from each edge: n.(p - v) = n.p - n.v
            Nc = N[chunk]
            off = (Nc * V[chunk]).sum(-1)                                                  # (n,3)
            dist = (px[..., None, 0] * Nc[:, None, None, :, 0] + px[..., None, 1] * Nc[:, None, None, :, 1]
                    - off[:, None, None, :])                                               # (n,K,K,3)
            inside = (dist >= -0.5).all(-1)
            ix, iy = px[..., 0][inside], px[..., 1][inside]
            ok = (ix >= 0) & (iy >= 0) & (ix < img.shape[1]) & (iy < img.shape[0])
            img[iy[ok], ix[ok]] = True
        K = K + 1 if K < 5 else 2 * K
    return float(np.count_nonzero(img) * res * res)


@lru_cache(maxsize=4)
def _flat_area(key) -> float:
    uv, faces = key
    return footprint_area(np.frombuffer(uv).reshape(-1, 2), np.frombuffer(faces, dtype=np.int64).reshape(-1, 3))


def fold_metrics_from(rest_uv: np.ndarray, faces: np.ndarray, xyz: np.ndarray, table_z: float) -> dict:
    """fold_metrics for raw arrays (e.g. read back from an episode file)."""
    flat = _flat_area((np.ascontiguousarray(rest_uv, float).tobytes(),
                       np.ascontiguousarray(faces, np.int64).tobytes()))
    area = footprint_area(np.asarray(xyz)[:, :2], np.asarray(faces))
    h = float(np.asarray(xyz)[:, 2].max() - table_z)
    return {"coverage": area / flat if flat > 0 else 0.0, "height": h, "lifted": h > 0.05}


def fold_metrics(info: ClothInfo, xyz: np.ndarray, table_z: float) -> dict:
    """Simple, task-agnostic folding progress measures.

    coverage: footprint area / flat footprint area (1 = spread flat, ~0.25 = folded in quarters)
    height:   max vertex height above the table [m]
    lifted:   True if any part of the garment is more than 5 cm above the table
    """
    return fold_metrics_from(info.rest_uv, info.faces, xyz, table_z)


# ---------------------------------------------------------------- textures
def _rng(seed=7):
    return np.random.default_rng(seed)


def denim_textures(g: GarmentSpec, px_per_m: int = 1400) -> tuple[bytes, bytes]:
    """Procedural front and back textures (PNG bytes) laid out in pattern space.

    Image x = across the garment (s), image y = from waistband (top) to hem (bottom),
    matching `texcoords` (t = 1 at the waistband, OpenGL convention).
    """
    from PIL import Image, ImageDraw, ImageFilter

    half = g.hip / 2 + 0.02
    W = int(2 * half * px_per_m)
    H = int(g.length * px_per_m)
    rng = _rng()

    def X(v):   # pattern v [m] -> pixel x
        return (v + half) * px_per_m

    def Y(u):   # pattern u [m] -> pixel y
        return u * px_per_m

    # --- base denim: indigo warp with white weft, 3/1 right-hand twill
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    twill = 0.5 + 0.5 * np.sin((xx + yy * 1.0) * (2 * np.pi / 4.0))
    from .textures import _smooth_noise
    slub = (_smooth_noise(rng, H, W, 5) - 0.5) * 2.5         # uneven yarn thickness
    slub += (_smooth_noise(rng, H, W, 40) - 0.5) * 1.5       # large-scale mottling
    streak = np.repeat(rng.normal(0, 1, (1, W)).astype(np.float32), H, 0)   # vertical warp streaks
    noise = rng.normal(0, 1, (H, W)).astype(np.float32)
    v_coord = (xx / px_per_m) - half
    u_coord = yy / px_per_m
    # fading: lighter along the thighs and knees (worn areas), darker at seams/hems
    fade = np.zeros_like(xx)
    for leg in (-1, 1):
        cx = leg * (g.hip / 4 + 0.005)
        fade += np.exp(-((v_coord - cx) / 0.055) ** 2) * np.exp(-((u_coord - 0.42) / 0.2) ** 2) * 0.9
        fade += np.exp(-((v_coord - cx) / 0.05) ** 2) * np.exp(-((u_coord - 0.62) / 0.05) ** 2) * 0.4
    # whiskers: thin light creases fanning out from the fly towards the hips
    for k in range(5):
        for leg in (-1, 1):
            ang = np.deg2rad(-12 + 9 * k)                 # angle below horizontal
            u0 = g.rise - 0.045 + 0.012 * k
            d_ = np.array([leg * np.cos(ang), np.sin(ang)])
            px_, py_ = v_coord - leg * 0.03, u_coord - u0
            along = px_ * d_[0] + py_ * d_[1]
            perp = np.abs(px_ * d_[1] - py_ * d_[0])
            w_ = np.clip(along / 0.02, 0, 1) * np.clip((0.11 - along) / 0.05, 0, 1)
            fade += 0.28 * np.exp(-(perp / 0.0025) ** 2) * w_
    base = np.array([0.105, 0.19, 0.36], np.float32)     # indigo
    light = np.array([0.52, 0.62, 0.75], np.float32)     # faded blue
    shade = (0.10 * (twill - 0.5) + 0.035 * slub + 0.03 * streak + 0.05 * noise)[..., None]
    f = np.clip(0.18 + 0.55 * fade, 0, 1)[..., None]
    img = base * (1 - f) + light * f
    img = img * (1 + shade * 1.6)
    img = np.clip(img, 0, 1)

    front = Image.fromarray((img * 255).astype(np.uint8), "RGB")
    back = Image.fromarray((np.clip(img * 0.97, 0, 1) * 255).astype(np.uint8), "RGB")

    thread = (206, 146, 56)
    thread_dark = (170, 112, 40)

    def outline_mask(img_):
        """Transparent outside the jeans silhouette so seam darkening follows the edges."""
        return img_

    def seam(dr, pts, width=2, gap=None, dash=(9, 5), color=thread):
        """Dashed topstitching along a polyline of pattern points [(u, v), ...]."""
        pts = [(X(v), Y(u)) for u, v in pts]
        on = True
        acc = 0.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(x1 - x0, y1 - y0))
            t = 0.0
            while t < L:
                step = dash[0] if on else dash[1]
                step = min(step - acc, L - t)
                if on:
                    a = t / L; b = (t + step) / L
                    dr.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                             (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=color, width=width)
                t += step
                acc += step
                if acc >= (dash[0] if on else dash[1]) - 1e-6:
                    acc = 0.0
                    on = not on

    def offset_line(pts, d):
        """Offset a polyline of (u, v) points sideways by d metres (in pattern space)."""
        pts = np.asarray(pts, float)
        out = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]; b = pts[min(len(pts) - 1, i + 1)]
            t = b - a
            t /= np.linalg.norm(t) + 1e-9
            n = np.array([-t[1], t[0]])
            out.append(pts[i] + d * n)
        return [tuple(p) for p in out]

    def seam_shadow(dr, pts, width=7):
        dr.line([(X(v), Y(u)) for u, v in pts], fill=(0, 0, 0, 70), width=width)

    uv, _ = jeans_pattern(g)
    # silhouette edges, from the pattern itself
    def edge(side):
        m_cols = int(round(g.hip / 2 / g.spacing))
        # sample outline analytically (same formulas as the pattern)
        pts_outer, pts_inner = [], []
        for u in np.linspace(0, g.length, 60):
            if u <= g.rise:
                t = u / g.rise
                w = g.waist + (g.hip - g.waist) * t ** 0.6
                pts_outer.append((u, side * w / 2))
            else:
                t = (u - g.rise) / (g.length - g.rise)
                wl_eff, outer = _leg(g, t)
                pts_outer.append((u, side * outer))
                pts_inner.append((u, side * (outer - wl_eff)))
        return pts_outer, [(g.rise, 0.0)] + pts_inner

    for im, is_front in ((front, True), (back, False)):
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        # waistband: darker folded band with double stitching
        wb = 0.038
        dr.rectangle([X(-g.waist / 2 - 0.01), Y(0), X(g.waist / 2 + 0.01), Y(wb)], fill=(10, 20, 45, 70))
        dr.line([(X(-g.waist / 2 - 0.01), Y(wb)), (X(g.waist / 2 + 0.01), Y(wb))], fill=(0, 0, 0, 90), width=4)
        for side in (-1, 1):
            outer, inner = edge(side)
            seam_shadow(dr, outer, 8)
            seam_shadow(dr, inner, 8)
            seam(dr, offset_line(outer, -side * 0.006), 2)
            seam(dr, offset_line(outer, -side * 0.012), 2) if not is_front else None
            seam(dr, offset_line(inner, side * 0.006), 2)
            seam(dr, offset_line(inner, side * 0.012), 2)
            # hem: folded band with a single row of stitching
            u_h = g.length - 0.025
            t = 1.0
            wl_h, out_h = _leg(g, 1.0)
            outer_v = side * out_h
            inner_v = side * (out_h - wl_h)
            dr.rectangle([min(X(outer_v), X(inner_v)), Y(u_h), max(X(outer_v), X(inner_v)), Y(g.length)],
                         fill=(0, 0, 0, 55))
            seam(dr, [(u_h + 0.004, outer_v - side * 0.004), (u_h + 0.004, inner_v + side * 0.004)], 2)
        for y in (0.007, wb - 0.007):
            seam(dr, [(y, -g.waist / 2), (y, g.waist / 2)], 2)
        # belt loops
        for lv in ((-0.17, -0.09, 0.09, 0.17) if is_front else (-0.16, 0.0, 0.16)):
            dr.rectangle([X(lv - 0.006), Y(-0.001), X(lv + 0.006), Y(wb + 0.006)], fill=(22, 40, 78, 255),
                         outline=(12, 22, 45, 255))
            seam(dr, [(0.004, lv - 0.004), (0.004, lv + 0.004)], 1, dash=(3, 2))
            seam(dr, [(wb + 0.002, lv - 0.004), (wb + 0.002, lv + 0.004)], 1, dash=(3, 2))
        if is_front:
            # front pockets (the scoop), coin pocket, fly J-stitch, button, rivets
            for side in (-1, 1):
                w0 = g.waist / 2
                curve = [(wb, side * (w0 - 0.105))] + [
                    (wb + 0.085 * np.sin(a) ** 1.3, side * (w0 - 0.105 * np.cos(a) ** 0.8 + 0.005 * np.sin(a)))
                    for a in np.linspace(0.05, np.pi / 2, 14)]
                curve = [(u, v) for u, v in curve]
                seam_shadow(dr, curve, 6)
                seam(dr, offset_line(curve, side * 0.004), 2)
                seam(dr, offset_line(curve, side * 0.009), 2)
                # rivet at the pocket corner
                for (ru, rv) in ((wb + 0.003, side * (w0 - 0.103)), (wb + 0.083, side * (w0 + 0.002))):
                    cx, cy = X(rv), Y(ru)
                    dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(150, 98, 52, 255), outline=(95, 60, 30, 255), width=2)
                    dr.ellipse([cx - 3, cy - 3, cx + 1, cy + 1], fill=(220, 170, 110, 255))
            # coin pocket on the garment's right (image left)
            cu0, cv0 = wb + 0.004, -(g.waist / 2 - 0.035)
            box = [(cu0, cv0), (cu0, cv0 + 0.065), (cu0 + 0.06, cv0 + 0.065), (cu0 + 0.06, cv0)]
            seam(dr, [box[0], box[3]], 2); seam(dr, [box[1], box[2]], 2); seam(dr, [box[2], box[3]], 2)
            seam(dr, [(cu0 + 0.012, cv0), (cu0 + 0.012, cv0 + 0.065)], 2)
            # fly: J-shaped double stitch on the garment's left (image right of centre)
            j = [(wb, 0.035), (0.13, 0.035)] + [(0.13 + 0.03 * np.sin(a), 0.035 * np.cos(a)) for a in np.linspace(0, np.pi / 2, 8)]
            seam_shadow(dr, [(wb, 0.0), (g.rise - 0.02, 0.0)], 5)
            seam(dr, j, 2)
            seam(dr, offset_line(j, -0.006), 2)
            # button
            cx, cy = X(0.0), Y(wb / 2)
            dr.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(125, 110, 95, 255), outline=(70, 60, 50, 255), width=3)
            dr.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=(180, 165, 150, 255), width=2)
        else:
            # yoke, centre back seam, back pockets, leather patch
            yoke = [(wb + 0.07, -g.waist / 2 - 0.005), (wb + 0.03, 0.0), (wb + 0.07, g.waist / 2 + 0.005)]
            seam_shadow(dr, yoke, 6)
            seam(dr, offset_line(yoke, 0.004), 2); seam(dr, offset_line(yoke, 0.009), 2)
            cb = [(wb, 0.0), (g.rise, 0.0)]
            seam_shadow(dr, cb, 6)
            seam(dr, offset_line(cb, 0.004), 2); seam(dr, offset_line(cb, 0.009), 2)
            for side in (-1, 1):
                c = side * 0.095
                u0 = wb + 0.085
                pw, ph = 0.075, 0.075
                poly = [(u0, c - pw), (u0, c + pw), (u0 + ph, c + pw * 0.95), (u0 + ph + 0.03, c), (u0 + ph, c - pw * 0.95)]
                dr.polygon([(X(v), Y(u)) for u, v in poly], fill=(16, 30, 62, 60))
                seam_shadow(dr, poly + [poly[0]], 6)
                seam(dr, offset_line(poly + [poly[0]], 0.004), 2)
                seam(dr, [(u0 + 0.012, c - pw), (u0 + 0.012, c + pw)], 2)
                # decorative arcuate stitching
                arc1 = [(u0 + 0.04 + 0.02 * np.sin(np.pi * (x - c + pw) / pw), x) for x in np.linspace(c - pw + 0.01, c, 8)]
                arc2 = [(u0 + 0.04 + 0.02 * np.sin(np.pi * (x - c) / pw), x) for x in np.linspace(c, c + pw - 0.01, 8)]
                seam(dr, arc1, 2, color=thread_dark); seam(dr, arc2, 2, color=thread_dark)
            # leather patch on the garment's right side of the waistband
            px0, px1 = X(-0.15), X(-0.075)
            dr.rounded_rectangle([px0, Y(0.004), px1, Y(wb + 0.018)], 6, fill=(150, 104, 62, 255), outline=(100, 66, 36, 255), width=3)
            dr.line([(px0 + 14, Y(0.02)), (px1 - 14, Y(0.02))], fill=(110, 72, 40, 255), width=4)
            dr.line([(px0 + 20, Y(0.03)), (px1 - 20, Y(0.03))], fill=(110, 72, 40, 255), width=3)
        ov = ov.filter(ImageFilter.GaussianBlur(0.6))
        im.paste(ov, (0, 0), ov)

    def png(im):
        buf = BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()

    return png(front), png(back)


# ------------------------------------------------------------------ grasping
def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point to p on each triangle (a[i], b[i], c[i]); vectorised (Ericson, RTCD 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap); d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp); d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp); d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    v = vb / denom
    w = vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]          # interior
    # vertex / edge regions
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (d1 <= 0) & (d2 <= 0); out[m] = a[m]
        m = (d3 >= 0) & (d4 <= d3); out[m] = b[m]
        m = (d6 >= 0) & (d5 <= d6); out[m] = c[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.clip(d1 / (d1 - d3), 0, 1); out[m] = (a + ab * t[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.clip(d2 / (d2 - d6), 0, 1); out[m] = (a + ac * t[:, None])[m]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0, 1); out[m] = (b + (c - b) * t[:, None])[m]
    return out


class PinchGrasp:
    """Pinch grasp of fabric between the 2F-85 finger pads.

    Why a grasp model: the jaws close to a gap of a few millimetres, while the
    simulated fabric is resolved at ~4 cm. The fold of fabric that a real gripper
    squeezes between its pads cannot form on such a mesh, so friction alone never
    holds the cloth. This class adds exactly that missing piece, under the same
    conditions as a real pinch:

      * it triggers only while the jaws are *closing* (not when a closed gripper
        is pushed onto the cloth), once the jaws are nearly shut or stalled on fabric;
      * the fabric surface must be between the fingertips at that moment (the
        closest point on the cloth surface lies inside the jaw volume);
      * the pinched patch is held at the pose it had relative to the gripper, via soft
        `connect` equality constraints: the triangle under the fingertips *in every layer*
        between the pads (both panels of the jeans, or all layers of a folded stack), as
        real jaws squeeze everything between them. Holding only one layer would drag it
        through the others;
      * it slips out if the pull exceeds what a pinch can hold (`slip_force`),
        and is released as soon as the jaws open: once the pads are `release_gap`
        further apart than when they pinched, nothing squeezes the fabric any more
        (a small flick of the stick lets go, as with a real gripper).
    Everything else (contact with the table, the pads pushing fabric, the fabric
    colliding with itself, draping, bending) is ordinary MuJoCo physics.
    """
    MAX_LAYERS = 4             # fabric layers one pinch can hold (one triangle each)
    K = 3 * MAX_LAYERS         # vertices held per gripper at most
    LAYER_GAP = 0.008          # a second triangle of the same panel counts as another (folded)
                               # layer if its surface is this far away along the closing axis [m]
    JAW_TRIGGER = 0.80         # jaw closure (0 open .. 1 closed) at which a pinch is attempted
    JAW_STALL = 0.45           # ... or at which a stalled closing jaw counts as "on fabric"
    JAW_RELEASE = 0.40
    RELEASE_GAP = 0.004        # pads this much further apart than at the pinch [m]: fabric is free
    CMD_CLOSE = 0.5
    # jaw volume in the pinch-site frame (x: across the pads, y: closing axis, z: toward the tips)
    HALF_WIDTH = 0.011 + 0.010     # pad half width + margin
    HALF_GAP_MARGIN = 0.008
    Z_RANGE = (-0.020, 0.0177 + 0.008)

    def __init__(self, m: mujoco.MjModel, cloth: ClothInfo, side: str, eq_ids: list[int],
                 slip_force: float = 30.0, release_gap: float = RELEASE_GAP):
        self.cloth = cloth
        self.side = side
        self.eq_ids = list(eq_ids)
        self.body = m.body(f"{side}_gripper_base").id
        self.site = m.site(f"{side}_gripper_pinch").id
        self.driver_q = m.jnt_qposadr[m.joint(f"{side}_gripper_right_driver_joint").id]
        self.slip_force = slip_force
        self.release_gap = release_gap
        self.held: list[int] = []            # cloth vertex indices (0..N-1)
        self._gap0 = 0.0                     # tightest pad gap while pinching [m]
        self.slips = 0                       # vertices pulled out of a closed pinch (diagnostics)
        self._armed = True
        self._stall_t = 0.0
        self._prev_jaw = 0.0
        self.vert_body = cloth.body_ids.copy()

    @property
    def holding(self) -> bool:
        return bool(self.held)

    def reset(self, m, d):
        self.release(m, d)
        self._armed = True
        self._stall_t = 0.0

    def release(self, m, d):
        for e in self.eq_ids:
            d.eq_active[e] = 0
        self.held = []

    def _jaw(self, d):
        return float(np.clip(d.qpos[self.driver_q] / 0.8, 0, 1))

    @staticmethod
    def pad_gap(jaw: float) -> float:
        """Distance between the 2F-85 finger pads [m] at jaw closure 0 (open) .. 1 (closed)."""
        return 2 * (0.0466 * (1 - jaw) + 0.0042 * jaw)

    def update(self, m: mujoco.MjModel, d: mujoco.MjData, grip_cmd: float, dt: float):
        jaw = self._jaw(d)
        closing = grip_cmd >= self.CMD_CLOSE
        if not closing or jaw < self.JAW_RELEASE:
            if self.held:
                self.release(m, d)
            if not closing:
                self._armed = True
            self._stall_t = 0.0
        elif self.held:
            gap = self.pad_gap(jaw)
            self._gap0 = min(self._gap0, gap)  # the jaws keep closing after the pinch triggers
            if gap > self._gap0 + self.release_gap:
                self.release(m, d)             # jaws opened: the pads no longer squeeze the fabric
            else:
                self._check_slip(m, d)
        elif self._armed:
            stalled = jaw > self.JAW_STALL and abs(jaw - self._prev_jaw) < 0.02 * dt / 0.01
            self._stall_t = self._stall_t + dt if stalled else 0.0
            if jaw >= self.JAW_TRIGGER or self._stall_t > 0.12:
                self._armed = False            # one attempt per closing motion
                self._try_grasp(m, d, jaw)
        self._prev_jaw = jaw

    def _try_grasp(self, m, d, jaw):
        V = verts(self.cloth, d)
        F = self.cloth.faces
        R = d.site_xmat[self.site].reshape(3, 3)
        p = d.site_xpos[self.site]
        centre = p + R @ np.array([0.0, 0.0, 0.006])
        # quick reject: triangles far away
        near = np.linalg.norm(V - centre, axis=1) < 0.08
        cand = np.where(near[F].any(axis=1))[0]
        if len(cand) == 0:
            return
        a, b, c = V[F[cand, 0]], V[F[cand, 1]], V[F[cand, 2]]
        q = closest_points_on_triangles(centre[None, :], a, b, c)
        loc = (q - p) @ R                        # site frame
        half_gap = self.pad_gap(jaw) / 2 + self.HALF_GAP_MARGIN
        ok = ((np.abs(loc[:, 0]) <= self.HALF_WIDTH) & (np.abs(loc[:, 1]) <= half_gap)
              & (loc[:, 2] >= self.Z_RANGE[0]) & (loc[:, 2] <= self.Z_RANGE[1]))
        if not ok.any():
            return
        dist = np.linalg.norm(q - centre, axis=1)
        # nearest triangle first, then one per further layer between the pads: the other panel,
        # or the same panel folded over (its surface clearly apart along the closing axis)
        layers: list[tuple[int, float]] = []        # (panel, closing-axis offset) of each held layer
        held: list[int] = []
        for i in np.where(ok)[0][np.argsort(dist[ok])]:
            panel, y = int(self.cloth.face_layer[cand[i]]), float(loc[i, 1])
            if all(panel != pl or abs(y - yl) > self.LAYER_GAP for pl, yl in layers):
                layers.append((panel, y))
                held += [int(v) for v in F[cand[i]] if v not in held]
                if len(layers) == self.MAX_LAYERS:
                    break
        Rb = d.xmat[self.body].reshape(3, 3)
        pb = d.xpos[self.body]
        for e, vi in zip(self.eq_ids, held):
            m.eq_obj2id[e] = self.vert_body[vi]
            m.eq_data[e, 0:3] = Rb.T @ (V[vi] - pb)     # anchor in the gripper frame
            m.eq_data[e, 3:6] = 0.0                     # the vertex itself
            d.eq_active[e] = 1
        self.held = held
        self._gap0 = self.pad_gap(jaw)

    def _check_slip(self, m, d):
        if d.nefc == 0:
            return
        eq_rows = d.efc_type[:d.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        ids = d.efc_id[:d.nefc]
        keep = []
        for e, vi in zip(self.eq_ids, self.held):
            rows = eq_rows & (ids == e)
            f = float(np.linalg.norm(d.efc_force[:d.nefc][rows])) if rows.any() else 0.0
            if f > self.slip_force:
                d.eq_active[e] = 0
                self.slips += 1
            else:
                keep.append(vi)
        if len(keep) != len(self.held):
            self.held = keep
            if not keep:
                self.release(m, d)


def add_grasp_constraints(spec: mujoco.MjSpec, cloth: ClothInfo, side: str) -> list[str]:
    """Pre-allocate K inactive connect constraints between a gripper and the cloth."""
    names = []
    for k in range(PinchGrasp.K):
        e = spec.add_equality()
        e.name = f"{side}_pinch_{k}"
        e.type = mujoco.mjtEq.mjEQ_CONNECT
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = f"{side}_gripper_base"
        e.name2 = cloth.vert_bodies[k]
        e.active = False
        e.solref = [0.01, 1.0]
        e.solimp = [0.9, 0.95, 0.001, 0.5, 2]
        names.append(e.name)
    return names


# --------------------------------------------------------------- rendering
def loop_subdivision(faces: np.ndarray, nvert: int, levels: int = 2):
    """Loop subdivision as a sparse linear map, so a client can smooth the simulated
    mesh every frame with one sparse mat-vec: X_fine = W @ X_coarse.

    Returns (rows, cols, vals, fine_faces, n_fine) with W in COO form (float32 vals)."""
    W = {i: {i: 1.0} for i in range(nvert)}      # current level vertices as combos of coarse ones
    F = np.asarray(faces, np.int64)
    n = nvert
    for _ in range(levels):
        # edges and their opposite vertices
        edge_opp: dict[tuple[int, int], list[int]] = {}
        for a, b, c in F:
            for u, v, o in ((a, b, c), (b, c, a), (c, a, b)):
                edge_opp.setdefault((min(u, v), max(u, v)), []).append(int(o))
        nbrs: dict[int, set] = {i: set() for i in range(n)}
        bnd_nbrs: dict[int, list] = {i: [] for i in range(n)}
        for (u, v), opp in edge_opp.items():
            nbrs[u].add(v); nbrs[v].add(u)
            if len(opp) == 1:
                bnd_nbrs[u].append(v); bnd_nbrs[v].append(u)
        step: dict[int, dict[int, float]] = {}
        for i in range(n):
            if bnd_nbrs[i]:
                if len(bnd_nbrs[i]) == 2:
                    a, b = bnd_nbrs[i]
                    step[i] = {i: 0.75, a: 0.125, b: 0.125}
                else:
                    step[i] = {i: 1.0}
            else:
                k = len(nbrs[i])
                beta = 3.0 / (8.0 * k) if k > 3 else 3.0 / 16.0
                step[i] = {i: 1.0 - k * beta, **{j: beta for j in nbrs[i]}}
        edge_id = {}
        for (u, v), opp in edge_opp.items():
            idx = n + len(edge_id)
            edge_id[(u, v)] = idx
            if len(opp) == 2:
                step[idx] = {u: 0.375, v: 0.375, opp[0]: 0.125, opp[1]: 0.125}
            else:
                step[idx] = {u: 0.5, v: 0.5}
        newF = []
        for a, b, c in F:
            ab = edge_id[(min(a, b), max(a, b))]
            bc = edge_id[(min(b, c), max(b, c))]
            ca = edge_id[(min(c, a), max(c, a))]
            newF += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        # compose: new vertex = sum_j step[j] * W[j]
        W2 = {}
        for i, row in step.items():
            acc: dict[int, float] = {}
            for j, wj in row.items():
                for k, wk in W[j].items():
                    acc[k] = acc.get(k, 0.0) + wj * wk
            W2[i] = acc
        W = W2
        F = np.array(newF, np.int64)
        n = n + len(edge_id)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j, w in W[i].items():
            if abs(w) > 1e-6:
                rows.append(i); cols.append(j); vals.append(w)
    return (np.array(rows, np.uint32), np.array(cols, np.uint32), np.array(vals, np.float32),
            F.astype(np.uint32), n)


def linear_subdivision_uv(uv: np.ndarray, faces: np.ndarray, levels: int = 2) -> np.ndarray:
    """Texture coordinates for the Loop-subdivided mesh (midpoints, same vertex order)."""
    P = np.asarray(uv, float)
    F = np.asarray(faces, np.int64)
    for _ in range(levels):
        edge_id = {}
        pts = list(P)
        for a, b, c in F:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key not in edge_id:
                    edge_id[key] = None
        # same edge order as loop_subdivision (dict insertion order over faces)
        ordered = {}
        for a, b, c in F:
            for u, v, _o in ((a, b, c), (b, c, a), (c, a, b)):
                key = (min(u, v), max(u, v))
                if key not in ordered:
                    ordered[key] = len(P) + len(ordered)
                    pts.append(0.5 * (P[key[0]] + P[key[1]]))
        newF = []
        for a, b, c in F:
            ab = ordered[(min(a, b), max(a, b))]; bc = ordered[(min(b, c), max(b, c))]
            ca = ordered[(min(c, a), max(c, a))]
            newF += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        P = np.array(pts)
        F = np.array(newF, np.int64)
    return P
