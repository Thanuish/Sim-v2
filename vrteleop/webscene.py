"""Export the visual geometry of a compiled MjModel for the Three.js client.

Only geoms in visual groups 0-2 are exported (MuJoCo's default viewer groups);
collision geoms (group 3 in the Menagerie models) stay hidden. Identical meshes
(e.g. the same UR5e link used by both arms) are sent once.

Materials (colour, PBR hints, 2D textures with their repeat settings) are exported
too, so the web view shows the same surfaces as MuJoCo's own renderer. Deformable
cloth (flex) is sent as a coarse simulation mesh plus a Loop-subdivision matrix;
the client smooths the streamed vertex positions every frame.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json

import mujoco
import numpy as np

from . import cloth as C
from . import textures as T

GEOM_TYPES = {int(k): v for k, v in {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}.items()}

RGB_ROLE = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _data_url(img_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("ascii")


class _Textures:
    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.index: dict[int, int] = {}
        self.items: list[str] = []

    def add_model_texture(self, texid: int) -> int:
        if texid in self.index:
            return self.index[texid]
        m = self.m
        w, h, c = m.tex_width[texid], m.tex_height[texid], m.tex_nchannel[texid]
        a = m.tex_adr[texid]
        img = m.tex_data[a:a + w * h * c].reshape(h, w, c)[..., :3]
        # tex_data rows are top-down like the source image; three.js flips on upload (flipY),
        # which gives the same (s, t) mapping as MuJoCo
        self.index[texid] = len(self.items)
        self.items.append(_data_url(T.encode(np.ascontiguousarray(img), "JPEG", 88), "image/jpeg"))
        return self.index[texid]

    def add_png(self, png: bytes) -> int:
        img = T.decode_png(png)
        self.items.append(_data_url(T.encode(img, "JPEG", 88), "image/jpeg"))
        return len(self.items) - 1


def _materials(m: mujoco.MjModel, tex: _Textures, used: set[int]):
    mats = {}
    for i in sorted(used):
        texid = int(m.mat_texid[i, RGB_ROLE])
        ok_tex = texid >= 0 and m.tex_type[texid] == int(mujoco.mjtTexture.mjTEXTURE_2D)
        mats[i] = {
            "name": m.material(i).name,
            "rgba": [round(float(c), 4) for c in m.mat_rgba[i]],
            "tex": tex.add_model_texture(texid) if ok_tex else -1,
            "texrepeat": m.mat_texrepeat[i].round(4).tolist(),
            "texuniform": bool(m.mat_texuniform[i]),
            "emission": round(float(m.mat_emission[i]), 3),
            "specular": round(float(m.mat_specular[i]), 3),
            "shininess": round(float(m.mat_shininess[i]), 3),
            "roughness": round(float(m.mat_roughness[i]), 3),
            "metallic": round(float(m.mat_metallic[i]), 3),
        }
    return mats


def export_scene(m: mujoco.MjModel, max_group: int = 2, cloth: C.ClothInfo | None = None,
                 web_textures: dict | None = None, subdiv_levels: int = 2, cloth_material: int = -1):
    """Returns (render_body_ids, gzipped JSON bytes). `cloth` is the garment pattern; a cloth
    simulated outside MuJoCo (no flex, flex_id < 0) gets its material from `cloth_material`."""
    body_index: dict[int, int] = {}
    render_bodies: list[int] = []
    geoms = []
    meshes = []
    mesh_lookup: dict[int, int] = {}
    mesh_hash: dict[str, int] = {}
    used_mats: set[int] = set()
    tex = _Textures(m)

    for g in range(m.ngeom):
        gtype = int(m.geom_type[g])
        if gtype not in GEOM_TYPES or m.geom_group[g] > max_group:
            continue
        # a cloth simulated outside MuJoCo is streamed as a flex below; its MuJoCo render mesh
        # (updated only for MuJoCo's own renderer) would show as a second, motionless garment
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == C.ClothRenderMesh.NAME:
            continue
        matid = int(m.geom_matid[g])
        rgba = m.mat_rgba[matid] if matid >= 0 else m.geom_rgba[g]
        if rgba[3] <= 0.01:
            continue
        b = int(m.geom_bodyid[g])
        if b not in body_index:
            body_index[b] = len(render_bodies)
            render_bodies.append(b)
        entry = {
            "body": body_index[b],
            "name": m.geom(g).name,
            "type": GEOM_TYPES[gtype],
            "size": m.geom_size[g].round(6).tolist(),
            "pos": m.geom_pos[g].round(6).tolist(),
            "quat": m.geom_quat[g].round(6).tolist(),
            "rgba": [round(float(c), 4) for c in rgba],
            "mat": matid,
        }
        if matid >= 0:
            used_mats.add(matid)
        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            if mid not in mesh_lookup:
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                vert = m.mesh_vert[va:va + vn].astype(np.float32)
                face = m.mesh_face[fa:fa + fn].astype(np.uint32)
                na, nn = m.mesh_normaladr[mid], m.mesh_normalnum[mid]
                normal = m.mesh_normal[na:na + nn].astype(np.float32)
                fnorm = m.mesh_facenormal[fa:fa + fn]
                if nn == vn and np.array_equal(fnorm, face):
                    mesh = {"indexed": True, "vert": _b64(vert), "normal": _b64(normal),
                            "face": _b64(face)}
                else:  # normals indexed separately: expand to per-corner arrays
                    mesh = {"indexed": False, "vert": _b64(vert[face].reshape(-1, 3)),
                            "normal": _b64(normal[fnorm].reshape(-1, 3))}
                h = hashlib.md5(vert.tobytes() + face.tobytes()).hexdigest()
                if h in mesh_hash:
                    mesh_lookup[mid] = mesh_hash[h]
                else:
                    mesh_hash[h] = mesh_lookup[mid] = len(meshes)
                    meshes.append(mesh)
            entry["mesh"] = mesh_lookup[mid]
        geoms.append(entry)

    flexes = []
    if cloth is not None:
        fid = cloth.flex_id
        matid = int(m.flex_matid[fid]) if fid >= 0 else int(cloth_material)
        if matid >= 0:
            used_mats.add(matid)
        # Render mesh: one vertex per (sim vertex, texture coordinate) pair, so the seams
        # between the front and back panel keep both textures (and a crisp crease).
        pairs = {}
        rfaces = np.zeros_like(cloth.faces)
        for f in range(len(cloth.faces)):
            for c in range(3):
                key = (int(cloth.faces[f, c]), int(cloth.face_tc[f, c]))
                rfaces[f, c] = pairs.setdefault(key, len(pairs))
        src = np.array([v for v, _ in pairs], np.uint32)          # render vertex -> sim vertex
        ruv = np.array([cloth.texcoord[t] for _, t in pairs])
        rows, cols, vals, fine_faces, n_fine = C.loop_subdivision(rfaces, len(pairs), subdiv_levels)
        cols = src[cols]                                          # weights act on sim vertices
        uv_fine = C.linear_subdivision_uv(ruv, rfaces, subdiv_levels).astype(np.float32)
        back = (web_textures or {}).get("denim_back")
        flexes.append({
            "name": cloth.name,
            "nvert": cloth.nvert,
            "vertoffset": 0,                # offset (in vertices) in the streamed flex block
            "mat": matid,
            "back_tex": tex.add_png(back) if back else -1,
            "nfine": int(n_fine),
            "w_rows": _b64(rows), "w_cols": _b64(cols), "w_vals": _b64(vals),
            "faces": _b64(fine_faces), "uv": _b64(uv_fine),
            "thickness": float(m.flex_radius[fid]) if fid >= 0 else 0.0,
        })

    payload = {
        "bodies": [m.body(b).name for b in render_bodies],
        "geoms": geoms,
        "meshes": meshes,
        "materials": _materials(m, tex, used_mats),
        "textures": tex.items,
        "flexes": flexes,
    }
    return np.array(render_bodies, dtype=np.int64), gzip.compress(json.dumps(payload).encode(), 5)
