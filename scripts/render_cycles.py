"""Render a recorded jeans episode photorealistically with Blender's Cycles path tracer.

    python scripts/render_cycles.py data/jeans/<session>/episode_0000.hdf5 --out fold.mp4
    python scripts/render_cycles.py episode.hdf5 --out preview.mp4 --width 960 --height 540 --samples 16 --every 3

replay.py draws episodes with MuJoCo's real-time OpenGL renderer (one shadow map, no bounced
light); this traces light the way a film renderer does, with soft shadows from the ceiling panel,
light bouncing off the table and the walls, and a fabric shader on the denim (sheen, the weave
as a bump map, thickness at the openings).

Two stages, two Python environments, one file:
  1. here (the project's environment): the episode is replayed from its recorded states and the
     scene is written out: every visible geom with its material and texture, the robot's poses
     and the cloth's vertices for every frame (an .npz and PNG textures in a work folder);
  2. Blender's Python module (bpy, which needs Python 3.11: --blender-python, default
     $BLENDER_PYTHON) builds the scene from that folder, renders the frames on the GPU (OptiX,
     denoised) and ffmpeg joins them into an MP4.

    uv venv -p 3.11 .venv-blender && uv pip install -p .venv-blender/bin/python bpy==4.2.0 numpy

The robot and the cloth are exactly the recorded ones; only the picture is new. Episodes from
any cloth engine that records its vertices (gpu, newton) work.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# MuJoCo geom types this exports (the scene's visible geoms are boxes, planes and meshes)
PLANE, BOX, MESH = 0, 6, 7


# ------------------------------------------------------------------ stage 1: export
def export(episode: str, work: Path, every: int):
    import h5py
    import mujoco
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from vrteleop import cloth_config as CC
    from vrteleop import scene as S

    h = h5py.File(episode, "r")
    attr = lambda k, dflt=None: (lambda v: v.decode() if isinstance(v, bytes) else v)(h.attrs.get(k, dflt))
    if attr("task") != "jeans":
        sys.exit("render_cycles.py renders jeans episodes")
    engine = attr("cloth_engine", "mujoco")
    if engine == "mujoco":
        sys.exit("this episode's cloth lives in MuJoCo's state: record with --cloth-engine gpu or newton")
    cc = CC.ClothConfig.from_json(attr("cloth_config"))
    sc = S.build_scene("jeans", cc, engine)
    m, d = sc.model, mujoco.MjData(sc.model)
    _, J, _ = S.external_garment(cc, engine)
    faces = h["cloth/faces"][:]
    if not np.array_equal(faces, J["faces"]):
        sys.exit("the garment built from the episode's settings differs from the recorded one")

    qpos, qvel = h["observations/full_qpos"], h["observations/full_qvel"]
    frames = range(0, len(qpos), every)
    fps = float(h.attrs["fps"]) / every

    # visible geoms (the groups MuJoCo's renderer shows), except the cloth's render mesh
    geoms = [g for g in range(m.ngeom) if m.geom_group[g] <= 2 and m.geom(g).name != "jeans_render"
             and (m.geom_matid[g] >= 0 or m.geom_rgba[g][3] > 0) and m.geom_type[g] in (PLANE, BOX, MESH)]
    moving = [g for g in geoms if m.body_rootid[m.geom_bodyid[g]] != 0 or m.body_mocapid[m.geom_bodyid[g]] >= 0]
    textures = {}
    for t in range(m.ntex):
        if m.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_2D:
            continue
        w, ht, c = m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
        img = m.tex_data[m.tex_adr[t]:m.tex_adr[t] + w * ht * c].reshape(ht, w, c)
        path = work / f"tex_{t}.png"
        Image.fromarray(img[..., :3]).save(path)
        textures[t] = path.name
    mats = []
    for i in range(m.nmat):
        tex = [int(t) for t in np.atleast_1d(m.mat_texid[i]) if t >= 0 and t in textures]
        mats.append(dict(name=m.material(i).name, rgba=m.mat_rgba[i].tolist(), specular=float(m.mat_specular[i]),
                         shininess=float(m.mat_shininess[i]), reflectance=float(m.mat_reflectance[i]),
                         emission=float(m.mat_emission[i]), texrepeat=m.mat_texrepeat[i].tolist(),
                         texture=textures[tex[0]] if tex else None))
    meshes = {}
    for g in geoms:
        if m.geom_type[g] == MESH:
            k = int(m.geom_dataid[g])
            if k not in meshes:
                va, vn = m.mesh_vertadr[k], m.mesh_vertnum[k]
                fa, fn = m.mesh_faceadr[k], m.mesh_facenum[k]
                meshes[k] = (m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy())
    d.qpos[:] = qpos[0]
    mujoco.mj_forward(m, d)
    static = {g: (d.geom_xpos[g].copy(), d.geom_xmat[g].copy()) for g in geoms}
    pos = np.zeros((len(frames), len(moving), 3), np.float32)
    mat = np.zeros((len(frames), len(moving), 9), np.float32)
    for n, i in enumerate(frames):
        d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_kinematics(m, d)
        pos[n], mat[n] = d.geom_xpos[moving], d.geom_xmat[moving]
    arrays = dict(cloth=h["observations/cloth_verts"][list(frames)].astype(np.float32),
                  cloth_faces=faces, cloth_tc=J["tc"], cloth_face_tc=J["face_tc"], pos=pos, mat=mat)
    scene = dict(fps=fps, table_z=S.TABLE_Z, table_center=list(S.TABLE_CENTER), moving=moving,
                 cloth_material=int(m.material("denim").id),
                 geoms=[dict(id=g, type=int(m.geom_type[g]), size=m.geom_size[g].tolist(),
                             mat=int(m.geom_matid[g]), rgba=m.geom_rgba[g].tolist(),
                             mesh=int(m.geom_dataid[g]) if m.geom_type[g] == MESH else -1,
                             pos=static[g][0].tolist(), xmat=static[g][1].tolist()) for g in geoms],
                 materials=mats)
    for k, (V, F) in meshes.items():
        arrays[f"mesh_{k}_v"], arrays[f"mesh_{k}_f"] = V, F
    np.savez(work / "scene.npz", **arrays)
    (work / "scene.json").write_text(json.dumps(scene))
    return len(frames), fps


# ------------------------------------------------------------------ stage 2: Blender
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def blender(work: Path, a):
    import bpy
    from mathutils import Matrix, Vector

    sc = json.loads((work / "scene.json").read_text())
    A = np.load(work / "scene.npz")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    def material(i, name=None):
        mt = sc["materials"][i]
        bm = bpy.data.materials.new(name or mt["name"])
        bm.use_nodes = True
        nt = bm.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        col = srgb_to_linear(mt["rgba"][:3])
        bsdf.inputs["Base Color"].default_value = (*col, 1.0)
        bsdf.inputs["Roughness"].default_value = float(np.clip(1.0 - 0.9 * mt["shininess"], 0.25, 0.95))
        n = mt["name"]
        if "metal" in n or "aluminium" in n:
            bsdf.inputs["Metallic"].default_value = 0.85
            bsdf.inputs["Roughness"].default_value = 0.35
        if "silicone" in n:
            bsdf.inputs["Roughness"].default_value = 0.7
        if mt["emission"] > 0:
            bsdf.inputs["Emission Color"].default_value = (*col, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 6.0 * mt["emission"]
        if mt["texture"]:
            img = nt.nodes.new("ShaderNodeTexImage")
            img.image = bpy.data.images.load(str(work / mt["texture"]))
            nt.links.new(img.outputs["Color"], bsdf.inputs["Base Color"])
            return bm, img
        return bm, None

    mats = {}

    def mat_for(i):
        if i not in mats:
            mats[i] = material(i)[0]
        return mats[i]

    def place(obj, p, R):
        M = Matrix([list(R[0:3]) + [p[0]], list(R[3:6]) + [p[1]], list(R[6:9]) + [p[2]], [0, 0, 0, 1]])
        obj.matrix_world = M

    def box_mesh(name, hx, hy, hz, repeat):
        """A box with its faces' UVs in metres x texrepeat (MuJoCo's uniform texture mapping)."""
        me = bpy.data.meshes.new(name)
        s = np.array([hx, hy, hz])
        V, F, UV = [], [], []
        for ax in range(3):
            for sign in (-1, 1):
                u, v = [k for k in range(3) if k != ax]
                c = []
                for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = np.zeros(3)
                    p[ax], p[u], p[v] = sign * s[ax], du * s[u], dv * s[v]
                    c.append(p)
                    UV.append((p[u] * repeat[0], p[v] * repeat[1]))
                idx = list(range(len(V), len(V) + 4))
                V += c
                F.append(idx if sign > 0 else idx[::-1])
        me.from_pydata([tuple(x) for x in V], [], F)
        uvl = me.uv_layers.new()
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = UV[me.loops[li].vertex_index]
        return me

    objs = {}
    for gd in sc["geoms"]:
        g = gd["id"]
        rep = sc["materials"][gd["mat"]]["texrepeat"] if gd["mat"] >= 0 else [1, 1]
        if gd["type"] == BOX:
            me = box_mesh(f"geom{g}", *gd["size"], rep)
        elif gd["type"] == PLANE:
            sx, sy = gd["size"][:2]
            me = box_mesh(f"geom{g}", sx, sy, 0.0, rep)
        else:
            V, F = A[f"mesh_{gd['mesh']}_v"], A[f"mesh_{gd['mesh']}_f"]
            me = bpy.data.meshes.get(f"mesh{gd['mesh']}")
            if me is None:
                me = bpy.data.meshes.new(f"mesh{gd['mesh']}")
                me.from_pydata(V.tolist(), [], F.tolist())
                me.shade_smooth()
                me.set_sharp_from_angle(angle=np.radians(35))
        ob = bpy.data.objects.new(f"g{g}", me)
        scene.collection.objects.link(ob)
        if gd["mat"] >= 0:
            ob.data.materials.clear()
            ob.data.materials.append(mat_for(gd["mat"]))
        else:
            bm = bpy.data.materials.new(f"rgba{g}")
            bm.use_nodes = True
            bm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
                *srgb_to_linear(gd["rgba"][:3]), 1.0)
            ob.data.materials.append(bm)
        place(ob, gd["pos"], gd["xmat"])
        objs[g] = ob

    # ---- the jeans: the recorded surface, smoothed, given a thickness, with a fabric shader
    F, TC, FTC = A["cloth_faces"], A["cloth_tc"], A["cloth_face_tc"]
    X = A["cloth"]
    cm = bpy.data.meshes.new("jeans")
    cm.from_pydata(X[0].tolist(), [], F.tolist())
    uvl = cm.uv_layers.new()
    uv = TC[FTC.ravel()]
    uvl.data.foreach_set("uv", uv.astype(np.float32).ravel())
    cm.shade_smooth()
    cloth = bpy.data.objects.new("jeans", cm)
    scene.collection.objects.link(cloth)
    sub = cloth.modifiers.new("smooth", "SUBSURF")
    sub.levels, sub.render_levels = 0, 1
    sol = cloth.modifiers.new("thickness", "SOLIDIFY")
    sol.thickness, sol.offset = 0.0009, 0.0
    denim, img = material(sc["cloth_material"], "denim")
    nt = denim.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Sheen Weight"].default_value = 0.2
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    bsdf.inputs["Sheen Tint"].default_value = (0.75, 0.8, 1.0, 1.0)
    # the twill weave as a fine bump: a wave texture along the diagonal, ~2 mm period
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type, wave.bands_direction = "BANDS", "DIAGONAL"
    wave.inputs["Scale"].default_value = 260.0
    wave.inputs["Distortion"].default_value = 1.5
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(tc.outputs["UV"], wave.inputs["Vector"])
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    cloth.data.materials.append(denim)

    # ---- light: the ceiling panel as an area light, a soft window light, a dim room
    tz = sc["table_z"]
    cx, cy = sc["table_center"]
    for name, loc, size, power, rot in (
            ("ceiling", (0.45, 0.1, 2.55), (1.2, 0.6), 170.0, (0, 0, 0)),
            ("window", (cx - 0.6, cy + 2.0, tz + 1.1), (2.0, 1.2), 110.0, (np.radians(-62), 0, np.radians(180))),
            ("fill", (cx + 2.2, cy - 1.4, tz + 0.9), (1.5, 1.5), 35.0, (np.radians(65), 0, np.radians(55)))):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape, ld.size, ld.size_y, ld.energy = "RECTANGLE", size[0], size[1], power
        lo = bpy.data.objects.new(name, ld)
        lo.location, lo.rotation_euler = loc, rot
        scene.collection.objects.link(lo)
    world = bpy.data.worlds.new("room")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.56, 0.6, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.2
    scene.world = world

    # ---- camera: three-quarter view of the table, eye height of a person standing at it
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width = a.lens, 36.0
    cd.dof.use_dof, cd.dof.aperture_fstop = True, 5.6
    cam = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(cam)
    eye = Vector(a.eye) if a.eye else Vector((cx + 1.3, cy - 0.95, tz + 0.75))
    target = Vector((cx - 0.02, cy, tz + 0.08))
    cam.location = eye
    cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
    cd.dof.focus_distance = (target - eye).length
    scene.camera = cam

    # ---- Cycles on the GPU
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            if any(dv.type == dev_type for dv in prefs.devices):
                break
        except TypeError:
            continue
    for dv in prefs.devices:
        dv.use = dv.type != "CPU"
    scene.cycles.device = "GPU"
    scene.cycles.samples = a.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX" if prefs.compute_device_type == "OPTIX" else "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    scene.render.use_persistent_data = True         # keep the static scene's BVH between frames
    scene.render.resolution_x, scene.render.resolution_y = a.width, a.height
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.4
    print(f"cycles on {prefs.compute_device_type}: {[dv.name for dv in prefs.devices if dv.use]}", flush=True)

    out = work / "frames"
    out.mkdir(exist_ok=True)
    moving = sc["moving"]
    first = a.first
    n = len(X) - first if a.frames is None else min(len(X) - first, a.frames)
    import time
    t0 = time.time()
    for i in range(n):
        for k, g in enumerate(moving):
            place(objs[g], A["pos"][first + i, k], A["mat"][first + i, k])
        cm.vertices.foreach_set("co", X[first + i].ravel())
        cm.update()
        scene.render.filepath = str(out / f"{first + i:05d}.png")
        bpy.ops.render.render(write_still=True)
        if i % 10 == 0 or i == n - 1:
            el = time.time() - t0
            print(f"  frame {i + 1}/{n}  {el / (i + 1):.1f} s/frame", flush=True)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", nargs="?")
    p.add_argument("--out", default=None, help="the MP4 to write (default: next to the episode)")
    p.add_argument("--work", default=None, help="folder for the exported scene and the frames")
    p.add_argument("--blender-python", default=os.environ.get("BLENDER_PYTHON", str(ROOT / ".venv-blender/bin/python")))
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=64, help="paths per pixel (denoised)")
    p.add_argument("--lens", type=float, default=32.0, help="focal length [mm], 36 mm sensor")
    p.add_argument("--eye", type=float, nargs=3, default=None, help="camera position [m]")
    p.add_argument("--every", type=int, default=1, help="render every n-th recorded frame")
    p.add_argument("--frames", type=int, default=None, help="stop after this many frames (previews)")
    p.add_argument("--first", type=int, default=0, help="start at this (exported) frame (previews)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted render in --work: keep the export and the finished frames")
    p.add_argument("--stage", choices=["all", "blender"], default="all", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.stage == "blender":
        blender(Path(a.work), a)
        return
    ep = Path(a.episode)
    work = Path(a.work or ep.with_suffix(".render"))
    work.mkdir(parents=True, exist_ok=True)
    if a.resume and (work / "scene.json").exists():
        fps = json.loads((work / "scene.json").read_text())["fps"]
        done = sorted((work / "frames").glob("*.png"))
        a.first = max(0, len(done) - 1)                 # the last one may be half written
        print(f"resuming at frame {a.first}", flush=True)
    else:
        n, fps = export(str(ep), work, a.every)
        print(f"exported {n} frames to {work}", flush=True)
    args = [a.blender_python, __file__, "--stage", "blender", "--work", str(work), "--width", str(a.width),
            "--height", str(a.height), "--samples", str(a.samples), "--lens", str(a.lens)]
    if a.eye:
        args += ["--eye", *map(str, a.eye)]
    args += ["--first", str(a.first)]
    if a.frames:
        args += ["--frames", str(a.frames)]
    subprocess.run(args, check=True)
    import imageio_ffmpeg
    out = Path(a.out or ep.with_suffix(".cycles.mp4"))
    first = min(int(f.stem) for f in (work / "frames").glob("*.png"))     # frames are numbered by episode frame
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
                    "-start_number", str(first), "-i", str(work / "frames" / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
