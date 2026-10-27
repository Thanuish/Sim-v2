"""MuJoCo dual-UR5e teleoperation server for Meta Quest (WebXR).

    python -m vrteleop.server                # jeans folding task, HTTPS on :8443 (self-signed)
    python -m vrteleop.server --task blocks  # pick-and-place blocks into a bin
    python -m vrteleop.server --cameras      # also record camera images
    python -m vrteleop.server --http         # plain HTTP on :8080 (use with `adb reverse`)

Open the printed URL in the Quest browser, press "Enter VR".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import ssl
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import WSMsgType, web

from . import cloth as C
from . import cloth_config as CC
from . import scene as S
from .arms import DualArms
from .cloth_engine import make_engine
from .fold_game import FoldGame
from .pipeline import Session
from .recorder import EpisodeRecorder
from .webscene import export_scene

WEB_DIR = Path(__file__).resolve().parent / "web"
GRIPPER_JOINT_MAX = 0.8     # 2F-85 driver joint range [0, 0.8] rad
PROTOCOL_VERSION = 5        # bump when server <-> web client messages change


class TeleopSim:
    def __init__(self, args):
        self.args = args
        self.latest_input: dict = {}
        self.latest_input_t = 0.0
        self.input_count = 0
        self.input_hz = 0.0
        self._input_log_t = time.perf_counter()
        self.clients: set[web.WebSocketResponse] = set()
        self.rtf = 1.0
        self.scene_id = 0                 # bumped on every rebuild: web clients reload the scene
        self._pending_reload = None       # ClothConfig to switch to (applied by the sim loop)
        self.session = Session(
            Path(args.data_dir), args.task, name=args.session or None, operator=args.operator,
            preview=not args.no_preview,
            settings={k: v for k, v in vars(args).items() if k not in ("host",)})
        print(f"[data] recording to {self.session.dir}")
        self.last_saved = None
        self.message = ""
        self.game = FoldGame()            # jeans task: the folding steps and the clock
        # the fold score (~15 ms of numpy) runs on a worker so the sim loop never stalls on it
        self._metrics_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics")
        self._metrics_job = None          # (future, verts, holding, generation)
        self._generation = 0              # bumped on reset: drops a fold score of the old scene
        args.cloth_engine = args.cloth_engine or "mujoco"
        self._build(cloth_config_from_args(args))

    def _build(self, cloth_cfg):
        """Everything that depends on the compiled scene (rebuilt when the garment changes)."""
        args = self.args
        self.info = S.build_scene(args.task, cloth_cfg, args.cloth_engine)
        self.m = self.info.model
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(args.seed)
        self.arms = DualArms(self.info, args.task, args.scale, feedforward=not args.no_feedforward)
        self.cloth = make_engine(self.info, args.cloth_engine)     # None for the blocks task
        self.cloth_metrics = {}
        self.cloth_sent = None            # cloth version in the last pose packet
        self._metrics_t = 0.0
        pattern = self.cloth.info if self.cloth is not None else None
        self.render_bodies, self.scene_gz = export_scene(
            self.m, cloth=pattern, web_textures=self.info.web_textures,
            subdiv_levels=self.cloth.render_subdiv if self.cloth is not None else 2,
            cloth_material=self.cloth.material_id if self.cloth is not None else -1)
        print(f"[scene] task '{args.task}': {S.TASKS[args.task]}"
              + (f"  (jeans: {pattern.nvert} cloth vertices, {self.cloth.name} cloth engine)"
                 if pattern is not None else ""))
        print(f"[scene] {len(self.render_bodies)} bodies, web payload {len(self.scene_gz) / 1e6:.1f} MB (gzip)")

        cams = []
        self.renderer = None           # one offscreen renderer shared by recording + VR streaming
        if args.cameras:
            cams = [c for c in (args.camera_names.split(",") if args.camera_names else self.info.cameras)]
        # Live camera feeds shown as floating screens in VR (toggle: left stick click)
        self.stream_cams = [c for c in args.stream_cams.split(",") if c] if args.stream_cams else []
        self.cams_on = bool(self.stream_cams)
        self.recorder = EpisodeRecorder(
            self.session.dir, args.fps, self.info.xml, list(self.info.objects), self._joint_names(), cams,
            task=args.task,
            cloth_faces=pattern.faces if pattern is not None else None,
            cloth_rest_uv=pattern.rest_uv if pattern is not None else None)
        self.reset(randomize=0.0)
        self.scene_id += 1

    def request_reload(self, cloth_cfg) -> str:
        """Switch to another garment / cloth settings; the sim loop rebuilds the scene."""
        if self.recorder.active:
            raise RuntimeError("stop or discard the recording first")
        self._pending_reload = cloth_cfg
        return f"reloading with garment '{cloth_cfg.name}'" if cloth_cfg else "reloading"

    # ------------------------------------------------------------------ state
    def _joint_names(self):
        names = []
        for s in S.SIDES:
            names += [f"{s}_{j}" for j in S.ARM_JOINTS] + [f"{s}_gripper"]
        return names

    def reset(self, randomize: float | None = None):
        r = self.args.randomize if randomize is None else randomize
        S.reset(self.info, self.d, self.rng, randomize=r, cloth_init=self.args.cloth_init)
        if self.cloth is not None:
            self.cloth.reset(self.d, rng=self.rng, randomize=r, cloth_init=self.args.cloth_init)
        self.cloth_metrics = {}
        self._generation += 1
        self.game.reset()
        self.arms.reset(self.d)

    def get_renderer(self):
        if self.renderer is None:
            try:
                self.renderer = mujoco.Renderer(self.m, self.args.img_h, self.args.img_w)
            except Exception as e:        # no OpenGL (e.g. a server without a display)
                raise RuntimeError(f"MuJoCo can't render camera images ({e}); on a machine without "
                                   "a display, set MUJOCO_GL=egl (NVIDIA) or MUJOCO_GL=osmesa") from e
        return self.renderer

    def camera_packet(self, i: int) -> bytes:
        """Render stream camera i and JPEG-encode it: [2, cam_index, 0, 0] + jpeg."""
        from io import BytesIO
        from PIL import Image
        r = self.get_renderer()
        if self.cloth is not None:
            self.cloth.sync_render(r, self.d)
        r.update_scene(self.d, camera=self.stream_cams[i])
        buf = BytesIO()
        Image.fromarray(r.render()).save(buf, "JPEG", quality=self.args.jpeg_quality)
        return bytes([2, i, 0, 0]) + buf.getvalue()

    def ee_pose(self, s):
        return self.arms.ee_pose(self.d, s)

    # ---------------------------------------------------------------- control
    def control(self, dt: float):
        fresh = (time.perf_counter() - self.latest_input_t) < self.args.input_timeout
        self.arms.control(self.d, self.latest_input if fresh else None, dt)
        if self.cloth is not None:
            self.cloth.update(self.d, {s: self.arms.gripper(s) for s in S.SIDES}, dt)
            if self.game.t_start is None and any(self.arms.engaged(s) for s in S.SIDES):
                self.game.start()

    def update_cloth_metrics(self):
        if self.cloth is not None:
            V = self.cloth.verts(self.d)
            self._apply_metrics(self.cloth.metrics_of(V), V, any(self.cloth.holding(s) for s in S.SIDES))

    def _apply_metrics(self, metrics, verts, holding):
        self.cloth_metrics = metrics
        msg = self.game.update(verts, metrics, holding)
        if msg:
            self.message = msg

    def poll_cloth_metrics(self, start: bool) -> bool:
        """The fold score in the background: apply a finished one; with `start`, begin the next.
        Returns True when a computation was started."""
        job = self._metrics_job
        if job is not None:
            if not job[0].done():
                return False
            self._metrics_job = None
            if job[3] == self._generation and self.cloth is not None:
                self._apply_metrics(job[0].result(), job[1], job[2])
        if not start or self.cloth is None:
            return False
        V = self.cloth.verts(self.d)          # a snapshot: the engine replaces it every frame
        self._metrics_job = (self._metrics_pool.submit(self.cloth.metrics_of, V), V,
                             any(self.cloth.holding(s) for s in S.SIDES), self._generation)
        return True

    # -------------------------------------------------------------- recording
    def record_frame(self):
        d, info = self.d, self.info
        qpos, qvel, act = [], [], []
        ee_p, ee_q, tp, tq, gc, eng, cp = [], [], [], [], [], [], []
        for s in S.SIDES:
            qa, va = info.arm_qpos_adr[s], info.arm_dof_adr[s]
            ga = info.grip_qpos_adr[s]
            gv = self.m.jnt_dofadr[self.m.joint(f"{s}_gripper_right_driver_joint").id]
            qpos += list(d.qpos[qa]) + [d.qpos[ga] / GRIPPER_JOINT_MAX]
            qvel += list(d.qvel[va]) + [d.qvel[gv] / GRIPPER_JOINT_MAX]
            tel = self.arms.teleop[s]
            act += list(self.arms.q_cmd[s]) + [tel.gripper]
            p, q = self.ee_pose(s)
            ee_p.append(p); ee_q.append(q)
            tp.append(tel.target_pos); tq.append(tel.target_quat); gc.append(tel.gripper)
            eng.append(tel.engaged)
            c = self.latest_input.get(s) or {}
            cp.append(c.get("pose") or [np.nan] * 7)
        objs = []
        for n in info.objects:
            a = info.obj_qpos_adr[n]
            objs.append(d.qpos[a:a + 7].copy())
        frame = dict(time=d.time, qpos=qpos, qvel=qvel, ee_pos=ee_p, ee_quat=ee_q,
                     object_pose=objs, full_qpos=d.qpos.copy(), full_qvel=d.qvel.copy(),
                     action=act, target_pos=tp, target_quat=tq, gripper_cmd=gc, engaged=eng,
                     head_pose=self.latest_input.get("head") or [np.nan] * 7,
                     controller_pose=cp)
        if self.cloth is not None:
            k = self.cloth.max_held
            frame["cloth_verts"] = self.cloth.verts(d).copy()
            frame["cloth_grasp"] = [(list(self.cloth.held(s)) + [-1] * k)[:k] for s in S.SIDES]
            frame["cloth_contact_force"] = self.cloth.forces()
        if self.recorder.camera_names:
            r = self.get_renderer()
            if self.cloth is not None:
                self.cloth.sync_render(r, d)
            for cam in self.recorder.camera_names:
                r.update_scene(d, camera=cam)
                frame[f"img_{cam}"] = r.render().copy()
        self.recorder.add(frame)

    def log_input(self):
        """Print a one-line summary of what the headset is sending (helps debug controls)."""
        parts = [f"[input] {self.input_hz:4.0f} Hz"]
        for s in S.SIDES:
            c = self.latest_input.get(s)
            if not c:
                parts.append(f"{s[0].upper()}: no controller")
                continue
            st = c.get("stick") or [0, 0]
            pressed = [i for i, b in enumerate(c.get("buttons") or []) if b]
            parts.append(f"{s[0].upper()}: grip {c.get('grip', 0):.2f} stick ({st[0]:+.2f},{st[1]:+.2f}) "
                         f"btn {pressed} {'ENGAGED' if self.arms.engaged(s) else ''} "
                         f"grip% {self.arms.gripper(s) * 100:.0f}")
        print("  ".join(parts), flush=True)

    COMMANDS = ("record_toggle", "save_fail", "discard", "cams_toggle", "reset")

    def command(self, cmd: str) -> str:
        rec = self.recorder
        if cmd not in self.COMMANDS:
            raise ValueError(f"unknown command '{cmd}', choose from {', '.join(self.COMMANDS)}")
        if cmd == "record_toggle":
            if rec.active:
                self.save_episode(success=True)
            else:
                rec.start(self.d.time)
                self.message = f"Recording episode {rec.next_index()}"
        elif cmd == "save_fail":
            if rec.active:
                self.save_episode(success=False)
        elif cmd == "discard":
            if rec.active:
                rec.discard()
                self.message = "Episode discarded"
            self.reset()
        elif cmd == "cams_toggle":
            self.cams_on = not self.cams_on and bool(self.stream_cams)
            self.message = f"Camera screens {'on' if self.cams_on else 'off'}"
        elif cmd == "reset":
            if rec.active:
                rec.discard()
                self.message = "Episode discarded (reset)"
            else:
                self.message = "Scene reset"
            self.reset()
        print(f"[cmd] {cmd}: {self.message}")
        return self.message

    def save_episode(self, success: bool):
        extra = {}
        if self.cloth is not None:
            self.update_cloth_metrics()
            extra = {"final_coverage": self.cloth_metrics["coverage"],
                     "final_height": self.cloth_metrics["height"],
                     "cloth_spacing": self.cloth.info.garment.spacing,
                     "cloth_engine": self.cloth.name,
                     "garment": self.info.cloth_config.name,
                     "cloth_config": self.info.cloth_config.to_json(),   # rebuilds it exactly
                     "cloth_engine_settings": json.dumps(self.cloth.settings())}
        n_frames = self.recorder.num_frames
        label = "success" if success else "failure"

        def saved(path):              # on the recorder's writer thread, once the file is complete
            self.session.episode_saved(path, {
                "success": bool(success),
                "duration_s": round(n_frames / self.args.fps, 2),
                "frames": n_frames,
                "final_coverage": round(extra["final_coverage"], 3) if extra else "",
                "final_height_m": round(extra["final_height"], 3) if extra else "",
                "notes": "",
            })
            print(f"[data] saved {self.session.relpath(path)}")
            self.message = f"Saved {path.name} ({label})"

        path = self.recorder.stop(self.m.opt.timestep, success=success, extra_attrs=extra, on_saved=saved)
        if path:
            self.last_saved = path.name
            self.message = f"Saving {path.name} ({label})..."
        else:
            self.message = "Episode too short, not saved"
        self.reset()

    # -------------------------------------------------------------- streaming
    def pose_packet(self) -> bytes:
        d = self.d
        tel = self.arms.teleop
        flags = (1 if tel["left"].engaged else 0) | (2 if tel["right"].engaged else 0) \
            | (4 if self.recorder.active else 0)
        rb = self.render_bodies
        poses = np.concatenate([d.xpos[rb], d.xquat[rb]], axis=1).astype(np.float32)
        tgt = np.array([np.concatenate([tel[s].target_pos, tel[s].target_quat])
                        for s in S.SIDES], dtype=np.float32)
        head = self.latest_input.get("head")
        head_live = head is not None and (time.perf_counter() - self.latest_input_t) < 1.0
        if head_live:
            flags |= 8
        head_arr = np.asarray(head if head_live else [0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        # the cloth (~6400 points with the GPU engine) only when it changed since the last packet
        cloth = np.zeros((0, 3), np.float32)
        if self.cloth is not None and self.cloth.version(d) != self.cloth_sent:
            self.cloth_sent = self.cloth.version(d)
            cloth = self.cloth.verts(d).astype(np.float32)
        header = np.array([d.time, flags, len(rb), len(cloth)], dtype=np.float32)
        # layout: header(4) | bodies(n*7) | targets(2*7) | operator head pose(7) | cloth verts(N*3)
        # (all in the MuJoCo frame)
        return (bytes([1, 0, 0, 0]) + header.tobytes() + poses.tobytes() + tgt.tobytes()
                + head_arr.tobytes() + cloth.tobytes())

    def status(self, rtf: float) -> str:
        return json.dumps(self.status_dict(rtf))

    def status_dict(self, rtf: float | None = None) -> dict:
        cc = self.info.cloth_config
        return {
            "type": "status",
            "version": PROTOCOL_VERSION,
            "scene_id": self.scene_id,
            "garment": cc.name if cc else None,
            "task": self.args.task,
            "task_title": S.TASKS[self.args.task],
            "cloth_engine": self.cloth.name if self.cloth is not None else None,
            "holding": {s: self.cloth.holding(s) for s in S.SIDES} if self.cloth is not None else {},
            "cloth": {k: round(float(v), 3) for k, v in self.cloth_metrics.items()},
            "game": self.game.status() if self.cloth is not None else None,
            "recording": self.recorder.active,
            "frames": self.recorder.num_frames,
            "rec_time": self.recorder.num_frames / self.args.fps,
            "next_episode": self.recorder.next_index(),
            "last_saved": self.last_saved,
            "message": self.message,
            "sim_time": round(self.d.time, 2),
            "rtf": round(self.rtf if rtf is None else rtf, 2),
            "engaged": {s: self.arms.engaged(s) for s in S.SIDES},
            "gripper": {s: round(self.arms.gripper(s), 2) for s in S.SIDES},
            "cams": self.stream_cams,
            "cams_on": self.cams_on,
            "input_fresh": (time.perf_counter() - self.latest_input_t) < self.args.input_timeout,
            "input_hz": round(self.input_hz, 1),
        }

    async def broadcast(self, data, binary=True):
        dead = []
        # a snapshot: clients connect / disconnect while a send is awaited
        for ws in list(self.clients):
            try:
                if binary:
                    await ws.send_bytes(data)
                else:
                    await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def run(self):
        while True:
            await self._run_scene()               # returns when a reload was requested
            cfg, self._pending_reload = self._pending_reload, None
            previous = self.info.cloth_config
            try:
                self._build(cfg)
                self.message = f"Loaded garment '{cfg.name}'" if cfg else "Scene reloaded"
            except Exception as e:                # keep the server alive on the old scene
                self._build(previous)
                self.message = f"Reload failed ({e}); kept the previous scene"
            print(f"[scene] {self.message}")

    async def _run_scene(self):
        m, d, cloth = self.m, self.d, self.cloth
        ctrl_dt = 1.0 / self.args.control_hz
        pub_dt = 1.0 / self.args.stream_hz
        rec_dt = 1.0 / self.args.fps
        wall0, sim0 = time.perf_counter(), d.time
        next_ctrl = d.time
        next_rec = d.time
        next_pub = next_status = next_cam = 0.0
        cam_dt = 1.0 / max(self.args.cam_stream_hz, 0.1)
        cam_i = 0
        rtf_w, rtf_s, rtf = time.perf_counter(), d.time, 1.0
        while self._pending_reload is None:
            now = time.perf_counter()
            target = sim0 + (now - wall0)
            steps = 0
            while d.time < target and steps < 25:
                if d.time >= next_ctrl - 1e-9:
                    self.control(ctrl_dt)
                    next_ctrl = d.time + ctrl_dt
                if self.recorder.active and d.time >= next_rec - 1e-9:
                    self.record_frame()
                    next_rec = d.time + rec_dt
                elif not self.recorder.active:
                    next_rec = d.time
                mujoco.mj_step(m, d)
                if cloth is not None:
                    cloth.step(d)
                steps += 1
                if d.time < sim0:            # reset() rewound the clock
                    break
            behind = d.time < target - 0.004       # physics has not caught up with the wall clock
            if d.time < target - 0.1 or d.time < sim0:   # fell behind / reset: resync clock
                wall0, sim0 = time.perf_counter(), d.time
                next_ctrl = next_rec = d.time
            if now >= next_pub and self.clients:
                await self.broadcast(self.pose_packet())
                next_pub = now + pub_dt
            if self.cams_on and self.clients and now >= next_cam and not behind:
                # Physics first: camera screens are only rendered while the sim keeps real time.
                # One camera per loop iteration (round-robin) so physics is never starved.
                # If rendering is slow, the stream rate drops automatically (<= ~20% of wall time).
                t_r = time.perf_counter()
                try:
                    packet = self.camera_packet(cam_i)
                except RuntimeError as e:         # the screens are optional: keep simulating
                    print(f"[cams] {e}\n[cams] camera screens turned off")
                    self.stream_cams, self.cams_on = [], False
                    self.message = "Camera screens unavailable (see the server window)"
                    continue
                await self.broadcast(packet)
                cam_i = (cam_i + 1) % len(self.stream_cams)
                spent = time.perf_counter() - t_r
                next_cam = now + max(cam_dt / len(self.stream_cams), 4.0 * spent)
            if now - self._input_log_t >= 2.0:
                self.input_hz = self.input_count / (now - self._input_log_t)
                self.input_count = 0
                self._input_log_t = now
                if self.input_hz > 0:
                    self.log_input()
            since = now - self._metrics_t       # fold score: every 1 s when idle time allows, 3 s at most
            if self.poll_cloth_metrics(since > 3.0 or (since > 1.0 and not behind)):
                self._metrics_t = now
            if now >= next_status:
                if now - rtf_w > 0.5:
                    rtf = (d.time - rtf_s) / (now - rtf_w) if d.time >= rtf_s else 1.0
                    rtf_w, rtf_s = now, d.time
                    self.rtf = rtf
                if self.clients:
                    await self.broadcast(self.status(rtf), binary=False)
                next_status = now + 0.2
            await asyncio.sleep(0.001)


# ---------------------------------------------------------------------- web
def make_app(sim: TeleopSim) -> web.Application:
    @web.middleware
    async def no_cache_app_js(request, handler):
        resp = await handler(request)
        if request.path.startswith("/static/") and "/vendor/" not in request.path:
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app = web.Application(middlewares=[no_cache_app_js])

    boot_id = str(int(time.time()))
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8").replace(
        "/static/main.js", f"/static/main.js?v={boot_id}")   # never run a stale cached client

    async def index(_):
        return web.Response(text=index_html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

    async def scene_json(_):
        return web.Response(body=sim.scene_gz, content_type="application/json",
                            headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
        await ws.prepare(request)
        sim.clients.add(ws)
        sim.cloth_sent = None             # the newcomer needs the cloth in the next packet
        peer = request.remote
        print(f"[ws] client connected: {peer} ({len(sim.clients)} total)")
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                t = data.get("type")
                if t == "input":
                    sim.latest_input = data
                    sim.latest_input_t = time.perf_counter()
                    sim.input_count += 1
                elif t == "cmd":
                    try:
                        sim.command(data.get("cmd", ""))
                    except ValueError as e:
                        print(f"[cmd] {e}")
        finally:
            sim.clients.discard(ws)
            print(f"[ws] client disconnected: {peer}")
        return ws

    # ---- control API (this PC only): used by the MCP server (vrteleop/mcp_server.py) and scripts
    def local_only(handler):
        async def wrapped(request):
            if request.remote not in ("127.0.0.1", "::1"):
                return web.json_response({"error": "the control API only accepts requests from this PC"},
                                         status=403)
            try:
                return await handler(request)
            except (ValueError, RuntimeError, CC.ConfigError) as e:
                return web.json_response({"error": str(e)}, status=400)
        return wrapped

    async def api_status(_):
        return web.json_response(sim.status_dict())

    async def api_command(request):
        body = await request.json()
        msg = sim.command(str(body.get("cmd", "")))
        return web.json_response({"message": msg, "status": sim.status_dict()})

    async def api_cloth(_):
        cc = sim.info.cloth_config
        return web.json_response({
            "garment": cc.name if cc else None,
            "config_file": str(args_cloth_path()),
            "garments": CC.names(args_cloth_path()),
            "settings": json.loads(cc.to_json()) if cc else None})

    async def api_reload(request):
        """{"garment": "shorts", "overrides": {"spacing": 0.045}}: rebuild the scene (the cloth
        config file is re-read, so edits to it take effect)."""
        body = await request.json() if request.can_read_body else {}
        if sim.args.task != "jeans":
            raise ValueError("the blocks task has no cloth settings")
        cur = sim.info.cloth_config
        name = body.get("garment") or (cur.name if cur else None)
        overrides = body.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be an object of garment settings")
        # validated here, so mistakes go back to the caller and the running scene is untouched
        cfg = cloth_config_from_args(sim.args, name, overrides)
        msg = sim.request_reload(cfg)
        return web.json_response({"message": msg})

    def args_cloth_path():
        return Path(sim.args.cloth_config) if sim.args.cloth_config else CC.DEFAULT_PATH

    app.router.add_get("/", index)
    app.router.add_get("/scene.json", scene_json)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", WEB_DIR, show_index=False)
    app.router.add_get("/api/status", local_only(api_status))
    app.router.add_post("/api/command", local_only(api_command))
    app.router.add_get("/api/cloth", local_only(api_cloth))
    app.router.add_post("/api/reload", local_only(api_reload))

    async def start_sim(app_):
        app_["sim_task"] = asyncio.create_task(sim.run())

        def crashed(task):            # never let the sim loop die silently
            if not task.cancelled() and task.exception() is not None:
                import traceback
                traceback.print_exception(task.exception())
                sim.message = f"SIMULATION STOPPED: {task.exception()!r} (see the server window)"
        app_["sim_task"].add_done_callback(crashed)

    async def stop_sim(app_):
        app_["sim_task"].cancel()
        if sim.recorder.active:
            sim.save_episode(success=False)
        sim.recorder.wait()                   # finish writing episodes before exiting

    app.on_startup.append(start_sim)
    app.on_cleanup.append(stop_sim)
    return app


def lan_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=None, help="default 8443 (https) / 8080 (http)")
    p.add_argument("--http", action="store_true", help="serve plain HTTP (for adb reverse / localhost)")
    p.add_argument("--data-dir", default=str(S.ROOT / "data"),
                   help="root folder; episodes go to <data-dir>/<task>/<session>/")
    p.add_argument("--session", default="",
                   help="session folder name (default: date and time); reuse a name to add to it")
    p.add_argument("--operator", default=os.environ.get("USER") or os.environ.get("USERNAME", ""),
                   help="stored with every episode")
    p.add_argument("--no-preview", action="store_true", help="don't render an MP4 preview per episode")
    p.add_argument("--fps", type=float, default=30.0, help="recording rate (sim time)")
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--stream-hz", type=float, default=90.0)
    p.add_argument("--scale", type=float, default=1.0, help="hand->robot translation scale")
    p.add_argument("--no-feedforward", action="store_true",
                   help="plain PD joint servos (the arm trails your hand by ~0.1 s; for comparison)")
    p.add_argument("--task", choices=list(S.TASKS), default=S.DEFAULT_TASK,
                   help="jeans: fold a pair of jeans (cloth); blocks: pick-and-place into a bin")
    p.add_argument("--cloth-init", choices=["flat", "crumpled"], default="flat",
                   help="jeans task: start spread flat, or dropped into a random heap")
    p.add_argument("--cloth-engine", choices=["mujoco", "gpu", "newton"], default=None,
                   help="jeans task: cloth simulation. gpu (the server's default): XPBD cloth on any GPU "
                        "(Vulkan, CUDA or Metal; settings in [gpu] of the cloth config); newton: Newton's VBD "
                        "solver with full self-collision (NVIDIA CUDA, requirements-newton.txt; settings in "
                        "[newton]; slower than real time, see docs/newton_engine.md); mujoco (the default of "
                        "the scripts, and the fallback): flex cloth in the MuJoCo model (5.5 cm)")
    p.add_argument("--garment", default=None,
                   help="jeans task: garment from the cloth config (default: its `default`), e.g. "
                        "jeans, shorts, stretch_jeans")
    p.add_argument("--cloth-config", default=None,
                   help="jeans task: cloth settings file (default: config/cloth.toml)")
    p.add_argument("--cloth-spacing", type=float, default=None,
                   help="jeans task: override the garment's simulation resolution [m] (0.045 is finer "
                        "but needs a fast CPU)")
    p.add_argument("--cloth-fast", action="store_true",
                   help="jeans task: cheaper cloth collisions (~40 %% less CPU), but folded or crumpled "
                        "fabric can cut through itself")
    p.add_argument("--slip-force", type=float, default=None,
                   help="jeans task: override the pull [N] at which fabric slips out of a pinch")
    p.add_argument("--randomize", type=float, default=0.05,
                   help="object xy jitter on reset [m] (jeans: also +/- 3x this in heading [rad])")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cameras", action="store_true", help="record camera images into episodes")
    p.add_argument("--camera-names", default="", help="comma list; default = all cameras")
    p.add_argument("--img-w", type=int, default=320)
    p.add_argument("--img-h", type=int, default=240)
    p.add_argument("--stream-cams", default="left_gripper_wrist_cam,head_cam,right_gripper_wrist_cam",
                   help="cameras shown as live screens in VR (comma list, '' to disable)")
    p.add_argument("--cam-stream-hz", type=float, default=15.0)
    p.add_argument("--input-timeout", type=float, default=0.35,
                   help="disengage arms if no controller data for this long [s]")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--public-host", default=os.environ.get("TELEOP_PUBLIC_HOST", ""),
                   help="address(es) the headset uses to reach this machine, comma separated "
                        "(needed in Docker, where the container can't see the host's IP)")
    return p


def cloth_config_from_args(args, garment: str | None = None, overrides: dict | None = None):
    """The garment from the cloth config file (re-read every call), with command-line overrides
    applied; `garment` / `overrides` (from the control API) take precedence."""
    if args.task != "jeans":
        return None
    over = {}
    if args.cloth_spacing:
        over["spacing"] = args.cloth_spacing
    if args.cloth_fast:
        over.update(C.FAST_COLLISIONS)
    over.update(overrides or {})
    cfg = CC.load(garment or args.garment, args.cloth_config, over)
    if args.slip_force is not None:
        cfg.slip_force = args.slip_force
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    if sys.platform == "win32":
        # Windows wakes sleeping threads every 15.6 ms by default, so the sim loop's 1 ms sleep
        # took 15.6 ms: the headset got ~64 uneven updates a second instead of 90
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)

    if not S.UR5E_XML.exists() or not S.GRIPPER_XML.exists():
        sys.exit("Robot assets missing. Run:  python setup_assets.py")

    try:
        if args.cloth_engine is None and args.task == "jeans":
            sim = None
            # newton is not in this list yet: on an RTX 4090 it isn't real time at denim-like stretch
            # (docs/newton_engine.md); start it with --cloth-engine newton
            for engine, fallback in (("gpu", "the MuJoCo cloth"),):
                args.cloth_engine = engine
                try:
                    sim = TeleopSim(args)
                    break
                except (ImportError, RuntimeError) as e:
                    print(f"[cloth] the {engine} cloth engine didn't start ({e}); trying {fallback}")
            if sim is None:
                args.cloth_engine = "mujoco"
                sim = TeleopSim(args)
        else:
            sim = TeleopSim(args)
    except CC.ConfigError as e:
        sys.exit(f"Cloth config error: {e}")
    if sim.info.cloth_config is not None:
        cc = sim.info.cloth_config
        print(f"[cloth] garment '{cc.name}' from {args.cloth_config or CC.DEFAULT_PATH}: "
              f"spacing {cc.garment.spacing} m, mass {cc.garment.mass} kg, timestep {cc.timestep * 1e3:.0f} ms")
    app = make_app(sim)
    public = [h.strip() for h in args.public_host.split(",") if h.strip()]
    ips = public + ["127.0.0.1"] if public else lan_ips()
    ssl_ctx = None
    if not args.http:
        from .certs import ensure_cert
        cert, key = ensure_cert(S.ROOT / "certs", ips)
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(cert, key)
    port = args.port or (8080 if args.http else 8443)
    scheme = "http" if args.http else "https"
    print("\n  Open on your Quest browser:")
    for ip in ips:
        if ip != "127.0.0.1":
            print(f"    {scheme}://{ip}:{port}")
    print(f"    {scheme}://localhost:{port}   (desktop / adb reverse)\n")
    if not args.http:
        print("  First visit: accept the self-signed certificate warning (Advanced -> Proceed).\n")
    web.run_app(app, host=args.host, port=port, ssl_context=ssl_ctx, print=None)


if __name__ == "__main__":
    main()
