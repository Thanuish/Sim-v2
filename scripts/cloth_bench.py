"""Headless cloth benchmark: a scripted two-arm jeans fold with physical quality metrics.

    python scripts/cloth_bench.py                      # default cloth, report + side-view sheet
    python scripts/cloth_bench.py --cloth-spacing 0.045 --out bench_fine

It drives the real TeleopSim through the same path as the headset (clutch mapping, IK,
gripper thumbstick, pinch grasp), but in *simulation* time: the result does not depend on
how fast this machine is, so runs are comparable across machines and code changes. The same
fold as scripts/fake_client.py --fold.

Metrics (worst over the run unless noted):
  stretch      thread (warp/weft) strain [%]; real denim stretches < 2-3 %
  penetration  deepest contact of a cloth vertex with anything [mm]
  crossings    cloth triangles passing through each other at the end of any phase; the jeans are a
               closed tube apart from the waist and hems, so each one can show the pale inside (0)
  jitter       RMS vertex speed while the jeans lie still after settling [mm/s]; ~0 when at rest
  slips        vertices that slipped out of a closed pinch
  follow       how far released fabric keeps rising with the opening gripper [mm]; real
               fabric stays down (~0), > 10 mm means it sticks to the pads
  stuck        grippers still holding fabric after the operator opened them (the last release
               is a short flick of the stick, jaws about a third open); should be 0
  coverage     footprint / flat footprint at the end (1 = flat, ~0.3 = folded)
  cost         wall time per simulated second, per phase (< 1 s = faster than real time)

--test release  pinch / set down / open trials (short flick, full open, open in the air)
--test poke     open and closed grippers pushed into the jeans or swept through a leg, then
                lifted: nothing should hang on an unpinched gripper
Unknown options go to the server (e.g. --cloth-fast, --cloth-spacing 0.045, --cloth-init crumpled).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco  # noqa: E402

import mesh_check as M  # noqa: E402
from vrteleop import cloth as C  # noqa: E402
from vrteleop import scene as S  # noqa: E402
from vrteleop.server import TeleopSim, build_parser  # noqa: E402

CTRL_BASE = {"left": np.array([0.0, 0.25, 1.1]), "right": np.array([0.0, -0.25, 1.1])}
INPUT_HZ = 72.0          # like the Quest browser


class Bench:
    LAY_HEIGHT = 0.003       # [m] the laid-over leg is let go this far above grasp height

    def __init__(self, server_args: list[str], snapshots: bool = True):
        data = tempfile.mkdtemp(prefix="cloth-bench-")
        args = build_parser().parse_args(["--http", "--no-preview", "--stream-cams", "",
                                          "--data-dir", data, "--randomize", "0", "--seed", "0",
                                          *server_args])
        self.sim = sim = TeleopSim(args)
        self.m, self.d = sim.m, sim.d
        # every measurement goes through the cloth engine's interface (cloth_engine.ClothEngine)
        self.engine = sim.cloth
        self.cloth = sim.cloth.info
        self.ctrl_dt = 1.0 / args.control_hz
        self.home = {s: sim.arms.teleop[s].target_pos.copy() for s in S.SIDES}
        self.target = {s: self.home[s].copy() for s in S.SIDES}
        self.stick = {s: 0.0 for s in S.SIDES}
        self._next_input = 0.0
        self.worst = {"stretch": 0.0, "penetration": 0.0}
        self.stretch_p99 = 0.0             # 99th percentile over edges, worst over time
        self.solver_iters: list[int] = []
        self.stuck = 0                     # grippers still holding fabric after the operator opened them
        self.follow: list[float] = []
        self._released: list[tuple[float, str, list[int], float]] = []
        self._held = {s: [] for s in S.SIDES}
        self._slips = {s: 0 for s in S.SIDES}
        self.phases: list[tuple[str, float, float]] = []    # (name, sim seconds, wall seconds)
        self.jitter = float("nan")
        self.snaps: list[tuple[str, np.ndarray]] = []
        self.crossings = 0                 # worst number of self-intersecting triangle pairs
        self.renderer = mujoco.Renderer(self.m, 360, 480) if snapshots else None
        if args.randomize:                 # TeleopSim itself always starts from the nominal pose
            self.reset(randomize=args.randomize)

    def reset(self, randomize: float = 0.0):
        """Back to the start pose (sim time restarts at 0); `randomize` jitters the jeans' pose."""
        self.sim.reset(randomize=randomize)
        self.home = {s: self.sim.arms.teleop[s].target_pos.copy() for s in S.SIDES}
        self.target = {s: self.home[s].copy() for s in S.SIDES}
        self.stick = {s: 0.0 for s in S.SIDES}
        self._next_input = 0.0
        self._held = {s: [] for s in S.SIDES}
        self._slips = self.engine.slips()
        self._released = []

    # ------------------------------------------------------------- stepping
    def _input(self):
        msg = {"type": "input", "head": [-0.4, 0.0, 1.65, 1, 0, 0, 0]}
        for s in S.SIDES:
            pose = list(CTRL_BASE[s] + self.target[s] - self.home[s]) + [1, 0, 0, 0]
            msg[s] = {"pose": pose, "grip": 1.0, "stick": [0.0, self.stick[s]], "epoch": 0}
        return msg

    def advance(self, dur: float):
        """Step physics for `dur` seconds of sim time (controller at control_hz, input at 72 Hz)."""
        m, d, sim = self.m, self.d, self.sim
        t_end = d.time + dur
        while d.time < t_end - 1e-9:
            if d.time >= self._next_input - 1e-9:
                sim.latest_input = self._input()
                self._next_input = d.time + 1.0 / INPUT_HZ
            sim.latest_input_t = time.perf_counter()        # input is always "fresh"
            sim.control(self.ctrl_dt)
            t_ctrl = d.time + self.ctrl_dt
            while d.time < t_ctrl - 1e-9:
                mujoco.mj_step(m, d)
                self.engine.step(d)
            self._measure()

    def V(self) -> np.ndarray:
        return self.engine.verts(self.d)

    def _measure(self):
        d, e = self.d, self.engine
        version = e.version(d)
        if version == getattr(self, "_measured", None):     # the cloth hasn't changed since
            return
        self._measured = version
        strain = e.thread_strain(d)
        self.worst["stretch"] = max(self.worst["stretch"], float(strain.max()))
        self.stretch_p99 = max(self.stretch_p99, float(np.percentile(strain, 99)))
        if e.name == "mujoco":
            self.solver_iters.append(int(d.solver_niter[0]))
        pen = e.penetration(d)
        for key, v in pen.items():
            self.worst[f"pen_{key}"] = max(self.worst.get(f"pen_{key}", 0.0), v) if not np.isnan(v) else v
        self.worst["penetration"] = max([self.worst["penetration"]] + [v for v in pen.values() if not np.isnan(v)])
        slips = e.slips()
        for s in S.SIDES:
            before, now = self._held[s], list(e.held(s))
            if before and not now and slips == self._slips:      # let go, not pulled out
                z0 = float(self.V()[before, 2].mean())
                self._released.append((d.time, s, before, z0))
            self._held[s] = now
        self._slips = slips
        keep = []
        for t0, s, vs, z0 in self._released:
            if d.time - t0 >= 0.8:
                z = float(self.V()[vs, 2].mean())
                self.follow.append(max(0.0, z - z0))
            else:
                keep.append((t0, s, vs, z0))
        self._released = keep

    def n_crossings(self) -> int:
        """Pairs of cloth triangles that pass through each other (the garment is a closed tube
        apart from the waist and hems, so each one can show the pale inside)."""
        return len(M.intersecting_pairs(self.V(), self.cloth.faces))

    def phase(self, name: str, fn):
        t_sim, t_wall = self.d.time, time.perf_counter()
        fn()
        self.phases.append((name, self.d.time - t_sim, time.perf_counter() - t_wall))
        self.crossings = max(self.crossings, self.n_crossings())
        if "release" in name:        # the operator has opened the gripper(s) and moved away
            self.stuck += sum(self.engine.holding(s) for s in S.SIDES)
        self.snapshot(name)

    def snapshot(self, label: str, lookat=None, distance: float = 1.45):
        if self.renderer is None:
            return
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.45, 0.0, S.TABLE_Z + 0.08] if lookat is None else lookat
        cam.distance, cam.azimuth, cam.elevation = distance, 200.0, -18.0
        self.engine.sync_render(self.renderer, self.d)       # a cloth simulated outside MuJoCo
        self.renderer.update_scene(self.d, camera=cam)
        self.snaps.append((label, self.renderer.render().copy()))

    # --------------------------------------------------------------- script
    def move(self, goal: dict, dur: float, close: dict | None = None):
        start = {s: self.target[s].copy() for s in S.SIDES}
        n = max(1, int(round(dur / self.ctrl_dt)))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            for s in S.SIDES:
                if s in goal:
                    self.target[s] = start[s] + (np.asarray(goal[s], float) - start[s]) * a
                c = (close or {}).get(s)
                self.stick[s] = 0.0 if c is None else (1.0 if c else -1.0)   # +y (down) closes
            self.advance(self.ctrl_dt)
        self.stick = {s: 0.0 for s in S.SIDES}

    def settle(self):
        """Let the jeans lie still, then measure how much they still move (RMS vertex speed between
        successive cloth states over 0.5 s)."""
        self.advance(1.5)
        prev, v = (self.d.time, self.V().copy()), []
        t_end = self.d.time + 0.5
        while self.d.time < t_end or not v:
            self.advance(self.ctrl_dt)
            t, X = self.d.time, self.V()
            if not np.array_equal(X, prev[1]):
                v.append(np.sqrt(np.mean(np.sum((X - prev[1]) ** 2, axis=1))) / (t - prev[0]))
                prev = (t, X.copy())
        self.jitter = float(np.mean(v))

    def fold(self):
        """Same choreography as scripts/fake_client.py --fold."""
        V = self.V()
        far = V[:, 0] > np.median(V[:, 0])
        idx = np.where(far)[0]
        w = idx[np.argmax(V[far, 1] * 5 + V[far, 0])]
        h = idx[np.argmax(-V[far, 1] * 5 + V[far, 0])]
        z0 = S.TABLE_Z + 0.018
        gl = np.r_[V[w, :2] + [-0.02, -0.02], z0]
        gr = np.r_[V[h, :2] + [-0.02, 0.02], z0]
        # laid over as a mirror image: each grasp lands as far inside the other leg's corner as it
        # was inside its own, and is let go 3 mm above grasp height. Further out (1.5 cm past the
        # corner) dragged the leg taut across the table (15 % stretch by the end); set down at grasp
        # height the pinched patch crushed the layers below it (24 crossings). Let go, the stiff fold
        # at the hips still pushes the top leg ~13 cm out over the other one's edge (footprint 0.81
        # after this fold); the hems fold takes the jeans to ~0.45
        near = np.where(~far)[0]
        tl = np.r_[V[near[np.argmax(V[~far, 1] * 5 - V[~far, 0])], 0] + 0.02, gl[1], z0 + self.LAY_HEIGHT]
        tr = np.r_[V[near[np.argmax(-V[~far, 1] * 5 - V[~far, 0])], 0] + 0.02, gr[1], z0 + self.LAY_HEIGHT]
        self.phase("settle", self.settle)
        self.phase("approach", lambda: (self.move({}, 0.3),
                                        self.move({"left": gl + [0, 0, 0.08], "right": gr + [0, 0, 0.08]}, 2.0),
                                        self.move({"left": gl, "right": gr}, 1.0)))
        self.phase("pinch", lambda: self.move({}, 1.0, close={"left": True, "right": True}))
        self.phase("lift", lambda: self.move({"left": gl + [0, 0, 0.16], "right": gr + [0, 0, 0.16]}, 1.2))
        self.phase("carry", lambda: self.move({"left": tl + [0, 0, 0.16], "right": tr + [0, 0, 0.16]}, 2.2))
        self.phase("lay down", lambda: self.move({"left": tl, "right": tr}, 1.0))
        self.phase("release", lambda: (self.move({}, 0.9, close={"left": False, "right": False}),
                                       self.move({"left": tl + [0, 0, 0.1], "right": tr + [0, 0, 0.05]}, 0.8),
                                       self.advance(0.5)))
        V = self.V()
        hem = V[:, 1] < V[:, 1].min() + 0.03
        gh = np.r_[V[hem, 0].mean(), V[hem, 1].max() - 0.01, z0]
        self.phase("hems pinch", lambda: (self.move({"right": gh + [0, 0, 0.08]}, 1.5),
                                          self.move({"right": gh}, 0.9),
                                          self.move({}, 1.0, close={"right": True})))
        self.phase("hems carry", lambda: (self.move({"right": gh + [0, 0, 0.2]}, 1.2),
                                          self.move({"right": [gh[0], 0.02, z0 + 0.2]}, 2.2)))
        # a short flick of the stick (jaws ~1/3 open), as operators do, rather than opening fully
        self.phase("hems release", lambda: (self.move({"right": [gh[0], 0.02, z0 + 0.03]}, 1.0),
                                            self.move({}, 0.2, close={"right": False}),
                                            self.move({"right": [gh[0], 0.0, z0 + 0.15]}, 0.8),
                                            self.advance(1.5)))

    def release_trials(self, n_points: int = 6):
        """Pinch the far leg's outer edge at n points, lift, set it down, open the gripper (a short
        flick and fully), lift away. Released fabric should stay on the table."""
        results = []
        for flick in (0.2, 0.9, "air"):
            for k in range(n_points):
                self.reset()
                self.advance(0.8)
                V = self.V()
                far = np.where(V[:, 1] < np.percentile(V[:, 1], 30))[0]       # the right arm's leg
                u = np.linspace(V[far, 0].min() + 0.08, V[far, 0].max() - 0.08, n_points)[k]
                edge = far[np.argsort(np.abs(V[far, 0] - u) + 5 * (V[far, 1] - V[far, 1].min()))[:1]][0]
                z0 = S.TABLE_Z + 0.018
                g = np.r_[V[edge, :2] + [0.0, 0.02], z0]
                self.move({}, 0.3)
                self.move({"right": g + [0, 0, 0.08]}, 1.2)
                self.move({"right": g}, 0.8)
                self.move({}, 0.8, close={"right": True})
                held = self.engine.holding("right")
                if flick == "air":
                    # lift the corner high, open fully while the leg hangs: the fabric must drop
                    self.move({"right": g + [0, 0, 0.30]}, 1.5)
                    self.advance(0.5)
                    lifted = self.near_gripper_height()              # still carried after hanging 0.5 s
                    self.move({}, 0.9, close={"right": False})
                    self.advance(1.5)
                    still = self.engine.holding("right")
                    # the highest fabric within 10 cm of the opened gripper (a leg that falls against
                    # the other arm is not hanging on this gripper)
                    V = self.V()
                    near = np.linalg.norm(V - self.sim.ee_pose("right")[0], axis=1) < 0.10
                    hang = float(V[near, 2].max() - S.TABLE_Z) if near.any() else 0.0
                    results.append((flick, k, held, still, hang, lifted))
                    self.snapshot(f"opened in the air {k}", lookat=self.sim.ee_pose("right")[0] - [0, 0, 0.12],
                                  distance=0.7)
                    continue
                self.move({"right": g + [0, 0, 0.10]}, 1.0)
                lifted = self.near_gripper_height()
                self.move({"right": g + [0, 0, 0.01]}, 1.0)
                n_follow = len(self.follow)
                self.move({}, flick, close={"right": False})
                self.move({"right": g + [0, 0, 0.12]}, 0.8)
                self.advance(0.2)
                still = self.engine.holding("right")
                f = self.follow[n_follow] if len(self.follow) > n_follow else float("nan")
                results.append((flick, k, held, still, f, lifted))
        print(f"\n  release trials ({n_points} grasp points x short flick / full open on the table /"
              f" full open in the air)")
        print("    open   point  pinched  carried [mm]  still held  follow [mm] (carried: highest cloth within 10 cm"
              " of the gripper after lifting it 10 / 30 cm; air follow: the same 1.5 s after opening)")
        for flick, k, held, still, f, lifted in results:
            lab = "air " if flick == "air" else f"{flick:3.1f}s"
            print(f"    {lab}   {k:5d}  {str(held):7s}  {1e3 * lifted:12.1f}  {str(still):10s}  {1e3 * f:8.1f}")
        table = [r for r in results if r[2] and r[0] != "air"]
        air = [r for r in results if r[2] and r[0] == "air"]
        fol = np.array([r[4] for r in table if not np.isnan(r[4])])
        # carried: the fabric came up with the gripper (table trials lift 10 cm, air trials 30 cm)
        carried = [r for r in results if r[5] > (0.05 if r[0] != "air" else 0.15)]
        summary = {"pinched": sum(r[2] for r in results), "trials": len(results),
                   "carried": len(carried),
                   "still_held": sum(r[3] for r in table + air),
                   "follow_mean_mm": 1e3 * float(fol.mean()) if len(fol) else float("nan"),
                   "follow_max_mm": 1e3 * float(fol.max()) if len(fol) else float("nan"),
                   "sticky": int((fol > 0.010).sum()),
                   "hung_in_air": sum(1 for r in air if r[4] > 0.05),
                   "air_hang_max_mm": 1e3 * max((r[4] for r in air), default=float("nan"))}
        print(f"    pinched {summary['pinched']}/{summary['trials']}, carried {summary['carried']}/{summary['trials']}, "
              f"still held after opening "
              f"{summary['still_held']}, fabric followed > 10 mm: {summary['sticky']}, "
              f"follow mean {summary['follow_mean_mm']:.1f} / max {summary['follow_max_mm']:.1f} mm; "
              f"opened in the air but fabric still hanging > 50 mm: {summary['hung_in_air']}/{len(air)}")
        return summary

    def near_gripper_height(self, side: str = "right") -> float:
        """Height above the table of the highest fabric within 10 cm of the gripper [m]."""
        V = self.V()
        near = np.linalg.norm(V - self.sim.ee_pose(side)[0], axis=1) < 0.10
        return float(V[near, 2].max() - S.TABLE_Z) if near.any() else 0.0

    def poke_trials(self, n_points: int = 4):
        """Push an open / a closed gripper down into the jeans (without pinching), drag it a little
        and lift it away. Nothing should come up with it: fabric that rises is hooked on a finger."""
        results = []
        for motion in ("poke", "sweep"):
            for closed in (False, True):
                for k in range(n_points):
                    self.reset()
                    self.advance(0.8)
                    if closed:
                        self.move({}, 0.8, close={"right": True})    # close in the air: no pinch
                    V = self.V()
                    leg = np.where(V[:, 1] < np.percentile(V[:, 1], 40))[0]
                    u = np.linspace(V[leg, 0].min() + 0.1, V[leg, 0].max() - 0.1, n_points)[k]
                    row = leg[np.abs(V[leg, 0] - u) < 0.03]
                    y0, y1 = V[row, 1].min(), V[row, 1].max()
                    self.move({}, 0.3)
                    if motion == "poke":    # straight down at an edge (odd k) or mid-leg (even k)
                        p = np.r_[u, y0 + (0.015 if k % 2 else 0.5 * (y1 - y0)), S.TABLE_Z + 0.004]
                        self.move({"right": p + [0, 0, 0.12]}, 1.0)
                        self.move({"right": p}, 1.0)
                        self.move({"right": p + [0.03, 0.02, 0]}, 0.6)       # a small drag
                    else:                   # sideways through the leg, just above the table
                        p = np.r_[u, y0 - 0.05, S.TABLE_Z + 0.006]
                        self.move({"right": p + [0, 0, 0.12]}, 1.0)
                        self.move({"right": p}, 1.0)
                        p = np.r_[u, y1 - 0.03, S.TABLE_Z + 0.006]
                        self.move({"right": p}, 1.5)
                    self.move({"right": p + [0.0, 0.0, 0.25]}, 1.2)
                    self.advance(0.5)
                    z = self.V()[:, 2] - S.TABLE_Z
                    results.append((motion, closed, k, self.engine.holding("right"),
                                    float(z.max()), int((z > 0.03).sum()), self.n_crossings()))
                    self.snapshot(f"{motion} {'closed' if closed else 'open'} {k}",
                                  lookat=[u, 0.5 * (y0 + y1), S.TABLE_Z + 0.08], distance=0.9)
        print(f"\n  poke trials (open / closed gripper pushed down into the jeans or swept sideways through"
              f" a leg, then lifted 25 cm)")
        print("    motion gripper  point  pinched  highest cloth [mm]  vertices > 30 mm  crossings"
              "   (lying flat: 12.5 mm)")
        for motion, closed, k, held, zmax, n, cr in results:
            print(f"    {motion:6s} {'closed' if closed else 'open  '}   {k:5d}  {str(held):7s}"
                  f"  {1e3 * zmax:17.1f}  {n:16d}  {cr:9d}")
        # the gripper ends 25 cm up: fabric above 15 cm hangs on it (a pushed or swept leg that just
        # bunches up or flips over stays well below that)
        hooked = sum(1 for r in results if r[4] > 0.15)
        print(f"    fabric hanging on an unpinched gripper (> 150 mm up) in {hooked}/{len(results)} trials, "
              f"panels crossed afterwards in {sum(1 for r in results if r[6] > 0)}")
        return {"poke_hooked": hooked, "poke_trials": len(results),
                "poke_zmax_mm": 1e3 * max(r[4] for r in results)}

    # --------------------------------------------------------------- report
    def report(self) -> dict:
        cov = C.fold_metrics(self.cloth, self.V(), S.TABLE_Z)["coverage"]
        sim_t = sum(p[1] for p in self.phases)
        wall_t = sum(p[2] for p in self.phases)
        r = {"vertices": self.cloth.nvert, "timestep_ms": self.m.opt.timestep * 1e3,
             "stretch_pct": 100 * self.worst["stretch"], "stretch_p99_pct": 100 * self.stretch_p99,
             "engine": self.engine.name,
             "solver_iters": float(np.mean(self.solver_iters)) if self.solver_iters else float("nan"),
             "solver_iters_max": max(self.solver_iters, default=0),
             "penetration_mm": 1e3 * self.worst["penetration"],
             **{f"{k}_mm": 1e3 * self.worst.get(k, 0.0) for k in ("pen_world", "pen_self", "pen_robot")},
             "jitter_mm_s": 1e3 * self.jitter, "slips": self.engine.slips(),
             "follow_mm": 1e3 * max(self.follow, default=0.0), "releases": len(self.follow),
             "stuck": self.stuck, "crossings": self.crossings,
             "coverage": cov, "cost_s_per_s": wall_t / sim_t}
        print(f"\n  engine {r['engine']}   vertices {r['vertices']}   robot timestep {r['timestep_ms']:.1f} ms")
        print(f"  stretch      {r['stretch_pct']:6.2f} %      (worst edge; 99th pct {r['stretch_p99_pct']:.2f} %;"
              f" real denim < 2-3 %)")
        if self.solver_iters:
            print(f"  solver       {r['solver_iters']:6.1f} iterations mean, {r['solver_iters_max']} max"
                  f" (cap {self.m.opt.iterations})")
        print(f"  penetration  {r['penetration_mm']:6.2f} mm     (table {r['pen_world_mm']:.2f}, cloth-cloth "
              f"{r['pen_self_mm']:.2f}, grippers {r['pen_robot_mm']:.2f}"
              + (f"; sphere radius {1e3 * self.cloth.garment.sphere_r:.1f})" if self.engine.name == "mujoco"
                 else "; outside MuJoCo: fabric below the table top, nan = not measured)"))
        print(f"  crossings    {r['crossings']:6d}       (cloth triangles passing through each other, worst"
              f" phase end; the pale inside shows; real jeans 0)")
        print(f"  jitter       {r['jitter_mm_s']:6.2f} mm/s   (at rest after settling)")
        print(f"  slips        {r['slips']:6d}")
        print(f"  follow       {r['follow_mm']:6.1f} mm     (released fabric rising with the gripper,"
              f" {r['releases']} releases)")
        print(f"  stuck        {r['stuck']:6d}       (grippers still holding after the operator opened them)")
        print(f"  coverage     {r['coverage']:6.3f}       (1 flat .. ~0.3 folded)")
        print(f"  cost         {r['cost_s_per_s']:6.2f} s wall per sim s overall")
        for name, ts, tw in self.phases:
            print(f"      {name:13s} {tw / ts:5.2f}")
        return r

    def save_sheet(self, path: Path):
        if not self.snaps:
            return
        from PIL import Image, ImageDraw
        cols = 4
        h, w = self.snaps[0][1].shape[:2]
        rows = (len(self.snaps) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, h * rows), "white")
        for k, (label, img) in enumerate(self.snaps):
            tile = Image.fromarray(img)
            ImageDraw.Draw(tile).text((8, 8), label, fill=(255, 255, 255))
            sheet.paste(tile, ((k % cols) * w, (k // cols) * h))
        sheet.save(path)
        print(f"  snapshots -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="", help="write the side-view snapshot sheet to <out>.png")
    p.add_argument("--no-snapshots", action="store_true")
    p.add_argument("--test", choices=["fold", "release", "poke", "all"], default="fold",
                   help="fold: the scripted fold; release: repeated pinch / set down / open trials; "
                        "poke: grippers pushed into the jeans and lifted (hooking)")
    p.add_argument("--opt", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override mjOption fields after compiling, e.g. solver=2 iterations=50")
    a, server_args = p.parse_known_args()
    b = Bench(server_args, snapshots=not a.no_snapshots)
    for kv in a.opt:
        k, v = kv.split("=", 1)
        setattr(b.m.opt, k, type(getattr(b.m.opt, k))(float(v)))
        print(f"  opt.{k} = {getattr(b.m.opt, k)}")
    if a.test in ("fold", "all"):
        b.fold()
        b.report()
        if a.out:
            b.save_sheet(Path(a.out).with_suffix(".png"))
    if a.test in ("release", "all"):
        b.snaps = []
        b.release_trials()
        if a.out:
            b.save_sheet(Path(a.out + "_release").with_suffix(".png"))
    if a.test in ("poke", "all"):
        b.snaps = []
        b.poke_trials()
        if a.out:
            b.save_sheet(Path(a.out + "_poke").with_suffix(".png"))


if __name__ == "__main__":
    main()
