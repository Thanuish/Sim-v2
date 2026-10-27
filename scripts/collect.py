"""Headless data collection: a scripted operator folds the jeans, recorded as ordinary episodes.

    python scripts/collect.py --episodes 5                          # Newton cloth, random starts
    python scripts/collect.py --episodes 3 --cloth-engine gpu --randomize 0.03
    python scripts/collect.py --episodes 2 --cloth-init crumpled --video

Runs in simulation time, so a cloth engine slower than real time (the Newton VBD cloth at
denim-like stretch runs at ~0.3 x real time on an RTX 4090, docs/newton_engine.md) records
exactly what it would in real time. The operator is the scripted fold of scripts/cloth_bench.py,
driven through the same path as a headset (clutch mapping, IK, gripper thumbstick, grasp), and
the episodes go to data/<task>/<session>/ in the VR episodes' HDF5 layout (pipeline.py,
recorder.py). An episode counts as a success when the jeans end at most 62 % of their flat
footprint (the folding game's first step: one leg over the other).

--video renders each saved episode to an MP4 next to it (scripts/replay.py, front camera).
Unknown options go to the server (e.g. --garment shorts, --cloth-spacing ...).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

import cloth_bench as CB  # noqa: E402
from vrteleop import scene as S  # noqa: E402

SUCCESS_COVERAGE = 0.62


class Collector(CB.Bench):
    """The benchmark's scripted operator, recording at the recorder's rate while it works."""

    def __init__(self, server_args, data_dir: str, seed: int):
        super().__init__(["--data-dir", data_dir, "--seed", str(seed), *server_args], snapshots=False)
        self.sim.args.data_dir = data_dir
        self.rec_dt = 1.0 / self.sim.args.fps
        self._next_rec = 0.0

    def advance(self, dur: float):
        m, d, sim = self.m, self.d, self.sim
        t_end = d.time + dur
        while d.time < t_end - 1e-9:
            if d.time >= self._next_input - 1e-9:
                sim.latest_input = self._input()
                self._next_input = d.time + 1.0 / CB.INPUT_HZ
            sim.latest_input_t = time.perf_counter()
            sim.control(self.ctrl_dt)
            t_ctrl = d.time + self.ctrl_dt
            while d.time < t_ctrl - 1e-9:
                if sim.recorder.active and d.time >= self._next_rec - 1e-9:
                    sim.record_frame()
                    self._next_rec = d.time + self.rec_dt
                mujoco.mj_step(m, d)
                self.engine.step(d)
            self._measure()

    def episode(self, randomize: float) -> tuple[Path | None, float]:
        self.reset(randomize=randomize)
        self.phases, self.crossings = [], 0
        self.sim.recorder.start(self.d.time)
        self._next_rec = self.d.time
        self.fold()
        cov = self.engine.metrics(self.d)["coverage"]
        self.sim.save_episode(success=cov <= SUCCESS_COVERAGE)
        return self.sim.last_saved and self.sim.session.dir / self.sim.last_saved, cov

    def n_crossings(self) -> int:          # not needed for recording, and costly on fine cloth
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--cloth-engine", default="newton", choices=["newton", "gpu", "mujoco"])
    p.add_argument("--randomize", type=float, default=0.03, help="start pose jitter (see scene.reset)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--video", action="store_true", help="render each episode to an MP4 (front camera)")
    a, server_args = p.parse_known_args()
    c = Collector(["--cloth-engine", a.cloth_engine, *server_args], a.data_dir, a.seed)
    print(f"collecting {a.episodes} episode(s) with the {c.engine.name} cloth ({c.engine.info.nvert} points) "
          f"into {c.sim.session.dir}")
    saved = []
    for k in range(a.episodes):
        t0, s0 = time.perf_counter(), c.d.time
        path, cov = c.episode(a.randomize)
        wall = time.perf_counter() - t0
        print(f"  episode {k}: {path.name if path else 'not saved'}  coverage {cov:.3f} "
              f"({'success' if cov <= SUCCESS_COVERAGE else 'failure'})  "
              f"{wall:.0f} s wall for the fold ({c.engine.name})", flush=True)
        if path:
            saved.append(path)
    c.sim.recorder.wait()
    if a.video:
        for path in saved:
            out = path.with_suffix(".mp4")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "replay.py"), str(path), "--video", str(out),
                            "--camera", "front", "--width", "960", "--height", "720"], check=True)
            print(f"  video -> {out}")


if __name__ == "__main__":
    main()
