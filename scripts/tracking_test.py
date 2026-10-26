"""How closely the robot follows the operator's hand: a scripted hand motion through the real
teleop path (clutch mapping, IK, joint servos), with the headset's input rate.

    python scripts/tracking_test.py                    # both arms, with and without feedforward
    python scripts/tracking_test.py --input-hz 90

Reports per arm the gripper's position error against the commanded target (mean / max), the
delay that best explains it (how far the robot trails your hand), and the overshoot when the hand
stops. Uses the blocks task (no cloth), so it only measures the arms.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from vrteleop import scene as S  # noqa: E402
from vrteleop.server import TeleopSim, build_parser  # noqa: E402

CTRL = {"left": np.array([0.0, 0.25, 1.1]), "right": np.array([0.0, -0.25, 1.1])}


def hand_path(t):
    """Hand offset from where the grip was squeezed [m]: a horizontal figure eight (0.5 Hz,
    peak ~0.5 m/s), then a fast 18 cm reach that stops dead."""
    if t < 4.0:
        w = 2 * np.pi * 0.5
        return np.array([0.10 * np.sin(w * t), 0.12 * np.sin(2 * w * t) / 2, 0.05 * np.sin(w * t)])
    a = np.clip((t - 4.5) / 0.5, 0, 1)                   # 0.5 s minimum-jerk reach
    s = a ** 3 * (10 - 15 * a + 6 * a * a)
    return s * np.array([0.15, 0.0, 0.10])                # up and out: clear of the blocks


def run(feedforward: bool, input_hz: float, duration: float = 6.0):
    args = build_parser().parse_args(
        ["--http", "--no-preview", "--stream-cams", "", "--randomize", "0", "--task", "blocks",
         "--data-dir", tempfile.mkdtemp(prefix="tracking-")] + ([] if feedforward else ["--no-feedforward"]))
    sim = TeleopSim(args)
    m, d = sim.m, sim.d
    ctrl_dt = 1.0 / args.control_hz
    sim.reset(randomize=0.0)
    for _ in range(50):                                  # settle at home
        mujoco.mj_step(m, d)
    t0, next_in, msg = d.time, d.time, None
    log = {s: [] for s in S.SIDES}
    while d.time - t0 < duration:
        t = d.time - t0
        if d.time >= next_in - 1e-9:                     # the headset's input rate
            msg = {"head": [-0.4, 0, 1.65, 1, 0, 0, 0], "t": t}      # t: the headset clock
            for k, s in enumerate(S.SIDES):
                p = hand_path(t) * (1 if k == 0 else np.array([1, -1, 1]))
                msg[s] = {"pose": list(CTRL[s] + p) + [1, 0, 0, 0], "grip": 1.0, "stick": [0, 0], "epoch": 0}
            next_in += 1.0 / input_hz
        sim.latest_input = msg
        sim.latest_input_t = __import__("time").perf_counter()
        sim.control(ctrl_dt)
        t_c = d.time + ctrl_dt
        while d.time < t_c - 1e-9:
            mujoco.mj_step(m, d)
        for s in S.SIDES:
            log[s].append((d.time - t0, sim.arms.teleop[s].target_pos.copy(), sim.ee_pose(s)[0]))
    out = {}
    for s in S.SIDES:
        t = np.array([r[0] for r in log[s]])
        tp = np.array([r[1] for r in log[s]])
        ep = np.array([r[2] for r in log[s]])
        moving = t > 0.5                                 # skip the clutch engaging
        err = np.linalg.norm(tp - ep, axis=1)[moving]
        n = int(round(0.25 / ctrl_dt))
        lag = min(range(n), key=lambda L: np.mean(np.linalg.norm(tp[moving][:len(err) - L] - ep[moving][L:], axis=1)))
        stop = t > 5.0                                   # after the reach: overshoot past the target
        goal = tp[-1]
        direction = (goal - tp[np.searchsorted(t, 4.5)])
        direction /= np.linalg.norm(direction)
        over = max(0.0, float(((ep[stop] - goal) @ direction).max()))
        out[s] = dict(mean=1e3 * err.mean(), max=1e3 * err.max(), lag=1e3 * lag * ctrl_dt,
                      overshoot=1e3 * over, settle=1e3 * np.linalg.norm(ep[-1] - goal))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-hz", type=float, default=72.0, help="headset input rate (Quest: 72-120)")
    a = ap.parse_args()
    print(f"\n  input {a.input_hz:.0f} Hz      error mean / max [mm]   delay [ms]   overshoot [mm]   final [mm]")
    for ff in (False, True):
        r = run(ff, a.input_hz)
        for s in S.SIDES:
            v = r[s]
            print(f"  {'feedforward' if ff else 'plain PD   '}  {s:5s}  {v['mean']:7.1f} / {v['max']:5.1f}"
                  f"          {v['lag']:5.0f}        {v['overshoot']:6.1f}          {v['settle']:5.1f}")


if __name__ == "__main__":
    main()
