# Demo: folding jeans with Newton VBD cloth

What to show, in order, and the commands behind each step. Everything runs on the Ubuntu box
with the RTX 4090 (`MUJOCO_GL=egl` when there is no display).

## 1. The video (2 min)

`jeans_fold_v2.mp4`: a scripted two-arm fold recorded with the Newton VBD cloth engine and
rendered with Blender Cycles. It shows:

- **Cloth:** 5 mm sewn denim jeans, 36,179 points, full self-collision.
- **Fabric:** stiffness fitted to textile tests (ASTM D1388 cantilever, Kawabata stretch).
- **Robots:** two UR5e arms with Robotiq 2F-85 grippers, driven through the same IK and grasp
  path as the VR headset.

Talking points:

- **The fold.** The far leg is laid over the near one as a mirror image, then the hems are
  folded up. The jeans end at 0.43 of their flat footprint (success is 0.62 or less).
- **Physics.** Layers never pass through each other (0 crossings). Thread stretch is about 3 %
  at the 99th percentile, against 1-3 % for real denim. Nothing sticks to the grippers.
- **Why it isn't live.** VBD at denim-like stiffness is about 30x slower than real time on a
  4090, so it is used for offline data collection. The VR teleop runs MuJoCo or GPU cloth
  live. See `docs/newton_engine.md`.

## 2. Record a new episode live (start it at the beginning, ~14 min)

```bash
MUJOCO_GL=egl .venv/bin/python scripts/collect.py --episodes 1 --cloth-engine newton --data-dir demo_data
```

It prints the coverage and success when it finishes. The episode is written in the same HDF5
layout as the VR recordings: robot state, actions and every cloth vertex for every frame.

```bash
.venv/bin/python scripts/replay.py demo_data/jeans/*/episode_0000.hdf5 --info
.venv/bin/python scripts/replay.py demo_data/jeans/*/episode_0000.hdf5 --video quick.mp4 --camera front   # ~1 min
```

## 3. The physics numbers (optional, ~15 min)

```bash
MUJOCO_GL=egl .venv/bin/python scripts/cloth_bench.py --task jeans --cloth-engine newton --out bench
```

This prints the stretch, penetration, crossings, stuck grippers and coverage, and writes
`bench.png`, a contact sheet of every phase of the fold.

## 4. Photoreal render of any episode (~25 min at 1080p)

```bash
.venv/bin/python scripts/render_cycles.py demo_data/jeans/*/episode_0000.hdf5 --out fold.mp4
# add --resume to continue an interrupted render, or --width 960 --height 540 --samples 16 for a quick preview
```

## Known limits (say them before someone asks)

- **Not real time.** Newton VBD at 5 mm takes about 30 s of compute per simulated second.
- **The first fold doesn't stay put.** Let go, the laid-over leg slides ~13 cm out, pushed by
  the stiff waistband fold. The hems fold then makes up for it.
- **Stretch.** About 3 % of stretch remains along the curved fold line at the inseam. A finer
  mesh would reduce it.
