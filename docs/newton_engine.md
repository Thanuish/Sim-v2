# The Newton VBD cloth engine (Phase 1)

`--cloth-engine newton` simulates the jeans with Newton's VBD solver (Vertex Block Descent),
next to MuJoCo, which only simulates the robots (`vrteleop/newton_cloth.py`). Measured on an
RTX 4090 with Newton 1.7.0 and Warp 1.18.0 (`docs/baseline.md` has the workstation).

**Outcome: the Phase 1 target is not met on this GPU.** No setting is both real time and
denim-like. The engine is therefore not the VR default: VR keeps the GPU (Taichi XPBD) engine,
and Newton records data offline in simulation time (`scripts/collect.py`), where speed doesn't
matter.

## What it does

- **Robot:** every collision geom of the arms and grippers is a kinematic Newton body with its
  real shape: capsules, cylinders, boxes, and meshes as the convex hulls MuJoCo collides with.
  Poses at the start and end of each 1/60 s frame come from MuJoCo and are interpolated across
  the substeps. The solver gets the start of a substep as the previous pose, so its friction
  sees the robot move. Coupling is one-way (robot -> cloth); the cloth's force on each arm is
  read from VBD's contact forces and recorded as `teleop/cloth_contact_force`.
- **Cloth:** membrane stretch and bending, self-contact (vertex-triangle and edge-edge,
  detected every substep), and robot / table contact on the whole surface (edges and faces).
  Each frame runs as one CUDA graph.
- **Grasp:** `[grasp] model = "pinch"` makes the points between the pads kinematic, riding with
  the gripper until the pads open by `release_gap`. While a gripper holds, its fingers stop
  touching the fabric. `"friction"` attaches nothing: the pads squeeze the fabric and friction
  holds it.
- **Tethers:** long-range attachments to the held points (Kim et al. 2012), as the GPU engine
  has. In friction mode they anchor on the points between closed pads; they hold nothing and
  only stop overstretching.
- **Starts:** a settle before each episode, plus `--randomize` and `--cloth-init crumpled` (also
  added to the GPU engine, which lacked them).

Patterns taken from RoboSimStudio (Apache-2.0, see `NOTICE`): kinematic held particles, fingers
that stop colliding while holding, per-shape pad friction through Newton's
`sqrt(cloth mu x shape mu)` rule, the solver's previous body pose taken from the substep start,
and a settle before the episode.

## Why it isn't real time

**Cost.** Per-kernel GPU timing at 14,111 points (4 substeps x 8 iterations, full self-contact):

| Kernel group | ms per frame |
|---|---|
| self-contact force accumulation (every iteration) | 9.2 |
| edge-edge contact detection | 3.4 |
| displacement truncation | 2.7 |
| robot / table contact | 2.0 |
| elasticity solve | 1.6 |
| total | ~23.9 |

The flat jeans are two layers in contact everywhere, so contact work runs over the whole
garment in every iteration. Full-surface robot contact is *cheaper* than points-only here
(23 vs 36 ms): the signed-distance path beats per-point mesh queries.

**Stretch.** VBD needs about 128 sweeps per frame (substeps x iterations) before a lifted
garment stops stretching. On a bare Newton sheet, without any of this project's code (6,000
points, a corner patch lifted 20 cm in 1 s; stretch = 99th percentile of the threads):

| substeps x iterations | 4 x 8 | 8 x 4 | 10 x 4 | 16 x 4 | 4 x 32 | 16 x 8 |
|---|---|---|---|---|---|---|
| stretch | 23.4 % | 18.7 % | 14.1 % | 4.6 % | 8.1 % | 1.5 % |

Ten times the areal density or ten times the stretch stiffness changes nothing: this is the
solver's convergence, not the fabric. Small time steps matter more than iterations.

**The trade-off.** The jeans, same lift, with tethers and full self-contact:

| spacing | points | substeps x iterations | stretch | GPU ms / frame | real-time factor |
|---|---|---|---|---|---|
| 8 mm | 14,111 | 4 x 8 (no tethers) | 25.5 % | 19.7 | ~0.8 |
| 8 mm | 14,111 | 4 x 8 | 12.4 % | 19.5 | ~0.8 |
| 12 mm | ~6,400 | 8 x 8 | 3.5 % | 22.3 | ~0.75 |
| 12 mm | ~6,400 | 16 x 8 | 2.2 % | 40.3 | ~0.4 |
| 10 mm | 8,887 | 16 x 8 | 2.4 % | 51.5 | ~0.3 |
| 8 mm | 14,111 | 16 x 8 | 2.7 % | 66.5 | ~0.25 |

Real denim stretches 1-3 %. The largest garment that stays real time here (~6,000 points)
already stretches 3.5 %, and the spec's minimum of 15,000 points runs at about a quarter of real
time once it is denim-like. `config/cloth.toml` ships the accurate setting (10 mm, 16 x 8).

## Benchmarks

`python scripts/cloth_bench.py --test all --cloth-engine newton` at the shipped settings (10 mm,
8,887 points, 16 x 8, pinch grasp), next to the other two engines (MuJoCo: the same bench;
GPU: `gpu_cloth_test.py` and `teleop_trials.py`, since the bench's MuJoCo-only measurements
were only made engine-independent in this phase):

| | MuJoCo cloth | GPU (Taichi) cloth | Newton VBD |
|---|---|---|---|
| points | 303 | 8,887 | 8,887 |
| triangle crossings (fold) | 0 | 17-25 | **0** |
| thread stretch, 99th pct / worst edge | 2.3 % / 7.6 % | not measured | 5.1 % / 77 % (next to the pinch) |
| rest jitter | 0.95 mm/s | not measured | **0.26 mm/s** |
| fabric below the table top | 12.0 mm (contact spheres) | not measured | 0.0 mm |
| pinched (release trials) | 18/18 | 8/8 per arm (teleop trials) | 18/18 |
| released fabric following the gripper > 10 mm (short flick) | 1/6 | not measured | **6/6, up to 92 mm** |
| opened in the air, fabric still hanging > 50 mm | 6/6 | not measured | 3/6 (2 fully, 334 mm) |
| fabric on a gripper that never pinched (poke / sweep) | 0/16 | 0/16 per arm | 0/16 |
| cost, s wall per s simulated | 1.00 | 0.09 | 3.41 |

The Newton cloth never passes through itself and lies still, but released fabric sticks to the
fingers after a short flick of the stick: the fingers start touching the fabric again while it
is still wrapped around them, and friction holds it. A full open on the table releases cleanly.
This is the next thing to fix; `friction` grasp mode is not measured yet.

The MuJoCo column is with the arm feedforward of `fd59ed8`. Without it the same fold stretches
1.9 % and the table penetration is 3.2 mm: the arms now move at the commanded speed instead of
lagging 0.1 s behind it, which pulls the coarse MuJoCo cloth harder.

## Offline data collection

`scripts/collect.py` records the scripted fold as ordinary episodes (the VR HDF5 layout, the
session folder, `index.csv`, optionally an MP4), in simulation time:

    python scripts/collect.py --episodes 5 --video

First episode on this workstation: 25.1 s, 752 frames, footprint 61.6 % of flat (success),
189 s of wall time (with the benchmark sharing the GPU).
