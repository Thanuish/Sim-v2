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
| thread stretch, 99th pct / worst edge | 2.3 % / 7.6 % | not measured | 4.1 % / 15.9 % |
| rest jitter | 0.95 mm/s | not measured | **0.26 mm/s** |
| fabric below the table top | 12.0 mm (contact spheres) | not measured | 0.0 mm |
| pinched (release trials) | 18/18 | 8/8 per arm (teleop trials) | 18/18 |
| released fabric following the gripper > 10 mm | 1/12 | not measured | **0/12** |
| opened in the air, fabric still hanging on the gripper | 6/6 | not measured | **0/6** |
| fabric on a gripper that never pinched (poke / sweep) | 0/16 | 0/16 per arm | 0/16 |
| final coverage of the fold (lower = more compact) | 0.53 | | 0.46 |
| cost, s wall per s simulated | 1.00 | 0.09 | 3.34 |

Three fixes got the Newton column here; before them released fabric stuck to the fingers (6/12)
and pinching a folded stack pushed its layers through each other (89-286 crossings):

- **Fingers wait for the fabric to leave.** While a gripper holds, its fingers don't touch the
  fabric, so the fabric beside the pinch drapes through the finger links. Colliding again at the
  moment of release hooked it on them. Now the fingers collide again only once no fabric is
  inside their region (`NewtonVBDCloth.FINGER_BOX`).
- **The pinch fires when the fingers squeeze fabric.** MuJoCo's jaws never feel this cloth
  (one-way coupling), so they closed to 8 mm on a folded leg four layers thick and crushed it.
  The pinch now also triggers when the fabric pushes on the fingers with more than 2 N while the
  pads are within 25 mm of each other.
- **Tethers never pull.** A tether is at least as long as the point's distance to its anchor when
  the pinch starts; before, tethers on a stretched stack snapped points through their neighbours.

The air test now counts only fabric within 10 cm of the opened gripper: in two trials the leg
had fallen against the other arm, which the old global measure counted as hanging.

The MuJoCo column is with the arm feedforward of `fd59ed8`. Without it the same fold stretches
1.9 % and the table penetration is 3.2 mm: the arms now move at the commanded speed instead of
lagging 0.1 s behind it, which pulls the coarse MuJoCo cloth harder.

## Grasp modes: pinch and friction

The same benchmark with `--grasp-model pinch` and `--grasp-model friction`. "Carried" is the
highest fabric within 10 cm of the gripper after lifting it 10 cm (table trials) or 30 cm (air
trials, measured after hanging 0.5 s).

| | pinch | friction |
|---|---|---|
| carried, release trials | 18/18 (115 mm / 313 mm) | 18/18 (123 mm / 319 mm) |
| dropped after a full open, on the table | 6/6 | 6/6 |
| dropped after a full open, in the air | 6/6 | 6/6 |
| dropped after a short flick of the stick, on the table | 6/6 | not measured (see below) |
| fold: gripper still holding after the operator opened it | 0 | **1** (the hems, short flick) |
| fold: released fabric rising with the gripper | 0 mm | **23.5 mm** |
| fold: triangle crossings | 0 | 0 |
| fold: thread stretch, 99th pct / worst edge | 4.1 % / 15.9 % | 6.4 % / 61 % |
| fold: final coverage | 0.46 | 0.43 |
| poke / sweep: fabric on a gripper that never pinched | 0/16 | 0/16 |
| cost, s wall per s simulated | 3.34 | 3.82 |

Friction holds the fabric with nothing attached, as reliably as the pinch. Two things are worse:

- **A short flick doesn't reliably let go.** Pinch mode releases once the pads open 4 mm (the
  operator's flick). With friction, the jaws a third open still had the folded hems between the
  fingers, and the fabric went up with the gripper (23.5 mm, counted as still holding). The bench
  couldn't measure the table flick trials in this mode: its follow measure starts when the engine
  stops reporting fabric between the pads, which didn't happen. A full open always releases.
- **More stretch.** Without kinematic held points the squeezed patch is held only by friction, so
  the threads next to the fingers stretch more (worst edge 61 %).

Recommendation: keep `pinch` as the default. `friction` is the more physical model and carries
fabric as well, so it is usable for data whose releases are full opens; the short flick needs
work first.

## Offline data collection

`scripts/collect.py` records the scripted fold as ordinary episodes (the VR HDF5 layout, the
session folder, `index.csv`, optionally an MP4), in simulation time:

    python scripts/collect.py --episodes 5 --video

First episode on this workstation: 25.1 s, 752 frames, footprint 61.6 % of flat (success),
189 s of wall time (with the benchmark sharing the GPU).
