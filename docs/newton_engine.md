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

## Calibrated denim, sewn jeans, path-traced video

The numbers above are for the 10 mm sheet at 16 x 8. The Newton engine's defaults are now 5 mm
points, 16 x 16 sweeps, fitted denim and a sewn garment (`config/cloth.toml [newton]`).

**Calibration** (`scripts/fabric_calibrate.py`). VBD's stiffnesses are per hinge and per
triangle, so the values that make cloth behave like denim depend on the spacing and on the
sweeps. The script runs textile tests on a strip or disc of 12 oz denim (0.407 kg/m²) with the
engine's own solver and fits:

| Test | Target | 10 mm, 16 x 8 | 5 mm, 16 x 16 (defaults) |
|---|---|---|---|
| Stretch at 49 N/m (Kawabata EMT) | 1 % (denim with its yarns straight) | tri_ke 655 → 3 % (fitted to 3 %) | tri_ke 2890 → 0.99 % |
| Cantilever droop at 7 cm (ASTM D1388, bending length 3.5 cm) | 41.5° | edge_ke 0.32 → 41.4° | edge_ke 6.61 → 41.2° |
| Cusick drape coefficient (check, not fitted) | 0.6–0.85 | 0.95 | 0.92 |

- **The 10 mm mesh can't drape like denim.** A 30 cm disc stays nearly flat at any bending stiffness.
- **The drape at 5 mm is a little stiffer than real denim.**
- **The old values were far off at 5 mm.** edge_ke 5 drooped 75° and tri_ke 1e4 stretched 0.95 %.
- **At 16 x 8, VBD stops converging for stiff hinges at 5 mm.** The droop hardly depends on the
  stiffness there, and the script warns when that happens.
- **The stretch target is 1 %, not the 3 % a tester reads.** Most of the 3 % is the crimp of the
  yarns straightening at almost no load. Fitted to 3 % (tri_ke 820), the linear model let the
  folded jeans stretch 16 % (p99), and the stretch stayed.

**Sewn jeans** (`cloth.GARMENT_PARTS`). The waistband, fly, hems, out- and inseams, back
pockets and front pocket bags are heavier and stiffer than plain denim, by the layers sewn
there. The jeans weigh 0.46 kg. Real 12 oz jeans weigh 0.6–0.7 kg; the rest is hardware and
seam allowances, which the model leaves out.

**Fold, scripted operator, seed 0**

| | 10 mm sheet, 16 x 8 | 5 mm sewn, 16 x 16 |
|---|---|---|
| Points | 8 887 | 36 179 |
| Thread stretch p99, median over the episode / worst moment / end | 2.4 % / 3.9 % / — | 4.2 % / 6.7 % / 3.4 % |
| Layers through each other | 0 | 0 |
| Final footprint | 0.45 (success) | 0.50 (success) |
| Wall time for the 25 s fold | 94 s | 414 s |

**Settings tried on the way, 5 mm sewn**

| Settings | Stretch p99 |
|---|---|
| tri_ke 820, 16 x 8 | 16 % |
| tri_ke 1e4, 16 x 8 | 7.8 % |
| tri_ke 1e4, 16 x 16 | 6.3 % |
| tri_ke 820, 16 x 8, no structure | 27 % (84 % at the end) |
| tri_ke 820, 16 x 8, strain limiting (Provot) at 4 % | 7.4 % |

The structure helps.

Strain limiting didn't work and was removed. The stretch at the end sits along the inseams of
the folded stack, and no layers cross. With friction switched off it relaxes only partly. A
strain limiter shortened the threads, and the solver lengthened them again within a frame.
What pushes them is not found yet.

**Path-traced video** (`scripts/render_cycles.py`). Episodes from the gpu or newton cloth are
rendered with Blender Cycles (bpy 4.2, Python 3.11, OptiX, denoised), from the recorded robot
states and cloth vertices. The jeans are subdivided, given 0.9 mm thickness and a fabric shader
(sheen, twill bump). The ceiling panel and a window are area lights. At 1080p and 64 samples a
frame takes ~2 s on the RTX 4090, so a 25 s episode renders in ~30 min.

### Where the stretch came from

These were measured on the scripted fold with the 5 mm sewn jeans. Stretch is the 99th percentile
of the threads; "median" is over the episode.

| Change | Median | End | Footprint |
|---|---|---|---|
| Baseline (this section's defaults) | 4.2 % | 3.4 % | 0.50 |
| Seam creases at rest flat (all rest angles 0) | 5.6 % | 5.6 % | 0.38 |
| Seam crease hinges free | 4.4 % | 4.3 % | 0.61 |
| Layers 1 mm apart instead of 2 | 4.0 % | 3.8 % | 0.51 |
| 32 x 8 sweeps instead of 16 x 16 | 3.7 % | 3.5 % | 0.61 |
| Seam band bending x2 instead of x6 | 3.3 % | 2.8 % | 0.63 |
| **Tethers along the fabric, relaxed pinch, slack scaled (kept)** | **3.2 %** | **2.8 %** | **0.625** |

The seam-band row was measured with the kept tethers already in place, so compare it with the
kept row: it changed nothing, and the factor stays 6.

What was found and kept:

- **The pinch froze stretched fabric.** The closing fingers drag the fabric between the pads
  outward before the pinch triggers: threads were 11 % stretched a frame before it and 16 % while
  held. The patch is now relaxed to its fabric lengths when the pinch closes, and held threads
  stay within 4.5 %.
- **The tethers didn't reach the rest of the garment.** They ran in straight pattern lines
  within one panel, so the other leg and the hips were never tethered. A leg laid over the other
  dragged them across the table faster than VBD's local sweeps carry a pull, and table friction
  then held the stretch. Tethers now run along the fabric (multi-source Dijkstra over the mesh,
  through the seams): long-range attachments as Kim et al. (2012) define them.
- **The tether slack was fixed at 1 mm.** That was tuned at 10 mm and is a fifth of an edge at
  5 mm. It is now a tenth of the spacing.

At the end, 368 threads are more than 4 % stretched, against 534 before. 96 % of them are
within 3 cm of a seam, mostly along the fold line of the leg laid over. A sheet can't be folded
flat along a curved line (the tapered inseam) without stretching or wrinkling. Real denim
wrinkles there finer than 5 mm points can form.

**The scripted fold now ends at 0.625, just over the 0.62 success line.** Its fixed 0.40 m
carry lays the far leg ~20 cm past the near one, so the second pinch catches one leg. The old
straight-across tethers dragged the other leg along, which a real gripper wouldn't do. Mirroring
the carry and an arc over the leg didn't fix it: the free hem end swings past and lands beside
the other leg. The choreography (scripts/cloth_bench.py fold) needs reworking; it is unchanged
here so the numbers stay comparable.
