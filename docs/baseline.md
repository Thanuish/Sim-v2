# Baseline (Phase 0)

The numbers every later change is compared against: the two cloth engines as they were at commit
`fd59ed8`, measured with the repository's own checks. Measured 2026-10-26.

## The workstation

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (24564 MiB) |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (driver); Warp reports CUDA Toolkit 13.4, device `cuda:0`, sm_89 |
| CPU | 13th Gen Intel Core i9-13900KF, 32 threads |
| RAM | 62 GB |
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0-38-generic, no display (`MUJOCO_GL=egl` for offscreen rendering) |
| Python | 3.12.3, venv made with `uv` (the system Python has no `ensurepip`) |
| MuJoCo / Taichi | 3.14.0 / 1.7.4 (CUDA backend) |
| Newton / Warp | 1.7.0 / 1.18.0 (installed for Phase 1; `wp.get_cuda_devices()` -> `['cuda:0']`) |

## MuJoCo cloth engine (`--cloth-engine mujoco`, 5.5 cm, 303 vertices)

`python scripts/cloth_bench.py --test all` (simulation time, a scripted two-arm fold):

| Metric | Value | Note |
|---|---|---|
| stretch, worst thread | 4.28 % | 99th percentile 2.03 %; real denim < 2-3 % |
| solver iterations | 27.7 mean, 30 max | cap 30 |
| penetration | 6.29 mm | table 6.29, cloth-cloth 2.29, grippers 0.70 (sphere radius 5.0 mm) |
| triangle crossings | 0 | worst phase end |
| rest jitter | 0.56 mm/s | RMS vertex speed lying still |
| slips out of a pinch | 0 | |
| follow (released fabric rising with the gripper) | 24.7 mm | 3 releases |
| stuck after opening | 0 | |
| final coverage | 0.552 | 1 flat .. ~0.3 folded |
| cost | 1.01 s wall per sim s | real time; phases 0.70-1.09 |

Release trials (6 points x short flick / full open / open in the air): pinched 18/18, still held
after opening 0, follow mean 3.0 / max 17.3 mm; **opened in the air, fabric still hanging > 50 mm in
6/6** (the coarse cloth stays draped over the open fingers).

Poke trials (open / closed gripper pushed into or swept through a leg, then lifted 25 cm): fabric
hanging on an unpinched gripper in 0/16, crossings afterwards 0/16.

`python scripts/teleop_trials.py --cloth-engine mujoco`: lifted 8/8 per arm, left on the open
gripper 0/8, poke/sweep 0/16 per arm; real-time factor 1.06.

## GPU cloth engine (`--cloth-engine gpu`, Taichi XPBD)

`cloth_bench.py` does not run on this engine: it reads the MuJoCo engine's internals
(`sim.cloth.grasp`, the flex) and stops with `AttributeError: 'GPUCloth' object has no attribute
'grasp'`. Phase 1 makes it engine-independent. The existing GPU checks:

`python scripts/gpu_cloth_test.py` (the scripted fold without robots, pinned patches as grippers):

| Phase | 1.2 cm (6383 pts), ms/frame | crossings | 1.0 cm (8887 pts, the shipped setting), ms/frame | crossings |
|---|---|---|---|---|
| settle | 1.2 | 0 | 1.3 | 0 |
| lift | 1.6 | 0 | 1.7 | 0 |
| carry | 1.2 | 0 | 1.2 | **19** |
| lay down | 1.2 | 0 | 1.3 | **17** |
| release | 0.9 | 0 | 1.0 | **17** |
| hems lift | 1.0 | 0 | 1.1 | **17** |
| hems carry | 1.0 | **8** | 1.2 | **21** |
| hems down | 1.1 | **24** | 1.2 | **21** |
| settle | 0.9 | 0 | 1.0 | **25** |
| real-time factor (worst phase) | 10.7 | | 10.0 | |
| final footprint / height | 0.29 x 0.76 m, 88 mm | | 0.29 x 0.74 m, 69 mm | |

At the shipped 1 cm resolution the XPBD cloth passes through itself from the first carry on and
never recovers; folded layers rest 9.5-12 mm apart (point-point self-collision at `thickness` x
spacing), several times a real denim stack.

`python scripts/teleop_trials.py --cloth-engine gpu`: lifted 8/8 per arm, left on the open gripper
0/8, poke/sweep 0/16 per arm; real-time factor 11.1.

Not measured on this engine by any existing tool: thread stretch while hanging, penetration, rest
jitter, slip and follow. Phase 1's engine-independent bench adds them for every engine.

## Full server, scripted operator

`python -m vrteleop.server --http` plus `python scripts/fake_client.py --fold` (real time, GPU cloth,
three camera screens streaming): real-time factor 0.99-1.0 throughout.

Profile of the sim loop (py-spy, busy time only): GPU cloth kernels 50 %; the fold score 22 %
(a per-triangle Python rasteriser, 45 ms per call once a second: a visible hitch in VR; fixed in
`fd59ed8`, now 13 ms on a worker thread); camera screens 11 %.

## Arm tracking

`python scripts/tracking_test.py` (blocks task, 72 Hz headset input):

| | error mean / max [mm] | delay [ms] | overshoot [mm] |
|---|---|---|---|
| plain PD servos (before `fd59ed8`) | 23.5 / 53.6 | 90 | 0.0 |
| velocity feedforward (since `fd59ed8`) | 2.4 / 7.9 | 0 | 3.4 |
