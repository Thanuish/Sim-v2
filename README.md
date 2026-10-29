# MuJoCo VR Teleop: fold jeans with two robot arms in VR

Put on a **Meta Quest** headset and control two simulated robot arms (Universal Robots
**UR5e** with **Robotiq 2F-85** grippers) with your hands. The task is to fold a pair of
jeans lying on a table. Everything you do is recorded as training data for robot learning
(imitation learning), in the same format as ALOHA / ACT.

- The physics runs in [MuJoCo](https://mujoco.org) on your computer; the jeans are simulated
  on your graphics card.
- The headset only shows the scene in its web browser. Nothing has to be installed on it.
- There is a folding game: three folds in order, against the clock.
- Offline, the jeans can be simulated with NVIDIA Newton's VBD solver (calibrated, sewn denim)
  and rendered with Blender Cycles: see [docs/DEMO.md](docs/DEMO.md) and
  [docs/newton_engine.md](docs/newton_engine.md).

```
  Meta Quest (web browser)                       Your computer (Python)
 ┌──────────────────────────────┐            ┌─────────────────────────────────────────┐
 │ shows the robots and jeans   │ ◄───────── │ robot poses and cloth, up to 90x / s    │
 │ sends hand + head poses      │ ─────────► │ your hand motion -> arm joints (IK)     │
 │ buttons: record, reset, ...  │            │ MuJoCo physics (robots), GPU (jeans)    │
 └──────────────────────────────┘            │ records each demonstration to a file    │
                                             └─────────────────────────────────────────┘
```

---

## Contents

1. [What you need](#1-what-you-need)
2. [Install and start (Windows)](#2-install-and-start-windows)
3. [Install and start (Mac, Linux, Docker)](#3-install-and-start-mac-linux-docker)
4. [Using it in VR](#4-using-it-in-vr)
5. [The folding game](#5-the-folding-game)
6. [Recorded data](#6-recorded-data)
7. [The cloth simulation and its settings](#7-the-cloth-simulation-and-its-settings)
8. [Testing without a headset](#8-testing-without-a-headset)
9. [Control from Claude (MCP) and the control API](#9-control-from-claude-mcp-and-the-control-api)
10. [All server options](#10-all-server-options)
11. [Troubleshooting](#11-troubleshooting)
12. [How the code is organised](#12-how-the-code-is-organised)

---

## 1. What you need

| | |
|---|---|
| **Headset** | Meta Quest 2, 3, 3S or Pro, with the built-in Quest Browser |
| **Computer** | Windows 10/11, macOS or Linux. A graphics card for the jeans: any card with Vulkan (AMD, NVIDIA, Intel, including laptop graphics), NVIDIA CUDA, or Apple Metal. Without one the simulator still works, with a coarser cloth (see [section 7](#7-the-cloth-simulation-and-its-settings)). |
| **Software** | [Python 3.12](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads). For a USB connection to the headset: [Android platform-tools](https://developer.android.com/tools/releases/platform-tools) (the `adb` program). |
| **Cable** | A USB-C cable from the headset to the computer (recommended; Wi-Fi also works) |

Laptops: plug in the charger. On battery the graphics card slows down and the simulation runs
in slow motion.

---

## 2. Install and start (Windows)

### One-time setup

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/). In the
   installer, tick **"Add python.exe to PATH"**.
2. Install **Git** from [git-scm.com](https://git-scm.com/downloads).
3. Download [Android platform-tools](https://developer.android.com/tools/releases/platform-tools),
   unzip it, and add the `platform-tools` folder to your PATH so that the command `adb` works in
   a terminal.
4. Get the project. In a terminal, go to the folder where you want it and run:

   ```bash
   git clone https://github.com/Thanuish/Sim.git
   ```

5. Open the new `Sim` folder and double-click **`setup_windows.bat`**. It creates a private
   Python environment (`.venv`) and installs everything. This takes a few minutes.

6. On the headset, turn on **developer mode** once (Meta Horizon phone app → your headset →
   Developer mode), so it can be reached over USB.

### Every time

1. Plug the headset into the computer with the USB-C cable. In the headset, allow USB
   debugging if it asks ("Always allow from this computer").
2. Double-click **`start.bat`**. Wait until the window shows `Open on your Quest browser`.
   The first start takes a little longer (about 30 s).
3. In the headset, open the **Quest Browser** and go to **`http://localhost:8080`**.
4. Press **Enter VR**. You are standing behind the robots, which are mounted on the table in
   front of you. Press **Y** if the view is not centred.

To stop, close the `start.bat` window.

**Unplugged the headset?** Plug it back in, then run `adb reverse tcp:8080 tcp:8080` in a
terminal (or just restart `start.bat`, which does this for you), and reload the page in the
Quest Browser.

**Wi-Fi instead of USB:** run `start.bat --https`, then open `https://<your-PC's-IP>:8443` on
the headset (the window prints the address). The headset warns about the certificate the
first time: choose **Advanced → Proceed**. Both devices must be on the same network.

---

## 3. Install and start (Mac, Linux, Docker)

### Mac

```bash
git clone https://github.com/Thanuish/Sim.git
cd Sim
./setup_mac.command          # one-time setup (installs Python 3.12 via uv, no admin needed)
./start.command              # start the server (HTTPS: open the printed https://... address on the headset)
```

For USB: install platform-tools (`brew install android-platform-tools`), plug in the headset,
run `adb reverse tcp:8080 tcp:8080`, start with `./start.command --http` and open
`http://localhost:8080` on the headset.

### Linux (or any system, by hand)

```bash
git clone https://github.com/Thanuish/Sim.git
cd Sim
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_assets.py                 # only if the assets/ folder is missing
python -m vrteleop.server --http       # then: adb reverse tcp:8080 tcp:8080, open http://localhost:8080
```

### Docker

Docker runs everything in a container. A container normally can't use the graphics card, so
the server uses the coarser MuJoCo cloth there (see [section 7](#7-the-cloth-simulation-and-its-settings)).
For the full GPU cloth, install it directly (above).

```bash
docker compose build          # once (and after changing the code)
docker compose up             # starts the server on http://localhost:8080
```

Connect the headset over USB with `adb reverse tcp:8080 tcp:8080` on the computer. For Wi-Fi,
tell the container the computer's IP address:

```bash
TELEOP_PUBLIC_HOST=192.168.1.50 TELEOP_ARGS="" docker compose up                     # Linux / Mac
```
```powershell
$env:TELEOP_PUBLIC_HOST="192.168.1.50"; $env:TELEOP_ARGS=""; docker compose up      # Windows
```

and open `https://192.168.1.50:8443` on the headset. With an NVIDIA card,
`docker compose up teleop-gpu` renders the camera screens on the GPU (needs the NVIDIA
Container Toolkit). Recordings are saved to the `data/` folder on your computer. Extra server
options go in `TELEOP_ARGS`, for example `TELEOP_ARGS="--http --task blocks"`.

---

## 4. Using it in VR

### Controls

| Button | What it does |
|---|---|
| **Grip** (side button, hold) | Take control of that arm. While you hold it, the gripper follows your hand. Let go to move your hand without moving the robot (like lifting a computer mouse). |
| **Thumbstick up / down** | Close / open that arm's gripper. It stays where you leave it. |
| **Thumbstick click** | Show or hide the live camera screens |
| **A** (right hand) | Start recording; press again to stop and save the demonstration as a success |
| **B** (right hand) | Throw away the current recording and reset |
| **X** (left hand) | Reset the scene (and start a new game) |
| **Y** (left hand) | Recenter the view on the robots |
| **Left trigger + left stick** | Move your viewpoint forward / back / sideways |
| **Right trigger + right stick up/down** | Move your viewpoint up / down |
| **Left trigger + Y** | Back to the default viewpoint |

A small panel on your left wrist shows the recording, the gripper openings, which arms you
control (■), a ✋ when a gripper holds fabric, and the folding game.

### How to pick up the jeans

1. Hold **Grip** and bring the **open** gripper down until the fingertips are just above the
   fabric.
2. Push the **thumbstick up** to close it. The fingertips swing down as they close and
   pinch the fabric. ✋ appears on the panel.
3. Move your hand: the fabric comes along. Every layer between the fingers is held, so you
   can pick up an edge (both layers) or a folded stack.
4. Push the **thumbstick down** a little to let go.

A gripper that is already closed does not grab fabric when you push it into the jeans, and
fabric slides off open fingers, just like with a real gripper.

### A typical fold

1. Left arm pinches the far leg at the waistband, right arm pinches the same leg at the hem.
   Lift it and lay it over the other leg.
2. Right arm pinches both hems and brings them up to the waist.
3. Fold once more.

### How your hand motion maps to the robot

The mapping is relative: when you squeeze the grip, your controller's position at that moment
is linked to the gripper's position. From then on the gripper moves and turns by the same
amount as your hand. You never need to calibrate, and you can cover large distances in several
strokes (grip, move, let go, move back, grip again). `--scale 1.5` makes the robot move 1.5x
your hand motion.

When you enter VR, your eyes are placed at the robot's "head": centred between the two arms,
slightly behind and above them. The arms then reach out in front of you like your own.

### Watching on the computer

Open `http://localhost:8080` in a browser on the computer. The view follows the headset
wearer's head; press `V` to orbit freely with the mouse, `H` to hide the panels, `F` for full
screen. The page also has Record / Reset buttons (`Space` = record, `R` = reset). Add `?lite`
to the address on older headsets to turn off shadows and reflections.

### Camera screens

Three live robot camera views (left wrist, head, right wrist) float behind the table, and
appear as thumbnails on the computer's page. Choose others with `--stream-cams`, for example
`--stream-cams overhead,front`, or turn them off with `--stream-cams ""` (this makes everything
run faster).

---

## 5. The folding game

The panels show the current step, a clock and your best time:

| Step | Goal | Counts when the jeans cover at most |
|---|---|---|
| 1 | Lay one leg over the other | 62 % of the table area they cover lying flat |
| 2 | Fold in half: bring the hems up to the waist | 35 % |
| 3 | Fold once more into a compact stack | 22 % |

A step only counts once nothing is being held and the fabric has stopped moving for a second,
so throwing the jeans around does not score. The clock starts when you first take control of
an arm and stops at the last step. The best time is kept while the server runs. Press **X** for
a new game. The steps and thresholds are in `vrteleop/fold_game.py`.

---

## 6. Recorded data

### Recording

Press **A** to start and **A** again to save (or **B** to throw it away). Every saved
demonstration is one file plus a summary and a preview video:

```
data/
  jeans/                         one folder per task
    index.csv                    one line per saved demonstration, all sessions (opens in Excel)
    2026-09-28_14-05-12/         one folder per server start
      session.json               who recorded it, on what computer, all settings
      episode_0000.hdf5          the demonstration (contents below)
      episode_0000.json          success, duration, fold result
      episode_0000.mp4           video of what the operator saw in the headset
```

Options: `--operator NAME` (default: your login name), `--session NAME` (a fixed folder name),
`--no-preview` (no MP4), `--data-dir PATH`. `python scripts/episodes.py` lists everything
recorded (`--list`, `--failed`, `--latest`).

### What is inside an episode file (HDF5)

| Key | Shape | Content |
|---|---|---|
| `observations/qpos` | (T, 14) | `[left arm 6 joints, left gripper 0..1, right arm 6, right gripper 0..1]` |
| `observations/qvel` | (T, 14) | joint velocities |
| `observations/ee_pos`, `ee_quat` | (T, 2, 3 / 4) | measured gripper pose (quaternion wxyz) |
| `observations/cloth_verts` | (T, N, 3) | every simulated point of the jeans |
| `observations/images/<camera>` | (T, H, W, 3) | camera images, only with `--cameras` |
| `observations/full_qpos`, `full_qvel` | (T, nq / nv) | the full MuJoCo state, for exact replay |
| `action` | (T, 14) | joint command `[left q 6, left gripper, right q 6, right gripper]` |
| `action_ee/pos`, `quat`, `gripper` | | the gripper targets the operator commanded |
| `teleop/engaged`, `head_pose`, `controller_pose` | | the raw VR signals |
| `teleop/cloth_grasp` | (T, 2, K) | the points each gripper pinches (-1 = none) |
| `cloth/faces`, `cloth/rest_uv` | | the jeans' sewing pattern |

Attributes: `fps` (30), `timestep`, `success`, `task`, `joint_names`, `camera_names`,
`final_coverage` (1 = spread flat, about 0.25 = folded in quarters), `final_height`,
`cloth_engine`, `cloth_config` (the exact cloth settings used), `model_xml`. The layout follows
ALOHA / ACT conventions, so converting to LeRobot is straightforward.

Camera images (`head_cam`, `overhead`, `front`, `left_gripper_wrist_cam`,
`right_gripper_wrist_cam`):

```bash
python -m vrteleop.server --cameras                         # all cameras, 320x240
python -m vrteleop.server --cameras --camera-names front,left_gripper_wrist_cam --img-w 640 --img-h 480
```

### Replay

```bash
python scripts/replay.py data/jeans/<session>/episode_0000.hdf5 --info           # what is inside
python scripts/replay.py data/jeans/<session>/episode_0000.hdf5 --video out.mp4 --camera front
python scripts/replay.py data/jeans/<session>/episode_0000.hdf5                  # interactive 3D viewer
```

On Windows, **`replay.bat`** opens the newest recording in the viewer (Mac:
`./replay.command`; macOS needs `mjpython` instead of `python` for the viewer).
`--camera operator` films what the headset wearer saw.

For a photorealistic video of a jeans episode recorded with the `gpu` or `newton` cloth, render
it with Blender's Cycles path tracer (NVIDIA GPU; Blender's Python module needs Python 3.11):

```bash
uv venv -p 3.11 .venv-blender && uv pip install -p .venv-blender/bin/python bpy==4.2.0 numpy
python scripts/render_cycles.py data/jeans/<session>/episode_0000.hdf5 --out fold.mp4   # ~2 s/frame at 1080p
```

---

## 7. The cloth simulation and its settings

### The jeans

A full-size pair (1.0 m long, 45 cm across the hips, 0.65 kg) built like real jeans: a front
and a back panel sewn together along the outer seams and the inseams, open at the waist and at
both hems, so the legs are tubes. They have the fly, pockets, rivets, the yoke, back pockets
and the leather patch, and a paler inside.

### Two cloth engines

| | **GPU cloth** (default) | **MuJoCo cloth** (fallback) |
|---|---|---|
| Runs on | the graphics card | the processor |
| Detail | 1 cm, ~9000 points | 5.5 cm, ~280 points |
| Looks | round folds, sharp creases, limp drape | coarser folds |
| Used when | a GPU is available | no GPU (Docker, some PCs), or `--cloth-engine mujoco` |

The server picks the GPU cloth by itself and switches to the MuJoCo cloth if no graphics card
is usable (it says so in the window).

How the GPU cloth works, in short (`vrteleop/gpu_cloth.py`, XPBD written in Taichi):

- **Threads** along and across the legs barely stretch; the fabric shears easily on the bias,
  so it drapes; denim bending stiffness keeps folds round.
- **No passing through itself**: every point keeps a layer's thickness away from every other
  part of the fabric. Fabric on fabric has friction (μ 0.5), so folds stay where you put them.
- **Robots**: the arms' and fingers' collision shapes push the fabric away; the table has
  friction (μ 0.4).
- **Holding**: closing the gripper with fabric between the fingers pinches every layer in
  between. While held, each point of the jeans is tied to its nearest pinched point by an
  invisible thread of the fabric's true length (long-range attachments, Kim et al. 2012), so a
  lifted leg hangs at its real length instead of stretching like rubber.
- **Letting go**: only a gripper that is pinching has friction on the fabric, so fabric slides
  off open fingers instead of hanging on them.

### Changing the cloth: `config/cloth.toml`

Every value is in **`config/cloth.toml`**, each with a comment explaining what it does, its
unit and sensible values: the garment's size and weight, stretch, bending, shear, friction,
the GPU cloth's resolution (`[gpu]`), and the grip. Edit a value and restart the server.

The file has several garments (`jeans`, `shorts`, `stretch_jeans`). To add your own, copy a
block as `[garments.NAME]`, usually starting with `inherits = "jeans"` and changing only what
differs, then start with `start.bat --garment NAME`. Spelling mistakes and impossible sizes are
reported when the server starts. Each recording stores the exact settings it used, so old
recordings replay correctly after you change the file.

---

## 8. Testing without a headset

| Command | What it checks |
|---|---|
| `python scripts/fake_client.py --fold` (with the server running with `--http`) | pretends to be a headset: folds the jeans with both arms and records one episode |
| `python scripts/teleop_trials.py --cloth-engine gpu` | pinch / lift / release and poke / sweep trials with both arms: does a pinch hold, does fabric stay on an open gripper, speed |
| `python scripts/teleop_trials.py --cloth-engine gpu --human EPISODE.hdf5` | replays a recorded VR session into the current cloth |
| `python scripts/gpu_cloth_test.py` | scripted fold of the GPU cloth alone: speed, self-crossings, pictures |
| `python scripts/cloth_bench.py --test all` | the MuJoCo cloth: stretch, penetration, release, cost |

---

## 9. Control from Claude (MCP) and the control API

While the server runs it accepts commands from the same computer (not from the network):

| Request | Does |
|---|---|
| `GET /api/status` | everything the panels show, as JSON |
| `POST /api/command` `{"cmd": "reset"}` | `reset`, `record_toggle`, `save_fail`, `discard`, `cams_toggle` |
| `GET /api/cloth` | the current garment and cloth settings |
| `POST /api/reload` `{"garment": "shorts"}` | switch garment without restarting (the headset reloads by itself) |

`vrteleop/mcp_server.py` makes these available to Claude (or any MCP client): read the status,
send commands, switch garments, read and change cloth settings (comments in the file are kept,
values are checked before saving), run the cloth test, list and inspect recordings. Claude
Code finds it through `.mcp.json` in this folder; approve the `teleop-sim` server once. On Mac /
Linux change `command` in `.mcp.json` to `.venv/bin/python`.

---

## 10. All server options

`start.bat` / `start.command` pass their arguments on, for example `start.bat --task blocks`.

```
--http               plain HTTP on port 8080 (for USB with adb reverse; start.bat uses this)
--https              (start.bat only) HTTPS on port 8443, for Wi-Fi
--port N             another port
--task jeans|blocks  fold the jeans (default), or put blocks into a bin
--scale 1.0          robot motion per hand motion
--randomize 0.05     random object placement on reset [m]
--fps 30             recording rate
--control-hz 100     arm control rate
--stream-hz 90       how often robot poses are sent to the headset
--stream-cams ...    live camera screens (default: left wrist, head, right wrist; "" = off)
--cam-stream-hz 15   camera screen frame rate
--cameras            record camera images (--camera-names, --img-w, --img-h)
--data-dir data      where recordings go
--operator NAME      recorded as the operator
--no-preview         don't make an MP4 for each recording
--garment NAME       garment from config/cloth.toml
--cloth-config FILE  another cloth settings file
--cloth-engine gpu|mujoco   cloth simulation (default: gpu, falls back to mujoco)
--cloth-spacing M    MuJoCo cloth: resolution (smaller = finer, slower)
--cloth-fast         MuJoCo cloth: cheaper collisions (folds may cut through themselves)
--slip-force N       MuJoCo cloth: how hard you can pull before a pinch slips
--cloth-init crumpled  MuJoCo cloth: start from a random heap
```

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| **The page doesn't load on the headset** | Is `start.bat` running? Over USB: run `adb reverse tcp:8080 tcp:8080` again (after every replug) and check `adb devices` shows the headset; allow USB debugging in the headset. |
| **"VR NOT SUPPORTED"** | You opened the page on the computer (that is only the viewer), or over Wi-Fi without `https://`. On the headset use `http://localhost:8080` (USB) or the `https://` address (Wi-Fi). |
| **Slow motion / low frame rate** | Plug in the laptop charger, close other programs, turn the camera screens off (thumbstick click or `--stream-cams ""`). `rtf` on the computer's page shows the speed (1 = real time). |
| **"VERSION MISMATCH" on the panel** | The page is older than the server: reload it in the browser. |
| **The window says it uses the MuJoCo cloth** | No usable graphics card was found (normal in Docker). Update the graphics driver; the line above it says why. |
| **A message about `nvcuda.dll`** at start | Harmless: there is no NVIDIA card, so the GPU cloth uses Vulkan instead. |
| **The arms stop following** | Hold **Grip** again. Both arms let go when the headset stops sending (for example when it goes to sleep). |
| **The view is off-centre** | Press **Y**. |

---

## 12. How the code is organised

```
start.bat / start.command      start the server (Windows / Mac)
setup_windows.bat / setup_mac.command   one-time setup
replay.bat / replay.command    open the newest recording
vrteleop/
  server.py        the main program: real-time loop, recording, streaming to the headset, control API
  arms.py          the two arms: hand motion -> gripper target -> joint commands
  teleop.py        the relative (clutch) mapping from controller to gripper target
  ik.py            inverse kinematics: gripper target -> arm joint angles
  scene.py         the scene: room, table, robots, cameras, tasks
  cloth_engine.py  the cloth interface, with the GPU and the MuJoCo cloth behind it
  gpu_cloth.py     the GPU cloth (XPBD in Taichi)
  cloth.py         the jeans pattern, the MuJoCo cloth, the pinch grasp, fold measures, textures
  cloth_config.py  reads and checks config/cloth.toml
  fold_game.py     the folding game
  recorder.py      writes the episode files
  webscene.py      sends the scene to the headset
  mcp_server.py    the MCP server for Claude
  web/             the headset / browser app (three.js, works offline)
scripts/           replay, tests and benchmarks (see section 8)
config/cloth.toml  garments and cloth settings
assets/            robot models from MuJoCo Menagerie (UR5e, Robotiq 2F-85; see their LICENSE files)
```

Common changes:

- **Task and scene**: `TASKS` and `build_spec()` in `vrteleop/scene.py` (jeans position:
  `JEANS_POS`). The headset picks up scene changes by itself.
- **Robot placement**: `ARM_X`, `ARM_Y`, `TABLE_Z`, `HOME_Q` in `scene.py`.
- **How the arms respond**: the `DiffIK` gains and `max_joint_vel` in `ik.py`.
- **Where your eyes are**: `HEAD_POS` in `scene.py` and `OPERATOR_EYE_MJ` in `web/main.js`
  (keep the two the same).

Tested with MuJoCo 3.14 and Python 3.12. The controller models in VR are loaded from the
internet (WebXR input profiles); offline they are invisible but tracking still works.
