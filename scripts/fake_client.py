"""Headset-free smoke test: pretends to be a Quest and records one episode.

    python -m vrteleop.server --http --task blocks   # terminal 1
    python scripts/fake_client.py                    # terminal 2: drives both arms in circles

    python -m vrteleop.server --http                 # jeans task
    python scripts/fake_client.py --fold             # scripted two-arm fold of the jeans

--fold reads the cloth vertices streamed to the client, grabs the far leg's waistband
and hem corners with both grippers, lays that leg over the other one, then folds the
hems up to the knees with the right arm. It goes through exactly the same path as a
real operator (clutch mapping, IK, gripper thumbstick, pinch grasp, recorder).
"""
import argparse
import asyncio
import json
import math
import ssl
import struct
import time

import aiohttp
import numpy as np

LAY_HEIGHT = 0.003       # [m] the laid-over leg is let go this far above grasp height
HOME = {"left": np.array([0.45, 0.22, 0.95]), "right": np.array([0.45, -0.22, 0.95])}
TABLE_Z = 0.75
CTRL_BASE = {"left": np.array([0.0, 0.25, 1.1]), "right": np.array([0.0, -0.25, 1.1])}


class Pacer:
    """Sends at a fixed rate against absolute deadlines. A plain asyncio.sleep(1/72) can return up
    to ~16 ms early on Windows (Python 3.12's 15.6 ms monotonic clock), which made the scripted
    moves run several times too fast and flooded the server with input."""
    def __init__(self, hz: float):
        self.dt = 1.0 / hz
        self.next = time.perf_counter()

    async def wait(self):
        self.next += self.dt
        await asyncio.sleep(max(0.0, self.next - time.perf_counter()))


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.status = {}
        self.cloth = None
        self.t0 = time.time()

    async def reader(self):
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.status = json.loads(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY and msg.data[0] == 1:
                f = np.frombuffer(msg.data, np.float32, offset=4)
                nb, nc = int(f[2]), int(f[3])
                k = 4 + 7 * nb + 14 + 7
                if nc > 0:
                    self.cloth = f[k:k + 3 * nc].reshape(nc, 3).copy()

    async def cmd(self, c):
        await self.ws.send_str(json.dumps({"type": "cmd", "cmd": c}))

    async def send(self, ee_offset, grip_engaged, stick):
        """ee_offset: side -> desired EE displacement from the pose at engage time."""
        msg = {"type": "input", "t": time.time() - self.t0, "head": [-0.4, 0, 1.65, 1, 0, 0, 0]}
        for s in ("left", "right"):
            pose = list(CTRL_BASE[s] + ee_offset[s]) + [1, 0, 0, 0]
            msg[s] = {"pose": pose, "grip": 1.0 if grip_engaged else 0.0, "stick": [0.0, stick[s]],
                      "epoch": 0}
        await self.ws.send_str(json.dumps(msg))


async def circles(cl: Client, seconds: float):
    await cl.cmd("record_toggle")
    t0 = time.time()
    pace = Pacer(72)
    while (t := time.time() - t0) < seconds:
        off = {}
        for side, sgn in (("left", 1), ("right", -1)):
            r = 0.08 * min(t, 1.0)
            off[side] = np.array([r * math.cos(2 * t), sgn * r * math.sin(2 * t), -0.1 * min(t, 1.0)])
        st = 1.0 if t < seconds / 2 else -1.0
        await cl.send(off, t > 0.2, {"left": st, "right": st})
        await pace.wait()


async def fold(cl: Client):
    while cl.cloth is None:
        await asyncio.sleep(0.05)
    V = cl.cloth
    far = V[:, 0] > np.median(V[:, 0])
    idx = np.where(far)[0]
    w = idx[np.argmax(V[far, 1] * 5 + V[far, 0])]      # far-leg waistband corner (operator's left)
    h = idx[np.argmax(-V[far, 1] * 5 + V[far, 0])]     # far-leg hem corner (operator's right)
    z0 = TABLE_Z + 0.018
    gl = np.r_[V[w, :2] + [-0.02, -0.02], z0]
    gr = np.r_[V[h, :2] + [-0.02, 0.02], z0]
    # laid over as a mirror image: each grasp lands as far inside the other leg's corner as it was
    # inside its own, let go just above grasp height (see scripts/cloth_bench.py)
    near = np.where(~far)[0]
    tl = np.r_[V[near[np.argmax(V[~far, 1] * 5 - V[~far, 0])], 0] + 0.02, gl[1], z0 + LAY_HEIGHT]
    tr = np.r_[V[near[np.argmax(-V[~far, 1] * 5 - V[~far, 0])], 0] + 0.02, gr[1], z0 + LAY_HEIGHT]
    target = {s: HOME[s].copy() for s in HOME}
    grip = {"left": 0.0, "right": 0.0}               # 0 open .. 1 closed (tracked like the server)

    async def move(goal, dur, close=None):
        start = {s: target[s].copy() for s in HOME}
        n = max(1, int(dur * 72))
        pace = Pacer(72)
        for k in range(n):
            a = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / n)
            stick = {}
            for s in HOME:
                if s in goal:
                    target[s] = start[s] + (np.asarray(goal[s]) - start[s]) * a
                c = (close or {}).get(s)
                stick[s] = 0.0 if c is None else (1.0 if c else -1.0)   # +y (down) closes
            await cl.send({s: target[s] - HOME[s] for s in HOME}, True, stick)
            await pace.wait()

    def report(label):
        s = cl.status
        print(f"  {label:22s} holding={s.get('holding')} cloth={s.get('cloth')} rtf={s.get('rtf')}", flush=True)

    await cl.cmd("record_toggle")
    await move({}, 0.3)                                                   # engage both clutches
    await move({"left": gl + [0, 0, 0.08], "right": gr + [0, 0, 0.08]}, 2.0); report("approach")
    await move({"left": gl, "right": gr}, 1.0)
    await move({}, 1.0, close={"left": True, "right": True}); report("pinch")
    await move({"left": gl + [0, 0, 0.16], "right": gr + [0, 0, 0.16]}, 1.2); report("lift")
    await move({"left": tl + [0, 0, 0.16], "right": tr + [0, 0, 0.16]}, 2.2); report("carry over")
    await move({"left": tl, "right": tr}, 1.0)
    await move({}, 0.9, close={"left": False, "right": False}); report("release")
    await move({"left": tl + [0, 0, 0.1], "right": tr + [0, 0, 0.05]}, 0.8)
    await asyncio.sleep(0.5)
    V = cl.cloth
    hem = V[:, 1] < V[:, 1].min() + 0.03
    gh = np.r_[V[hem, 0].mean(), V[hem, 1].max() - 0.01, z0]
    await move({"right": gh + [0, 0, 0.08]}, 1.5)
    await move({"right": gh}, 0.9)
    await move({}, 1.0, close={"right": True}); report("pinch hems")
    await move({"right": gh + [0, 0, 0.2]}, 1.2)
    await move({"right": [gh[0], 0.02, z0 + 0.2]}, 2.2); report("carry hems")
    await move({"right": [gh[0], 0.02, z0 + 0.03]}, 1.0)
    await move({}, 0.9, close={"right": False})
    await move({"right": [gh[0], 0.0, z0 + 0.15]}, 0.8)
    await asyncio.sleep(1.5); report("done")


async def main(a):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(a.url, ssl=ctx if a.url.startswith("wss") else None,
                                max_msg_size=1 << 24) as ws:
            cl = Client(ws)
            rt = asyncio.create_task(cl.reader())
            await asyncio.sleep(0.5)
            if a.fold:
                if cl.status.get("task") not in (None, "jeans"):
                    print("server is not running the jeans task (start it with --task jeans)")
                    return
                await fold(cl)
            else:
                await circles(cl, a.seconds)
            print("status before stop:", {k: cl.status.get(k) for k in ("recording", "frames", "engaged", "rtf")})
            await cl.cmd("record_toggle")
            await asyncio.sleep(0.8)
            print("status after stop:", cl.status.get("message"))
            rt.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8080/ws")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fold", action="store_true", help="scripted jeans fold (jeans task)")
    main_args = p.parse_args()
    asyncio.run(main(main_args))
