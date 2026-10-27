"""Episode recorder -> one HDF5 file per demonstration.

Layout (T = number of frames, sampled at a fixed rate in *sim* time):

    /time                           (T,)        sim time [s]
    /observations/qpos              (T, 14)     [L arm 6, L gripper, R arm 6, R gripper]
    /observations/qvel              (T, 14)
    /observations/ee_pos            (T, 2, 3)   measured pinch-site pose [left, right]
    /observations/ee_quat           (T, 2, 4)   wxyz
    /observations/object_pose       (T, K, 7)   xyz + wxyz of each free object (blocks task)
    /observations/cloth_verts       (T, N, 3)   world positions of the garment's N sim vertices (jeans task)
    /observations/full_qpos         (T, nq)     full MuJoCo state (for exact replay)
    /observations/full_qvel         (T, nv)
    /observations/images/<cam>      (T, H, W, 3) uint8   (only with --cameras)
    /action                         (T, 14)     joint-space command [L q_cmd 6, L grip, R q_cmd 6, R grip]
    /action_ee/pos                  (T, 2, 3)   EE target (what the operator asked for)
    /action_ee/quat                 (T, 2, 4)
    /action_ee/gripper              (T, 2)      0 open .. 1 closed
    /teleop/engaged                 (T, 2)      bool
    /teleop/head_pose               (T, 7)      operator head in MuJoCo frame (nan if unknown)
    /teleop/controller_pose         (T, 2, 7)
    /teleop/cloth_grasp             (T, 2, K)   cloth vertices pinched by [left, right] gripper (-1 = none;
                                                K = 12 with the MuJoCo cloth, 96 with the GPU and Newton
                                                cloth; int16, int32 for garments over 32767 vertices)
    /teleop/cloth_contact_force     (T, 2, 3)   force of the cloth on the [left, right] arm [N, world]
                                                (Newton engine; NaN where the engine doesn't measure it)

Cloth episodes also store /cloth/faces (F, 3) and /cloth/rest_uv (N, 2) (the flat pattern
in metres), and root attrs final_coverage / final_height (see cloth.fold_metrics),
cloth_engine, cloth_config (every garment and engine setting) and cloth_engine_settings (the
engine's versions, e.g. Newton and Warp, and its derived values).

Attributes on the root: task, fps, timestep, date, object_names, camera_names,
joint_names, success (bool, user-flagged), and the full model XML.
The layout mirrors ALOHA/ACT-style datasets, so it is easy to convert to LeRobot.
"""
from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path

import h5py
import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: Path, fps: float, model_xml: str, object_names: list[str],
                 joint_names: list[str], camera_names: list[str] | None = None,
                 task: str = "blocks", cloth_faces=None, cloth_rest_uv=None):
        self.out_dir = Path(out_dir)
        self.task = task
        self.cloth_faces = cloth_faces
        self.cloth_rest_uv = cloth_rest_uv
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.model_xml = model_xml
        self.object_names = object_names
        self.joint_names = joint_names
        self.camera_names = camera_names or []
        self.active = False
        self.frames: dict[str, list] = {}
        self.t_start = 0.0
        self._writers: list[threading.Thread] = []

    @property
    def num_frames(self) -> int:
        return len(self.frames.get("time", []))

    def next_index(self) -> int:
        existing = sorted(self.out_dir.glob("episode_*.hdf5"))
        if not existing:
            return 0
        return max(int(p.stem.split("_")[1]) for p in existing) + 1

    def start(self, sim_time: float):
        self.frames = {}
        self.active = True
        self.t_start = sim_time

    def add(self, frame: dict):
        for k, v in frame.items():
            self.frames.setdefault(k, []).append(np.asarray(v))

    def discard(self):
        self.active = False
        self.frames = {}

    def stop(self, timestep: float, success: bool | None = None,
             extra_attrs: dict | None = None, on_saved=None) -> Path | None:
        """End the episode. The file is written on a background thread (compressing a long
        episode takes a second or more, which must not stall the simulation); `on_saved(path)`
        runs once it is complete. Returns the path (its name is reserved right away)."""
        self.active = False
        if self.num_frames < 2:
            self.frames = {}
            return None
        idx = self.next_index()
        path = self.out_dir / f"episode_{idx:04d}.hdf5"
        path.touch()                          # reserve the name: the next episode gets the next one
        frames, self.frames = self.frames, {}
        t = threading.Thread(target=self._write, name=f"write {path.name}",
                             args=(path, frames, self.t_start, timestep, success, extra_attrs, on_saved))
        t.start()
        self._writers = [w for w in self._writers if w.is_alive()] + [t]
        return path

    def wait(self):
        """Block until every episode is on disk (call before exiting)."""
        for w in self._writers:
            w.join()

    def _write(self, path, f, t_start, timestep, success, extra_attrs, on_saved):
        with h5py.File(path, "w") as h:
            h.attrs["task"] = self.task
            h.attrs["fps"] = self.fps
            h.attrs["timestep"] = timestep
            h.attrs["date"] = _dt.datetime.now().isoformat()
            h.attrs["object_names"] = json.dumps(self.object_names)
            h.attrs["joint_names"] = json.dumps(self.joint_names)
            h.attrs["camera_names"] = json.dumps(self.camera_names)
            h.attrs["success"] = bool(success) if success is not None else False
            h.attrs["model_xml"] = self.model_xml
            for k, v in (extra_attrs or {}).items():
                h.attrs[k] = v
            h.create_dataset("time", data=np.stack(f["time"]) - t_start)
            obs = h.create_group("observations")
            for k in ("qpos", "qvel", "ee_pos", "ee_quat", "object_pose", "full_qpos", "full_qvel"):
                data = np.stack(f[k]).astype(np.float32 if not k.startswith("full") else np.float64)
                if k == "object_pose" and data.ndim == 2:          # no rigid objects in this task
                    data = data.reshape(len(data), 0, 7)
                obs.create_dataset(k, data=data)
            if "cloth_verts" in f:
                obs.create_dataset("cloth_verts", data=np.stack(f["cloth_verts"]).astype(np.float32),
                                   compression="gzip", compression_opts=4)
            if self.camera_names:
                img = obs.create_group("images")
                for cam in self.camera_names:
                    arr = np.stack(f[f"img_{cam}"])
                    img.create_dataset(cam, data=arr, chunks=(1,) + arr.shape[1:],
                                       compression="gzip", compression_opts=4)
            h.create_dataset("action", data=np.stack(f["action"]).astype(np.float32))
            ae = h.create_group("action_ee")
            ae.create_dataset("pos", data=np.stack(f["target_pos"]).astype(np.float32))
            ae.create_dataset("quat", data=np.stack(f["target_quat"]).astype(np.float32))
            ae.create_dataset("gripper", data=np.stack(f["gripper_cmd"]).astype(np.float32))
            tg = h.create_group("teleop")
            tg.create_dataset("engaged", data=np.stack(f["engaged"]).astype(bool))
            tg.create_dataset("head_pose", data=np.stack(f["head_pose"]).astype(np.float32))
            tg.create_dataset("controller_pose", data=np.stack(f["controller_pose"]).astype(np.float32))
            if "cloth_grasp" in f:
                g = np.stack(f["cloth_grasp"])
                tg.create_dataset("cloth_grasp", data=g.astype(np.int16 if g.max(initial=0) < 32767 else np.int32))
            if "cloth_contact_force" in f:
                tg.create_dataset("cloth_contact_force", data=np.stack(f["cloth_contact_force"]).astype(np.float32))
            if self.cloth_faces is not None:
                cg = h.create_group("cloth")
                cg.create_dataset("faces", data=np.asarray(self.cloth_faces, np.int32))
                cg.create_dataset("rest_uv", data=np.asarray(self.cloth_rest_uv, np.float32))
        if on_saved is not None:
            on_saved(path)
