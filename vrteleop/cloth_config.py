"""Cloth settings from a TOML file (config/cloth.toml by default).

The file has one table per garment (its size and fabric: every field of cloth.GarmentSpec), plus
the simulation and grasp settings. A garment can inherit another one and change a few values.

    cfg = load()                                   # the file's default garment
    cfg = load("shorts", overrides={"spacing": 0.045})
    cfg = load("jeans", path="my_cloth.toml")

Unknown or misspelled keys are errors (with a suggestion), so a typo never goes unnoticed.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import tomllib
from pathlib import Path

from .cloth import GarmentSpec

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "cloth.toml"


@dataclasses.dataclass
class GPUSettings:
    """[gpu]: the GPU cloth engine (vrteleop/gpu_cloth.py, XPBD)."""
    spacing: float = 0.010             # distance between points [m] (the garment's `spacing` is
                                       # the MuJoCo engine's)
    substeps: int = 20                 # solver substeps per 1/60 s frame
    self_collision_every: int = 2      # self-collision on every n-th substep (the costliest part)
    fps: int = 60
    thickness: float = 0.95            # self-collision distance, as a fraction of spacing
    stretch_compliance: float = 1e-8   # threads (inverse stiffness; 0 = rigid)
    shear_compliance: float = 2e-5     # bias diagonals
    bend_compliance: float = 2e-3      # across every fold line (larger = softer folds)
    cloth_friction: float = 0.5        # Coulomb friction coefficient, fabric on fabric
    table_friction: float = 0.4        # ... fabric on the table
    gripper_friction: float = 0.3      # ... fabric on an arm while its gripper pinches (else 0)
    air_drag: float = 0.5              # [1/s]
    relaxation: float = 1.5            # Jacobi over-relaxation of the constraint solve


@dataclasses.dataclass
class NewtonSettings:
    """[newton]: the Newton VBD cloth engine (vrteleop/newton_cloth.py). SI units; the friction
    coefficients are the real pairwise values (the engine converts them to Newton's
    sqrt(cloth mu x shape mu) combination)."""
    device: str = "cuda"               # cuda (an NVIDIA GPU) or cpu (slow; headless tests)
    spacing: float = 0.005             # distance between points [m]
    substeps: int = 16                 # solver substeps per 1/60 s frame
    iterations: int = 16               # VBD iterations per substep
    fps: int = 60
    self_contact: bool = True          # fabric on fabric, vertex-triangle and edge-edge
    tri_ke: float = 2890.0             # in-plane stretch stiffness [N/m] (2D Lame mu)
    tri_ka: float = 2890.0             # in-plane area stiffness [N/m] (2D Lame lambda)
    tri_kd: float = 1.5e-5             # in-plane damping [N s/m]
    edge_ke: float = 6.61              # bending stiffness [N m/rad] per hinge (VBD scales it by the edge length)
    edge_kd: float = 1.0e-3            # bending damping
    fabric_density: float = 0.407      # one layer of the denim [kg/m^2] (12 oz per square yard), with structure
    structure: bool = True             # sewn parts heavier and stiffer (cloth.GARMENT_PARTS); false = a
                                       # uniform sheet weighing the garment's mass
    particle_radius: float = 0.001     # contact radius of a point against the robot and the table [m]
    self_contact_margin: float = 0.002  # layers of fabric rest this far apart [m]
    self_contact_gap: float = 0.001    # extra reach of the self-contact search [m]
    body_contact_gap: float = 0.002    # extra reach of the robot / table contact search [m]
    full_surface_contact: bool = True  # robot and table touch edges and faces, not only the points
    contact_ke: float = 1.0e4          # contact stiffness [N/m]
    contact_kd: float = 1.0e-2         # contact damping [N s/m]
    cloth_friction: float = 0.5        # friction coefficient, fabric on fabric
    table_friction: float = 0.4        # ... fabric on the table
    pad_friction: float = 1.5          # ... fabric on the silicone finger pads
    gripper_friction: float = 0.5      # ... fabric on the rest of the gripper and the arm
    settle_frames: int = 30            # frames simulated at reset before the clock starts


@dataclasses.dataclass
class ClothConfig:
    name: str
    garment: GarmentSpec
    gpu: GPUSettings = dataclasses.field(default_factory=GPUSettings)
    newton: NewtonSettings = dataclasses.field(default_factory=NewtonSettings)
    # [simulation]
    timestep: float = 0.004
    solver_iterations: int = 30
    # [grasp]
    slip_force: float = 30.0
    release_gap: float = 0.004
    pad_friction: float = 1.0
    grasp_model: str = "pinch"         # [grasp] model: "pinch" or "friction" (newton engine)

    def to_json(self) -> str:
        """Everything needed to rebuild this cloth exactly (stored with every episode)."""
        d = dataclasses.asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> "ClothConfig":
        d = json.loads(text)
        g = _coerce_fields(GarmentSpec, d.pop("garment"), "recorded garment")
        gpu = GPUSettings(**_coerce_fields(GPUSettings, d.pop("gpu", {}), "recorded gpu"))
        newton = NewtonSettings(**_coerce_fields(NewtonSettings, d.pop("newton", {}), "recorded newton"))
        return cls(garment=GarmentSpec(**g), gpu=gpu, newton=newton, **d)


SECTIONS = {"simulation": ("timestep", "solver_iterations"),
            "grasp": ("slip_force", "release_gap", "pad_friction", "model")}
ENGINE_SECTIONS = {"gpu": GPUSettings, "newton": NewtonSettings}     # per-engine settings
SECTION_KEYS = {("grasp", "model"): "grasp_model"}                  # file key -> ClothConfig field
GRASP_MODELS = ("pinch", "friction")


class ConfigError(ValueError):
    pass


def _suggest(key, valid):
    close = difflib.get_close_matches(key, list(valid), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _coerce_fields(cls, values: dict, where: str) -> dict:
    """Check keys against a dataclass and convert TOML values to the field types."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    out = {}
    for k, v in values.items():
        if k not in fields:
            raise ConfigError(f"{where}: unknown setting '{k}'{_suggest(k, fields)}")
        default = fields[k].default
        if isinstance(default, bool):
            if not isinstance(v, bool):
                raise ConfigError(f"{where}: '{k}' must be true or false, got {v!r}")
        elif isinstance(default, int):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"{where}: '{k}' must be a whole number, got {v!r}")
        elif isinstance(default, float):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(f"{where}: '{k}' must be a number, got {v!r}")
            v = float(v)
        elif isinstance(default, str):
            if not isinstance(v, str):
                raise ConfigError(f"{where}: '{k}' must be text, got {v!r}")
        elif isinstance(default, tuple):
            if not isinstance(v, (list, tuple)) or len(v) != len(default):
                raise ConfigError(f"{where}: '{k}' must be a list of {len(default)} numbers, got {v!r}")
            v = tuple(float(x) for x in v)
        out[k] = v
    return out


def _resolve(garments: dict, name: str, path: Path, chain=()) -> dict:
    """The garment's table merged over the ones it inherits from."""
    if name not in garments:
        raise ConfigError(f"{path}: no garment '{name}'{_suggest(name, garments)}; "
                          f"available: {', '.join(sorted(garments))}")
    if name in chain:
        raise ConfigError(f"{path}: garments inherit in a circle: {' -> '.join(chain + (name,))}")
    table = dict(garments[name])
    parent = table.pop("inherits", None)
    merged = _resolve(garments, parent, path, chain + (name,)) if parent else {}
    for k, v in table.items():
        if (k in SECTIONS or k in ENGINE_SECTIONS) and isinstance(v, dict):
            merged[k] = {**merged.get(k, {}), **v}
        else:
            merged[k] = v
    return merged


def load(name: str | None = None, path: str | Path | None = None,
         overrides: dict | None = None) -> ClothConfig:
    """Load garment `name` (default: the file's `default`) from `path`; `overrides` replace
    garment fields (e.g. from command-line options)."""
    path = Path(path) if path else DEFAULT_PATH
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"cloth config not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    for k in data:
        if k not in ("default", "garments", *ENGINE_SECTIONS, *SECTIONS):
            raise ConfigError(f"{path}: unknown top-level setting '{k}'"
                              f"{_suggest(k, ['default', 'garments', *ENGINE_SECTIONS, *SECTIONS])}")
    garments = data.get("garments", {})
    name = name or data.get("default")
    if not name:
        raise ConfigError(f"{path}: no garment given and no 'default = \"...\"' in the file")
    fields = _resolve(garments, name, path)
    settings = {}
    for section, keys in SECTIONS.items():
        values = {**data.get(section, {}), **fields.pop(section, {})}
        for k, v in values.items():
            if k not in keys:
                raise ConfigError(f"{path}: [{section}] unknown setting '{k}'{_suggest(k, keys)}")
            settings[SECTION_KEYS.get((section, k), k)] = v
    settings = _coerce_fields(ClothConfig, settings, f"{path}")
    if settings.get("grasp_model", "pinch") not in GRASP_MODELS:
        raise ConfigError(f"{path}: [grasp] model must be one of {', '.join(GRASP_MODELS)}, "
                          f"got {settings['grasp_model']!r}")
    engines = {}
    for section, cls in ENGINE_SECTIONS.items():
        values = {**data.get(section, {}), **fields.pop(section, {})}
        engines[section] = cls(**_coerce_fields(cls, values, f"{path} [{section}]"))
    if engines["newton"].device not in ("cuda", "cpu"):
        raise ConfigError(f"{path} [newton]: device must be cuda or cpu, got {engines['newton'].device!r}")
    g = _coerce_fields(GarmentSpec, fields, f"{path} [garments.{name}]")
    g.update(overrides or {})
    garment = GarmentSpec(**g)
    for k in ("thigh", "hem"):     # wider legs would overlap each other in the flat pattern
        if getattr(garment, k) > garment.hip / 2 + 1e-9:
            raise ConfigError(f"{path} [garments.{name}]: {k} ({getattr(garment, k)}) must be <= "
                              f"hip / 2 ({garment.hip / 2}): the legs would overlap")
    if garment.length - garment.rise < 3 * garment.spacing:
        raise ConfigError(f"{path} [garments.{name}]: the legs (length - rise = "
                          f"{garment.length - garment.rise:.3f} m) need at least 3 x spacing")
    if garment.layer_gap <= 2 * garment.sphere_r:
        raise ConfigError(f"{path} [garments.{name}]: layer_gap ({garment.layer_gap}) must be > "
                          f"2 * sphere_r ({2 * garment.sphere_r})")
    return ClothConfig(name=name, garment=garment, **engines, **settings)


def names(path: str | Path | None = None) -> list[str]:
    """Garments defined in the file."""
    with open(Path(path) if path else DEFAULT_PATH, "rb") as f:
        return sorted(tomllib.load(f).get("garments", {}))
