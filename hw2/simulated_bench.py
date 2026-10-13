from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Region",
    "NoiseModel",
    "DMD",
    "Measurement",
    "SimulatedBench",
    "region_metrics",
    "surrounding_background_regions",
]


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _shape2(name: str, shape) -> tuple[int, int]:
    try:
        rows, cols = shape
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (rows, cols) pair, got {shape!r}") from None
    return _positive_int(f"{name}[0]", rows), _positive_int(f"{name}[1]", cols)


def _finite_number(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


@dataclass(frozen=True)
class Region:


    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    name: str = ""

    def __post_init__(self):
        for attr in ("row_start", "row_stop", "col_start", "col_stop"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Region.{attr} must be an integer, got {value!r}")
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError(f"Region {self.name!r} has a negative start: {self}")
        if self.row_stop <= self.row_start or self.col_stop <= self.col_start:
            raise ValueError(f"Region {self.name!r} is empty: {self}")

    @property
    def num_pixels(self) -> int:
        return (self.row_stop - self.row_start) * (self.col_stop - self.col_start)

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.row_start, self.row_stop), slice(self.col_start, self.col_stop)

    def overlaps(self, other: "Region") -> bool:
        return (self.row_start < other.row_stop and other.row_start < self.row_stop
                and self.col_start < other.col_stop and other.col_start < self.col_stop)

    def fits(self, shape: tuple[int, int]) -> bool:
        return self.row_stop <= shape[0] and self.col_stop <= shape[1]


def _validate_regions(shape: tuple[int, int], focus: Region, backgrounds) -> tuple[Region, ...]:
    if not isinstance(focus, Region):
        raise ValueError(f"focus must be a Region, got {focus!r}")
    backgrounds = tuple(backgrounds)
    if not backgrounds:
        raise ValueError("at least one background region is required")
    for region in (focus, *backgrounds):
        if not isinstance(region, Region):
            raise ValueError(f"background regions must be Region objects, got {region!r}")
        if not region.fits(shape):
            raise ValueError(f"region {region} does not fit inside the {shape} camera image")
    for i, a in enumerate(backgrounds):
        if a.overlaps(focus):
            raise ValueError(f"background region {a} overlaps the focus region {focus}")
        for b in backgrounds[i + 1:]:
            if a.overlaps(b):
                raise ValueError(f"background regions {a} and {b} overlap")
    return backgrounds


def surrounding_background_regions(focus: Region, camera_shape, gap: int = 1, width: int = 2) -> tuple[Region, ...]:


    rows, cols = _shape2("camera_shape", camera_shape)
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError(f"gap must be a nonnegative integer, got {gap!r}")
    width = _positive_int("width", width)
    f = focus
    candidates = [
        ("top", f.row_start - gap - width, f.row_start - gap, f.col_start, f.col_stop),
        ("bottom", f.row_stop + gap, f.row_stop + gap + width, f.col_start, f.col_stop),
        ("left", f.row_start, f.row_stop, f.col_start - gap - width, f.col_start - gap),
        ("right", f.row_start, f.row_stop, f.col_stop + gap, f.col_stop + gap + width),
    ]
    regions = []
    for name, r0, r1, c0, c1 in candidates:
        r0, r1 = max(r0, 0), min(r1, rows)
        c0, c1 = max(c0, 0), min(c1, cols)
        if r1 > r0 and c1 > c0:
            regions.append(Region(int(r0), int(r1), int(c0), int(c1), name=name))
    if not regions:
        raise ValueError("no background region fits inside the image around this focus")
    return tuple(regions)


@dataclass(frozen=True)
class Measurement:
    image: np.ndarray
    focus_mean: float
    background_region_means: tuple
    background_mean: float
    enhancement: float
    background_floored: bool
    acquisition_index: int = -1


def region_metrics(image, focus: Region, backgrounds, background_floor: float = 1e-12) -> Measurement:

    image = np.asarray(image)
    if image.ndim != 2 or not np.isrealobj(image) or not np.all(np.isfinite(image)):
        raise ValueError("image must be a finite, real 2-D array")
    if np.any(image < 0):
        raise ValueError("image must be nonnegative")
    background_floor = _finite_number("background_floor", background_floor, positive=True)
    backgrounds = _validate_regions(image.shape, focus, backgrounds)

    focus_mean = float(image[focus.slices].mean())
    region_means = tuple(float(image[b.slices].mean()) for b in backgrounds)
    total = sum(float(image[b.slices].sum()) for b in backgrounds)
    background_mean = total / sum(b.num_pixels for b in backgrounds)
    floored = background_mean < background_floor
    enhancement = focus_mean / max(background_mean, background_floor)
    return Measurement(
        image=image,
        focus_mean=focus_mean,
        background_region_means=region_means,
        background_mean=background_mean,
        enhancement=enhancement,
        background_floored=floored,
    )


@dataclass(frozen=True)
class NoiseModel:


    enabled: bool = True
    photons_per_unit: float | None = 1000.0
    read_noise_std: float = 0.01

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"NoiseModel.enabled must be bool, got {self.enabled!r}")
        if self.photons_per_unit is not None:
            _finite_number("photons_per_unit", self.photons_per_unit, positive=True)
        _finite_number("read_noise_std", self.read_noise_std, nonnegative=True)


class DMD:


    def __init__(self, shape, illumination_amplitude: float = 1.0):
        self.shape = _shape2("dmd_shape", shape)
        self.illumination_amplitude = _finite_number(
            "illumination_amplitude", illumination_amplitude, positive=True
        )

    @property
    def num_macropixels(self) -> int:
        return self.shape[0] * self.shape[1]

    def validate_pattern(self, pattern) -> np.ndarray:

        p = np.asarray(pattern)
        if p.shape not in (self.shape, (self.num_macropixels,)):
            raise ValueError(f"pattern shape {p.shape} does not match the DMD {self.shape} "
                             f"or ({self.num_macropixels},)")
        if p.dtype == bool:
            return p.reshape(self.shape).copy()
        if not np.issubdtype(p.dtype, np.integer) and not np.issubdtype(p.dtype, np.floating):
            raise ValueError(f"pattern must be boolean or 0/1 numbers, got dtype {p.dtype}")
        if not np.all((p == 0) | (p == 1)):
            raise ValueError("pattern must be binary: every macropixel must be 0 or 1")
        return p.reshape(self.shape).astype(bool)

    def input_field(self, pattern) -> np.ndarray:

        p = self.validate_pattern(pattern)
        return self.illumination_amplitude * p.ravel().astype(np.float64)


class SimulatedBench:


    def __init__(self, dmd_shape, camera_shape, focus: Region, backgrounds, *, seed: int = 0,
                 noise: NoiseModel | None = None, illumination_amplitude: float = 1.0,
                 background_floor: float = 1e-12, transmission_matrix=None):
        self.dmd = DMD(dmd_shape, illumination_amplitude)
        self.camera_shape = _shape2("camera_shape", camera_shape)
        self.focus = focus
        self.backgrounds = _validate_regions(self.camera_shape, focus, backgrounds)
        self.noise = noise if noise is not None else NoiseModel()
        if not isinstance(self.noise, NoiseModel):
            raise ValueError(f"noise must be a NoiseModel, got {self.noise!r}")
        self.background_floor = _finite_number("background_floor", background_floor, positive=True)
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a nonnegative integer, got {seed!r}")
        self.seed = int(seed)


        t_seq, noise_seq = np.random.SeedSequence(self.seed).spawn(2)
        n_out = self.camera_shape[0] * self.camera_shape[1]
        n_in = self.dmd.num_macropixels
        if transmission_matrix is None:
            rng = np.random.default_rng(t_seq)
            t = (rng.standard_normal((n_out, n_in)) + 1j * rng.standard_normal((n_out, n_in)))
            t *= 1.0 / math.sqrt(2.0 * n_in)
        else:
            t = np.array(transmission_matrix, dtype=np.complex128, copy=True)
            if t.shape != (n_out, n_in):
                raise ValueError(f"transmission_matrix must have shape {(n_out, n_in)}, got {t.shape}")
            if not np.all(np.isfinite(t)):
                raise ValueError("transmission_matrix must be finite")
        t.flags.writeable = False
        self._transmission = t
        self._noise_rng = np.random.default_rng(noise_seq)
        self._acquisitions = 0


    @property
    def num_macropixels(self) -> int:
        return self.dmd.num_macropixels

    @property
    def num_acquisitions(self) -> int:
        return self._acquisitions

    def transmission_fingerprint(self) -> str:

        return hashlib.sha256(self._transmission.tobytes()).hexdigest()


    def _ideal_intensity(self, pattern) -> np.ndarray:
        e_out = self._transmission @ self.dmd.input_field(pattern)
        return (np.abs(e_out) ** 2).reshape(self.camera_shape)

    def acquire(self, pattern) -> np.ndarray:

        image = self._ideal_intensity(pattern)
        if self.noise.enabled:
            if self.noise.photons_per_unit is not None:
                scale = self.noise.photons_per_unit
                image = self._noise_rng.poisson(image * scale) / scale
            if self.noise.read_noise_std > 0:
                image = image + self._noise_rng.normal(0.0, self.noise.read_noise_std, image.shape)
            image = np.clip(image, 0.0, None)
        self._acquisitions += 1
        return np.asarray(image, dtype=np.float64)

    def measure(self, pattern) -> Measurement:

        index = self._acquisitions
        m = region_metrics(self.acquire(pattern), self.focus, self.backgrounds, self.background_floor)
        return replace(m, acquisition_index=index)
