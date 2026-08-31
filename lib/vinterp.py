"""
Vertical interpolation of LENS2 monthly-climatology T, Q, Z3 from native
CESM hybrid-sigma levels onto fixed pressure levels (the 37 standard ERA5
levels), plus relative humidity -- pure Python/numpy, replacing an NCL
`vinth2p_ecmwf` implementation that was abandoned after producing
physically impossible values (T up to ~14000 K) whose badness scaled with
extrapolation distance below the surface. Root cause not fully isolated
(NCL's documented units for that function are `psfc` in Pa and `p0` in
hPa/mb, which is what was passed, matching the pattern already used and
apparently working in the sibling pipeline's 04_write_intermediate.ncl --
but there the pressure-level output was never numerically sanity-checked
either, so that script's TonP/ZonP/RonP should be treated as unverified
too). Rather than keep guessing at a black-box function's unit contract,
this module implements the interpolation directly so every step is
inspectable.

Method
------
- Native level pressure: P = hyam*P0 + hybm*PS (P0 = 100000 Pa), the
  standard CESM hybrid-sigma formula -- unambiguous, no library involved.
- Within the native profile (P_top <= target <= P_surface): linear
  interpolation in ln(P) for T, Q, and Z3.
- Below the surface (target > P_surface, common wherever local terrain
  makes PS less than the 1000 hPa etc. reference levels): extrapolate with
  the hypsometric equation using a fixed standard-atmosphere lapse rate
  (6.5 K/km) referenced to the lowest model level -- the standard
  "hydrostatic + constant lapse rate" family of methods (e.g. the
  ECMWF/Trenberth reduction-to-sea-level approach), bounded by
  construction (no runaway growth).
- Above the model top (target < P_top -- CESM's L32 top is ~3.6 hPa,
  above the 1/2/3 hPa ERA5 levels): isothermal extrapolation (T, Q held at
  the top-level value), Z extended hydrostatically.
- Relative humidity is computed from the interpolated T, Q at each target
  pressure (Bolton 1980 saturation vapor pressure), not interpolated
  directly -- avoids extra nonlinear-interpolation error in a derived
  quantity.

This is a reasonable, transparent scheme for a PGW *delta* in particular:
both the current and future climatologies go through the identical
extrapolation formula, so any systematic bias it introduces largely cancels
in (future - current). It does not aim to reproduce ECMWF's exact
operational extrapolation formula bit-for-bit.
"""

from __future__ import annotations

import numpy as np

RD = 287.05      # J / (kg K), dry air gas constant
G = 9.80665      # m / s^2
LAPSE = 0.0065   # K / m, US standard atmosphere
P0 = 100000.0    # Pa, CESM hybrid-coefficient reference pressure


def hybrid_pressure(hyam: np.ndarray, hybm: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """hyam, hybm: (nlev,) in native (top-down) order. ps: (lat, lon) Pa.
    Returns P: (nlev, lat, lon) Pa, increasing with level index (top->surface).
    """
    return hyam[:, None, None] * P0 + hybm[:, None, None] * ps[None, :, :]


def saturation_vapor_pressure_hpa(t_k: np.ndarray) -> np.ndarray:
    """Bolton (1980) saturation vapor pressure over water, hPa. t_k in K."""
    t_c = t_k - 273.15
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def relative_humidity(t_k: np.ndarray, q_kgkg: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """t_k, q_kgkg, p_pa same shape. Returns RH in percent, clipped [0, 100]."""
    eps = 0.622
    es_hpa = saturation_vapor_pressure_hpa(t_k)
    p_hpa = p_pa / 100.0
    qs = eps * es_hpa / (p_hpa - (1.0 - eps) * es_hpa)
    rh = 100.0 * q_kgkg / qs
    return np.clip(rh, 0.0, 100.0)


def interp_to_pressure_levels(t, q, z, p_native, target_pa, ps):
    """t, q, z, p_native: (nlev, lat, lon), p_native increasing with level
    index (top->surface). target_pa: (ntarget,) Pa. ps: (lat, lon) Pa.

    Returns dict with 'T', 'Q', 'Z' arrays shaped (ntarget, lat, lon).
    """
    nlev, ny, nx = t.shape
    ntarget = len(target_pa)

    t_top, q_top, z_top, p_top = t[0], q[0], z[0], p_native[0]
    t_bot, q_bot, z_bot = t[-1], q[-1], z[-1]
    # p_native[-1] should equal ps by construction (hybm[-1]~1, hyam[-1]~0),
    # but use ps directly since that's the physically exact surface pressure
    p_bot = ps

    out_t = np.empty((ntarget, ny, nx), dtype=np.float64)
    out_q = np.empty((ntarget, ny, nx), dtype=np.float64)
    out_z = np.empty((ntarget, ny, nx), dtype=np.float64)

    ln_p_native = np.log(p_native)

    for k, target in enumerate(target_pa):
        above = target < p_top                 # higher than model top
        below = target > p_bot                 # underground
        within = ~above & ~below

        # --- within: log-P linear interpolation ---
        # number of native levels with pressure <= target, minus 1
        # (p_native increases with index, so this locates the bracketing pair)
        idx_lo = np.clip(np.sum(p_native <= target, axis=0) - 1, 0, nlev - 2)
        idx_hi = idx_lo + 1
        iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        p_lo = p_native[idx_lo, iy, ix]
        p_hi = p_native[idx_hi, iy, ix]
        w = (np.log(target) - np.log(p_lo)) / (np.log(p_hi) - np.log(p_lo))
        w = np.clip(w, 0.0, 1.0)

        t_lo, t_hi = t[idx_lo, iy, ix], t[idx_hi, iy, ix]
        q_lo, q_hi = q[idx_lo, iy, ix], q[idx_hi, iy, ix]
        z_lo, z_hi = z[idx_lo, iy, ix], z[idx_hi, iy, ix]

        t_within = t_lo + w * (t_hi - t_lo)
        q_within = q_lo + w * (q_hi - q_lo)
        z_within = z_lo + w * (z_hi - z_lo)

        # --- below surface: hypsometric + constant lapse rate ---
        dz_below = (RD * t_bot / G) * np.log(p_bot / np.maximum(target, 1e-6))
        z_below = z_bot + dz_below            # dz_below < 0 here (going down)
        t_below = t_bot - LAPSE * dz_below     # warmer below ground
        q_below = q_bot                        # hold specific humidity

        # --- above top: isothermal + hydrostatic ---
        dz_above = (RD * t_top / G) * np.log(p_top / np.maximum(target, 1e-6))
        z_above = z_top + dz_above
        t_above = t_top
        q_above = q_top

        out_t[k] = np.where(within, t_within, np.where(below, t_below, t_above))
        out_q[k] = np.where(within, q_within, np.where(below, q_below, q_above))
        out_z[k] = np.where(within, z_within, np.where(below, z_below, z_above))

    return {"T": out_t, "Q": out_q, "Z": out_z}
