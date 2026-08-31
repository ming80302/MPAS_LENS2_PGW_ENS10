#!/usr/bin/env python3
"""
Stage 02: interpolate LENS2 10-member-ensemble-mean monthly-climatology T, Q, Z3 from
native CESM hybrid levels onto the 37 standard ERA5 pressure levels, and
compute RH -- pure Python (lib/vinterp.py), no NCL dependency. See
lib/vinterp.py's module docstring for why this replaced an NCL
`vinth2p_ecmwf`-based version (that approach produced physically impossible
temperatures, e.g. >10000 K, that scaled with extrapolation distance below
the surface -- abandoned rather than trusted).

Usage:
    python 02_interp_pressure_rh.py --period current
    python 02_interp_pressure_rh.py --period future --months 6,7,8
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

import config
import lib.vinterp as vinterp


def _mkey(month) -> str:
    """'ann' passes through for the annual-mean pseudo-month (config.py's
    *_ann.nc convention, see 01b_ensemble_climatology.py's build_annual_mean);
    a real calendar month is zero-padded as usual."""
    return "ann" if month == "ann" else f"{int(month):02d}"


def _load(var: str, period: str, month) -> xr.DataArray:
    f = config.RAW_DIR / period / f"{var}_{_mkey(month)}.nc"
    with xr.open_dataset(f) as ds:
        return ds[var].load()


def interp_one(period: str, month, hyam: np.ndarray, hybm: np.ndarray,
                overwrite: bool = False) -> None:
    out_dir = config.INTERP_DIR / period
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_mkey(month)}.nc"
    if out_file.exists() and not overwrite:
        return

    T = _load("T", period, month).values
    Q = _load("Q", period, month).values
    Z = _load("Z3", period, month).values
    PS = _load("PS", period, month).values
    lat = _load("T", period, month)["lat"].values
    lon = _load("T", period, month)["lon"].values

    p_native = vinterp.hybrid_pressure(hyam, hybm, PS)
    target_pa = np.array(config.ERA5_PRESSURE_LEVELS_PA, dtype=np.float64)

    out = vinterp.interp_to_pressure_levels(T, Q, Z, p_native, target_pa, PS)
    rh = vinterp.relative_humidity(out["T"], out["Q"], target_pa[:, None, None])

    ds_out = xr.Dataset(
        data_vars=dict(
            GHT=(("lev_pa", "lat", "lon"), out["Z"].astype(np.float32)),
            TT=(("lev_pa", "lat", "lon"), out["T"].astype(np.float32)),
            RH=(("lev_pa", "lat", "lon"), rh.astype(np.float32)),
        ),
        coords=dict(lev_pa=target_pa, lat=lat, lon=lon),
    )
    tmp = out_file.with_suffix(".nc.tmp")
    ds_out.to_netcdf(tmp)
    tmp.rename(out_file)
    print(f"wrote {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", choices=list(config.PERIODS), required=True)
    p.add_argument("--months", default=None)
    p.add_argument("--annual", action="store_true",
                    help="interpolate the single annual-mean ensemble-mean "
                         "climatology (RAW_DIR/period/{var}_ann.nc, from "
                         "01b_ensemble_climatology.py --annual) instead of "
                         "the 12 per-month climatologies")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    config.ensure_dirs()
    if args.annual:
        months = ["ann"]
    else:
        months = [int(m) for m in args.months.split(",")] if args.months else config.MONTHS

    with xr.open_dataset(config.HYBRID_COEFF_FILE) as hc:
        hyam = hc["hyam"].values
        hybm = hc["hybm"].values

    for month in months:
        interp_one(args.period, month, hyam, hybm, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
