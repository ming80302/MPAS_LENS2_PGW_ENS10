#!/usr/bin/env python3
"""
Stage 03: compute the PGW delta (future - current) for GHT, TT, RH, SST,
SKINTEMP and regrid it onto the exact ERA5 grid (1440x721, 0.25deg,
confirmed via a real file + rd_intermediate.exe -- see config.py) that the
IM files stage 04 perturbs use.

SST/SKINTEMP deltas are computed in whatever units the source LENS2 fields
are in (degC for SST) *before* any unit conversion -- a delta is
unit-offset-invariant (T_future_K - T_current_K == T_future_C - T_current_C),
so no explicit degC->K conversion is needed here; the delta gets added
directly onto the real ERA5 slab (which is already in K) in stage 04.

SST is regridded POP gx1v6 -> CAM fv0.9x1.25 (sparse ESMF weights) -> ERA5
0.25deg (bilinear); GHT/TT/RH/SKINTEMP are already on the CAM grid and only
need the CAM -> ERA5 bilinear step. Land-masked (NaN) ocean cells and any
edge-of-domain interpolation artifacts are zeroed out defensively (the
*actual* land/ocean masking that matters happens in stage 04, which only
adds the SST delta where the real ERA5 record's own value isn't already its
land fill value -- zeroing here just prevents propagating a NaN into a nearby
valid ocean cell through the bilinear stencil).

Usage:
    python 03_compute_delta_regrid.py
    python 03_compute_delta_regrid.py --months 6,7,8
    python 03_compute_delta_regrid.py --annual
    python 03_compute_delta_regrid.py --label JJASO
        # needs INTERP_DIR/{current,future}/JJASO.nc and RAW_DIR/{current,future}/
        # {SST,TS}_JJASO.nc, i.e. run 02_interp_pressure_rh.py --label JJASO for
        # both periods first (01b_ensemble_climatology.py --label's RAW_DIR
        # output covers SST/TS too)
    python 03_compute_delta_regrid.py --period-current current --period-future other --label JJASO
        # delta between any two config.PERIODS entries, not just the default
        # current/future pair ("other" here stands for whatever third period
        # name you've added to config.PERIODS) -- needs stage 01/02 output for
        # BOTH period names given. Non-default period pairs write to
        # DELTA_DIR/<period-future>_minus_<period-current>/ instead of flat
        # DELTA_DIR/, so they never collide with the default future-vs-current
        # delta (which 04_apply_perturbation.py reads from flat DELTA_DIR/
        # unchanged); the default --period-current/--period-future values
        # ("current"/"future") preserve the original flat-DELTA_DIR layout.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

import config
import lib.regrid_latlon as regrid


def _mkey(month) -> str:
    """Non-numeric pseudo-months ('ann', or a custom --label like 'JJASO')
    pass through as-is; a real calendar month is zero-padded as usual (see
    02_interp_pressure_rh.py)."""
    return month if isinstance(month, str) else f"{int(month):02d}"


def _load_interp(period: str, month) -> xr.Dataset:
    f = config.INTERP_DIR / period / f"{_mkey(month)}.nc"
    return xr.open_dataset(f)


def _load_raw(var: str, period: str, month) -> xr.DataArray:
    f = config.RAW_DIR / period / f"{var}_{_mkey(month)}.nc"
    with xr.open_dataset(f) as ds:
        return ds[var].load()


def _delta_dir(period_current: str, period_future: str):
    """Flat config.DELTA_DIR for the original current/future pair (so
    04_apply_perturbation.py's existing lookups need no changes); any other
    period pair gets its own subdirectory so it can never collide with that
    default delta or with another period pair's delta."""
    if period_current == "current" and period_future == "future":
        return config.DELTA_DIR
    return config.DELTA_DIR / f"{period_future}_minus_{period_current}"


def compute_month(month, pop_regridder: regrid.SparseRegridder,
                   period_current: str = "current", period_future: str = "future",
                   overwrite: bool = False) -> None:
    out_dir = _delta_dir(period_current, period_future)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_mkey(month)}.nc"
    if out_file.exists() and not overwrite:
        return

    cur = _load_interp(period_current, month)
    fut = _load_interp(period_future, month)
    cam_lat, cam_lon = cur["lat"].values, cur["lon"].values

    delta_ght = (fut["GHT"].values - cur["GHT"].values)
    delta_tt = (fut["TT"].values - cur["TT"].values)
    delta_rh = (fut["RH"].values - cur["RH"].values)

    ght_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_ght,
                                             config.ERA5_LAT, config.ERA5_LON)
    tt_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_tt,
                                            config.ERA5_LAT, config.ERA5_LON)
    rh_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_rh,
                                            config.ERA5_LAT, config.ERA5_LON)

    sst_cur_cam = pop_regridder(_load_raw("SST", period_current, month).values)
    sst_fut_cam = pop_regridder(_load_raw("SST", period_future, month).values)
    delta_sst_cam = np.nan_to_num(sst_fut_cam - sst_cur_cam, nan=0.0)
    sst_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_sst_cam,
                                             config.ERA5_LAT, config.ERA5_LON)

    ts_cur = _load_raw("TS", period_current, month).values
    ts_fut = _load_raw("TS", period_future, month).values
    delta_ts_cam = ts_fut - ts_cur
    ts_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_ts_cam,
                                            config.ERA5_LAT, config.ERA5_LON)

    ds_out = xr.Dataset(
        data_vars=dict(
            GHT=(("lev_pa", "lat", "lon"), np.nan_to_num(ght_era5).astype(np.float32)),
            TT=(("lev_pa", "lat", "lon"), np.nan_to_num(tt_era5).astype(np.float32)),
            RH=(("lev_pa", "lat", "lon"), np.nan_to_num(rh_era5).astype(np.float32)),
            SST=(("lat", "lon"), np.nan_to_num(sst_era5).astype(np.float32)),
            SKINTEMP=(("lat", "lon"), np.nan_to_num(ts_era5).astype(np.float32)),
        ),
        coords=dict(lev_pa=cur["lev_pa"].values, lat=config.ERA5_LAT, lon=config.ERA5_LON),
        attrs=dict(month=month, period_current=period_current, period_future=period_future,
                   description=f"{period_future} minus {period_current} LENS2 "
                                "10-member-ensemble-mean climatology"),
    )
    cur.close()
    fut.close()

    tmp = out_file.with_suffix(".nc.tmp")
    ds_out.to_netcdf(tmp)
    tmp.rename(out_file)
    print(f"wrote {out_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", default=None)
    p.add_argument("--period-current", default="current", choices=list(config.PERIODS),
                    help="period name (config.PERIODS) used as the 'current' "
                         "(subtracted) side of the delta, default: current")
    p.add_argument("--period-future", default="future", choices=list(config.PERIODS),
                    help="period name (config.PERIODS) used as the 'future' "
                         "(minuend) side of the delta, default: future")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--annual", action="store_true",
                        help="compute a single annual-mean delta (DELTA_DIR/ann.nc) "
                             "from the annual-mean current/future ensemble-mean "
                             "climatologies (INTERP_DIR/period/ann.nc, from "
                             "02_interp_pressure_rh.py --annual) instead of the 12 "
                             "per-month deltas")
    group.add_argument("--label", default=None,
                        help="compute a single custom seasonal delta "
                             "(DELTA_DIR/<label>.nc) from the seasonal-mean "
                             "current/future ensemble-mean climatologies "
                             "(INTERP_DIR/period/<label>.nc, from "
                             "02_interp_pressure_rh.py --label <label>) instead "
                             "of the 12 per-month deltas, e.g. --label JJASO")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    config.ensure_dirs()
    if args.annual:
        months = ["ann"]
    elif args.label:
        months = [args.label]
    else:
        months = [int(m) for m in args.months.split(",")] if args.months else config.MONTHS
    pop_regridder = regrid.SparseRegridder(config.POP_TO_CAM_WEIGHTS)

    for month in months:
        compute_month(month, pop_regridder,
                      period_current=args.period_current, period_future=args.period_future,
                      overwrite=args.overwrite)


if __name__ == "__main__":
    main()
