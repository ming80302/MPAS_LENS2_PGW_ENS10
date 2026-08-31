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
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

import config
import lib.regrid_latlon as regrid


def _mkey(month) -> str:
    """'ann' passes through for the annual-mean pseudo-month; a real
    calendar month is zero-padded as usual (see 02_interp_pressure_rh.py)."""
    return "ann" if month == "ann" else f"{int(month):02d}"


def _load_interp(period: str, month) -> xr.Dataset:
    f = config.INTERP_DIR / period / f"{_mkey(month)}.nc"
    return xr.open_dataset(f)


def _load_raw(var: str, period: str, month) -> xr.DataArray:
    f = config.RAW_DIR / period / f"{var}_{_mkey(month)}.nc"
    with xr.open_dataset(f) as ds:
        return ds[var].load()


def compute_month(month, pop_regridder: regrid.SparseRegridder,
                   overwrite: bool = False) -> None:
    out_file = config.DELTA_DIR / f"{_mkey(month)}.nc"
    if out_file.exists() and not overwrite:
        return

    cur = _load_interp("current", month)
    fut = _load_interp("future", month)
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

    sst_cur_cam = pop_regridder(_load_raw("SST", "current", month).values)
    sst_fut_cam = pop_regridder(_load_raw("SST", "future", month).values)
    delta_sst_cam = np.nan_to_num(sst_fut_cam - sst_cur_cam, nan=0.0)
    sst_era5 = regrid.bilinear_regular_grid(cam_lat, cam_lon, delta_sst_cam,
                                             config.ERA5_LAT, config.ERA5_LON)

    ts_cur = _load_raw("TS", "current", month).values
    ts_fut = _load_raw("TS", "future", month).values
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
        attrs=dict(month=month, description="future minus current LENS2 10-member-ensemble-mean climatology"),
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
    p.add_argument("--annual", action="store_true",
                    help="compute a single annual-mean delta (DELTA_DIR/ann.nc) "
                         "from the annual-mean current/future ensemble-mean "
                         "climatologies (INTERP_DIR/period/ann.nc, from "
                         "02_interp_pressure_rh.py --annual) instead of the 12 "
                         "per-month deltas")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    config.ensure_dirs()
    if args.annual:
        months = ["ann"]
    else:
        months = [int(m) for m in args.months.split(",")] if args.months else config.MONTHS
    pop_regridder = regrid.SparseRegridder(config.POP_TO_CAM_WEIGHTS)

    for month in months:
        compute_month(month, pop_regridder, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
