#!/usr/bin/env python3
"""
Stage 01a: build each of the 10 LENS2 ensemble members' own monthly
climatologies for T, Q, Z3, TS, SST, for the "current" (1970-1999) and
"future" (2066-2095) periods -- one climatology per (member, period, month,
var), written to config.MEMBER_RAW_DIR. Stage 01b then averages the 10
members' climatologies together (equal weight per member) into the
single ensemble-mean climatology stages 02-04 consume.

This is a straight per-member adaptation of
/glade/u/home/mingge/MPAS/MPAS_LENS2_PGW/01_climatology.py (see that
script's docstring for the streaming/end-of-period-stamping rationale,
which applies unchanged here) -- the only difference is that this loops
over config.MEMBERS instead of a single hardcoded member.

Usage:
    python 01a_member_climatology.py --period current
    python 01a_member_climatology.py --period future --vars T,Q,Z3
    python 01a_member_climatology.py --members 1011.001,1031.002
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

import config
import lib.catalog as catalog

_TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def _load_month_value(var: str, year: int, month: int, ens_mem: str, var_id: str) -> xr.DataArray:
    freq = config.FREQ_OF[var]
    # day_1 chunks are named/bounded by their literal on-disk timestamps,
    # and (per the end-of-period stamping below) the record we actually need
    # first is day 2 of the month -- looking chunks up by day=1 can pick the
    # prior decade's chunk when day 1 00:00 still falls inside its end bound.
    lookup_day = 2 if freq == "day_1" else 1
    src = catalog.find_file(var, year, month, ens_mem, var_id, day=lookup_day)
    with xr.open_dataset(src, decode_times=_TIME_CODER, decode_timedelta=False) as ds:
        if freq == "month_1":
            # CAM monthly-mean output stamps each record with the *end* of
            # the averaging period, so the January mean carries a February
            # timestamp -- select the following month to land on it.
            label_year, label_month = (year + 1, 1) if month == 12 else (year, month + 1)
            sub = ds[var].sel(time=f"{label_year}-{label_month:02d}")
            return sub.isel(time=0).load()
        else:  # day_1 -> reduce to this year's monthly mean immediately
            # POP daily output is also end-of-period stamped, so day 1's
            # mean carries a day-2 timestamp and the month's last day's mean
            # carries next month's day-1 timestamp. Slicing on that range
            # (rather than computing the month length ourselves) also
            # sidesteps Gregorian vs. noleap-calendar day-count mismatches.
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            sub = ds[var].sel(time=slice(f"{year}-{month:02d}-02",
                                          f"{next_year}-{next_month:02d}-01"))
            if sub.sizes.get("time", 0) == 0:
                raise ValueError(f"empty selection for {var} {year}-{month:02d} from {src}")
            return sub.mean(dim="time").load()


def build_member_climatology(ens_mem: str, var_id: str, period_name: str, months: list,
                              variables: list, overwrite: bool = False) -> None:
    y0, y1 = config.PERIODS[period_name]
    n_years = y1 - y0 + 1
    out_dir = config.MEMBER_RAW_DIR / period_name / f"{ens_mem}.{var_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        for month in months:
            out_file = out_dir / f"{var}_{month:02d}.nc"
            if out_file.exists() and not overwrite:
                continue

            running_sum = None
            template = None
            n_ok = 0
            for year in range(y0, y1 + 1):
                try:
                    da = _load_month_value(var, year, month, ens_mem, var_id)
                except FileNotFoundError as e:
                    print(f"[MISSING] {ens_mem}.{var_id} {var} {year}-{month:02d}: {e}",
                          file=sys.stderr)
                    continue
                if template is None:
                    template = da
                    running_sum = np.array(da.values, dtype=np.float64, copy=True)
                else:
                    if da.shape != template.shape:
                        raise ValueError(f"{ens_mem}.{var_id} {var} {year}-{month:02d} shape "
                                          f"{da.shape} != template {template.shape}")
                    running_sum += da.values
                n_ok += 1

            if template is None:
                print(f"[SKIP] no data for {ens_mem}.{var_id} {var} month={month:02d} "
                      f"({period_name})", file=sys.stderr)
                continue
            if n_ok != n_years:
                print(f"[WARN] {ens_mem}.{var_id} {var} month={month:02d} ({period_name}): "
                      f"{n_ok}/{n_years} years found", file=sys.stderr)

            clim = template.copy(data=(running_sum / n_ok).astype(template.dtype))
            clim.attrs["n_years_averaged"] = n_ok
            clim.attrs["period_start_year"] = y0
            clim.attrs["period_end_year"] = y1
            clim.attrs["ens_mem"] = ens_mem
            clim.attrs["var_id"] = var_id
            clim.name = var

            tmp = out_file.with_suffix(".nc.tmp")
            clim.to_dataset().to_netcdf(tmp)
            tmp.rename(out_file)
            print(f"wrote {out_file} (n_years={n_ok})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", choices=list(config.PERIODS) + ["all"], default="all")
    p.add_argument("--months", default=None)
    p.add_argument("--vars", default=None)
    p.add_argument("--members", default=None,
                    help="comma-separated ens_mem.var_id pairs, e.g. 1011.001,1031.002; "
                         "default: all 10 in config.MEMBERS")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    config.ensure_dirs()
    periods = list(config.PERIODS) if args.period == "all" else [args.period]
    months = [int(m) for m in args.months.split(",")] if args.months else config.MONTHS
    variables = args.vars.split(",") if args.vars else config.ALL_LENS2_VARS
    if args.members:
        members = [tuple(m.split(".")) for m in args.members.split(",")]
    else:
        members = config.MEMBERS

    for ens_mem, var_id in members:
        for period_name in periods:
            build_member_climatology(ens_mem, var_id, period_name, months, variables,
                                      overwrite=args.overwrite)


if __name__ == "__main__":
    main()
