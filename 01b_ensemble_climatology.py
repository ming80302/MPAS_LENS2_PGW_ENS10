#!/usr/bin/env python3
"""
Stage 01b: average the 10 members' own climatologies (stage 01a,
config.MEMBER_RAW_DIR) into a single equal-weight ensemble-mean
climatology per (period, var, month), written to config.RAW_DIR in
exactly the same layout MPAS_LENS2_PGW's single-member stage 01 produces
(RAW_DIR/<period>/<var>_<month>.nc) -- so stages 02-04 are reused
completely unchanged from that pipeline; they never know an ensemble mean
was involved.

Equal weight per member (not per member-year): each member's 30-year
climatology counts once, regardless of that member's own valid-year count,
matching standard ensemble-mean practice and
/glade/u/home/mingge/LENS2/MPAS_PGW/02_ensemble_climatology.py's approach.

Usage:
    python 01b_ensemble_climatology.py --period current
    python 01b_ensemble_climatology.py --period future --vars T,Q,Z3
    python 01b_ensemble_climatology.py --period current --annual
    python 01b_ensemble_climatology.py --period current --months 6,7,8,9,10 --label JJASO
        # builds the 5 monthly ensemble-mean climatologies (if not already
        # present) then collapses them into {var}_JJASO.nc
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import xarray as xr

import config


def combine_variable(period_name: str, var: str, month: int, overwrite: bool = False) -> None:
    out_dir = config.RAW_DIR / period_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{var}_{month:02d}.nc"
    if out_file.exists() and not overwrite:
        return

    running_sum = None
    template = None
    n_ok = 0
    for ens_mem, var_id in config.MEMBERS:
        f = config.MEMBER_RAW_DIR / period_name / f"{ens_mem}.{var_id}" / f"{var}_{month:02d}.nc"
        if not f.exists():
            print(f"[MISSING] {f} -- run 01a_member_climatology.py for {ens_mem}.{var_id} first",
                  file=sys.stderr)
            continue
        with xr.open_dataset(f) as ds:
            da = ds[var].load()
        if template is None:
            template = da
            running_sum = np.array(da.values, dtype=np.float64, copy=True)
        else:
            if da.shape != template.shape:
                raise ValueError(f"{f}: shape {da.shape} != template {template.shape}")
            running_sum += da.values
        n_ok += 1

    if template is None:
        print(f"[SKIP] no member climatologies found for {var} month={month:02d} "
              f"({period_name})", file=sys.stderr)
        return
    if n_ok != len(config.MEMBERS):
        print(f"[WARN] {var} month={month:02d} ({period_name}): "
              f"{n_ok}/{len(config.MEMBERS)} members found", file=sys.stderr)

    ens_mean = template.copy(data=(running_sum / n_ok).astype(template.dtype))
    ens_mean.attrs.pop("ens_mem", None)   # template's attrs are from one member's file;
    ens_mean.attrs.pop("var_id", None)    # drop rather than let one member's identity leak through
    ens_mean.attrs["n_members_averaged"] = n_ok
    ens_mean.attrs["period_start_year"], ens_mean.attrs["period_end_year"] = config.PERIODS[period_name]
    ens_mean.name = var

    tmp = out_file.with_suffix(".nc.tmp")
    ens_mean.to_dataset().to_netcdf(tmp)
    tmp.rename(out_file)
    print(f"wrote {out_file} (n_members={n_ok})")


def build_mean(period_name: str, months: list, label: str, variables: list,
                overwrite: bool = False) -> None:
    """Seasonal/annual-mean ensemble-mean climatology = unweighted mean of
    the already-computed ensemble-mean monthly climatologies (RAW_DIR/period/
    {var}_MM.nc for each month in `months`), written as {var}_<label>.nc in
    the same directory -- exactly analogous to
    MPAS_LENS2_PGW/01_climatology.py's build_annual_mean, just generalized to
    an arbitrary month subset/label instead of always all 12 months under
    '_ann'. Averaging
    commutes here (equal weight per member and equal weight per month, both
    simple arithmetic means), so mean-over-months-of(mean-over-members)
    equals mean-over-members-of(mean-over-months) -- this function only
    needs to average combine_variable's monthly outputs, not re-touch the 10
    per-member climatologies."""
    out_dir = config.RAW_DIR / period_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        out_file = out_dir / f"{var}_{label}.nc"
        if out_file.exists() and not overwrite:
            continue

        month_files = [out_dir / f"{var}_{m:02d}.nc" for m in months]
        missing = [f.name for f in month_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{var} ({period_name}): {label} mean needs the ensemble-mean "
                f"monthly climatologies for months {months} first, missing {missing}")

        das = []
        for f in month_files:
            with xr.open_dataset(f) as ds:
                das.append(ds[var].load())
        stacked = np.stack([da.values for da in das], axis=0)
        mean = das[0].copy(data=stacked.mean(axis=0).astype(das[0].dtype))
        mean.attrs.pop("ens_mem", None)
        mean.attrs.pop("var_id", None)
        mean.attrs["n_months_averaged"] = len(das)
        mean.attrs["months_averaged"] = ",".join(str(m) for m in months)
        mean.attrs["n_members_averaged"] = das[0].attrs.get("n_members_averaged")
        mean.attrs["period_start_year"], mean.attrs["period_end_year"] = config.PERIODS[period_name]
        mean.attrs["climatology_mode"] = f"mean_of_ensemble_mean_monthly_climatologies_{label}"
        mean.name = var

        tmp = out_file.with_suffix(".nc.tmp")
        mean.to_dataset().to_netcdf(tmp)
        tmp.rename(out_file)
        print(f"wrote {out_file} (mean of {len(das)} ensemble-mean monthly climatologies)")


def build_annual_mean(period_name: str, variables: list, overwrite: bool = False) -> None:
    """Annual-mean ensemble-mean climatology -- build_mean specialized to
    all 12 months, label 'ann' (kept as a thin wrapper for backward
    compatibility with any existing callers)."""
    build_mean(period_name, config.MONTHS, "ann", variables, overwrite=overwrite)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", choices=list(config.PERIODS) + ["all"], default="all")
    p.add_argument("--months", default=None)
    p.add_argument("--vars", default=None)
    p.add_argument("--overwrite", action="store_true")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--annual", action="store_true",
                        help="also build an annual-mean ensemble-mean climatology "
                             "({var}_ann.nc) = unweighted mean of the 12 monthly "
                             "ensemble-mean climatologies, for a non-seasonal "
                             "(climate-mean) PGW delta. Builds the 12 monthly "
                             "ensemble-mean climatologies first if not already "
                             "present (ignores --months; requires 01a to have "
                             "already produced all 10 members' monthly "
                             "climatologies).")
    group.add_argument("--label", default=None,
                        help="also build a custom seasonal-mean ensemble-mean "
                             "climatology ({var}_<label>.nc) = unweighted mean "
                             "of the ensemble-mean monthly climatologies for "
                             "the months given in --months (required together "
                             "with --label), e.g. --months 6,7,8,9,10 --label "
                             "JJASO for a Jun-Oct mean. Builds those monthly "
                             "ensemble-mean climatologies first if not already "
                             "present (requires 01a to have already produced "
                             "all 10 members' climatologies for those months).")
    args = p.parse_args()

    if args.label and not args.months:
        p.error("--label requires --months")

    config.ensure_dirs()
    periods = list(config.PERIODS) if args.period == "all" else [args.period]
    variables = args.vars.split(",") if args.vars else config.ALL_LENS2_VARS

    for period_name in periods:
        if args.annual:
            for var in variables:
                for month in config.MONTHS:
                    combine_variable(period_name, var, month, overwrite=args.overwrite)
            build_annual_mean(period_name, variables, overwrite=args.overwrite)
        elif args.label:
            months = [int(m) for m in args.months.split(",")]
            for var in variables:
                for month in months:
                    combine_variable(period_name, var, month, overwrite=args.overwrite)
            build_mean(period_name, months, args.label, variables, overwrite=args.overwrite)
        else:
            months = [int(m) for m in args.months.split(",")] if args.months else config.MONTHS
            for var in variables:
                for month in months:
                    combine_variable(period_name, var, month, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
