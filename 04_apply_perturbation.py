#!/usr/bin/env python3
"""
Stage 04: add the LENS2 10-member-ensemble-mean PGW delta (stage 03) onto real ERA5 WPS
intermediate-format files, writing perturbed copies to
config.PERTURBED_IM_DIR (originals under config.ERA5_IM_DIR are never
modified). Pure Python -- lib/wps_intermediate.py -- built and
byte-for-byte validated against the real WPS Fortran modules and
rd_intermediate.exe (see that module's docstring), so no Fortran build is
needed.

Matching:
  - GHT, TT, RH: record's xlvl (Pa) is matched against the delta file's
    lev_pa coordinate (exact match expected -- both come from the same
    37-level ERA5 standard set, see config.ERA5_PRESSURE_LEVELS_PA).
  - SST: only perturbed where the ERA5 record's own value isn't already the
    land fill sentinel (~-1e30) -- SST is only meaningful over ocean, and
    the fill value must not be perturbed into a bogus finite number.
  - SKINTEMP: perturbed everywhere (defined over both land and ocean).
  - Every other field/record is copied through unchanged.
  - The delta grid (1440x721) is required to exactly match the ERA5
    record's own (nx, ny) -- both come from the same ERA5 grid definition
    (config.ERA5_LAT/LON), so no further regridding happens here; a
    mismatch raises rather than silently interpolating again.

A soft floor of 271.0 K (typical seawater freezing point) is applied to
perturbed SST to avoid an unphysical value at extreme grid points; nothing
else is clipped.

Also perturbs the separate periodic SST-update stream under
config.ERA5_IM_DIR/SST_<year>/ERA5_SST:<date> (a different file layout from
the main ERA5:* archive, one used by WPS's "SST update" mechanism), writing
the perturbed copies out renamed to SST_UPDATE:<date>. The SST/skin-temp
record's *field name* is inconsistent across years in the real archive
(some years "SST", some "SKINTEMP" -- confirmed by inspection, not assumed)
even though its content is always sea-surface temperature, so any record
named "SST" or "SKINTEMP" is perturbed. Some years' files carry additional
unrelated records alongside it (e.g. 2026's files also have a "SEAICE"
record) -- those are copied through unchanged, same as the main stream's
treatment of non-perturbed fields. Unlike the main stream's SST record,
the SST/SKINTEMP record here carries no land-fill sentinel (real,
non-sentinel skin-temperature values exist over land too), so land vs.
ocean is instead determined from the static LANDSEA mask read once from
any single main-stream ERA5:* file (time-invariant geography, safe to
reuse across every SST-update file/date).

Usage:
    python 04_apply_perturbation.py                      # every ERA5:* file
                                                           # and every SST
                                                           # update file
    python 04_apply_perturbation.py --years 1984,1998     # only files whose
                                                           # hdate/year falls
                                                           # in these years
    python 04_apply_perturbation.py --overwrite           # also redo files
                                                           # whose output
                                                           # already exists
                                                           # (default: skip)
    python 04_apply_perturbation.py --annual              # single climate-mean
                                                           # delta applied to
                                                           # every record
    python 04_apply_perturbation.py --label JJASO         # single seasonal
                                                           # delta (DELTA_DIR/
                                                           # JJASO.nc, from
                                                           # 03_compute_delta_regrid.py
                                                           # --label JJASO)
                                                           # applied to every
                                                           # record, written to
                                                           # PERTURBED_IM_DIR_JJASO
    python 04_apply_perturbation.py --label JJASO \\
        --delta-dir .../MPAS_LENS2_PGW_ENS10_scratch/03_delta/2011_2041_minus_current
        # read the delta from a non-default DELTA_DIR (e.g. the
        # <period-future>_minus_<period-current>/ subdirectory
        # 03_compute_delta_regrid.py --period-current/--period-future writes
        # for a non-default period pair) instead of config.DELTA_DIR; output
        # goes to PERTURBED_IM_DIR_JJASO_<delta-dir-basename> so it can never
        # collide with a default-period-pair JJASO run
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import config
import lib.wps_intermediate as wps

SST_FLOOR_K = 271.0
SST_FILL_THRESHOLD = -1.0e29  # anything below this is treated as "land, don't touch"


def load_deltas(delta_dir) -> dict:
    """month -> {'lev_pa': array, 'GHT':.., 'TT':.., 'RH':.., 'SST':.., 'SKINTEMP':..}"""
    deltas = {}
    for f in sorted(delta_dir.glob("*.nc")):
        if not f.stem.isdigit():
            continue  # 'ann' or a custom --label file, only used via --annual/--label
        month = int(f.stem)
        with xr.open_dataset(f) as ds:
            deltas[month] = {k: ds[k].values for k in
                              ["lev_pa", "GHT", "TT", "RH", "SST", "SKINTEMP"]}
    return deltas


def _load_broadcast_delta(f) -> dict:
    """Load a single pseudo-month delta file and broadcast it to every
    calendar month so perturb_one_file/perturb_one_sst_update_file need no
    changes -- they already look up deltas[month] per record."""
    if not f.exists():
        sys.exit(f"{f} not found -- run 03_compute_delta_regrid.py for it first")
    with xr.open_dataset(f) as ds:
        delta = {k: ds[k].values for k in ["lev_pa", "GHT", "TT", "RH", "SST", "SKINTEMP"]}
    return {m: delta for m in range(1, 13)}


def load_annual_delta(delta_dir) -> dict:
    """Single annual-mean delta (<delta_dir>/ann.nc, from
    03_compute_delta_regrid.py --annual)."""
    return _load_broadcast_delta(delta_dir / "ann.nc")


def load_label_delta(delta_dir, label: str) -> dict:
    """Single custom seasonal delta (<delta_dir>/<label>.nc, from
    03_compute_delta_regrid.py --label <label>), e.g. a Jun-Oct ('JJASO')
    mean -- broadcast the same way --annual's delta is."""
    return _load_broadcast_delta(delta_dir / f"{label}.nc")


def perturb_field(rec: wps.MetField, month_delta: dict) -> bool:
    """Mutates rec.slab in place if it's a field we perturb. Returns True
    if modified."""
    if rec.field in config.PERTURB_FIELDS_3D:
        if rec.xlvl == config.ERA5_SURFACE_LEVEL_PA:
            # ERA5 IM files also carry 2 m TT / 2 m RH records under these
            # same field names at the surface sentinel level -- a different
            # physical quantity from the 3-D pressure-level profile we have
            # a delta for. Left unperturbed by design, not an error.
            return False
        lev_pa = month_delta["lev_pa"]
        idx = np.argmin(np.abs(lev_pa - rec.xlvl))
        if abs(lev_pa[idx] - rec.xlvl) > 1.0:
            print(f"[WARN] {rec.field} xlvl={rec.xlvl} has no matching delta "
                  f"level (nearest {lev_pa[idx]}, diff > 1 Pa) -- skipped", file=sys.stderr)
            return False
        delta = month_delta[rec.field][idx]
        if delta.shape != rec.slab.shape:
            raise ValueError(f"{rec.field} delta shape {delta.shape} != "
                              f"record slab shape {rec.slab.shape}")
        rec.slab = rec.slab + delta.astype(rec.slab.dtype)
        if rec.field == "RH":
            rec.slab = np.clip(rec.slab, 0.0, 100.0)
        return True

    if rec.field in config.PERTURB_FIELDS_SFC:
        delta = month_delta[rec.field]
        if delta.shape != rec.slab.shape:
            raise ValueError(f"{rec.field} delta shape {delta.shape} != "
                              f"record slab shape {rec.slab.shape}")
        if rec.field == "SST":
            ocean = rec.slab > SST_FILL_THRESHOLD
            rec.slab = np.where(ocean,
                                 np.maximum(rec.slab + delta.astype(rec.slab.dtype), SST_FLOOR_K),
                                 rec.slab)
        else:  # SKINTEMP
            rec.slab = rec.slab + delta.astype(rec.slab.dtype)
        return True

    return False


def load_ocean_mask() -> np.ndarray:
    """Boolean (ny, nx) array, True over ocean -- read from the LANDSEA
    record of any single main-stream ERA5:* file (static geography, so one
    file covers every date). Needed for the SST-update stream, which
    (unlike the main stream's own SST record) carries no land-fill sentinel
    to derive an ocean mask from directly."""
    candidates = select_files(None)
    if not candidates:
        raise RuntimeError(f"no main-stream ERA5:* file found under {config.ERA5_IM_DIR} "
                            "to read a LANDSEA mask from (needed for the SST-update stream)")
    src = candidates[0]
    recs = wps.read_file(src)
    landsea = next((r for r in recs if r.field == "LANDSEA"), None)
    if landsea is None:
        raise RuntimeError(f"{src}: no LANDSEA record found to build an ocean mask from")
    return landsea.slab == 0.0


def perturb_sst_update_field(rec: wps.MetField, month_delta: dict, ocean_mask: np.ndarray) -> None:
    """Mutates rec.slab in place -- every record in an SST-update file is
    treated as SST regardless of its own field name (see module docstring:
    the real archive inconsistently labels it 'SST' or 'SKINTEMP')."""
    delta = month_delta["SST"]
    if delta.shape != rec.slab.shape:
        raise ValueError(f"SST-update delta shape {delta.shape} != "
                          f"record slab shape {rec.slab.shape}")
    if ocean_mask.shape != rec.slab.shape:
        raise ValueError(f"ocean mask shape {ocean_mask.shape} != "
                          f"record slab shape {rec.slab.shape}")
    rec.slab = np.where(ocean_mask,
                         np.maximum(rec.slab + delta.astype(rec.slab.dtype), SST_FLOOR_K),
                         rec.slab)


def perturb_one_file(src: Path, dst: Path, deltas: dict) -> None:
    recs = wps.read_file(src)
    if not recs:
        print(f"[WARN] {src}: no records, skipping", file=sys.stderr)
        return

    month = int(recs[0].hdate[5:7])
    if month not in deltas:
        raise KeyError(f"{src}: month {month} has no stage-03 delta file "
                        f"(run 03_compute_delta_regrid.py for it first)")
    month_delta = deltas[month]

    n_touched = 0
    for r in recs:
        # No blanket nx/ny check here: some real ERA5 IM files (e.g. 2026's,
        # built via era5_to_int) carry a handful of fields -- notably PSFC,
        # SOILHGT -- on a different native grid (era5_to_int sources those
        # from the ~0.281deg Gaussian model-level archive, not the 0.25deg
        # lat-lon surface archive) even though every perturbed field is on
        # the standard ERA5 grid. Those off-grid fields are never perturbed
        # (see module docstring: unmatched fields pass through unchanged),
        # so checking grid size here would wrongly reject an otherwise-valid
        # file. perturb_field() already does its own precise shape check
        # (delta.shape != rec.slab.shape) for every field it actually
        # touches, which is the only check that matters.
        if perturb_field(r, month_delta):
            n_touched += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    wps.write_file(tmp, recs)
    tmp.rename(dst)
    print(f"wrote {dst} ({n_touched} of {len(recs)} records perturbed)")


def perturb_one_sst_update_file(src: Path, dst: Path, deltas: dict, ocean_mask: np.ndarray) -> None:
    recs = wps.read_file(src)
    if not recs:
        print(f"[WARN] {src}: no records, skipping", file=sys.stderr)
        return

    month = int(recs[0].hdate[5:7])
    if month not in deltas:
        raise KeyError(f"{src}: month {month} has no stage-03 delta file "
                        f"(run 03_compute_delta_regrid.py for it first)")
    month_delta = deltas[month]

    n_touched = 0
    for r in recs:
        if r.field not in ("SST", "SKINTEMP"):
            continue  # e.g. SEAICE in 2026's files -- pass through unchanged
        if r.nx != config.ERA5_NX or r.ny != config.ERA5_NY:
            raise ValueError(f"{src}: field {r.field} is {r.nx}x{r.ny}, expected "
                              f"{config.ERA5_NX}x{config.ERA5_NY} (config.ERA5_LAT/LON) "
                              f"-- delta grid would not line up")
        perturb_sst_update_field(r, month_delta, ocean_mask)
        n_touched += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    wps.write_file(tmp, recs)
    tmp.rename(dst)
    print(f"wrote {dst} ({n_touched} of {len(recs)} SST-update record(s) perturbed)")


_IM_NAME_RE = re.compile(r"^" + re.escape(config.ERA5_IM_PREFIX) + r":\d{4}-\d{2}-\d{2}_\d{2}$")
# on-disk SST-update files still carry ungrib's own out_prefix (ERA5_SST);
# the perturbed copies are renamed to _SST_UPDATE_PREFIX on write.
_SST_UPDATE_SRC_PREFIX = "ERA5_SST"
_SST_UPDATE_PREFIX = "SST_UPDATE"
_SST_UPDATE_NAME_RE = re.compile(r"^" + re.escape(_SST_UPDATE_SRC_PREFIX) + r":\d{4}-\d{2}-\d{2}_\d{2}$")


def select_files(years: list) -> list:
    pattern = str(config.ERA5_IM_DIR / f"{config.ERA5_IM_PREFIX}:*")
    files = [Path(f) for f in glob.glob(pattern)]
    # exclude the separate periodic-SST-update stream (ERA5_SST: prefix) and
    # any non-IM sidecar file that happens to share the ERA5: prefix (e.g.
    # "ERA5:2015-06-01_00.nc", a NetCDF conversion of the IM file found
    # alongside the real ones under config.ERA5_IM_DIR) -- only the exact
    # "ERA5:YYYY-MM-DD_HH" naming is a genuine WPS intermediate file.
    files = [f for f in files if _IM_NAME_RE.match(f.name)]
    if years:
        files = [f for f in files if int(f.name.split(":")[1][:4]) in years]
    return sorted(files)


def select_sst_update_files(years: list) -> list:
    """The periodic SST-update stream lives under per-year subdirectories,
    e.g. config.ERA5_IM_DIR/SST_1984/ERA5_SST:1984-06-01_00 (perturbed copies
    are written out renamed to SST_UPDATE:1984-06-01_00)."""
    pattern = str(config.ERA5_IM_DIR / "SST_*" / f"{_SST_UPDATE_SRC_PREFIX}:*")
    files = [Path(f) for f in glob.glob(pattern)]
    files = [f for f in files if _SST_UPDATE_NAME_RE.match(f.name)]
    if years:
        files = [f for f in files if int(f.name.split(":")[1][:4]) in years]
    return sorted(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default=None,
                    help="comma-separated years to restrict to, default: all "
                         "ERA5:* files under config.ERA5_IM_DIR")
    p.add_argument("--overwrite", action="store_true",
                    help="re-perturb files whose output already exists under "
                         "config.PERTURBED_IM_DIR (default: skip them)")
    p.add_argument("--delta-dir", default=None,
                    help="override the directory delta files are read from "
                         "(default: config.DELTA_DIR). Use this to apply a delta "
                         "computed with 03_compute_delta_regrid.py's "
                         "--period-current/--period-future for a non-default "
                         "period pair, e.g. DELTA_DIR/2011_2041_minus_current/. "
                         "When set, the output directory name is suffixed with "
                         "this directory's basename so it can never collide with "
                         "a default-DELTA_DIR run")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--annual", action="store_true",
                        help="apply a single annual-mean, non-seasonal delta "
                             "(DELTA_DIR/ann.nc, from 03_compute_delta_regrid.py "
                             "--annual) to every record regardless of its own "
                             "calendar month, instead of the default per-month delta")
    group.add_argument("--label", default=None,
                        help="apply a single custom seasonal delta (DELTA_DIR/"
                             "<label>.nc, from 03_compute_delta_regrid.py --label "
                             "<label>) to every record regardless of its own "
                             "calendar month, instead of the default per-month "
                             "delta, e.g. --label JJASO. Output goes to a separate "
                             "PERTURBED_IM_DIR_<label> directory so it never "
                             "collides with the default or --annual output")
    args = p.parse_args()

    config.ensure_dirs()
    years = [int(y) for y in args.years.split(",")] if args.years else None
    delta_dir = Path(args.delta_dir) if args.delta_dir else config.DELTA_DIR

    if args.annual:
        deltas = load_annual_delta(delta_dir)
    elif args.label:
        deltas = load_label_delta(delta_dir, args.label)
    else:
        deltas = load_deltas(delta_dir)
        if not deltas:
            sys.exit(f"no delta files found under {delta_dir} -- run "
                      "03_compute_delta_regrid.py first")

    if args.annual:
        out_root = config.PERTURBED_IM_DIR_ANNUAL
    elif args.label:
        out_root = Path(str(config.PERTURBED_IM_DIR) + f"_{args.label}")
    else:
        out_root = config.PERTURBED_IM_DIR
    if args.delta_dir:
        # disambiguate from a default-DELTA_DIR run of the same --annual/--label
        out_root = Path(str(out_root) + f"_{delta_dir.name}")

    files = select_files(years)
    sst_update_files = select_sst_update_files(years)
    if not files and not sst_update_files:
        sys.exit(f"no ERA5:* or SST-update files found under {config.ERA5_IM_DIR}"
                  + (f" for years {years}" if years else ""))

    for src in files:
        dst = out_root / src.name
        if dst.exists() and not args.overwrite:
            continue
        perturb_one_file(src, dst, deltas)

    if sst_update_files:
        ocean_mask = load_ocean_mask()
        for src in sst_update_files:
            dst_name = _SST_UPDATE_PREFIX + src.name[len(_SST_UPDATE_SRC_PREFIX):]
            dst = out_root / src.parent.relative_to(config.ERA5_IM_DIR) / dst_name
            if dst.exists() and not args.overwrite:
                continue
            perturb_one_sst_update_file(src, dst, deltas, ocean_mask)


if __name__ == "__main__":
    main()
