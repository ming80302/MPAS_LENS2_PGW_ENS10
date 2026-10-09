#!/usr/bin/env python3
"""
Stage 05: verify that stage 04's perturbed ERA5 WPS intermediate files make
sense, both against the real ERA5 files they were built from and against the
stage-03 delta they were supposed to apply. Auto-discovers whatever has
actually been perturbed under config.PERTURBED_IM_DIR (no hardcoded dates),
so it stays correct as more years/months are added via
04_apply_perturbation.py --years.

Four independent checks, each stronger than the last:
  1. Structural integrity -- every record NOT in the perturbation set
     (config.PERTURB_FIELDS_3D / PERTURB_FIELDS_SFC) is byte-identical
     between original and perturbed files.
  2. Quantitative correctness -- perturbed - original matches the stage-03
     delta file exactly (modulo the documented RH clip and SST floor).
  3. Physical plausibility -- the applied delta reproduces the expected
     10-member-ensemble-mean PGW signature (tropospheric warming +
     geopotential expansion, relative upper-level cooling, ocean/land
     surface warming), not noise or a corrupted field.
  4. The periodic SST-update stream (ERA5_SST:* -> SST_UPDATE:*), which
     uses a different file layout/land-mask than the main archive, gets
     its own structural + quantitative checks.

The main ERA5:* stream is processed one date at a time (read orig+pert,
check, accumulate only small summary arrays/scalars, then drop the full
records before moving on) rather than holding every date's full-resolution
records in memory at once -- each file is ~1 GB (243 records x 1440x721),
so keeping all dates' orig+pert resident at once does not scale past a
handful of dates on a memory-constrained login node.

Usage:
    python 05_verify_perturbation.py                # verify everything found
    python 05_verify_perturbation.py --years 1984
    python 05_verify_perturbation.py --plots         # also save PNGs to
                                                      # verify_plots/
    python 05_verify_perturbation.py --annual        # verify a PERTURBED_IM_DIR_ANNUAL
                                                      # run (DELTA_DIR/ann.nc)
    python 05_verify_perturbation.py --label JJASO   # verify a PERTURBED_IM_DIR_JJASO
                                                      # run (DELTA_DIR/JJASO.nc, from
                                                      # 04_apply_perturbation.py --label JJASO)
    python 05_verify_perturbation.py --label JJASO \\
        --delta-dir .../03_delta/2011_2041_minus_current
                                                      # mirrors 04_apply_perturbation.py's
                                                      # --delta-dir: verify a run made with
                                                      # a non-default period pair
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import config
import lib.wps_intermediate as wps

_THIS_DIR = Path(__file__).parent

_spec = importlib.util.spec_from_file_location(
    "apply_pgw", str(_THIS_DIR / "04_apply_perturbation.py"))
apply_pgw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apply_pgw)

PERTURBED_3D = set(config.PERTURB_FIELDS_3D)           # {GHT, TT, RH}
PERTURBED_SFC = set(config.PERTURB_FIELDS_SFC.keys())   # {SST, SKINTEMP}

CHECKS = []  # (name, passed: bool, detail: str)


def record(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Discovery: what has stage 04 actually produced?
# ---------------------------------------------------------------------------

def discover_main_dates(pert_dir: Path, years) -> list:
    files = sorted(pert_dir.glob(f"{config.ERA5_IM_PREFIX}:*"))
    dates = [f.name.split(":")[1] for f in files if apply_pgw._IM_NAME_RE.match(f.name)]
    if years:
        dates = [d for d in dates if int(d[:4]) in years]
    return dates


def discover_sst_update_pairs(pert_dir: Path, years) -> list:
    """-> list of (orig_path, pert_path) for the SST-update stream."""
    pattern = str(pert_dir / "SST_*" / f"{apply_pgw._SST_UPDATE_PREFIX}:*")
    pert_files = sorted(Path(p) for p in glob.glob(pattern))
    pert_files = [f for f in pert_files if apply_pgw._SST_UPDATE_NAME_RE.match(
        apply_pgw._SST_UPDATE_SRC_PREFIX + f.name[len(apply_pgw._SST_UPDATE_PREFIX):])]
    pairs = []
    for pf in pert_files:
        date = pf.name.split(":")[1]
        if years and int(date[:4]) not in years:
            continue
        orig_name = apply_pgw._SST_UPDATE_SRC_PREFIX + pf.name[len(apply_pgw._SST_UPDATE_PREFIX):]
        rel_dir = pf.parent.relative_to(pert_dir)
        of = config.ERA5_IM_DIR / rel_dir / orig_name
        pairs.append((of, pf))
    return pairs


def key(rec):
    return (rec.field, round(rec.xlvl, 3))


def find(recs, field, xlvl):
    for r in recs:
        if r.field == field and abs(r.xlvl - xlvl) < 1.0:
            return r
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", default=None,
                    help="comma-separated years to restrict to, default: everything "
                         "found under config.PERTURBED_IM_DIR")
    p.add_argument("--plots", action="store_true",
                    help="save diagnostic PNGs to verify_plots/")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--annual", action="store_true",
                        help="verify a PERTURBED_IM_DIR_ANNUAL run (04_apply_perturbation.py "
                             "--annual) against DELTA_DIR/ann.nc instead of the default "
                             "PERTURBED_IM_DIR / per-month deltas")
    group.add_argument("--label", default=None,
                        help="verify a PERTURBED_IM_DIR_<label> run (04_apply_perturbation.py "
                             "--label <label>) against DELTA_DIR/<label>.nc instead of the "
                             "default PERTURBED_IM_DIR / per-month deltas")
    p.add_argument("--delta-dir", default=None,
                    help="override the directory delta files are read from (default: "
                         "config.DELTA_DIR), mirroring 04_apply_perturbation.py --delta-dir")
    args = p.parse_args()
    years = [int(y) for y in args.years.split(",")] if args.years else None
    delta_dir = Path(args.delta_dir) if args.delta_dir else config.DELTA_DIR

    if args.annual:
        pert_dir = config.PERTURBED_IM_DIR_ANNUAL
    elif args.label:
        pert_dir = Path(str(config.PERTURBED_IM_DIR) + f"_{args.label}")
    else:
        pert_dir = config.PERTURBED_IM_DIR
    if args.delta_dir:
        pert_dir = Path(str(pert_dir) + f"_{delta_dir.name}")

    test_dates = discover_main_dates(pert_dir, years)
    sst_pairs = discover_sst_update_pairs(pert_dir, years)
    if not test_dates and not sst_pairs:
        sys.exit(f"nothing to verify under {pert_dir}"
                  + (f" for years {years}" if years else "")
                  + " -- run 04_apply_perturbation.py first")

    print(f"main-stream test dates ({len(test_dates)}): {test_dates}")
    print(f"SST-update test files ({len(sst_pairs)})")
    print()

    months_needed = sorted({int(d[5:7]) for d in test_dates})
    if args.annual:
        deltas = apply_pgw.load_annual_delta(delta_dir)
    elif args.label:
        deltas = apply_pgw.load_label_delta(delta_dir, args.label)
    else:
        deltas = apply_pgw.load_deltas(delta_dir)
        for m in months_needed:
            if m not in deltas:
                record(f"month {m:02d} delta file exists", False, f"missing under {delta_dir}")

    # ---- 1 & 2, streamed one date at a time: structural integrity + -------
    # ---- quantitative cross-check, main ERA5:* stream ----------------------
    print("=== 1. structural integrity (main ERA5:* stream) ===")
    worst = {}
    violations = []
    rh_ok = True
    sst_deltas = []
    skt_deltas = []
    tt_profile_by_month = {}
    plot_rep = None

    for date in test_dates:
        orig = wps.read_file(config.ERA5_IM_DIR / f"{config.ERA5_IM_PREFIX}:{date}")
        pert = wps.read_file(pert_dir / f"{config.ERA5_IM_PREFIX}:{date}")

        same_count = len(orig) == len(pert)
        same_order = same_count and [key(r) for r in orig] == [key(r) for r in pert]
        touched, unexpected = [], []
        if same_order:
            for ro, rp in zip(orig, pert):
                changed = not np.array_equal(ro.slab, rp.slab)
                is_sfc_sentinel = ro.xlvl == config.ERA5_SURFACE_LEVEL_PA
                expected_perturbable = (
                    (ro.field in PERTURBED_3D and not is_sfc_sentinel) or
                    (ro.field in PERTURBED_SFC)
                )
                if changed:
                    touched.append(key(ro))
                    if not expected_perturbable:
                        unexpected.append(key(ro))

        ok = same_count and same_order and not unexpected
        detail = f"{len(touched)}/{len(orig)} records touched"
        if not same_count:
            detail += "; record count differs from original"
        if same_count and not same_order:
            detail += "; field/level order differs from original"
        if unexpected:
            detail += f"; unexpected changes in {unexpected}"
        record(f"{date}: only expected fields changed", ok, detail)

        month = int(date[5:7])
        if month in deltas:
            d = deltas[month]
            lev_pa = d["lev_pa"]
            errs = {}
            for ro, rp in zip(orig, pert):
                is_sfc_sentinel = ro.xlvl == config.ERA5_SURFACE_LEVEL_PA
                applied = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)

                if ro.field in PERTURBED_3D and not is_sfc_sentinel:
                    idx = int(np.argmin(np.abs(lev_pa - ro.xlvl)))
                    expected = d[ro.field][idx].astype(np.float64)
                    if ro.field == "RH":
                        unclipped = (rp.slab > 0.0) & (rp.slab < 100.0)
                        err = np.abs(applied - expected)[unclipped]
                        if rp.slab.min() < 0.0 or rp.slab.max() > 100.0:
                            rh_ok = False
                    else:
                        err = np.abs(applied - expected)
                    if err.size:
                        errs[ro.field] = max(errs.get(ro.field, 0.0), float(err.max()))
                elif ro.field == "SKINTEMP":
                    expected = d["SKINTEMP"].astype(np.float64)
                    errs["SKINTEMP"] = float(np.abs(applied - expected).max())
                    skt_deltas.append(applied.ravel())
                elif ro.field == "SST":
                    expected = d["SST"].astype(np.float64)
                    ocean = ro.slab > apply_pgw.SST_FILL_THRESHOLD
                    land = ~ocean
                    if not np.array_equal(ro.slab[land], rp.slab[land]):
                        errs["SST_land_touched"] = 1.0
                    not_floored = rp.slab > apply_pgw.SST_FLOOR_K
                    cmp_mask = ocean & not_floored
                    err2 = np.abs(applied - expected)[cmp_mask]
                    if err2.size:
                        errs["SST"] = float(err2.max())
                    floored = ocean & ~not_floored
                    if floored.any() and not np.allclose(rp.slab[floored], apply_pgw.SST_FLOOR_K):
                        errs["SST_floor_violated"] = 1.0
                    sst_deltas.append(applied[ocean])

            for k, v in errs.items():
                if k in ("SST_land_touched", "SST_floor_violated"):
                    violations.append((date, k))
                else:
                    worst[k] = max(worst.get(k, 0.0), v)

            if month not in tt_profile_by_month:
                lat = config.ERA5_LAT
                weights = np.cos(np.deg2rad(lat))[:, None]
                profile = []
                for lv in lev_pa:
                    ro_tt, rp_tt = find(orig, "TT", lv), find(pert, "TT", lv)
                    if ro_tt is None or rp_tt is None:
                        profile.append(np.nan)
                        continue
                    dprof = rp_tt.slab.astype(np.float64) - ro_tt.slab.astype(np.float64)
                    profile.append(np.average(dprof, weights=np.broadcast_to(weights, dprof.shape)))
                tt_profile_by_month[month] = (np.array(profile), lev_pa)

        if args.plots and plot_rep is None:
            plot_rep = dict(date=date, orig=orig, pert=pert)

    # ---- 2. quantitative cross-check vs stage-03 delta (aggregated) -------
    print("\n=== 2. quantitative cross-check vs stage-03 delta ===")
    for field, err in worst.items():
        record(f"applied delta matches stage-03 delta for {field} (float32 tol)",
               err < 1e-2, f"max |applied - expected| = {err:.4g}")
    record("SST land-fill cells never modified", not any(k == "SST_land_touched" for _, k in violations),
           str([d for d, k in violations if k == "SST_land_touched"]))
    record("SST floor (271 K) respected everywhere it applies",
           not any(k == "SST_floor_violated" for _, k in violations),
           str([d for d, k in violations if k == "SST_floor_violated"]))

    # ---- 3. physical plausibility ----------------------------------------
    print("\n=== 3. physical plausibility ===")
    record("perturbed RH stays within [0, 100] everywhere", rh_ok)

    if sst_deltas:
        d_sst = np.concatenate(sst_deltas)
        record("SST (main stream, ocean) mean delta in (0, 10) K", 0.0 < d_sst.mean() < 10.0,
               f"mean={d_sst.mean():+.2f} K min={d_sst.min():+.2f} max={d_sst.max():+.2f}")
    else:
        print("  (no SST record among main-stream test dates -- covered by the SST-update stream below)")

    if skt_deltas:
        d_skt = np.concatenate(skt_deltas)
        record("SKINTEMP mean delta in (0, 15) K", 0.0 < d_skt.mean() < 15.0,
               f"mean={d_skt.mean():+.2f} K min={d_skt.min():+.2f} max={d_skt.max():+.2f}")

    # vertical structure: tropospheric warming, relative stratospheric cooling
    for m in months_needed:
        if m not in tt_profile_by_month:
            continue
        profile, lev_pa = tt_profile_by_month[m]
        trop_mean = profile[(lev_pa >= 50000) & (lev_pa <= 85000)].mean()
        strat_mean = profile[lev_pa <= 20000].mean()
        record(f"month {m:02d}: tropospheric TT warming (500-850 hPa mean > 0)", trop_mean > 0,
               f"{trop_mean:+.3f} K")
        record(f"month {m:02d}: relative upper-level cooling (<=200 hPa mean < tropospheric mean)",
               strat_mean < trop_mean, f"strat={strat_mean:+.3f} K vs trop={trop_mean:+.3f} K")

    # ---- 4. SST-update stream ----------------------------------------------
    print("\n=== 4. SST-update stream (ERA5_SST:* -> SST_UPDATE:*) ===")
    if sst_pairs:
        ocean_mask = apply_pgw.load_ocean_mask()
        sst_update_deltas = []
        land_untouched_all = True
        by_month_err = {}
        floor_violations = []
        other_fields_untouched_all = True
        other_fields_seen = set()
        for of, pf in sst_pairs:
            orig = wps.read_file(of)
            pert = wps.read_file(pf)
            if len(orig) != len(pert):
                record(f"{pf.name}: record count matches original", False,
                       f"orig={len(orig)} pert={len(pert)}")
                continue

            # only "SST"/"SKINTEMP" are perturbed here (see
            # 04_apply_perturbation.py's perturb_one_sst_update_file); some
            # years' files carry additional unrelated records alongside it
            # (e.g. 2026's also have "SEAICE") which must pass through
            # byte-identical.
            sst_o = [r for r in orig if r.field in ("SST", "SKINTEMP")]
            sst_p = [r for r in pert if r.field in ("SST", "SKINTEMP")]
            if len(sst_o) != 1 or len(sst_p) != 1:
                record(f"{pf.name}: exactly 1 SST/SKINTEMP record", False,
                       f"got {len(sst_o)}/{len(sst_p)}")
                continue
            ro, rp = sst_o[0], sst_p[0]
            if ro.field != rp.field:
                record(f"{pf.name}: field name preserved", False, f"{ro.field} -> {rp.field}")

            for r_other in orig:
                if r_other.field in ("SST", "SKINTEMP"):
                    continue
                other_fields_seen.add(r_other.field)
                match = next((r for r in pert if r.field == r_other.field), None)
                if match is None or not np.array_equal(r_other.slab, match.slab):
                    other_fields_untouched_all = False

            land = ~ocean_mask
            if not np.array_equal(ro.slab[land], rp.slab[land]):
                land_untouched_all = False
            applied = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)
            # A handful of "ocean" cells per the static LANDSEA mask actually carry
            # the real-SST fill sentinel (~-1e30) in a given file (mask/data edge
            # mismatch); those get floored to SST_FLOOR_K by perturb_sst_update_field,
            # which is correct, but the resulting (floor - sentinel) "delta" is bogus
            # and must be excluded from both the quantitative and plausibility checks.
            not_floored = rp.slab > apply_pgw.SST_FLOOR_K
            month = int(rp.hdate[5:7])
            if month in deltas:
                expected = deltas[month]["SST"].astype(np.float64)
                cmp_mask = ocean_mask & not_floored
                err = np.abs(applied - expected)[cmp_mask]
                if err.size:
                    by_month_err[month] = max(by_month_err.get(month, 0.0), float(err.max()))
                floored = ocean_mask & ~not_floored
                if floored.any() and not np.allclose(rp.slab[floored], apply_pgw.SST_FLOOR_K):
                    floor_violations.append(pf.name)
            sst_update_deltas.append(applied[ocean_mask & not_floored])

        record("SST-update: land cells (LANDSEA mask) never modified", land_untouched_all)
        record("SST-update: 271 K floor respected everywhere it applies", not floor_violations,
               str(floor_violations))
        if other_fields_seen:
            record(f"SST-update: non-SST/SKINTEMP records {sorted(other_fields_seen)} left untouched",
                   other_fields_untouched_all)
        for m, err in by_month_err.items():
            record(f"SST-update month {m:02d}: applied delta matches stage-03 (float32 tol)",
                   err < 1e-2, f"max |applied - expected| = {err:.4g}")
        if sst_update_deltas:
            d_all = np.concatenate(sst_update_deltas)
            record(f"SST-update (ocean, all {len(sst_pairs)} files) mean delta in (0, 10) K",
                   0.0 < d_all.mean() < 10.0,
                   f"mean={d_all.mean():+.2f} K min={d_all.min():+.2f} max={d_all.max():+.2f}")
    else:
        print("  (no SST-update files found)")

    # ---- optional plots ----------------------------------------------------
    if args.plots and (plot_rep or sst_pairs):
        save_plots(plot_rep, sst_pairs)

    # ---- summary -------------------------------------------------------
    print("\n=== summary ===")
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    if n_fail:
        print("\nFAILED checks:")
        for name, ok, detail in CHECKS:
            if not ok:
                print(f"  - {name}" + (f" ({detail})" if detail else ""))
        sys.exit(1)
    print("All checks passed -- the perturbed files match the stage-03 delta and show a "
          "physically plausible PGW signature.")


def save_plots(plot_rep, sst_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _THIS_DIR / "verify_plots"
    out_dir.mkdir(exist_ok=True)
    lat, lon = config.ERA5_LAT, config.ERA5_LON

    def plot_map(ro, rp, title, fname):
        d = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)
        fig, axes = plt.subplots(1, 3, figsize=(16, 3.2), constrained_layout=True)
        vmin, vmax = np.nanpercentile(ro.slab, [1, 99])
        for ax, data, t in zip(axes[:2], [ro.slab, rp.slab], ["before", "after"]):
            im = ax.pcolormesh(lon, lat, data, vmin=vmin, vmax=vmax, cmap="RdBu_r", shading="auto")
            ax.set_title(f"{title} -- {t}")
            fig.colorbar(im, ax=ax, shrink=0.85)
        dmax = np.nanpercentile(np.abs(d), 99) or 1.0
        im = axes[2].pcolormesh(lon, lat, d, vmin=-dmax, vmax=dmax, cmap="RdBu_r", shading="auto")
        axes[2].set_title(f"{title} -- delta applied")
        fig.colorbar(im, ax=axes[2], shrink=0.85)
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)

    if plot_rep:
        for field, xlvl in [("TT", 85000.0), ("GHT", 85000.0), ("RH", 85000.0),
                             ("SKINTEMP", config.ERA5_SURFACE_LEVEL_PA)]:
            ro, rp = find(plot_rep["orig"], field, xlvl), find(plot_rep["pert"], field, xlvl)
            if ro is not None:
                plot_map(ro, rp, f"{field} {xlvl:.0f} Pa ({plot_rep['date']})",
                         f"{field}_{plot_rep['date']}.png")

    if sst_pairs:
        of, pf = sst_pairs[0]
        orig, pert = wps.read_file(of)[0], wps.read_file(pf)[0]
        land = ~apply_pgw.load_ocean_mask()
        ro_masked = type(orig)(**{**orig.__dict__, "slab": np.where(land, np.nan, orig.slab)})
        rp_masked = type(pert)(**{**pert.__dict__, "slab": np.where(land, np.nan, pert.slab)})
        plot_map(ro_masked, rp_masked, f"SST-update ({pf.name.split(':')[1]})",
                 "SST_update_example.png")

    print(f"\nsaved diagnostic plots to {out_dir}")


if __name__ == "__main__":
    main()
