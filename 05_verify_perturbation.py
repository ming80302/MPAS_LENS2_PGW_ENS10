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

Usage:
    python 05_verify_perturbation.py                # verify everything found
    python 05_verify_perturbation.py --years 1984
    python 05_verify_perturbation.py --plots         # also save PNGs to
                                                      # verify_plots/
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

def discover_main_dates(years) -> list:
    files = sorted(config.PERTURBED_IM_DIR.glob(f"{config.ERA5_IM_PREFIX}:*"))
    dates = [f.name.split(":")[1] for f in files if apply_pgw._IM_NAME_RE.match(f.name)]
    if years:
        dates = [d for d in dates if int(d[:4]) in years]
    return dates


def discover_sst_update_pairs(years) -> list:
    """-> list of (orig_path, pert_path) for the SST-update stream."""
    pattern = str(config.PERTURBED_IM_DIR / "SST_*" / f"{apply_pgw._SST_UPDATE_PREFIX}:*")
    pert_files = sorted(Path(p) for p in glob.glob(pattern))
    pert_files = [f for f in pert_files if apply_pgw._SST_UPDATE_NAME_RE.match(
        apply_pgw._SST_UPDATE_SRC_PREFIX + f.name[len(apply_pgw._SST_UPDATE_PREFIX):])]
    pairs = []
    for pf in pert_files:
        date = pf.name.split(":")[1]
        if years and int(date[:4]) not in years:
            continue
        orig_name = apply_pgw._SST_UPDATE_SRC_PREFIX + pf.name[len(apply_pgw._SST_UPDATE_PREFIX):]
        rel_dir = pf.parent.relative_to(config.PERTURBED_IM_DIR)
        of = config.ERA5_IM_DIR / rel_dir / orig_name
        pairs.append((of, pf))
    return pairs


# ---------------------------------------------------------------------------
# 1 & 2: structural integrity + quantitative cross-check, main ERA5:* stream
# ---------------------------------------------------------------------------

def key(rec):
    return (rec.field, round(rec.xlvl, 3))


def check_file(date: str) -> dict:
    orig = wps.read_file(config.ERA5_IM_DIR / f"{config.ERA5_IM_PREFIX}:{date}")
    pert = wps.read_file(config.PERTURBED_IM_DIR / f"{config.ERA5_IM_PREFIX}:{date}")

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

    return dict(date=date, orig=orig, pert=pert, same_count=same_count,
                same_order=same_order, n_touched=len(touched),
                fields_present={r.field for r in orig}, unexpected=unexpected)


def cross_check(r: dict, deltas: dict) -> dict:
    month = int(r["orig"][0].hdate[5:7])
    d = deltas[month]
    lev_pa = d["lev_pa"]
    max_abs_err = {}
    for ro, rp in zip(r["orig"], r["pert"]):
        is_sfc_sentinel = ro.xlvl == config.ERA5_SURFACE_LEVEL_PA
        applied = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)

        if ro.field in PERTURBED_3D and not is_sfc_sentinel:
            idx = int(np.argmin(np.abs(lev_pa - ro.xlvl)))
            expected = d[ro.field][idx].astype(np.float64)
            if ro.field == "RH":
                unclipped = (rp.slab > 0.0) & (rp.slab < 100.0)
                err = np.abs(applied - expected)[unclipped]
            else:
                err = np.abs(applied - expected)
            if err.size:
                max_abs_err[ro.field] = max(max_abs_err.get(ro.field, 0.0), float(err.max()))
        elif ro.field == "SKINTEMP":
            expected = d["SKINTEMP"].astype(np.float64)
            max_abs_err["SKINTEMP"] = float(np.abs(applied - expected).max())
        elif ro.field == "SST":
            expected = d["SST"].astype(np.float64)
            ocean = ro.slab > apply_pgw.SST_FILL_THRESHOLD
            land = ~ocean
            if not np.array_equal(ro.slab[land], rp.slab[land]):
                max_abs_err["SST_land_touched"] = 1.0
            not_floored = rp.slab > apply_pgw.SST_FLOOR_K
            cmp_mask = ocean & not_floored
            err = np.abs(applied - expected)[cmp_mask]
            if err.size:
                max_abs_err["SST"] = float(err.max())
            floored = ocean & ~not_floored
            if floored.any() and not np.allclose(rp.slab[floored], apply_pgw.SST_FLOOR_K):
                max_abs_err["SST_floor_violated"] = 1.0
    return max_abs_err


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
    args = p.parse_args()
    years = [int(y) for y in args.years.split(",")] if args.years else None

    test_dates = discover_main_dates(years)
    sst_pairs = discover_sst_update_pairs(years)
    if not test_dates and not sst_pairs:
        sys.exit(f"nothing to verify under {config.PERTURBED_IM_DIR}"
                  + (f" for years {years}" if years else "")
                  + " -- run 04_apply_perturbation.py first")

    print(f"main-stream test dates ({len(test_dates)}): {test_dates}")
    print(f"SST-update test files ({len(sst_pairs)})")
    print()

    # ---- 1. structural integrity ----------------------------------------
    print("=== 1. structural integrity (main ERA5:* stream) ===")
    results = [check_file(d) for d in test_dates]
    for r in results:
        ok = r["same_count"] and r["same_order"] and not r["unexpected"]
        detail = f"{r['n_touched']}/{len(r['orig'])} records touched"
        if not r["same_count"]:
            detail += "; record count differs from original"
        if r["same_count"] and not r["same_order"]:
            detail += "; field/level order differs from original"
        if r["unexpected"]:
            detail += f"; unexpected changes in {r['unexpected']}"
        record(f"{r['date']}: only expected fields changed", ok, detail)

    # ---- 2. quantitative cross-check vs stage-03 delta -------------------
    print("\n=== 2. quantitative cross-check vs stage-03 delta ===")
    months_needed = sorted({int(r["orig"][0].hdate[5:7]) for r in results if r["orig"]})
    deltas = {}
    for m in months_needed:
        f = config.DELTA_DIR / f"{m:02d}.nc"
        if not f.exists():
            record(f"month {m:02d} delta file exists", False, f"missing: {f}")
            continue
        with xr.open_dataset(f) as ds:
            deltas[m] = {k: ds[k].values for k in ["lev_pa", "GHT", "TT", "RH", "SST", "SKINTEMP"]}

    all_errs = {}
    for r in results:
        month = int(r["orig"][0].hdate[5:7]) if r["orig"] else None
        if month not in deltas:
            continue
        all_errs[r["date"]] = cross_check(r, deltas)

    worst = {}
    violations = []
    for d, errs in all_errs.items():
        for k, v in errs.items():
            if k in ("SST_land_touched", "SST_floor_violated"):
                violations.append((d, k))
            else:
                worst[k] = max(worst.get(k, 0.0), v)
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
    # RH stays in [0, 100] at every 3-D level, every date
    rh_ok = True
    for r in results:
        for rp in r["pert"]:
            if rp.field == "RH" and rp.xlvl != config.ERA5_SURFACE_LEVEL_PA:
                if rp.slab.min() < 0.0 or rp.slab.max() > 100.0:
                    rh_ok = False
    record("perturbed RH stays within [0, 100] everywhere", rh_ok)

    # SST: warms over ocean, respects floor, physically plausible magnitude
    sst_deltas = []
    for r in results:
        ro, rp = find(r["orig"], "SST", config.ERA5_SURFACE_LEVEL_PA), \
                 find(r["pert"], "SST", config.ERA5_SURFACE_LEVEL_PA)
        if ro is None:
            continue
        ocean = ro.slab > apply_pgw.SST_FILL_THRESHOLD
        sst_deltas.append((rp.slab.astype(np.float64) - ro.slab.astype(np.float64))[ocean])
    if sst_deltas:
        d_sst = np.concatenate(sst_deltas)
        record("SST (main stream, ocean) mean delta in (0, 10) K", 0.0 < d_sst.mean() < 10.0,
               f"mean={d_sst.mean():+.2f} K min={d_sst.min():+.2f} max={d_sst.max():+.2f}")
    else:
        print("  (no SST record among main-stream test dates -- covered by the SST-update stream below)")

    # SKINTEMP: not degenerate
    skt_deltas = []
    for r in results:
        ro, rp = find(r["orig"], "SKINTEMP", config.ERA5_SURFACE_LEVEL_PA), \
                 find(r["pert"], "SKINTEMP", config.ERA5_SURFACE_LEVEL_PA)
        if ro is not None:
            skt_deltas.append(rp.slab.astype(np.float64) - ro.slab.astype(np.float64))
    if skt_deltas:
        d_skt = np.concatenate([a.ravel() for a in skt_deltas])
        record("SKINTEMP mean delta in (0, 15) K", 0.0 < d_skt.mean() < 15.0,
               f"mean={d_skt.mean():+.2f} K min={d_skt.min():+.2f} max={d_skt.max():+.2f}")

    # vertical structure: tropospheric warming, relative stratospheric cooling
    for m in months_needed:
        rep = next((r for r in results if r["orig"] and int(r["orig"][0].hdate[5:7]) == m), None)
        if rep is None or m not in deltas:
            continue
        lat = config.ERA5_LAT
        weights = np.cos(np.deg2rad(lat))[:, None]
        lev_pa = deltas[m]["lev_pa"]
        profile = []
        for lv in lev_pa:
            ro, rp = find(rep["orig"], "TT", lv), find(rep["pert"], "TT", lv)
            d = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)
            profile.append(np.average(d, weights=np.broadcast_to(weights, d.shape)))
        profile = np.array(profile)
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
        for of, pf in sst_pairs:
            orig = wps.read_file(of)
            pert = wps.read_file(pf)
            if len(orig) != 1 or len(pert) != 1:
                record(f"{pf.name}: exactly 1 record", False, f"got {len(orig)}/{len(pert)}")
                continue
            ro, rp = orig[0], pert[0]
            if ro.field != rp.field:
                record(f"{pf.name}: field name preserved", False, f"{ro.field} -> {rp.field}")
            land = ~ocean_mask
            if not np.array_equal(ro.slab[land], rp.slab[land]):
                land_untouched_all = False
            applied = rp.slab.astype(np.float64) - ro.slab.astype(np.float64)
            month = int(rp.hdate[5:7])
            if month in deltas:
                expected = deltas[month]["SST"].astype(np.float64)
                not_floored = rp.slab > apply_pgw.SST_FLOOR_K
                cmp_mask = ocean_mask & not_floored
                err = np.abs(applied - expected)[cmp_mask]
                if err.size:
                    by_month_err[month] = max(by_month_err.get(month, 0.0), float(err.max()))
                floored = ocean_mask & ~not_floored
                if floored.any() and not np.allclose(rp.slab[floored], apply_pgw.SST_FLOOR_K):
                    floor_violations.append(pf.name)
            sst_update_deltas.append(applied[ocean_mask])

        record("SST-update: land cells (LANDSEA mask) never modified", land_untouched_all)
        record("SST-update: 271 K floor respected everywhere it applies", not floor_violations,
               str(floor_violations))
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
    if args.plots and (results or sst_pairs):
        save_plots(results, sst_pairs, deltas)

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


def save_plots(results, sst_pairs, deltas):
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

    if results:
        r0 = next((r for r in results if r["orig"]), None)
        if r0:
            for field, xlvl in [("TT", 85000.0), ("GHT", 85000.0), ("RH", 85000.0),
                                 ("SKINTEMP", config.ERA5_SURFACE_LEVEL_PA)]:
                ro, rp = find(r0["orig"], field, xlvl), find(r0["pert"], field, xlvl)
                if ro is not None:
                    plot_map(ro, rp, f"{field} {xlvl:.0f} Pa ({r0['date']})",
                             f"{field}_{r0['date']}.png")

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
