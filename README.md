# LENS2 10-member-ensemble-mean -> ERA5 PGW perturbation pipeline

[![GitHub](https://img.shields.io/badge/GitHub-ming80302%2FMPAS__LENS2__PGW__ENS10-blue?logo=github)](https://github.com/ming80302/MPAS_LENS2_PGW_ENS10)

Computes a LENS2 (CESM2-LE) **10-member ensemble-mean** climate-change
signal and adds it directly onto real ERA5 WPS intermediate-format files
already used to drive MPAS -- the same classic delta/"pseudo-global-warming"
perturbation design as
[`/glade/u/home/mingge/MPAS/MPAS_LENS2_PGW`](../MPAS_LENS2_PGW/README.md)
(single LENS2 member, "1031.002"), generalized to average across all 10
members. That pipeline's stages 02-04 (vertical interpolation, delta +
regrid, apply-to-ERA5) are reused here completely unchanged -- only the
climatology stage differs, so see that README for the full method writeup
(vertical interpolation formulas, WPS intermediate format handling, field
matching in stage 04, SST-update-stream handling, etc.); this README only
covers what's different.

Also a different task from
[`/glade/u/home/mingge/LENS2/MPAS_PGW`](../../LENS2/MPAS_PGW) (which
ensemble-means across these same 10 members, but drives MPAS *directly*
with the ensemble-mean climatology -- "PGW_C"/"PGW_F" -- with no delta ever
added to a real ERA5 file).

## What's different from the single-member pipeline

Only the climatology step. Everything from the delta onward (stage 03 in
both pipelines) is identical code, just fed a different climatology.

| Stage | Script | What it does |
|---|---|---|
| 01a | `01a_member_climatology.py` | Per-member LENS2 monthly climatology (current + future), one call per member in `config.MEMBERS` (10 total) -- otherwise identical to the single-member pipeline's stage 01. |
| 01b | `01b_ensemble_climatology.py` | Averages the 10 members' climatologies together, **equal weight per member** (not pooled over all 300 member-years) -- one ensemble-mean climatology per (period, var, month), written in the exact layout the single-member pipeline's stage 01 produces, so stages 02-04 need zero changes for the default per-month case. Also builds `--annual` (all 12 months) and `--label` (a custom month subset, e.g. Jun-Oct) seasonal means -- see "Seasonal-mean (--label) option" below. |
| 02 | `02_interp_pressure_rh.py` | Same vertical-interpolation/RH logic as the single-member pipeline; gained a `--label` option (alongside the existing `--annual`) to interpolate a custom seasonal-mean climatology instead of the 12 per-month ones. |
| 03 | `03_compute_delta_regrid.py` | Computes `future - current` from the two ensemble-mean climatologies (01b's output), i.e. **delta of ensemble-mean climatologies**, not a mean of 10 per-member deltas. Mathematically these are equal only for linear operations; regridding (stage 03) and pressure-level interpolation (stage 02) are *not* linear in general, so this pipeline's delta is not exactly the ensemble-mean of what you'd get running MPAS_LENS2_PGW's own stage 02/03 once per member -- it's the delta computed after averaging at the raw-climatology stage, chosen deliberately (see design discussion below) rather than as an approximation. Gained `--label`, plus `--period-current`/`--period-future` to compute a delta between *any* two `config.PERIODS` entries, not just the default `current`/`future` pair. |
| 04 | `04_apply_perturbation.py` | Applies the ensemble-mean delta to the real ERA5 IM files. Gained `--label` and a `--delta-dir` override (to apply a delta computed with a non-default `--period-current`/`--period-future` pair). The SST-update stream (`ERA5_SST:*` -> `SST_UPDATE:*`) now only perturbs the record actually named `SST`/`SKINTEMP`, passing any other record in the same file (e.g. a `SEAICE` record present in some years' archives) through byte-identical -- earlier it perturbed every record in the file, which silently corrupted `SEAICE` wherever that extra record existed. |
| 05 | `05_verify_perturbation.py` | New. Verifies a stage-04 run against the real ERA5 originals and the stage-03 delta: structural integrity (only the perturbed fields changed), quantitative correctness (applied delta matches stage-03 exactly, modulo the documented RH clip/SST floor), physical plausibility (expected warming/cooling signature), and a separate check for the periodic SST-update stream. Supports `--annual`/`--label`/`--delta-dir`, matching stage 04. Processes the main `ERA5:*` stream one date at a time rather than holding every date's full-resolution records in memory at once -- each file is ~1 GB, so this matters once you're checking more than a couple of dates on a memory-constrained login node. |

### Ensemble-mean method: average-of-climatologies, not average-of-deltas

Two different things are both called "ensemble-mean PGW delta" in practice:

1. **Delta of ensemble-mean climatologies** (what this pipeline does):
   average the 10 members' current-period climatology together and the 10
   members' future-period climatology together, *then* take one
   `delta = mean(future) - mean(current)`.
2. **Mean of per-member deltas**: compute each member's own
   `future - current` delta first, then average the 10 resulting deltas.

These are mathematically identical only where every downstream operation
(vertical interpolation, regridding, the delta subtraction itself) is
linear -- which the delta subtraction is, but stage 02's vertical
interpolation and stage 03's regridding are not exactly (interpolation
weights depend on each field's own values via the level-bracketing/
extrapolation logic in `lib/vinterp.py`). Method 1 was chosen here to match
`/glade/u/home/mingge/LENS2/MPAS_PGW/02_ensemble_climatology.py`'s approach
(average climatologies, not deltas) and because it's the cheaper of the
two -- one climatology, one interpolation, one regrid per period, rather
than 10 of each.

### The 10-member list

Same 10-member LENS2 "smbb" set (same order) as
`/glade/u/home/mingge/LENS2/run_get_cesm2_data.csh` and
`/glade/u/home/mingge/LENS2/MPAS_PGW/config.py`'s `ENSEMBLE_MEMBERS`,
verified directly against the live archive
(`lib.catalog.available_members()` confirms all 10 have monthly
T/Q/Z3/PS/TS and daily SST archived for both periods below):

```
1011.001  1031.002  1051.003  1071.004  1091.005
1111.006  1131.007  1151.008  1171.009  1191.010
```

`config.HYBRID_COEFF_FILE` reads static hybrid-sigma coefficients off just
one of these (the first, `1011.001`) since those coefficients are a fixed
property of the model's vertical grid, identical across every member.

## Annual-mean ("climate mean") option

By default the delta varies by calendar month (12 climatologies, 12
deltas). Pass `--annual` instead for a single non-seasonal delta -- the
unweighted mean of the 12 monthly ensemble-mean climatologies -- applied
uniformly to every date regardless of its own calendar month. Same option,
same semantics, as
[`MPAS_LENS2_PGW`](../MPAS_LENS2_PGW/README.md#annual-mean-climate-mean-option);
this section only covers what's different because the climatology here is
split across two stages (01a per-member, 01b ensemble-mean) instead of one.

**Whole pipeline (PBS, recommended):**

```
./run_pipeline.sh --annual
```

This submits the same 01a -> 01b -> 02 -> 03 -> 04 dependency chain as a
plain `./run_pipeline.sh`, with `-v ANNUAL=1` passed to every stage's qsub
call *except* 01a's -- stage 01a always builds all 12 real calendar
months' per-member climatologies unconditionally, since 01b's `--annual`
needs them as input either way (it builds the 12 ensemble-mean monthlies
first if missing, then averages those into the annual mean). Each of
01b/02/03/04's `jobs/*.pbs` scripts read `$ANNUAL` and append `--annual`
to their own `python` calls (and suffix their log filenames with `_annual`
so they don't collide with a prior month-specific run's logs). No stage
script needs editing or separate submission.

**Stage-by-stage (equivalent, if not using PBS):**

```
python 01a_member_climatology.py --period current   # unconditional, all 12
python 01a_member_climatology.py --period future     # months, all 10 members
python 01b_ensemble_climatology.py --period current --annual  # builds the 12
python 01b_ensemble_climatology.py --period future  --annual  # ensemble-mean
                                                                # monthlies if
                                                                # missing, then
                                                                # writes *_ann.nc
python 02_interp_pressure_rh.py --period current --annual
python 02_interp_pressure_rh.py --period future  --annual
python 03_compute_delta_regrid.py --annual        # writes DELTA_DIR/ann.nc
python 04_apply_perturbation.py --annual          # writes to
                                                   # config.PERTURBED_IM_DIR_ANNUAL
```

- **Definition**: `mean(clim_01, clim_02, ..., clim_12)` of the
  *ensemble-mean* monthly climatologies -- equal weight per calendar month.
  Averaging commutes here: mean-over-months-of(mean-over-members) equals
  mean-over-members-of(mean-over-months), since both are equal-weight
  arithmetic means -- so this is numerically identical to averaging each
  member's own annual mean across the 10 members. `01b_ensemble_climatology.py`'s
  `build_annual_mean` takes the cheaper path (average its own already-computed
  12 monthly ensemble-mean files) rather than re-touching the 10 members'
  per-member climatologies.
- **Requires** all 10 members' 12 monthly climatologies to already exist
  under `config.MEMBER_RAW_DIR` (stage 01a, unconditional -- run it
  normally first, `--annual` or not) and the 12 ensemble-mean monthly
  climatologies under `config.RAW_DIR` for stages 02/03 (01b's `--annual`
  builds these automatically if missing; stages 02/03 do not -- run them
  per-month first, or run 02 without `--annual` once).
- **Separate output directory**: stage 04's `--annual` output goes to
  `config.PERTURBED_IM_DIR_ANNUAL` (`..._ens10mean_annualmean`), not
  `config.PERTURBED_IM_DIR` -- both runs produce identically-named
  `ERA5:*` files, so sharing one directory would let the skip-if-exists /
  `--overwrite` logic silently mix monthly and annual-mean perturbations.
- Everything else (field matching, SST land-fill handling, the SST-update
  stream, byte-identical passthrough of unperturbed records) is unchanged
  from the default per-month run.

## Seasonal-mean (`--label`) option

Like `--annual` but for an arbitrary *subset* of calendar months instead of
all 12 -- e.g. a Jun-Oct ("JJASO") mean rather than a full-year mean. Same
idea as `--annual` (unweighted mean of the ensemble-mean monthly
climatologies, equal weight per month), just parameterized by `--months`
and a `--label` name for the output files instead of being hardcoded to
all 12 months / `"ann"`.

**Not wired into `./run_pipeline.sh`/PBS yet** -- `--annual` is, `--label`
is not, so run the stages by hand:

```
for p in current future; do
    python 01a_member_climatology.py --period $p --months 6,7,8,9,10   # unconditional
    python 01b_ensemble_climatology.py --period $p --months 6,7,8,9,10 --label JJASO
    python 02_interp_pressure_rh.py --period $p --label JJASO
done
python 03_compute_delta_regrid.py --label JJASO   # writes DELTA_DIR/JJASO.nc
python 04_apply_perturbation.py --label JJASO     # writes to
                                                   # config.PERTURBED_IM_DIR + "_JJASO"
python 05_verify_perturbation.py --label JJASO    # verify before trusting the output
```

- **`--months` is required together with `--label`** (unlike `--annual`,
  which always means all 12 months). `01b_ensemble_climatology.py --label`
  builds whichever monthly ensemble-mean climatologies it needs first if
  they're not already present (same auto-build behavior as `--annual`).
- **Cross-period delta**: `03_compute_delta_regrid.py` also accepts
  `--period-current`/`--period-future` to compute a delta between *any*
  two `config.PERIODS` entries, not just the default `current`/`future`
  pair (e.g. a 2011-2041-vs-current delta if you've added a third period
  to `config.PERIODS`). The default pair writes to flat `DELTA_DIR/` as
  before; any other pair writes to
  `DELTA_DIR/<period-future>_minus_<period-current>/` instead, so it can
  never collide with the default delta. To apply a delta from one of
  those subdirectories, pass the same path to
  `04_apply_perturbation.py --delta-dir ...` (and
  `05_verify_perturbation.py --delta-dir ...` to verify it) -- the output
  directory name gets the delta-dir's basename appended so it can't
  collide with a default-period-pair `--label` run either.
- **Separate output directories**, same reasoning as `--annual`: stage
  04's `--label JJASO` output goes to
  `config.PERTURBED_IM_DIR` + `"_JJASO"` (and further
  `+ "_<delta-dir-basename>"` if `--delta-dir` was also given), never the
  default or `--annual` directory.
- Everything else (field matching, SST land-fill handling, the SST-update
  stream, byte-identical passthrough of unperturbed records) is unchanged
  from the default per-month run.

## Output locations

- Per-member climatology: `config.MEMBER_RAW_DIR` under `config.SCRATCH_ROOT`
  (currently `/glade/campaign/mmm/c3we/mingge/MPAS_LENS2_PGW_ENS10`, not the
  derecho scratch filesystem despite the variable name -- it's large-ish,
  10x the single-member pipeline's raw-climatology size, and safe to
  delete once stage 01b has run).
- Ensemble-mean climatology: `config.RAW_DIR` (same layout as the
  single-member pipeline's `RAW_DIR`); `{var}_<label>.nc` alongside the
  `{var}_01.nc .. _12.nc` / `{var}_ann.nc` files when `--label` is used.
- Interpolated-onto-ERA5-levels climatology: `config.INTERP_DIR/<period>/<label>.nc`
  for a `--label` run (alongside the per-month/`ann.nc` files).
- Delta: `config.DELTA_DIR/<label>.nc` for the default `current`/`future`
  period pair; `config.DELTA_DIR/<period-future>_minus_<period-current>/<label>.nc`
  for any other `--period-current`/`--period-future` pair -- see
  "Seasonal-mean (`--label`) option" above.
- Perturbed ERA5 files:
  `/glade/campaign/mmm/c3we/mingge/wpsv4.5_mpas_LENS2PGW_ens10mean`
  (distinct from the single-member pipeline's
  `wpsv4.5_mpas_LENS2PGW_mem2`, so both can coexist and be compared).
  With `--annual`, this instead goes to
  `.../wpsv4.5_mpas_LENS2PGW_ens10mean_annualmean`
  (`config.PERTURBED_IM_DIR_ANNUAL`) -- see "Annual-mean" above. With
  `--label JJASO`, it goes to `..._ens10mean_JJASO` (further suffixed with
  the delta-dir's basename if `--delta-dir` was also given) -- see
  "Seasonal-mean (`--label`) option" above.

Run via `./run_pipeline.sh` (PBS dependency chain, stage 01a as a 10-task
job array, supports `--annual`) or stage-by-stage (required for `--label`,
which isn't wired into `run_pipeline.sh` yet); every stage skips outputs
that already exist unless `--overwrite` is passed. After stage 04, run
`05_verify_perturbation.py` (same `--annual`/`--label`/`--years` flags) to
confirm the perturbed files are structurally sound, quantitatively match
the stage-03 delta, and show a physically plausible signal before trusting
them downstream.
