"""
Central configuration for the LENS2 10-member-ensemble-mean -> MPAS PGW
*perturbation* pipeline.

Adapted directly from /glade/u/home/mingge/MPAS/MPAS_LENS2_PGW/config.py
(the single-member, "member 2" version of this same delta/PGW design --
see that pipeline's README for the full method writeup, which applies here
unchanged except for how the climatology is built). The only conceptual
difference from that pipeline: the current/future climatology at each
stage is the *ensemble mean* across all 10 LENS2 smbb members (equal
weight per member, i.e. each member's own 30-year climatology is computed
first and the 10 resulting climatologies are then averaged -- not a pooled
average over all 300 member-years), rather than a single member's own
climatology. Everything downstream of the climatology (vertical
interpolation, delta, regridding, applying to ERA5) is identical to the
single-member pipeline and reuses its stages 02-04 unchanged.

This is deliberately a separate pipeline directory, not a modification of
MPAS_LENS2_PGW in place -- that pipeline's single-member (1031.002) run
stays independently reproducible.

Different task from /glade/u/home/mingge/LENS2/MPAS_PGW (which also
ensemble-means across these same 10 members, but drives MPAS *directly*
with the ensemble-mean climatology -- "PGW_C"/"PGW_F" -- with no delta ever
added to real ERA5 files). This pipeline instead computes the 10-member
ensemble-mean warming signal and adds it directly onto real ERA5
WPS-intermediate-format files, exactly like MPAS_LENS2_PGW does for its
single member.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Ensemble members -- the 10-member LENS2 "smbb" set, same list (same order)
# as /glade/u/home/mingge/LENS2/run_get_cesm2_data.csh and
# /glade/u/home/mingge/LENS2/MPAS_PGW/config.py's ENSEMBLE_MEMBERS, verified
# directly against the live archive (lib.catalog.available_members()) --
# all 10 have monthly T/Q/Z3/PS/TS and daily SST archived for both periods
# below.
# ---------------------------------------------------------------------------
MEMBERS = [
    ("1011", "001"), ("1031", "002"), ("1051", "003"), ("1071", "004"),
    ("1091", "005"), ("1111", "006"), ("1131", "007"), ("1151", "008"),
    ("1171", "009"), ("1191", "010"),
]

# ---------------------------------------------------------------------------
# Climatology periods (delta = future minus current)
# ---------------------------------------------------------------------------
PERIODS = {
    "current": (1970, 1999),
    "future": (2066, 2095),
}
# Both periods fall entirely within one CESM2-LE experiment each (no
# mid-period BHISTsmbb/BSSP370smbb split needed).
HIST_CASE = "b.e21.BHISTsmbb.f09_g17.LE2"
SSP_CASE = "b.e21.BSSP370smbb.f09_g17.LE2"
HIST_SSP_SPLIT_YEAR = 2015


def case_for_year(year: int) -> str:
    return HIST_CASE if year < HIST_SSP_SPLIT_YEAR else SSP_CASE


MONTHS = list(range(1, 13))  # perturbation varies by calendar month

# ---------------------------------------------------------------------------
# Perturbation variables -> LENS2 monthly source variable(s) needed
# ---------------------------------------------------------------------------
LENS2_VARS_3D = ["T", "Q"]          # Z3 also needed for GHT
LENS2_VAR_Z3 = "Z3"
LENS2_VAR_PS = "PS"                 # needed to interpolate hybrid -> pressure levels
LENS2_VAR_SST = "SST"               # ocn, POP grid, monthly-mean of daily
LENS2_VAR_TS = "TS"                 # atm, CAM grid -> SKINTEMP

ALL_LENS2_VARS = LENS2_VARS_3D + [LENS2_VAR_Z3, LENS2_VAR_PS, LENS2_VAR_SST, LENS2_VAR_TS]

COMPONENT_OF = {
    "T": "atm", "Q": "atm", "Z3": "atm", "PS": "atm", "TS": "atm",
    "SST": "ocn",
}
FREQ_OF = {  # all monthly; SST is only archived daily, so monthly-mean it ourselves
    "T": "month_1", "Q": "month_1", "Z3": "month_1", "PS": "month_1", "TS": "month_1",
    "SST": "day_1",
}

# ---------------------------------------------------------------------------
# Source data locations
# ---------------------------------------------------------------------------
LENS2_ROOT = Path("/glade/campaign/cgd/cesm/CESM2-LE")
LENS2_TSERIES = LENS2_ROOT / "{component}" / "proc" / "tseries" / "{freq}"

# a representative monthly T file, purely to read static hybrid coefficients
# -- these are a fixed property of the model's vertical grid, identical
# across every member and every year, so any one member's file works.
_HYBRID_COEFF_ENS_MEM, _HYBRID_COEFF_VAR_ID = MEMBERS[0]
HYBRID_COEFF_FILE = (
    LENS2_ROOT / "atm/proc/tseries/month_1/T"
    / f"{HIST_CASE}-{_HYBRID_COEFF_ENS_MEM}.{_HYBRID_COEFF_VAR_ID}.cam.h0.T.192001-192912.nc"
)

# static surface geopotential (needed for vinth2p_ecmwf's extrapolation
# below the lowest hybrid level) -- same invariant file
# /glade/u/home/mingge/LENS2/MPAS_PGW/config.py uses, same LENS2 grid
# (f09_g17) so it applies here unchanged, and is member-independent.
ZSFC_FILE = Path(
    "/glade/campaign/mmm/c3we/mingge/CESM2_LE/Invariant_Data_CESM2/"
    "USGS-gtopo30_0.9x1.25_remap_c051027.nc"
)

# CAM 0.9x1.25 grid <-> gx1v6 POP grid regrid weights (member-independent).
POP_TO_CAM_WEIGHTS = Path(
    "/glade/campaign/mmm/c3we/mingge/CESM2_LE/Invariant_Data_CESM2/"
    "map_gx1v6_to_fv0.9x1.25_bilin_da_090309.nc"
)

# ---------------------------------------------------------------------------
# ERA5 WPS intermediate files this pipeline perturbs
# ---------------------------------------------------------------------------
ERA5_IM_DIR = Path("/glade/campaign/mmm/c3we/mingge/wpsv4.5_mpas")
ERA5_IM_PREFIX = "ERA5"

# Standard ERA5 37 pressure levels, in Pa, ascending -- verified against a
# real file (rd_intermediate.exe output), not assumed.
ERA5_PRESSURE_LEVELS_PA = [
    100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000, 12500,
    15000, 17500, 20000, 22500, 25000, 30000, 35000, 40000, 45000, 50000,
    55000, 60000, 65000, 70000, 75000, 77500, 80000, 82500, 85000, 87500,
    90000, 92500, 95000, 97500, 100000,
]
ERA5_SURFACE_LEVEL_PA = 200100  # WPS sentinel for "not a pressure level"

# ERA5 grid definition, read directly off a real file via rd_intermediate.exe
# (REF_LAT=90, REF_LON=0, DLAT=-0.25, DLON=0.25, I,J DIMS=1440,721).
import numpy as _np  # noqa: E402
ERA5_NY, ERA5_NX = 721, 1440
ERA5_LAT = 90.0 + _np.arange(ERA5_NY) * (-0.25)     # descending, 90 -> -90
ERA5_LON = 0.0 + _np.arange(ERA5_NX) * 0.25          # ascending, 0 -> 359.75

PERTURB_FIELDS_3D = ["GHT", "TT", "RH"]
PERTURB_FIELDS_SFC = {"SST": "SST", "SKINTEMP": "TS"}  # IM field -> LENS2 var

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
SCRATCH_ROOT = Path("/glade/derecho/scratch/mingge/MPAS_LENS2_PGW_ENS10")
MEMBER_RAW_DIR = SCRATCH_ROOT / "01a_member_raw"  # per-member, per-variable monthly climatology
RAW_DIR = SCRATCH_ROOT / "01b_ensemble_raw"       # 10-member equal-weight mean climatology
INTERP_DIR = SCRATCH_ROOT / "02_interp"           # on ERA5 pressure levels + RH, per period/month
DELTA_DIR = SCRATCH_ROOT / "03_delta"             # future-current, regridded to the ERA5 grid
PERTURBED_IM_DIR = Path(
    "/glade/campaign/mmm/c3we/mingge/wpsv4.5_mpas_LENS2PGW_ens10mean"
)  # perturbed copies of the ERA5 IM files, written next to (not over) the originals
# Separate output dir for the --annual (climate-mean, non-seasonal) delta
# (04_apply_perturbation.py --annual) so its output can never collide with /
# be silently skipped-as-already-existing against the default per-month run's
# files, since both would otherwise produce identically-named ERA5:* files.
# Same convention as MPAS_LENS2_PGW/config.py's PERTURBED_IM_DIR_ANNUAL.
PERTURBED_IM_DIR_ANNUAL = Path(str(PERTURBED_IM_DIR) + "_annualmean")


def ensure_dirs():
    for d in [MEMBER_RAW_DIR, RAW_DIR, INTERP_DIR, DELTA_DIR, PERTURBED_IM_DIR,
              PERTURBED_IM_DIR_ANNUAL]:
        d.mkdir(parents=True, exist_ok=True)
