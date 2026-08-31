#!/bin/bash
# Submit the full LENS2 10-member-ensemble-mean -> ERA5 PGW perturbation
# pipeline with PBS dependency chaining. All stages are plain Python -- no
# NCL/Fortran module needed.
#
# Usage: ./run_pipeline.sh            # default: 12 calendar-month deltas
#        ./run_pipeline.sh --annual   # single annual-mean (non-seasonal)
#                                      # delta -- see README's "Annual-mean"
#                                      # section. Passed through to stages
#                                      # 01b-04 via $ANNUAL in jobs/*.pbs;
#                                      # stage 01a is unaffected (it always
#                                      # builds all 12 real calendar months
#                                      # per member, which 01b's --annual
#                                      # needs as input either way).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ANNUAL=0
for arg in "$@"; do
    case "$arg" in
        --annual) ANNUAL=1 ;;
        *) echo "usage: $0 [--annual]" >&2; exit 1 ;;
    esac
done
SUFFIX=""
[ "$ANNUAL" = 1 ] && SUFFIX="_annual"

JOB1A=$(qsub jobs/01a_member_climatology.pbs)
echo "01a per-member climatology (10-member array):  $JOB1A"

# afterok on an array job's own id (already has a trailing "[]") waits for
# every array subjob to finish successfully before 01b runs -- "afterokarray"
# is a Torque-ism, not a valid dependency type on this system's OpenPBS/PBS
# Pro (see `man qsub`'s depend= list: after/afterok/afternotok/afterany/before*).
JOB1B=$(qsub -v ANNUAL=${ANNUAL} -N "lens2pgw_ens10_01b${SUFFIX}" -W "depend=afterok:${JOB1A}" jobs/01b_ensemble_climatology.pbs)
echo "01b ensemble-mean climatology:                  $JOB1B"

JOB2=$(qsub -v ANNUAL=${ANNUAL} -N "lens2pgw_ens10_02_interp${SUFFIX}" -W "depend=afterok:${JOB1B}" jobs/02_interp.pbs)
echo "02 pressure-level interp + RH:                  $JOB2"

JOB3=$(qsub -v ANNUAL=${ANNUAL} -N "lens2pgw_ens10_03_delta${SUFFIX}" -W "depend=afterok:${JOB2}" jobs/03_delta.pbs)
echo "03 delta + regrid to ERA5 grid:                 $JOB3"

JOB4=$(qsub -v ANNUAL=${ANNUAL} -N "lens2pgw_ens10_04_perturb${SUFFIX}" -W "depend=afterok:${JOB3}" jobs/04_apply_perturbation.pbs)
echo "04 apply perturbation to ERA5:                  $JOB4"

echo
if [ "$ANNUAL" = 1 ]; then
    echo "Perturbed files will land under config.PERTURBED_IM_DIR_ANNUAL"
    echo "(single non-seasonal delta applied to every date), ready to point"
    echo "an MPAS init_atmosphere_model namelist's config_met_prefix /"
    echo "config_sfc_prefix at (originals under config.ERA5_IM_DIR untouched)."
else
    echo "Perturbed files will land under config.PERTURBED_IM_DIR, ready to"
    echo "point an MPAS init_atmosphere_model namelist's config_met_prefix /"
    echo "config_sfc_prefix at (originals under config.ERA5_IM_DIR untouched)."
fi
