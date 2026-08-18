#!/bin/bash -l
# ======================================================================
#  JOB ARRAY  --  10 independent processes, each rendering 1/10th of
#                 the frames at the same time.
#
#  The "#PBS -J 0-9" line below is what makes this an array.  PBS starts
#  ten copies of this script; each copy sees a different value in the
#  variable PBS_ARRAY_INDEX (0,1,2,...,9) and uses it to work out which
#  slice of frames it owns.
#
#  Submit with:   mkdir -p logs && qsub render_array.pbs
#  Watch with:    qstat -u $USER -t          ( -t expands the subjobs )
#                 tail -f logs/render.0.log
#  Cancel with:   qdel '1234567[]'           ( quotes required )
#
#  All ten write into the SAME output directory.  That is fine: the
#  filenames are frame_NNNN.png with NNNN the global frame number, so
#  no two subjobs ever write the same file.
# ======================================================================
#PBS -N render_array
#PBS -A P93300042 
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=128GB
#PBS -l walltime=03:00:00
#PBS -J 0-9
#PBS -j oe
#PBS -o logs/render.^array_index^.log

set -e

# --- how the work is divided ------------------------------------------
#   247 frames / 10 subjobs = 25 each.
#   Subjob k renders frames [k*25, (k+1)*25).
#   Subjob 9 asks for [225:250); make_frames.py clamps 250 down to 247.
#
#   If you change "#PBS -J 0-9" you must change CHUNK to match:
#       CHUNK = ceil(total_frames / number_of_subjobs)
CHUNK=25

# Safety net: this variable only exists when PBS started us as an array.
if [ -z "$PBS_ARRAY_INDEX" ]; then
    echo "ERROR: PBS_ARRAY_INDEX is not set."
    echo "This script must be submitted as an array (the '#PBS -J' line)."
    echo "If you meant to run a single job, submit render_one.pbs instead."
    exit 1
fi

IDX=$PBS_ARRAY_INDEX
I0=$(( IDX * CHUNK ))
I1=$(( (IDX + 1) * CHUNK ))

cd $PBS_O_WORKDIR
source ./frame_opts.sh

# --- environment -------------------------------------------------------
# Batch jobs do NOT inherit your login environment, so everything must be
# loaded explicitly here.  'conda' lives inside the 'ncarenv' module, so
# ncarenv must be loaded first or Lmod reports that conda "exists but
# cannot be loaded".
module purge
module load ncarenv
module load conda
conda activate $CONDA_ENV

# Diagnostics: if this job ever fails again, these three lines tell you
# immediately whether the environment or the Python code is at fault.
echo "--- modules loaded ---"
module list 2>&1
echo "--- python in use ---"
which python
python -c "import xarray, matplotlib; print('imports OK')"
echo "----------------------"

echo "=============================================="
echo " JOB ARRAY subjob $IDX of 0-9"
echo " host      : $(hostname)"
echo " frames    : [$I0:$I1)"
echo " outdir    : $OUTDIR"
echo "=============================================="

python make_frames.py \
    --i0 $I0 \
    --i1 $I1 \
    --outdir $OUTDIR \
    --skip-existing \
    --cbar-label "$CBAR_LABEL" \
    $FRAME_OPTS

echo "subjob $IDX finished"
