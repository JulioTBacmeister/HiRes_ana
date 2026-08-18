#!/bin/bash -l
#PBS -N render_frames
#PBS -A P0000000
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=40GB
#PBS -l walltime=02:00:00
#PBS -J 0-9
#PBS -j oe
#PBS -o logs/render.^array_index^.log

# ----------------------------------------------------------------------
# Renders animation frames in 10 parallel chunks.
#
#   247 dates / 10 subjobs = 25 dates each.
#   Subjob k handles dates [k*25, (k+1)*25).
#   The last subjob overshoots (250 > 247); make_frames.py clamps it.
#
# Submit with:   mkdir -p logs && qsub render_frames.pbs
# Monitor with:  qstat -u $USER -t
# ----------------------------------------------------------------------

set -e

CHUNK=25
IDX=$PBS_ARRAY_INDEX

I0=$(( IDX * CHUNK ))
I1=$(( (IDX + 1) * CHUNK ))

# --- move to the directory the job was submitted from -----------------
cd $PBS_O_WORKDIR

# --- activate your python environment ---------------------------------
# Replace 'npl' with whatever conda environment has your xarray stack.
module load conda
conda activate npl

echo "subjob $IDX rendering dates [$I0:$I1) on $(hostname)"

python make_frames.py \
    --i0 $I0 \
    --i1 $I1 \
    --outdir frames_w \
    --field w_mpas_prt \
    --lev-index 3 \
    --skip-existing

echo "subjob $IDX finished"
