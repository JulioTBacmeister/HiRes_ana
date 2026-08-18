#!/bin/bash -l
# ======================================================================
#  SINGLE JOB  --  one process renders every frame, one after another.
#
#  There is NO "#PBS -J" line below.  That absence is the only thing
#  that makes this a single job rather than a job array.
#
#  Submit with:   mkdir -p logs && qsub render_one.pbs
#  Watch with:    qstat -u $USER
#                 tail -f logs/render_one.log
#
#  Use this for: test runs, small numbers of frames, or when you just
#  want the simplest possible thing that works.  247 frames in one
#  process will take a while -- hence the long walltime.
# ======================================================================
#PBS -N render_one
#PBS -A P93300042 
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=128GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/render_one.log

set -e

# --- which frames to render -------------------------------------------
# Whole sequence:
#I0=0
#I1=247
# For a quick test instead, comment out the two lines above and use:
I0=20
I1=23

cd $PBS_O_WORKDIR
source ./frame_opts.sh

module load conda
conda activate $CONDA_ENV

echo "=============================================="
echo " SINGLE JOB (not an array)"
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

echo "finished"
