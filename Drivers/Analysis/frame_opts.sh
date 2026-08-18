# ----------------------------------------------------------------------
# Shared plotting options for make_frames.py.
#
# This file is NOT run on its own.  Both render_one.pbs and
# render_array.pbs read it, so the two always render identical-looking
# frames.  Change a setting here and both pick it up.
#
# NOTE: no spaces inside any option value.  The list below gets split
# into words on spaces when it is used, so a value containing a space
# would be torn in half.  This is why the level lists are written as
# comma-separated strings with an '=' sign.
# ----------------------------------------------------------------------

FRAME_OPTS="
  --field w_mpas_prt
  --lev-index 2
  --linthresh 0.75
  --vmax 2
  --cbar-ticks=-2,-1,-0.5,-0.2,0,0.2,0.5,1,2
  --zeta
  --zeta-lev-index 0
  --zeta-levels=-2e-4,2e-4
  --zeta-color g
  --zeta-alpha 0.25
  --prec
  --prec-levels=3,25
  --prec-color gray
  --prec-alpha 0.2
  --dpi 300
  --wgtfile ../mpasa3p75_TO_UHR_Lat_0-80N_Lon_0-360_bilin.nc
"


# The colourbar label is kept OUT of FRAME_OPTS because it contains a
# space.  It is passed separately, in quotes, so the space survives.
CBAR_LABEL="w (m/s)"

# Where the PNGs go.
OUTDIR=frames_NH

# Conda environment holding xarray / matplotlib / netCDF4.
CONDA_ENV=npl
