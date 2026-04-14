#PBS -N MPproc
### Charging account
#PBS -A P93300042 
### Request one chunk of resources with N CPU and M GB of memory
#PBS -l select=1:ncpus=1:mem=200GB
###PBS -l select=1:ncpus=1:mem=300GB
### 
####PBS -l walltime=09:00:00
#PBS -l walltime=02:15:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe

module load conda

conda activate npl-2026a

#-------------------------------------
# The Python code called below is
# controlled by
#
#     config_ana.yaml
#
# Nothing to do here.
#--------------------------------------
echo "Cruising .... "

# remove logs that are more than 10 minutes old.
#-----------------------------------------------
find . -type f -name "MPproc.o*" -mmin +5 -exec rm {} \;

./regrid_MPAS3p75km.py
./recur.py --config config_mpas_ana.yaml --script PyBatch_MPASana.csh

