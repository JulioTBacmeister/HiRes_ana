#PBS -N HRproc
### Charging account
#PBS -A P93300042 
### Request one chunk of resources with N CPU and M GB of memory
#PBS -l select=1:ncpus=1:mem=64GB
### 
####PBS -l walltime=09:00:00
#PBS -l walltime=02:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe

module load conda

conda activate npl-2025b

#-------------------------------------
# The Python code called below is
# controlled by
#
#     config_ERA5regrid.yaml
#
# Nothing to do here.
#--------------------------------------
echo "Cruising .... "

# remove logs that are more than 15 minutes old.
#-----------------------------------------------
find . -type f -name "HRproc.o*" -mmin +15 -exec rm {} \;


./regrid_HRxLR.py

