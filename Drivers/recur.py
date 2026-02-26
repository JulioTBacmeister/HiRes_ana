#!/usr/bin/env python
# Import packages 
import os
import argparse
import subprocess as sp
import update_config as uc
from mpi4py import MPI

######################################################################
# This function is called by PyBatch_ERA5regrid.csh, and
# then may also resubmit PyBatch_ERA5regrid.csh after incrementing
# month and decrementing Resubmit in config_ERA5regrid file.
#####################################################################

def main():

    parser = argparse.ArgumentParser(
        description="Recursive resubmission driver for HR analysis"
    )
    parser.add_argument(
        "--config",
        dest="config_file_path",
        default="./config_ana.yaml",
        help="Path to YAML configuration file (default: ./config_ana.yaml)"
    )

    parser.add_argument(
        "--script",
        dest="shell_script",
        default="PyBatch_HRana.csh",
        help="Path to YAML configuration file (default: ./config_ana.yaml)"
    )

    args = parser.parse_args()
    config_file_path = args.config_file_path
 
    shell_script = args.shell_script  #"PyBatch_HRana.csh"

    config = uc.read_config_yaml( config_file_path )
    print( config , flush=True )
    
    #------------------------------
    if (config['StepBy'].lower() == 'day'):
        config = uc.increment_day( config ) #, NoLeapYear=True )
    if (config['StepBy'].lower() == 'month'):
        config = uc.increment_month( config ) #, NoLeapYear=True )
    if (config['StepBy'].lower() == 'hour'):
        config = uc.increment_hours( config , nhours=config['StepN']) #, NoLeapYear=True )

    config = uc.decrement_Resubmit( config )
    print( config , flush=True )
    uc.write_config_yaml(config_file_path, config)
   
    if ( (config['month']<=12) and (config['Resubmit']>=0) ):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if (rank == 0):
            print(f" Resubmitting me {shell_script} via qsub ", flush=True)
            
            sp.run(f"qsub {shell_script}", 
                   shell=True )
            print(f"PyBatch ... " , flush=True)
        
    
    

if __name__ == "__main__":
    main()
