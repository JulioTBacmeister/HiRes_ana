# HiRes_ana
Software to analyze hi resolution runs, ie., 1/8 degree. Emphasis on GW.

################################
Structure:
        root
            |- Drivers
                      |
                      |-- Regridder
                      |
                      |-- Utils
     

Key codes are in ./Drivers.  Main calculation driven in regrid_HRxLR.py.

Currently this code reads in "1/8 degree" (ne240pg3) and
   1) Calculates momentum fluxes 'upwp' and 'vpwp' using regridding to
      ne16pg3 to define background. and regrids resul to fv1x1 lat-lon
      grid
   2) Regrids U,V,T,OMEGA and PS from ne240pg3 to fv1x1

Code reads a YAML file called control_ana.yaml. A template is provided
in ./Drivers. To run the code, control_ana.yaml must exist in ./Drivers. 

A notebook is provided in ./Drivers to run  regrid_HRxLR.py interactively:
TestDriver.ipynb.

A C-shell batch script PyBatch_HRana.csh is also provided. 