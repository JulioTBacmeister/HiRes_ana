#!/bin/bash

#PBS -N esmf
#PBS -A P93300042
#PBS -l walltime=00:59:00
#PBS -q main
#PBS -l job_priority=premium
#PBS -j oe
#PBS -l select=4:ncpus=128:mpiprocs=128

#module load cesmdev/1.0 craype/2,7.20 mkl/2023.0.0 cmake/3.26.3 hdf5-mpi/1.12.2 parallel-netcdf/1.12.3
#module load ncarenv/23.06 intel/2023.0.0 ncarcompilers/1.0.0 cray-mpich/8.1.25 netcdf-mpi/4.9.2 
#module load esmf/8.6.0b04-debug

module load cray-mpich/8.1.29 mkl/2024.2.2 hdf5/1.12.3 parallel-netcdf/1.14.0
module load ncarenv/24.12 intel/2024.2.1 ncarcompilers/1.0.0 cray-mpich/8.1.29 netcdf/4.9.2 
module load esmf/8.8.0

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/work/juliob/GridFiles/Scrip/ne16pg3_scrip_170429.nc -d /glade/work/juliob/GridFiles/Scrip/fv0.9x1.25_141008.nc -m conserve -w ne16pg3_TO_fv1x1_cnsrv.nc --64bit_offset

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/work/juliob/GridFiles/Scrip/ne16pg3_scrip_170429.nc -d /glade/work/juliob/GridFiles/Scrip/fv0.9x1.25_141008.nc -m bilinear -w ne16pg3_TO_fv1x1_bilin.nc --64bit_offset

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/fv0.9x1.25_141008.nc -m conserve -w mpasa3p75_TO_fv1x1_cnsrv.nc --64bit_offset


# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/work/juliob/GridFiles/Scrip/ne16pg3_scrip_170429.nc  -d /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc  -m bilinear -w ne16pg3_TO_mpasa3p75_bilin.nc --64bit_offset

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/ne16pg3_scrip_170429.nc -m conserve -w mpasa3p75_TO_ne16pg3_cnsrv_yoohoo.nc --64bit_offset

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SAndesAP_scrip.nc -m conserve -w mpasa3p75_TO_UHR_SAndesAP_cnsrv.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SAndesAP_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SAndesAP_bilin_testing.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SEUS_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SEUS_bilin_testing.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/ne120pg3_scrip_170628.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SEUS_scrip.nc -m bilinear -w ne120pg3_TO_UHR_SEUS_bilin_testing.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/ne120pg3_scrip_170628.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SO-East_scrip.nc -m bilinear -w ne120pg3_TO_UHR_SO-East_bilin_testing.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen  -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SO-East_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SO-East_bilin_testing.nc --64bit_offset --dst_regional

# Run the executable for one file
mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen  -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SO-Global_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SO-Global_bilin.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SO-Indian_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SO-Indian_bilin.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_SAndesAP_v0_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_SAndesAP_v0_bilin.nc --64bit_offset --dst_regional

# Run the executable for one file
#mpiexec -np 512 -ppn 128 /glade/u/apps/cseg/derecho/23.06/spack/opt/spack/linux-sles15-x86_64_v3/oneapi-2023.0.0/esmf-8.6.0b04-mkg7dasd7hipsqte2ibfflqzfe7cwgos/bin/ESMF_RegridWeightGen -s /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/mpasa3.75_SCRIP_desc-20210803.nc -d /glade/work/juliob/GridFiles/Scrip/latlon_UHR_global_scrip.nc -m bilinear -w mpasa3p75_TO_UHR_global_bilin.nc --64bit_offset


#mpiexec -np 512 -ppn 128 /glade/u/apps/derecho/24.12/spack/opt/spack/esmf/8.8.0/cray-mpich/8.1.29/oneapi/2024.2.1/ypx5/bin/ESMF_RegridWeightGen -s /glade/work/aherring/grids/SCRIP_files/ERA5native_scrip.nc -d /glade/campaign/cesm/cesmdata/inputdata/share/scripgrids/ne120np4_pentagons_100310.nc -m bilinear -w ERA5ml_TO_ne120np4_bilin.nc --64bit_offs
