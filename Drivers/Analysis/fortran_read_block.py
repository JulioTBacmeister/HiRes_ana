from scipy.io import FortranFile

fortN = "/glade/work/juliob/GW_UnitTest/cases/xy-testfrontg/fort.111"

with FortranFile(fortN, "r") as f:
    nx, ny, nz = f.read_ints(np.int32)
    th = f.read_reals(np.float64).reshape((nx, ny, nz), order="F")
    u  = f.read_reals(np.float64).reshape((nx, ny, nz), order="F")
    v  = f.read_reals(np.float64).reshape((nx, ny, nz), order="F")

print(nx, ny, nz, th.flat[0:20])
