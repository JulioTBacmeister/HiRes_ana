###########################
# Source functions for GW
##########################

import numpy as np

def pysr_cx17(tilt_zg_zs=None, tilt_zl=None, u_zs=None, v_zs=None, precl=None, A=1.0, B=1.0, C=1.0 ):
    """GW momentum flux source, PySR complexity-17 form.

    Inputs (SI):
        tilt_zg_zs : s^-2,  mass-weighted |zeta dV/dz|, surface -> steering level
        tilt_zl    : s^-2,  same quantity at the launch level
        u_zs, v_zs : m s^-1, wind components at the steering level
        precl      : m s^-1, large-scale precipitation rate
    Returns:
        tau : Pa

    NOTE: coefficients are calibrated at sigma=2 horizontal smoothing
    (~180 x 210 km at 50 deg). Applied to bare model columns they will be
    biased; refit A, B, C (or the c_ coefficients) at the target scale.
    """
    D_ref   = 3.0e-7    # s^-2
    U_ref   = 10.0      # m s^-1
    P_ref   = 3.0e-8    # m s^-1
    tau_ref = 4.68e-3   # Pa

    # Tuning factors, all 1 for now. These scale terms in tau**2,
    # so A=2 gives sqrt(2) x the flux where T_conv dominates.
    # A, B, C = 1.0, 1.0, 1.0

    U_s = np.sqrt(u_zs**2 + v_zs**2) / U_ref
    P_o = precl / P_ref
    D_g = tilt_zg_zs / D_ref
    D_l = tilt_zl / D_ref

    tau_ratio_sq = (A*P_o**2 + B*(U_s*D_g)**2 + C*D_l) / 2.357

    return tau_ref * np.sqrt(np.maximum(tau_ratio_sq, 0.0))