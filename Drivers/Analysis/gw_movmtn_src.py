from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


# ============================================================
# Assumed utility routines corresponding to gw_utils
# ============================================================

def get_unit_vector(u: np.ndarray, v: np.ndarray):
    """
    Return unit-vector components (x, y) and magnitude.
    Safe for zero magnitude.
    """
    mag = np.sqrt(u**2 + v**2)
    x = np.zeros_like(u, dtype=np.float64)
    y = np.zeros_like(v, dtype=np.float64)

    mask = mag > 0.0
    x[mask] = u[mask] / mag[mask]
    y[mask] = v[mask] / mag[mask]

    return x, y, mag


def dot_2d(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray):
    """2D dot product row-by-row or elementwise broadcasted."""
    return u * x + v * y


def midpoint_interp(arr: np.ndarray):
    """
    Midpoint interpolation in vertical.
    If arr.shape = (ncol, pver), returns (ncol, pver-1)
    corresponding to averages between adjacent midpoint levels.
    """
    return 0.5 * (arr[:, :-1] + arr[:, 1:])


def index_of_nearest(values: np.ndarray, table: np.ndarray):
    """
    Return index of nearest table entry for each value.
    values can be 1D or 2D.
    """
    values = np.asarray(values)
    table = np.asarray(table)

    # broadcast to (..., ntab)
    diff = np.abs(values[..., None] - table[None, ...])
    return np.argmin(diff, axis=-1)


# ============================================================
# External configuration / lookup containers
# ============================================================

@dataclass
class Band:
    ngwv: int


@dataclass
class Desc:
    hd: np.ndarray       # heating-depth lookup axis
    min_hdepth: float
    uh: np.ndarray       # wind lookup axis
    mfcc: np.ndarray     # lookup table [hd, uh, phase_speed_bin]


@dataclass
class GwMovMtnConfig:
    source_type: int           # 1=vorticity, 2=shcu
    movmtn_klaunch: int = 0    # if >0 overrides launch level
    movmtn_ksteer: int = 0     # if >0 overrides steering level


# ============================================================
# Source helper routines
# ============================================================

def shcu_flux_src(
    xpwp_shcu: np.ndarray,
    alpha_gw_movmtn: float,
):
    """
    Python translation of shcu_flux_src

    Parameters
    ----------
    xpwp_shcu : ndarray, shape (ncol, pverx)
    alpha_gw_movmtn : float

    Returns
    -------
    xpwp_src : ndarray, shape (ncol,)
    steering_level : ndarray, shape (ncol,), int
    launch_level : ndarray, shape (ncol,), int
    """
    ncol, pverx = xpwp_shcu.shape

    # Fortran:
    # steering_level(:ncol) = (pverx - 1) - 5
    # launch_level(:ncol) = steering_level - 10
    #
    # In Python with 0-based indexing, the last valid index is pverx-1.
    steering_level = np.full(ncol, (pverx - 1) - 5, dtype=int)
    launch_level = steering_level - 10

    nlayers = 5

    # Fortran averaged xpwp_shcu(:, pverx-k) for k=0..nlayers-1
    # With Python 0-based indexing that becomes columns:
    #   (pverx-1), (pverx-2), ..., (pverx-nlayers)
    xpwp_src = alpha_gw_movmtn * np.mean(xpwp_shcu[:, -nlayers:], axis=1)

    return xpwp_src, steering_level, launch_level


def vorticity_flux_src(
    vorticity: np.ndarray,
    alpha_gw_movmtn: float,
):
    """
    Python translation of vorticity_flux_src

    Parameters
    ----------
    vorticity : ndarray, shape (ncol, pverx)
    alpha_gw_movmtn : float

    Returns
    -------
    vort_src : ndarray, shape (ncol,)
    steering_level : ndarray, shape (ncol,), int
    launch_level : ndarray, shape (ncol,), int
    """
    ncol, pverx = vorticity.shape

    steering_level = np.full(ncol, pverx - 20, dtype=int)
    launch_level = steering_level - 10

    scale_factor = 1.0e4
    nlayers = 10

    # Fortran:
    # vort_src += scale_factor * abs(vorticity(:, pverx-k)), k=0..nlayers-1
    vort_src = alpha_gw_movmtn * np.mean(
        scale_factor * np.abs(vorticity[:, -nlayers:]),
        axis=1,
    )

    return vort_src, steering_level, launch_level


# ============================================================
# Main routine
# ============================================================

def gw_movmtn_src(
    u: np.ndarray,
    v: np.ndarray,
    netdt: np.ndarray,
    xpwp_shcu: np.ndarray,
    vorticity: np.ndarray,
    zm: np.ndarray,
    alpha_gw_movmtn: float,
    use_gw_movmtn_pbl: bool,
    rair: float,
    gravit: float,
    band: Band,
    desc: Desc,
    config: GwMovMtnConfig,
):
    """
    Pythonic translation of gw_movmtn_src.

    Expected shapes
    ---------------
    u, v, netdt, vorticity, zm : (ncol, pver)
    xpwp_shcu                  : (ncol, pver+1) or similar interface array

    Returns
    -------
    result : dict of outputs
    """
    # ------------------------------------------------------------
    # Basic sizes / checks
    # ------------------------------------------------------------
    ncol, pver = u.shape
    ngwv = band.ngwv

    # phase-speed axis in Python:
    # Fortran used -ngwv:ngwv
    phase_speeds = np.arange(-ngwv, ngwv + 1, dtype=int)
    nphase = phase_speeds.size

    # ------------------------------------------------------------
    # Initialize outputs
    # ------------------------------------------------------------
    tau = np.zeros((ncol, nphase, pver + 1), dtype=np.float64)
    hdepth = np.zeros(ncol, dtype=np.float64)

    src_level = np.zeros(ncol, dtype=int)
    tend_level = np.zeros(ncol, dtype=int)

    ubm = np.zeros((ncol, pver), dtype=np.float64)
    ubi = np.zeros((ncol, pver), dtype=np.float64)   # mirrors Fortran intent(out) shape
    xv = np.zeros(ncol, dtype=np.float64)
    yv = np.zeros(ncol, dtype=np.float64)
    c = np.zeros((ncol, nphase), dtype=np.float64)

    usteer = np.zeros(ncol, dtype=np.float64)
    vsteer = np.zeros(ncol, dtype=np.float64)
    steer_level = np.zeros(ncol, dtype=np.float64)
    CS = np.zeros(ncol, dtype=np.float64)
    xpwp_src = np.zeros(ncol, dtype=np.float64)

    errmsg = ""
    errflg = 0

    # ------------------------------------------------------------
    # Local work arrays
    # ------------------------------------------------------------
    uwavef = np.zeros((ncol, pver), dtype=np.float64)
    vwavef = np.zeros((ncol, pver), dtype=np.float64)
    cell_retro_speed = np.zeros(ncol, dtype=np.float64)

    q0 = np.zeros(ncol, dtype=np.float64)
    qj = np.zeros(ncol, dtype=np.float64)

    uh = np.zeros(ncol, dtype=np.float64)
    CS1 = np.zeros(ncol, dtype=np.float64)
    udiff = np.zeros(ncol, dtype=np.float64)
    vdiff = np.zeros(ncol, dtype=np.float64)
    ubmsrc = np.zeros(ncol, dtype=np.float64)
    ubisrc = np.zeros(ncol, dtype=np.float64)
    ut = np.zeros(ncol, dtype=np.float64)
    taumm = np.zeros(ncol, dtype=np.float64)

    CF = 20.0
    AL = 1.0e5

    # ------------------------------------------------------------
    # Determine source / steering / launch levels
    # ------------------------------------------------------------
    if config.source_type == 1:
        xpwp_src, steer_k, launch_k = vorticity_flux_src(
            vorticity=vorticity,
            alpha_gw_movmtn=alpha_gw_movmtn,
        )
    elif config.source_type == 2:
        xpwp_src, steer_k, launch_k = shcu_flux_src(
            xpwp_shcu=xpwp_shcu,
            alpha_gw_movmtn=alpha_gw_movmtn,
        )
    else:
        raise ValueError(f"Unsupported source_type={config.source_type}")

    # override levels if requested
    if config.movmtn_klaunch > 0:
        launch_k[:] = config.movmtn_klaunch
    if config.movmtn_ksteer > 0:
        steer_k[:] = config.movmtn_ksteer

    # ------------------------------------------------------------
    # Steering winds
    # ------------------------------------------------------------
    col_idx = np.arange(ncol)
    usteer[:] = u[col_idx, steer_k]
    vsteer[:] = v[col_idx, steer_k]
    steer_level[:] = steer_k.astype(np.float64)

    xv_steer, yv_steer, umag_steer = get_unit_vector(usteer, vsteer)

    # As written, this is always zero because min(speed, 0) = 0
    # Preserving Fortran logic exactly.
    cell_retro_speed[:] = np.minimum(np.sqrt(usteer**2 + vsteer**2), 0.0)

    usteer[:] = usteer - xv_steer * cell_retro_speed
    vsteer[:] = vsteer - yv_steer * cell_retro_speed

    # ------------------------------------------------------------
    # Heating-depth logic
    # ------------------------------------------------------------
    boti = np.zeros(ncol, dtype=int)
    topi = np.zeros(ncol, dtype=int)

    if use_gw_movmtn_pbl:
        # Fortran set boti = pver and topi = Launch_k in 1-based indexing.
        # In Python, bottom physical model level is pver-1.
        boti[:] = pver - 1
        topi[:] = launch_k
    else:
        # Scan downward from bottom toward top in physical level index:
        # Fortran: do k = pver, 1, -1
        # Python:  k = pver-1, ..., 0
        for k in range(pver - 1, -1, -1):
            mask_boti_unset = (boti == 0)

            # outside max heating range
            mask_hi = mask_boti_unset & (zm[:, k] >= 20000.0)
            boti[mask_hi] = k
            topi[mask_hi] = k

            # first positive heating
            mask_heat = mask_boti_unset & (zm[:, k] < 20000.0) & (netdt[:, k] > 0.0)
            boti[mask_heat] = k

            # now topi not yet set, but boti was found
            mask_topi_unset = (boti != 0) & (topi == 0)

            mask_top_hi = mask_topi_unset & (zm[:, k] >= 20000.0)
            topi[mask_top_hi] = k

            mask_top_end = mask_topi_unset & (zm[:, k] < 20000.0) & ~(netdt[:, k] > 0.0)
            topi[mask_top_end] = k

            if np.all(topi != 0):
                break

    # heating depth
    hdepth[:] = zm[col_idx, topi] - zm[col_idx, boti]

    hd_idx = index_of_nearest(hdepth, desc.hd)

    # Fortran convention: hd_idx=0 means invalid/shallow.
    # But Python nearest-index returns a valid array index 0..N-1.
    # So we need a separate validity mask.
    valid_hd = hdepth >= max(desc.min_hdepth, desc.hd[0])

    # ------------------------------------------------------------
    # Maximum heating rate inside [topi, boti]
    # ------------------------------------------------------------
    for k in range(np.min(topi), np.max(boti) + 1):
        mask = (k >= topi) & (k <= boti)
        q0[mask] = np.maximum(q0[mask], netdt[mask, k])

    q0 *= CF
    qj[:] = gravit / rair * q0

    # ------------------------------------------------------------
    # Cell speed diagnostics
    # ------------------------------------------------------------
    CS1[:] = np.sqrt(usteer**2 + vsteer**2)

    # Preserving the literal Fortran expression, though it is odd:
    # CS = CS1*xv_steer + CS1*yv_steer = CS1*(xv_steer+yv_steer)
    CS[:] = CS1 * xv_steer + CS1 * yv_steer

    # ------------------------------------------------------------
    # Wave-frame winds
    # ------------------------------------------------------------
    uwavef[:] = u - usteer[:, None]
    vwavef[:] = v - vsteer[:, None]

    udiff[:] = uwavef[col_idx, topi]
    vdiff[:] = vwavef[col_idx, topi]

    xv[:], yv[:], ubisrc[:] = get_unit_vector(udiff, vdiff)

    # projection onto wavevector direction
    ubm[:] = dot_2d(uwavef, vwavef, xv[:, None], yv[:, None])
    ubmsrc[:] = ubm[col_idx, topi]

    # force source-level on-crest wind positive
    sgn = np.sign(ubmsrc)
    sgn[sgn == 0.0] = 1.0

    ubm *= sgn[:, None]
    xv *= sgn
    yv *= sgn

    # interface wind projection
    ubi[:, 0] = ubm[:, 0]
    ubi[:, 1:] = midpoint_interp(ubm)

    # ------------------------------------------------------------
    # Lookup-table wind
    # ------------------------------------------------------------
    ut[:] = ubm[col_idx, topi]
    uh[:] = ut - CS

    # set phase speeds; only central bin explicitly set to zero, like Fortran
    # c already initialized to zero everywhere
    center_bin = ngwv  # because phase_speeds = [-ngwv ... 0 ... +ngwv]

    # ------------------------------------------------------------
    # Source spectrum / source tau
    # ------------------------------------------------------------
    for i in range(ncol):
        if not use_gw_movmtn_pbl:
            if valid_hd[i]:
                uhmm_idx = index_of_nearest(np.array([uh[i]]), desc.uh)[0]

                taumm[i] = abs(desc.mfcc[hd_idx[i], uhmm_idx, 0])
                taumm[i] = taumm[i] * qj[i] * qj[i] / AL / 1000.0

                # assign sign based on ground-based phase speed = CS
                taumm[i] = -np.sign(CS[i]) * taumm[i] if CS[i] != 0.0 else 0.0

                # Fortran logic:
                # c0(i,:) = CS(i)
                # c_idx(i,:) = index_of_nearest(c0(i,:), c(i,:))
                #
                # Since c(i,:) is all zeros here, "nearest" is ambiguous.
                # Preserving literal intent is not really meaningful unless c is
                # later populated with real phase speeds.
                #
                # For now, place flux in the central phase-speed bin.
                c_idx_i = center_bin

                k0 = topi[i]
                if 0 <= k0 < pver:
                    tau[i, c_idx_i, k0] = taumm[i]
                if 0 <= k0 + 1 < (pver + 1):
                    tau[i, c_idx_i, k0 + 1] = taumm[i]

        else:
            ksrc = topi[i] + 1
            if 0 <= ksrc < (pver + 1):
                tau[i, center_bin, ksrc] = xpwp_src[i]

    src_level[:] = topi
    tend_level[:] = topi

    return {
        "src_level": src_level,
        "tend_level": tend_level,
        "tau": tau,
        "ubm": ubm,
        "ubi": ubi,
        "xv": xv,
        "yv": yv,
        "c": c,
        "hdepth": hdepth,
        "usteer": usteer,
        "vsteer": vsteer,
        "CS": CS,
        "steer_level": steer_level,
        "xpwp_src": xpwp_src,
        "errmsg": errmsg,
        "errflg": errflg,
        # useful internals for debugging
        "topi": topi,
        "boti": boti,
        "q0": q0,
        "qj": qj,
        "uh": uh,
        "ubmsrc": ubmsrc,
    }