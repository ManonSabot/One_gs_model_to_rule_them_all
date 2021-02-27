# -*- coding: utf-8 -*-

"""

This file is part of the TractLSM model.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------


"""

__title__ = ""
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (15.10.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, fLWP, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import calc_trans


# ======================================================================

def gas_exchange(p, fw, photo='Farquhar', res='low', dynamic=True, inf_gb=False,
                 iter_max=40, threshold_conv=0.1):

    # initial state
    Cs = p.CO2  # Pa
    Tleaf = p.Tair  # deg C

    # hydraulics
    P, E = hydraulics(p, res=res, kmax=p.kmaxT)

    # initialise gs over A
    g0 = 1.e-9  # g0 ~ 0, removing it entirely introduces errors
    Cs_umol_mol = Cs * conv.MILI / p.Patm  # umol mol-1
    gsoA = g0 + p.g1T * fw / Cs_umol_mol

    # iter on the solution until it is stable enough
    iter = 0

    while True:

        An, Aj, Ac, Ci = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                             gs_over_A=gsoA)

        # stomatal conductance, with fwsoil effect
        gs = np.maximum(cst.zero, conv.GwvGc * gsoA * An)

        # calculate new trans, gw, gb, etc.
        trans, real_zero, gw, gb, __ = calc_trans(p, Tleaf, gs, inf_gb=inf_gb)
        new_Tleaf, __ = leaf_temperature(p, trans, Tleaf=Tleaf, inf_gb=inf_gb)
        Pleaf = P[bn.nanargmin(np.abs(E - trans))]

        # new Cs (in Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb +
                                                       gs * conv.GcvGw)
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))
        Cs_umol_mol = Cs * conv.MILI / p.Patm

        # update gs over A
        gsoA = g0 + p.g1T * fw / Cs_umol_mol

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter >= 2) and
            real_zero and (abs(Tleaf - new_Tleaf) <= threshold_conv) and not
            np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate on leaf temperature
        Tleaf = new_Tleaf
        iter += 1

    if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
        (An > 0.)) or np.isclose(Ci, 0., rtol=cst.zero, atol=cst.zero) or
        (Ci < 0.) or np.isclose(Ci, p.CO2, rtol=cst.zero, atol=cst.zero) or
        (Ci > p.CO2) or (real_zero is None) or (not real_zero) or
       any(np.isnan([An, Ci, trans, gs, Tleaf, Pleaf]))):
        An, Ci, trans, gs, gb, Tleaf, Pleaf = (9999.,) * 7

    return An, Aj, Ac, Ci, trans, gs, gb, new_Tleaf, Pleaf


def fLWP_stable(p, photo='Farquhar', res='low', inf_gb=False):

    # hydraulics
    P, __ = hydraulics(p, res=res, kmax=p.kmaxT)

    # directionality?
    if np.isclose(p.LWP_ini, p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA):
        down = True  # downstream from Ps

    else:
        fw = fLWP(p, p.LWP_ini)  # previous stress factor
        __, __, __, __, __, __, __, __, Pleaf = gas_exchange(p, fw, photo=photo,
                                                             res=res,
                                                             dynamic=False,
                                                             inf_gb=inf_gb,
                                                             iter_max=5)

        if abs(Pleaf - p.LWP_ini) < P[0] - P[1]:  # no need to look further

            return fw

        elif Pleaf - p.LWP_ini > 0.:
            P = np.flip(P[P >= p.LWP_ini])
            down = False  # upstream from the previous LWP

        else:
            P = P[P <= p.LWP_ini]
            down = True  # downstream from the previous LWP

    for Psi in P:  # now look for most stable LWP

        fw = fLWP(p, Psi)  # update stress factor
        __, __, __, __, __, __, __, __, Pleaf = gas_exchange(p, fw, photo=photo,
                                                             res=res,
                                                             dynamic=False,
                                                             inf_gb=inf_gb,
                                                             iter_max=5)

        if ((down and (Pleaf - Psi > P[1] - P[0])) or
            ((not down) and (Pleaf - Psi < P[0] - P[1]))):  # converges

            return fw

    return 0.


def Tuzet(p, photo='Farquhar', res='low', inf_gb=False):

    """
    Checks the energy balance by looking for convergence of the new leaf
    temperature with the leaf temperature predicted by the previous
    iteration. Then returns the corresponding An, E, Ci, etc.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    threshold_conv: float
        convergence threshold for the new leaf temperature to be in
        energy balance

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    trans_can: float
        transpiration rate of canopy [mmol m-2 s-1] across leaves

    gs_can: float
        stomatal conductance of canopy [mol m-2 s-1] across leaves

    An_can: float
        C assimilation rate of canopy [umol m-2 s-1] across leaves

    Ci_can: float
        average intercellular CO2 concentration of canopy [Pa] across
        leaves

    rublim_can: string
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise.

    """

    # stability pre-requisites
    fw = fLWP_stable(p, photo=photo, res=res, inf_gb=inf_gb)

    # run the energy balance sub-routine
    An, Aj, Ac, Ci, trans, gs, gb, Tleaf, Pleaf = gas_exchange(p, fw,
                                                               photo=photo,
                                                               res=res,
                                                               inf_gb=inf_gb)
    rublim = rubisco_limit(Aj, Ac)  # lim?

    if not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
        trans *= conv.MILI  # mmol.m-2.s-1

    return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf
