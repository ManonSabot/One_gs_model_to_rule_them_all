# -*- coding: utf-8 -*-

"""
The Tuzet model, adapted for LSMs (in a similar manner to how MAESPA
does it), by iteration on the air temperature to get the leaf
temperature for which the Penman-Monteith energy balance conditions are
both satisfied and stable.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Duursma, R. A., & Medlyn, B. E. (2012). MAESPA: a model to study
  interactions between water limitation, environmental drivers and
  vegetation function at tree and stand levels, with an example
  application to [CO2] × drought interactions. Geoscientific Model
  Development, 5(4), 919-940.
* Tuzet, A., Perrier, A., & Leuning, R. (2003). A coupled model of
  stomatal conductance, photosynthesis and transpiration. Plant, Cell &
  Environment, 26(7), 1097-1116.

"""

__title__ = "Iterative solving with the Tuzet model"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, fLWP, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import calc_trans


# ======================================================================

def gas_exchange(p, fw, photo='Farquhar', res='low', iter_max=40,
                 threshold_conv=0.1, inf_gb=False):

    """
    Checks the energy balance by looking for convergence of the new leaf
    temperature with the leaf temperature predicted by the previous
    iteration. Then returns the corresponding An, E, Ci, etc.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    fw: float
        unitless hydraulic stress factor to be applied on gs

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to solve for leaf water
        potential

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    threshold_conv: float
        convergence threshold for the new leaf temperature to be in
        energy balance

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    An: float
        net photosynthetic assimilation rate [umol m-2 s-1]

    Aj: float
        electron transport-limited photosynthesis rate [μmol m-2 s-1]

    Ac: float
        rubisco-limited photosynthesis rate [μmol m-2 s-1]

    Ci: float
        intercellular CO2 concentration [Pa]

    trans: float
        transpiration rate [mmol m-2 s-1]

    gs: float
        stomatal conductance [mol m-2 s-1]

    gb: float
        leaf boundary layer conductance [mol m-2 s-1]

    new_Tleaf: float
        leaf temperature [degC]

    Pleaf: float
        leaf water potential [MPa]

    """

    # initial state
    Cs = p.CO2  # Pa

    try:   # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
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

        # calc. leaf water potential
        Pleaf = P[np.nanargmin(np.abs(E - trans))]

        try:  # is Tleaf one of the input fields?
            new_Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            new_Tleaf, __ = leaf_temperature(p, trans, Tleaf=Tleaf,
                                             inf_gb=inf_gb)

        # update Cs (Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb)
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


def fLWP_stable(p, photo='Farquhar', res='low', iter_max=5, inf_gb=False):

    """
    Leaf water potential can fluctuate a lot, so it first needs to be
    stabilized in order to estimate a stomatal reduction factor that
    makes sense and will not fluctuate equifinately.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to solve for leaf water
        potential

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    fw: float
        unitless hydraulic stress factor to be applied on gs

    """

    # hydraulics
    P, __ = hydraulics(p, res=res, kmax=p.kmaxT)

    # directionality?
    if np.isclose(p.LWP_ini,
                  p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA):
        down = True  # downstream from Ps

    else:
        fw = fLWP(p, p.LWP_ini)  # previous stress factor
        __, __, __, __, __, __, __, __, Pleaf = gas_exchange(p, fw,
                                                             photo=photo,
                                                             res=res,
                                                             iter_max=iter_max,
                                                             inf_gb=inf_gb)

        if abs(Pleaf - p.LWP_ini) < P[0] - P[1]:  # look no further

            return fw

        elif Pleaf - p.LWP_ini > 0.:
            P = np.flip(P[P >= p.LWP_ini])
            down = False  # upstream from the previous LWP

        else:
            P = P[P <= p.LWP_ini]
            down = True  # downstream from the previous LWP

    for Psi in P:  # now look for most stable LWP

        fw = fLWP(p, Psi)  # update stress factor
        __, __, __, __, __, __, __, __, Pleaf = gas_exchange(p, fw,
                                                             photo=photo,
                                                             res=res,
                                                             iter_max=iter_max,
                                                             inf_gb=inf_gb)

        if ((down and (Pleaf - Psi > P[1] - P[0])) or
           ((not down) and (Pleaf - Psi < P[0] - P[1]))):  # converges

            return fw

    return 0.


def Tuzet(p, photo='Farquhar', res='low', inf_gb=False):

    """
    Wrapper around the energy balance routine that first stabilizes the
    stomatal downregulation factor before finding model solutions.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to solve for leaf water
        potential

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    An: float
        net photosynthetic assimilation rate [umol m-2 s-1]

    Ci: float
        intercellular CO2 concentration [Pa]

    rublim: bool
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    trans: float
        transpiration rate [mmol m-2 s-1]

    gs: float
        stomatal conductance [mol m-2 s-1]

    gb: float
        leaf boundary layer conductance [mol m-2 s-1]

    Tleaf: float
        leaf temperature [degC]

    Pleaf: float
        leaf water potential [MPa]

    """

    # stability pre-requisites
    fw = fLWP_stable(p, photo=photo, res=res, inf_gb=inf_gb)

    # run the energy balance sub-routine
    An, Aj, Ac, Ci, trans, gs, gb, Tleaf, Pleaf = gas_exchange(p, fw,
                                                               photo=photo,
                                                               res=res,
                                                               inf_gb=inf_gb)

    # rubisco- or electron transport-limitation?
    rublim = rubisco_limit(Aj, Ac)

    if not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
        trans *= conv.MILI  # mmol.m-2.s-1

    return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf
