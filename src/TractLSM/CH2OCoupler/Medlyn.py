# -*- coding: utf-8 -*-

"""
The Medlyn (USO) model, adapted for LSMs, by iteration on the air
temperature to get the leaf temperature for which the Penman-Monteith
energy balance conditions are satisfied.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Kowalczyk, E. A., Wang, Y. P., Law, R. M., Davies, H. L., McGregor,
  J. L., & Abramowitz, G. (2006). The CSIRO Atmosphere Biosphere Land
  Exchange (CABLE) model for use in climate models and as an offline
  model. CSIRO Marine and Atmospheric Research Paper, 13, 42.
* Medlyn, B. E., Duursma, R. A., Eamus, D., Ellsworth, D. S., Prentice,
  I. C., Barton, C. V., ... & Wingate, L. (2011). Reconciling the
  optimal and empirical approaches to modelling stomatal conductance.
  Global Change Biology, 17(6), 2134-2144.
* Wang, Y. P., Kowalczyk, E., Leuning, R., Abramowitz, G., Raupach,
  M. R., Pak, B., ... & Luhar, A. (2011). Diagnosing errors in a land
  surface model (CABLE) in the time and frequency domains. Journal of
  Geophysical Research: Biogeosciences, 116(G1).

"""

__title__ = "Iterative solving with the USO model"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (19.11.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, fwWP
from TractLSM.SPAC import leaf_temperature, calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import calc_trans


# ======================================================================

def solve_std(p, sw, photo='Farquhar', res='low', iter_max=40,
              threshold_conv=0.1, inf_gb=False):

    """
    Checks the energy balance by looking for convergence of the new leaf
    temperature with the leaf temperature predicted by the previous
    iteration. Then returns the corresponding An, E, Ci, etc.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    sw: float
        mean volumetric soil moisture content [m3 m-3]

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

    new_Tleaf: float
        leaf temperature [degC]

    Pleaf: float
        leaf water potential [MPa]

    """

    # initial state
    Cs = p.CO2  # Pa
    Dleaf = np.maximum(0.05, p.VPD)  # gs model not valid < 0.05

    try:   # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf = p.Tair  # deg C

    # hydraulics
    P, E = hydraulics(p, res=res)

    if sw >= p.fc:
        g1 = p.g1

    else:
        g1 = p.g1 * fwWP(p, p.Ps)

    # initialise gs over A
    g0 = 1.e-9  # g0 ~ 0, removing it entirely introduces errors
    Cs_umol_mol = Cs * conv.MILI / p.Patm  # umol mol-1
    gsoA = g0 + (1. + g1 / (Dleaf ** 0.5)) / Cs_umol_mol

    # iter on the solution until it is stable enough
    iter = 0

    while True:

        An, Aj, Ac, Ci = calc_photosynthesis(p, 0., Cs, photo=photo,
                                             Tleaf=Tleaf, gs_over_A=gsoA)

        # stomatal conductance, with moisture stress effect
        gs = np.maximum(cst.zero, conv.GwvGc * gsoA * An)

        # calculate new trans, gw, gb, mol.m-2.s-1
        trans, real_zero, gw, gb, Dleaf = calc_trans(p, Tleaf, gs,
                                                     inf_gb=inf_gb)

        try:  # is Tleaf one of the input fields?
            new_Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            new_Tleaf, __ = leaf_temperature(p, trans, Tleaf=Tleaf,
                                             inf_gb=inf_gb)

        # update Cs (Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb)
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))
        Cs_umol_mol = Cs * conv.MILI / p.Patm

        # new leaf-air vpd, kPa
        if (np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) or
            np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero) or
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero)):
            Dleaf = np.maximum(0.05, p.VPD)  # kPa

        # update gs over A
        gsoA = g0 + (1. + g1 / (Dleaf ** 0.5)) / Cs_umol_mol

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter >= 1) and
            real_zero and (abs(Tleaf - new_Tleaf) <= threshold_conv) and not
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate on leaf temperature
        Tleaf = new_Tleaf
        iter += 1

    # calc. leaf water potential
    Pleaf = P[np.nanargmin(np.abs(trans - E))]

    # rubisco- or electron transport-limitation?
    rublim = rubisco_limit(Aj, Ac)

    if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
        (An > 0.)) or np.isclose(Ci, 0., rtol=cst.zero, atol=cst.zero) or
        (Ci < 0.) or np.isclose(Ci, p.CO2, rtol=cst.zero, atol=cst.zero) or
        (Ci > p.CO2) or (real_zero is None) or (not real_zero) or
       any(np.isnan([An, Ci, trans, gs, new_Tleaf, Pleaf]))):
        An, Ci, trans, gs, gb, new_Tleaf, Pleaf = (9999.,) * 7

    elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
        trans *= conv.MILI  # mmol.m-2.s-1

    return An, Ci, rublim, trans, gs, gb, new_Tleaf, Pleaf
