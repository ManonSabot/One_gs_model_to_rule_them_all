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
__version__ = "1.0 (16.10.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC.leaf import arrhen, adjust_low_T
from TractLSM.SPAC import vpsat, hydraulics
from TractLSM.SPAC import leaf_temperature, leaf_energy_balance
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import Ci_sup_dem, A_trans


# ======================================================================

def dVmaxoAdXi(p, trans, Ci, inf_gb=False):

    try:  # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

    # gamstar, Vmax, Kc and Ko are known at Tref, get their T dependency
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC

    # CO2 compensation point
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    # Michaelis-Menten constants
    #Kc = arrhen(p.Kc25, p.Ec, Tref, Tleaf)  # cst for carboxylation, Pa
    #Ko = arrhen(p.Ko25, p.Eo, Tref, Tleaf)  # cst for oxygenation, kPa
    #Ko = np.maximum(cst.zero, Ko)  # we don't want zeros in Km div

    # Michaelis-Menten constant for O2/CO2
    #Km = Kc * (1. + p.O2 / Ko)

    #dVmaxoA = -(p.CO2 * (Km + gamstar)) / ((Ci - gamstar) ** 2.)
    Vmax = arrhen(p.Vmax25, p.Ev, Tref, Tleaf, deltaS=p.deltaSv, Hd=p.Hdv)
    Vmax = adjust_low_T(Vmax, Tleaf)

    return Vmax  #dVmaxoA


def least_cost(p, photo='Farquhar', res='low', inf_gb=False, deriv=False):

    """
    Finds the instantaneous optimal C gain for a given C cost.
    First, the C gain equation is derived for gs, beta, Ci unknown.
    Then, the derived form of the equation is solved for Ci over a range of
    possible betas, gs, all of which are directly or indirectly leaf
    water potential P dependent.
    A check (check_solve) is performed to verify that the optimization satisfies
    the zero equality criteria and, finally, results are bound via a range of
    physically possible Ci values.
    N.B.: there can be several possible optimizations

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz model

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    gsOPT: float
        stomatal conductance [mol.m-2.s-1] for which the A(gs) is maximized

    AnOPT: float
        maximum C assimilation rate [μmol.m-2.s-1] given by the diffusive supply
        of CO2

    transOPT: float
        transpiration rate [mmol.m-2.s-1] for which the A(gs) is maximized

    CiOPT: float
        intercellular CO2 concentration [Pa] for which the A(gs) is maximized

    """

    # energy balance
    P, trans = hydraulics(p, res=res, kmax=p.kmaxLC)

    Ci, mask = Ci_sup_dem(p, trans, photo=photo, res=res, inf_gb=inf_gb)
    gc, gs, gb, ww = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)

    # expression of optimization
    expr = ((p.Eta * conv.MILI * trans[mask] +
            dVmaxoAdXi(p, trans[mask], Ci, inf_gb=inf_gb)) /
            A_trans(p, trans[mask], Ci, inf_gb=inf_gb))

    if deriv:
        where = np.where(np.diff(expr, 2) / np.diff(Ci / p.CO2, 2) > 0.)[0]
        expr = np.abs(np.gradient(expr, Ci / p.CO2))

    try:
        if inf_gb:  # check on valid range
            check = expr[gc > cst.zero]

            if deriv:
                check = expr[where][gc[where] > cst.zero]

        else:  # further constrain the realm of possible gs
            check = expr[np.logical_and(gc > cst.zero, gs < 1.5 * gb)]

            if deriv:
                check = expr[where][np.logical_and(gc[where] > cst.zero,
                                                   gs[where] < 1.5 * gb)]

        idx = np.isclose(expr, min(check))
        idx = [list(idx).index(e) for e in idx if e]

        if inf_gb:  # check for algo. "overshooting" due to inf. gb
            while Ci[idx[0]] < 2. * p.gamstar25:

                idx[0] -= 1

                if idx[0] < 3:
                    break

        # optimized where Ci for both photo models are close
        Ci = Ci[idx[0]]
        trans = trans[mask][idx[0]]  # mol.m-2.s-1
        gs = gs[idx[0]]
        Pleaf = P[mask][idx[0]]

        # rubisco limitation or electron transport-limitation?
        An, Aj, Ac = calc_photosynthesis(p, trans, Ci, photo=photo,
                                         inf_gb=inf_gb)
        rublim = rubisco_limit(Aj, Ac)

        # leaf temperature?
        Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

        if (np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
            (An > 0.)) or (idx[0] == len(P) - 1) or any(np.isnan([An, Ci, trans,
            gs, Tleaf, Pleaf])):
            An, Ci, trans, gs, gb, Tleaf, Pleaf = (9999.,) * 7

        elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
            trans *= conv.MILI  # mmol.m-2.s-1

        return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf

    except ValueError:  # no opt
        return (9999.,) * 8
