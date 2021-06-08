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
__version__ = ""
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC.leaf import arrhen
from TractLSM.SPAC import hydraulics, phiLWP
from TractLSM.SPAC import leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import Ci_sup_dem, A_trans


# ======================================================================

def MES(p, photo='Farquhar', res='low', inf_gb=False, deriv=False):

    """
    Finds the instantaneous optimal C gain for a given C cost.
    First, the C gain equation is derived for gs, beta, Ci, and ww unknown.
    Then, the derived form of the equation is solved for Ci over a range of
    possible betas, gs, and ww, all of which are directly or indirectly leaf
    water potential P dependent. ww can be both negative and positive for
    different P values within the same timestep, so its sign changes are checked
    using sing_change.
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

    # hydraulics and reduction factor
    P, trans = hydraulics(p, res=res, kmax=p.krlM, Pcrit=p.PcritM)
    phi = phiLWP(P, p.PcritM)

    # expression of optimisation
    Cc, mask = Ci_sup_dem(p, trans, photo=photo, res=res, phi=phi,
                          inf_gb=inf_gb)
    An, Aj, Ac = calc_photosynthesis(p, trans[mask], Cc, photo=photo,
                                     inf_gb=inf_gb)
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)
    expr = An

    if deriv:
        expr = np.abs(np.gradient(expr, gs))

    try:
        if inf_gb:  # check on valid range
            check = expr[gc > cst.zero]

        else:  # further constrain the realm of possible gs
            check = expr[np.logical_and(gc > cst.zero, gs < 1.5 * gb)]

        idx = np.isclose(expr, max(check))

        if deriv:
            idx = np.isclose(expr, min(check))

        idx = [list(idx).index(e) for e in idx if e]

        if inf_gb:  # check for algo. "overshooting" due to inf. gb
            while Cc[idx[0]] < p.gamstar25:

                idx[0] -= 1

                if idx[0] < 3:
                    break

        # optimized where Cis for both photo models are close
        An = An[idx[0]]
        trans = trans[mask][idx[0]]  # mol.m-2.s-1
        gs = gs[idx[0]]
        Pleaf = P[mask][idx[0]]

        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

        # Ci?
        gamstar = arrhen(p.gamstar25, p.Egamstar, p.Tref + conv.C_2_K, Tleaf)
        Ci = (Cc[idx[0]] - gamstar) / phi[mask][idx[0]] + gamstar

        # rubisco limitation or electron transport-limitation?
        rublim = rubisco_limit(Aj[idx[0]], Ac[idx[0]])

        if (np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
            (An > 0.)) or (idx[0] == len(P) - 1) or any(np.isnan([An, Ci,
            trans, gs, Tleaf, Pleaf])):
            An, Ci, trans, gs, gb, Tleaf, Pleaf = (9999.,) * 7

        elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
            trans *= conv.MILI  # mmol.m-2.s-1

        return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf

    except ValueError:  # no opt

        return (9999.,) * 8
