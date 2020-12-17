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
from TractLSM.CH2OCoupler import Ci_sup_dem, dAdgs


# ======================================================================

def CAP(p, photo='Farquhar', res='low', inf_gb=False):

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

    # hydraulics
    P = hydraulics(p, res=res, Kirchhoff=False, Pcrit=p.PcritC)
    ksr = np.maximum(p.ksrmaxC * (p.Psie / p.Ps) ** (2. + 3. / p.bch),
                     p.krlC / 100.)  # criteria to limit crazy low results
    ksl = 1. / (1. / ksr + 1. / p.krlC)  # soil-leaf hydraulic conductance
    trans = ksl * (p.Ps - P) * conv.FROM_MILI  # mol.s-1.m-2

    # reduction factor?
    phi = phiLWP(P, p.PcritC)

    # expression of optimisation
    sVmax25 = p.Vmax25 * phi
    Ci, mask = Ci_sup_dem(p, trans, photo=photo, res=res, Vmax25=sVmax25,
                          inf_gb=inf_gb)
    An, Aj, Ac = calc_photosynthesis(p, trans[mask], Ci, photo,
                                     Vmax25=sVmax25[mask], inf_gb=inf_gb)
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)
    expr = np.abs(np.gradient(An, gs))

    try:
        if inf_gb:  # check on valid range
            check = expr[gc > cst.zero]

        else:  # further constrain the realm of possible gs
            check = expr[np.logical_and(gc > cst.zero, gs < 1.5 * gb)]

        idx = np.isclose(expr, min(check))
        idx = [list(idx).index(e) for e in idx if e]

        if inf_gb:  # check for algo. "overshooting" due to inf. gb
            while Ci[idx[0]] < 2. * p.gamstar25:

                idx[0] -= 1

                if idx[0] < 3:
                    break

        # optimized where Ci for both photo models are close
        An = An[idx[0]]
        Ci = Ci[idx[0]]
        trans = trans[mask][idx[0]]  # mol.m-2.s-1
        gs = gs[idx[0]]
        Pleaf = P[mask][idx[0]]

        # rubisco limitation or electron transport-limitation?
        rublim = rubisco_limit(Aj[idx[0]], Ac[idx[0]])

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
