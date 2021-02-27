# -*- coding: utf-8 -*-

"""
Wolf et al. stomatal optimization model with competition for water, recreated
and adapted.
Reference:
----------
* Wolf et al. (2016). Optimal stomatal behavior with competition for water and
  risk of hydraulic impairment. Proceedings of the National Academy of Sciences,
  113(46), E7222-E7230.

"""

__title__ = "Wolf PNAS model"
__reference__ = "http://www.pnas.org/content/113/46/E7222.full.pdf"
__author__ = "Manon Sabot"
__version__ = "1.0 (02.01.2018)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter
from TractLSM.SPAC import hydraulics, leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import Ci_sup_dem, A_trans


#==============================================================================

def WUE_gs(p, photo='Farquhar', res='low', inf_gb=False, deriv=False):

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
    P, trans = hydraulics(p, res=res, kmax=p.kmaxWUE)

    # expression of optimization
    Ci, mask = Ci_sup_dem(p, trans, photo=photo, res=res, inf_gb=inf_gb)
    expr = (A_trans(p, trans[mask], Ci, inf_gb=inf_gb) -
            p.Lambda * conv.MILI * trans[mask])

    # deal with edge cases by rebounding the solution
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)

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
