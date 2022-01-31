# -*- coding: utf-8 -*-

"""
The carbon maximisation algorithm, adapted from Wolf et al. (2016)'s
hydraulics-limited stomatal optimization model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

Reference:
----------
* Wolf, A., Anderegg, W. R., & Pacala, S. W. (2016). Optimal stomatal
  behavior with competition for water and risk of hydraulic impairment.
  Proceedings of the National Academy of Sciences, 113(46), E7222-E7230.

"""

__title__ = "Carbon maximization algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (02.01.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, dcost_dpsi
from TractLSM.SPAC import leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import Ci_sup_dem, A_trans


# ======================================================================

def Cmax_gs(p, photo='Farquhar', res='low', inf_gb=False):

    """
    Finds the instantaneous maximum carbon intake, following the
    optimization criterion for which, at each instant in time, the
    stomata regulate canopy gas exchange and pressure to achieve the
    maximum difference between net photosynthetic intake and a cost
    function that increases exponentially with leaf water potential (P).
    Note, the algorithm uses derivative forms of the expressions.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    An: float
        net photosynthetic assimilation rate [umol m-2 s-1] when carbon
        intake is maximised

    Ci: float
        intercellular CO2 concentration [Pa] when carbon intake is
        maximised

    rublim: bool
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    trans: float
        transpiration rate [mmol m-2 s-1] when carbon intake is
        maximised

    gs: float
        stomatal conductance [mol m-2 s-1] when carbon intake is
        maximised

    gb: float
        leaf boundary layer conductance [mol m-2 s-1] when carbon
        intake is maximised

    Tleaf: float
        leaf temperature [degC] when carbon intake is maximised

    Pleaf: float
        leaf water potential [MPa] when carbon intake is maximised

    """

    # hydraulics
    P, trans = hydraulics(p, res=res, kmax=p.kmaxCM)

    # expression of optimization
    Ci, mask = Ci_sup_dem(p, trans, photo=photo, res=res, inf_gb=inf_gb)
    expr = np.abs(np.gradient(A_trans(p, trans[mask], Ci, inf_gb=inf_gb),
                              P[mask]) - dcost_dpsi(p, P[mask]))

    # deal with edge cases by rebounding the solution
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)

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
        Ci = Ci[idx[0]]
        trans = trans[mask][idx[0]]  # mol.m-2.s-1
        gs = gs[idx[0]]
        Pleaf = P[mask][idx[0]]

        # calc. optimal An
        An, Aj, Ac = calc_photosynthesis(p, trans, Ci, photo=photo,
                                         inf_gb=inf_gb)

        # rubisco limitation or electron transport-limitation?
        rublim = rubisco_limit(Aj, Ac)

        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

        if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
            (An > 0.)) or (idx[0] == len(P) - 1) or
           any(np.isnan([An, Ci, trans, gs, Tleaf, Pleaf]))):
            An, Ci, trans, gs, gb, Tleaf, Pleaf = (9999.,) * 7

        elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
            trans *= conv.MILI  # mmol.m-2.s-1

        return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf

    except ValueError:  # no opt

        return (9999.,) * 8
