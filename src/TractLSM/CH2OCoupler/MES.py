# -*- coding: utf-8 -*-

"""
The MES algorithm, adapted from Dewar et al. (2017)'s non-stomatal
optimization model where photosynthesis is limited by reductions in
mesophyll conductance induced by leaf water stress.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

Reference:
----------
* Dewar, R., Mauranen, A., Mäkelä, A., Hölttä, T., Medlyn, B., & Vesala,
  T. (2018). New insights into the covariation of stomatal, mesophyll
  and hydraulic conductances from optimization models incorporating
  nonstomatal limitations to photosynthesis. New Phytologist, 217(2),
  571-585.

"""

__title__ = "MES algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (14.12.2019)"
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
from TractLSM.CH2OCoupler import Ci_sup_dem


# ======================================================================

def MES(p, photo='Farquhar', res='low', inf_gb=False, deriv=False):

    """
    Finds the instantaneous optimal gs, following the MES optimality
    criterion for which, at each instant in time, the stomata regulate
    canopy gas exchange and pressure to maximize A(gs), considering that
    leaf water potentials (P) increasingly downregulate A as P drops.

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

    deriv: bool
        if True, uses the derivative form of the optimality criterion

    Returns:
    --------
    An: float
        net photosynthetic assimilation rate [umol m-2 s-1] when MES is
        satisfied

    Ci: float
        intercellular CO2 concentration [Pa] when MES is satisfied

    rublim: bool
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    trans: float
        transpiration rate [mmol m-2 s-1] when MES is satisfied

    gs: float
        stomatal conductance [mol m-2 s-1] when MES is satisfied

    gb: float
        leaf boundary layer conductance [mol m-2 s-1] when MES is
        satisfied

    Tleaf: float
        leaf temperature [degC] when MES is satisfied

    Pleaf: float
        leaf water potential [MPa] when MES is satisfied

    """

    # hydraulics and reduction factor
    P, trans = hydraulics(p, res=res, kmax=p.krlM, Pcrit=p.PcritM)
    phi = phiLWP(P, p.PcritM)

    # expression of optimization
    Cc, mask = Ci_sup_dem(p, trans, photo=photo, res=res, phi=phi,
                          inf_gb=inf_gb)
    An, Aj, Ac = calc_photosynthesis(p, trans[mask], Cc, photo=photo,
                                     inf_gb=inf_gb)
    expr = An

    # deal with edge cases by rebounding the solution
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)

    if deriv:  # derivative form
        expr = np.abs(np.gradient(expr, gs))

    try:
        if inf_gb:  # check on valid range
            check = expr[gc > cst.zero]

        else:  # further constrain the realm of possible gs
            check = expr[np.logical_and(gc > cst.zero, gs < 1.5 * gb)]

        idx = np.isclose(expr, max(check))

        if deriv:  # derivative form
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

        # infer Ci from Cc
        gamstar = arrhen(p.gamstar25, p.Egamstar, p.Tref + conv.C_2_K, Tleaf)
        Ci = (Cc[idx[0]] - gamstar) / phi[mask][idx[0]] + gamstar

        # rubisco limitation or electron transport-limitation?
        rublim = rubisco_limit(Aj[idx[0]], Ac[idx[0]])

        if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
            (An > 0.)) or (idx[0] == len(P) - 1) or
           any(np.isnan([An, Ci, trans, gs, Tleaf, Pleaf]))):
            An, Ci, trans, gs, gb, Tleaf, Pleaf = (9999.,) * 7

        elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
            trans *= conv.MILI  # mmol.m-2.s-1

        return An, Ci, rublim, trans, gs, gb, Tleaf, Pleaf

    except ValueError:  # no opt

        return (9999.,) * 8
