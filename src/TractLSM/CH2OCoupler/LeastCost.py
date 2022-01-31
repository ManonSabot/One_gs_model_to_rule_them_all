# -*- coding: utf-8 -*-

"""

The least cost algorithm, adapted from Prentice et al. (2014)'s stomatal
optimization model that balances carbon and water costs.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

Reference:
----------
* Prentice, I. C., Dong, N., Gleason, S. M., Maire, V., & Wright, I. J.
  (2014). Balancing the costs of carbon gain and water transport:
  testing a new theoretical framework for plant functional ecology.
  Ecology letters, 17(1), 82-91.

"""

__title__ = "Least cost algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (16.10.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC.leaf import arrhen, adjust_low_T
from TractLSM.SPAC import hydraulics, leaf_energy_balance
from TractLSM.SPAC import leaf_temperature, calc_photosynthesis
from TractLSM.SPAC import rubisco_limit
from TractLSM.CH2OCoupler import Ci_sup_dem, A_trans


# ======================================================================

def Vmax_T(p, trans, inf_gb=False):

    """
    Calculates Vcmax at any given temperature.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    Vcmax [umol m-2 s-1] at either a single reference temperature or
    across an array of temperatures determined by the transpiration
    array.

    """

    try:  # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

    # gamstar, Vmax, Kc and Ko are known at Tref, get their T dependency
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC

    # temperature adjustments
    Vmax = arrhen(p.Vmax25, p.Ev, Tref, Tleaf, deltaS=p.deltaSv, Hd=p.Hdv)
    Vmax = adjust_low_T(Vmax, Tleaf)

    return Vmax


def least_cost(p, photo='Farquhar', res='low', inf_gb=False, deriv=False):

    """
    Finds the instateneous least cost, following the optimization
    criterion for which, at each instant in time, the stomata regulate
    canopy gas exchange to minimise two costs: (i) the cost of
    maintaining the transpiration stream required to support
    assimilation under normal daytime conditions, and (ii) the cost of
    maintaining photosynthetic proteins at the level required to support
    assimilation at the same rate.

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
        net photosynthetic assimilation rate [umol m-2 s-1] at the point
        of least cost

    Ci: float
        intercellular CO2 concentration [Pa] at the point of least cost

    rublim: bool
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    trans: float
        transpiration rate [mmol m-2 s-1] at the point of least cost

    gs: float
        stomatal conductance [mol m-2 s-1] at the point of least cost

    gb: float
        leaf boundary layer conductance [mol m-2 s-1] at the point of
        least cost

    Tleaf: float
        leaf temperature [degC] at the point of least cost

    Pleaf: float
        leaf water potential [MPa] at the point of least cost

    """

    # hydraulics
    P, trans = hydraulics(p, res=res, kmax=p.kmaxLC)

    # expression of optimization
    Ci, mask = Ci_sup_dem(p, trans, photo=photo, res=res, inf_gb=inf_gb)
    expr = ((p.Eta * conv.MILI * trans[mask] +
             Vmax_T(p, trans[mask], inf_gb=inf_gb)) /
            A_trans(p, trans[mask], Ci, inf_gb=inf_gb))

    # deal with edge cases by rebounding the solution
    gc, gs, gb, __ = leaf_energy_balance(p, trans[mask], inf_gb=inf_gb)

    if deriv:  # derivative form
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

        # calc. optimal An
        An, Aj, Ac = calc_photosynthesis(p, trans, Ci, photo=photo,
                                         inf_gb=inf_gb)

        # rubisco- or electron transport-limitation?
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
