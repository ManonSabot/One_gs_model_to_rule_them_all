# -*- coding: utf-8 -*-

"""
Support functions for the coupling schemes.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "useful ancillary coupling functions"
__author__ = "Manon Sabot"
__version__ = "1.0 (02.01.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import vpsat, conductances, leaf_energy_balance
from TractLSM.SPAC import leaf_temperature, calc_photosynthesis
from TractLSM.SPAC.leaf import arrhen


# ======================================================================

def calc_trans(p, Tleaf, gs, inf_gb=False):

    """
    Calculates transpiration at the leaf level, accounting for effects
    of leaf temperature but not for e.g. radiative feedback on
    evaporation (as in PM).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Tleaf: float
        leaf temperature [degC]

    gs: float
        stomatal conductance [mol m-2 s-1]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    trans: float
        transpiration rate [mol m-2 s-1]

    real_zero: boolean
        True if the transpiration is really zero, False if Rnet is
        negative

    gw: float
        total leaf conductance to water vapour [mol m-2 s-1]

    gb: float
        leaf boundary layer conductance to water vapour [mol m-2 s-1]

    Dleaf: float
        leaf-to-air vapour pressure deficit [kPa]

    """

    # check that the trans value satisfies the energy balance
    real_zero = True

    # get conductances, mol m-2 s-1
    gw, __, gb, __ = conductances(p, Tleaf=Tleaf, gs=gs, inf_gb=inf_gb)

    try:  # is Tleaf one of the input fields?
        if np.isclose(Tleaf, p.Tleaf):
            Dleaf = p.VPD

        else:
            esat_l = vpsat(Tleaf)  # saturation vapour pressure of water
            esat_a = vpsat(p.Tair)  # air saturation vapour pressure
            Dleaf = (esat_l - (esat_a - p.VPD))  # leaf-air vpd, kPa

    except (IndexError, AttributeError, ValueError):  # calc. Dleaf
        esat_l = vpsat(Tleaf)  # saturation vapour pressure of water
        esat_a = vpsat(p.Tair)  # air saturation vapour pressure
        Dleaf = (esat_l - (esat_a - p.VPD))  # leaf-air vpd, kPa

    if np.isclose(gs, 0., rtol=cst.zero, atol=cst.zero):
        trans = cst.zero

    else:
        trans = gw * Dleaf / p.Patm

        if trans < 0.:  # non-physical trans
            real_zero = False

        trans = max(cst.zero, trans)  # mol m-2 s-1

    return trans, real_zero, gw, gb, Dleaf


def A_trans(p, trans, Ci, Tleaf=None, inf_gb=False):

    """
    Calculates the assimilation rate given the supply function, gc.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], an array of values depending on
        the possible leaf water potentials (P) and the Weibull
        parameters b, c

    Ci: float
        intercellular CO2 concentration [Pa], corresponding to a leaf
        water potential (P) for which the transpiration cost is minimal
        and the C assimilation gain is maximal

    Tleaf: float
        leaf temperature [degC]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    Calculates the photosynthetic gain A [umol m-2 s-1] for a given
    Ci(P) and over an array of Gc(P) values.

    """

    try:  # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        pass

    # get CO2 diffusive conduct.
    gc, __, __, __ = leaf_energy_balance(p, trans, Tleaf=Tleaf, inf_gb=inf_gb)
    A_P = conv.MILI * gc * (p.CO2 - Ci) / p.Patm

    try:
        A_P[np.isclose(np.squeeze(gc), cst.zero, rtol=cst.zero,
                       atol=cst.zero)] = cst.zero

    except TypeError:
        pass

    return A_P


def mtx_minimise(p, trans, all_Cis, photo, Vmax25=None, all_Ccs=None,
                 inf_gb=False):

    """
    Uses matrices to find each value of Ci for which An(supply) ~
    An(demand) on the transpiration stream.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    all_Cis: array
        all potential Ci [Pa] values over the transpiration stream (for
        each water potential, Ci values can be anywhere between a lower
        bound and Cs)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Vmax25: array
        all potential Vmax25 [umol m-2 s-1] values over the
        transpiration stream (for each water potential)

    all_Ccs: array
        all potential Cc [Pa] values over the transpiration stream (for
        each water potential, Cc values can be anywhere between a lower
        bound and Ci)

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    The value of Ci for which An(supply) is the closest to An(demand)
    (e.g. An(supply) - An(demand) closest to zero).

    """

    if Vmax25 is not None:  # non-stomatal limitations
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo,
                                             Vmax25=np.expand_dims(Vmax25,
                                                                   axis=1),
                                             inf_gb=inf_gb)

    elif all_Ccs is not None:  # non-stomatal limitations
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Ccs, photo, inf_gb=inf_gb)

    else:
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo, inf_gb=inf_gb)

    supply = A_trans(p, np.expand_dims(trans, axis=1), all_Cis, inf_gb=inf_gb)

    # find the meeting point between demand and supply
    idx = bn.nanargmin(np.abs(supply - demand), axis=1)  # closest ~0

    if all_Ccs is not None:
        all_Cis = all_Ccs

    # each Ci on the transpiration stream
    Ci = np.asarray([all_Cis[e, idx[e]] for e in range(len(trans))])
    Ci = np.ma.masked_where(idx == 0, Ci)
    Ci = np.ma.masked_where(idx == all_Cis.shape[1] - 1, Ci)

    return Ci


def Ci_sup_dem(p, trans, photo='Farquhar', res='low', Vmax25=None, phi=None,
               inf_gb=False):

    """
    Wrapper around the mtx_minimise function that allows to find each
    value of Ci for which An(supply) ~ An(demand) on the transpiration
    stream.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array
        transpiration [mol m-2 s-1], values depending on the possible
        leaf water potentials (P) and the Weibull parameters b, c

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    Vmax25: array
        all potential Vmax25 [umol m-2 s-1] values over the
        transpiration stream (for each water potential)

    phi: array
        linear unitless hydraulic cost factor applied to emulate
        non-stomatal limiations over the transpiration stream

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    The value of Ci for which An(supply) is the closest to An(demand)
    (e.g. An(supply) - An(demand) closest to zero). Also returns a mask
    associated to the valid values on that array.

    """

    # ref. photosynthesis
    A_ref, __, __ = calc_photosynthesis(p, trans, p.CO2, photo, Vmax25=Vmax25,
                                        inf_gb=inf_gb)

    # Cs < Ca, used to ensure physical solutions
    __, __, gb, __ = leaf_energy_balance(p, trans, inf_gb=inf_gb)
    boundary_CO2 = p.Patm * conv.FROM_MILI * A_ref / (gb * conv.GbcvGb)
    Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))

    # potential Ci values over the full range of transpirations
    if res == 'low':
        NCis = 500

    if res == 'med':
        NCis = 2000

    if res == 'high':
        NCis = 8000

    try:  # retrieve the appropriate Cis from the supply-demand
        Cis = np.asarray([np.linspace(0.1, Cs[e], NCis) for e in
                         range(len(trans))])

    except (IndexError, AttributeError, ValueError):
        Cis = np.asarray([np.linspace(0.1, Cs, NCis) for e in
                         range(len(trans))])

    if (Vmax25 is None) and (phi is not None):  # account for gm
        Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC

        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

        # CO2 compensation point
        gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

        try:  # now getting the Cc
            Ccs = np.asarray([phi[e] * (Cis[e] - gamstar[e]) + gamstar[e]
                             for e in range(len(trans))])

        except IndexError:  # only one Tleaf
            Ccs = np.asarray([phi[e] * (Cis[e] - gamstar) + gamstar
                             for e in range(len(trans))])

    if phi is None:  # no non-stomatal limitations
        Ccs = None

    Ci = mtx_minimise(p, trans, Cis, photo, Vmax25=Vmax25, all_Ccs=Ccs,
                      inf_gb=inf_gb)
    mask = ~Ci.mask

    try:
        if len(mask) > 0:
            pass

    except TypeError:
        mask = ([False] + [True, ] * len(Ci))[:-1]

    return Ci[mask], mask
