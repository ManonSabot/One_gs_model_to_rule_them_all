# -*- coding: utf-8 -*-

"""
Support functions

Reference:
----------
*

"""

__title__ = ""
__author__ = "Manon Sabot"
__version__ = "1.0 (02.01.2018)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

# general modules
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations

# own modules
from TractLSM import conv, cst  # unit converter
from TractLSM.SPAC import vpsat, conductances, leaf_energy_balance
from TractLSM.SPAC import leaf_temperature, calc_photosynthesis
from TractLSM.SPAC.leaf import arrhen


#==============================================================================

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
        boundary layer conductance to water vapour [mol m-2 s-1]

    """

    # check that the trans value satisfies the energy balance
    real_zero = True

    # get conductances, mol m-2 s-1
    gw, __, gb, __ = conductances(p, Tleaf=Tleaf, gs=gs, inf_gb=inf_gb)

    # saturation vapour pressure of water at Tair
    esat_a = vpsat(p.Tair)  # kPa
    esat_l = vpsat(Tleaf)  # vpsat at new Tleaf, kPa
    Dleaf = np.maximum(0.05, (esat_l - (esat_a - p.VPD)))

    if np.isclose(gs, 0., rtol=cst.zero, atol=cst.zero):
        trans = cst.zero

    else:
        trans = gw * Dleaf / p.Patm

        if trans < 0.:  # non-physical trans
            real_zero = False

        trans = max(cst.zero, trans)  # mol m-2 s-1

    return trans, real_zero, gw, gb, Dleaf


def dAdgs(p, gs, gb, Ci):

    dA = (conv.U * ((gb * conv.GbcvGb) ** 2.) * (p.CO2 - Ci) /
          (p.Patm * conv.MILI * (gb * conv.GbcvGb + gs * conv.GcvGw) ** 2.))

    return dA


def A_trans(p, trans, Ci, Tleaf=None, inf_gb=False):

    """
    Calculates the assimilation rate given the supply function, gc. No
    respiration here as this is the "physics" An.

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

    if Tleaf is not None:  # get CO2 diffusive conduct.
        gc, __, __, __ = leaf_energy_balance(p, trans, Tleaf=Tleaf,
                                             inf_gb=inf_gb)

    if Tleaf is None:
        __, gs, gb, __ = leaf_energy_balance(p, trans, inf_gb=inf_gb)
        gc = np.maximum(cst.zero, conv.GcvGw * (gb * gs) / (gb + gs))

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
        all potential Ci values over the transpiration stream (for each
        water potential, Ci values can be anywhere between a lower bound
        and Cs)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    The value of Ci for which An(supply) is the closest to An(demand)
    (e.g. An(supply) - An(demand) closest to zero).

    """

    if Vmax25 is not None:
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo,
                                             Vmax25=np.expand_dims(Vmax25,
                                                                   axis=1),
                                             inf_gb=inf_gb)

    elif all_Ccs is not None:
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Ccs, photo, inf_gb=inf_gb)

    else:
        demand, __, __ = calc_photosynthesis(p, np.expand_dims(trans, axis=1),
                                             all_Cis, photo, inf_gb=inf_gb)

    supply = A_trans(p, np.expand_dims(trans, axis=1), all_Cis)

    # find the meeting point between demand and supply
    demand[supply < 0.] = np.nan
    supply[supply < 0.] = np.nan
    idx = bn.nanargmin(np.abs(supply - demand), axis=1)  # closest ~0

    if all_Ccs is not None:
        all_Cis = all_Ccs  # Ci is Cc for the MES's photo routine

    # each Ci on the transpiration stream
    Ci = np.asarray([all_Cis[e, idx[e]] for e in range(len(trans))])
    Ci = np.ma.masked_where(idx == 0, Ci)
    Ci = np.ma.masked_where(idx == all_Cis.shape[1] - 1, Ci)

    return Ci


def Ci_sup_dem(p, trans, photo='Farquhar', res='low', Vmax25=None, phi=None,
               inf_gb=False):

    __, gs, gb, __ = leaf_energy_balance(p, trans, inf_gb=inf_gb)

    # ref. photosynthesis for which the dark respiration is set to 0
    A_ref, __, __ = calc_photosynthesis(p, trans, p.CO2, photo, Rleaf=0.,
                                        Vmax25=Vmax25, inf_gb=inf_gb)

    # Cs < Ca, used to ensure physical solutions
    boundary_CO2 = (p.Patm * conv.FROM_MILI * A_ref / (gb * conv.GbcvGb +
                    gs * conv.GcvGw))
    Cs = np.minimum(p.CO2, p.CO2 - boundary_CO2)  # Pa

    # potential Ci values over the full range of transpirations
    if res == 'low':
        NCis = 500

    if res == 'med':
        NCis = 8000

    if res == 'high':
        NCis = 50000

    # retrieve the appropriate Cis from the supply-demand
    Cis = np.asarray([np.linspace(0.1, Cs[e], NCis) for e in
                     range(len(trans))])

    if (Vmax25 is None) and (phi is not None):  # account for gm
        Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC
        Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

        # CO2 compensation point
        gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

        # now getting the Cc
        Ccs = np.asarray([phi[e] * (Cis[e] - gamstar[e]) + gamstar[e] for e in
                         range(len(trans))])

    if phi is None:
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
