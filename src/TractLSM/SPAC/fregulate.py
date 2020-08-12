# -*- coding: utf-8 -*-

"""
Functions used to down-regulate stomatal conductance.

This file is part of the TractLSM model.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Dewar, Roderick, et al. "New insights into the covariation of
  stomatal, mesophyll and hydraulic conductances from optimization
  models incorporating nonstomatal limitations to photosynthesis." New
  Phytologist 217.2 (2018): 571-585.
* Eller, Cleiton B., et al. "Modelling tropical forest responses to
  drought and El Niño with a stomatal optimization model based on xylem
  hydraulics." Philosophical Transactions of the Royal Society B:
  Biological Sciences 373.1760 (2018): 20170315.
* Lu, Yaojie, et al. "Optimal stomatal drought response shaped by
  competition for water and hydraulic risk can explain plant trait
  covariation." New Phytologist (2019).
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.
* Tuzet, A., A. Perrier, and R. Leuning. "A coupled model of stomatal
  conductance, photosynthesis and transpiration." Plant, Cell &
  Environment 26.7 (2003): 1097-1116.
* Wolf, Adam, William RL Anderegg, and Stephen W. Pacala. "Optimal
  stomatal behavior with competition for water and risk of hydraulic
  impairment." Proceedings of the National Academy of Sciences 113.46
  (2016): E7222-E7230.
* Zhou, Shuangxi, et al. "How should we model plant responses to
  drought? An analysis of stomatal and non-stomatal responses to water
  stress." Agricultural and Forest Meteorology 182 (2013): 204-214.

"""

__title__ = "Plant hydraulics module"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (29.11.2017)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators
from scipy.integrate import quad  # integrate on a range

# own modules
from TractLSM import cst  # general constants
from TractLSM.SPAC import f, Weibull_params


# ======================================================================

def fwsoil(p, sw):

    """
    Calculates the empirical soil moisture stress factor that determines
    the stomatal conductance and soil's responses to water limitation.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    sw: float
        mean volumetric soil moisture content [m3 m-3]

    Returns:
    --------
    The empirical stomatal conductance and soil's responses to soil
    moisture stress.

    """

    return np.maximum(cst.zero, np.minimum(1., (sw - p.pwp) / (p.fc - p.pwp)))


def fwLWPpd(p, Ppd):

    """
    Calculates the empirical soil water potential factor (i.e. a proxy
    for predawn leaf water potential) that determines the stomatal
    conductance's responses to water limitation (Zhou).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The empirical stomatal conductance's response to soil water
    potential.

    """

    return np.exp(p.sref * Ppd)


def fLWP(p, Pleaf):

    """
    Tuzet

    """

    return (1. + np.exp(p.srefT * p.PrefT)) / (1. + np.exp(p.srefT * (p.PrefT -
                                                                      Pleaf)))


def phiLWP(P, Pcrit):

    """
    Dewar.
    Calculates the hydraulic reduction factor that reflects the
    damage from cavitation and greater difficulty of moving up the
    transpiration stream with decreasing values of the hydraulic
    conductance. The reduction factor is: 1 - Pleaf / Pc

    Arguments:
    ----------

    P: array
        leaf water potential [MPa], an array of values from the
        soil water potential Ps to the critical water potential Pcrit
        for which cavitation of the xylem occurs

    Returns:
    --------
    phi: array
        reduction factor [unitless]

    """

    return np.maximum(0., np.minimum(1., 1. - P / Pcrit))


def fPLC(p, P, fmin=0., fmax=1.):

    """
    Lu: check the expression

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf water potential [MPa], an array of values from the
        soil water potential Ps to the critical water potential Pcrit
        for which cavitation of the xylem occurs
    Returns:
    --------
    cost: array
        hydraulic cost [unitless]

    f(P, b, c): array
        vulnerability curves of the the plant [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    # PLC
    PLC = 1. - f(P, b, c)  # normalized, unitless

    if (fmax == 1.) and (p.ratiocrit > 0.):
        fmax = 1. - p.ratiocrit

    return fmax - (fmax - PLC) - fmin


def hydraulic_cost(p, P):

    """
    Calculates the hydraulic cost function that reflects the increasing
    damage from cavitation and greater difficulty of moving up the
    transpiration stream with decreasing values of the hydraulic
    conductance. The cost is defined as: (kmax - k) / (kmax - kcrit)

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf water potential [MPa], an array of values from the
        soil water potential Ps to the critical water potential Pcrit
        for which cavitation of the xylem occurs

    Returns:
    --------
    cost: array
        hydraulic cost [unitless]

    f(P, b, c): array
        vulnerability curves of the the plant [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    # critical percentage below which cavitation occurs
    kcrit = p.ratiocrit * p.kmax  # xylem cannot recover past this

    # hydraulic conductance, from kmax @ Ps to kcrit @ Pcrit
    k = p.kmax * f(P, b, c)  # mmol s-1 m-2 MPa-1

    # current maximum hydraulic conductance of the plant
    kmax = p.kmax * f(p.Ps, b, c)  # mmol s-1 m-2 MPa-1

    # cost, from kmax @ Ps to kcrit @ Pcrit
    cost = (kmax - k) / (kmax - kcrit)  # normalized, unitless

    return cost, f(P, b, c)


def kcost(p, P, Ppd):

    """
    Calculates the

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf water potential [MPa], an array of values from the
        soil water potential Ps to the critical water potential Pcrit
        for which cavitation of the xylem occurs

    Returns:
    --------
    kcost: array
        hydraulic cost [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    # restrict the leaf water potentials P
    Pcrit = - b * np.log(1. / p.ratiocrit) ** (1. / c)  # MPa

    try:
        if len(P) > 1:
            P = P[P >= Pcrit]

    except TypeError:
        pass

    # from kmax @ Pleaf,pd to Pcrit
    cost = f(0.5 * (Ppd + P), b, c)  # normalized, unitless

    return cost, P


def dcost_dpsi(p, P, gs):

    """
    Wolf.

    Arguments:
    ----------

    Returns:
    --------
    The normalized cost of C assimilation to avoid a unit increase in xylem
    tension.

    """

    return np.gradient(P, gs) * (p.Alpha * P + p.Beta)
