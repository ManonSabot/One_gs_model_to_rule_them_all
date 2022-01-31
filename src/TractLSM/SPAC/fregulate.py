# -*- coding: utf-8 -*-

"""
Functions used to down-regulate stomatal conductance.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Anderegg et al. (2018). Woody plants optimise stomatal behaviour
  relative to hydraulic risk. Ecology letters, 21(7), 968-977.
* Dewar et al. (2018). New insights into the covariation of stomatal,
  mesophyll and hydraulic conductances from optimization models
  incorporating nonstomatal limitations to photosynthesis.
  New Phytologist, 217(2), 571-585.
* Eller at al. (2018). Modelling tropical forest responses to drought
  and El Niño with a stomatal optimization model based on xylem
  hydraulics. Philosophical Transactions of the Royal Society B:
  Biological Sciences, 373(1760), 20170315.
* Eller et al. (2020). Stomatal optimization based on xylem hydraulics
  (SOX) improves land surface model simulation of vegetation responses
  to climate. New Phytologist, 226(6), 1622-1637.
* Lu et al. (2020). Optimal stomatal drought response shaped by
  competition for water and hydraulic risk can explain plant trait
  covariation. New Phytologist, 225(3), 1206-1217.
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.
* Tuzet et al. (2003). A coupled model of stomatal conductance,
  photosynthesis and transpiration. Plant, Cell & Environment, 26(7),
  1097-1116.
* Wang et al. (2020). A theoretical and empirical assessment of stomatal
  optimization modeling. New Phytologist, 227(2), 311-325.
* Zhou et al. (2013). How should we model plant responses to drought? An
  analysis of stomatal and non-stomatal responses to water stress.
  Agricultural and Forest Meteorology, 182, 204-214.

"""

__title__ = "Moisture stress / water supply downregulators"
__author__ = "Manon E. B. Sabot"
__version__ = "8.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM.SPAC import f, Weibull_params  # hydraulics


# ======================================================================

def fwWP(p, Psi):

    """
    Calculates an empirical soil water potential factor that determines
    the stomatal conductance's responses to water limitation, following
    Zhou et al. (2013).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Psi: float
        soil water potential [MPa] (or predawn leaf water potential as
        a proxy)

    Returns:
    --------
    The unitless empirical stomatal conductance's response to soil water
    potential.

    """

    return np.exp(p.sref * Psi)


def fLWP(p, Pleaf):

    """
    Calculates an empirical logistic function used to describe the
    sensitivity of the stomates to leaf water potential, following
    Tuzet et al. (2003).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Pleaf: float
        leaf water potential [MPa]

    Returns:
    --------
    The unitless empirical stomatal conductance's response to leaf water
    potential.

    """

    return (1. + np.exp(p.srefT * p.PrefT)) / (1. + np.exp(p.srefT * (p.PrefT -
                                                                      Pleaf)))


def dkcost(p, Pxpd):

    """
    Approximates a hydraulic empirical reduction function using P50,
    which reflects the increasing cost of stomatal aperture with
    decreasing values of the hydraulic conductance, following
    Eller et al. (2020).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Pxpd: float
        leaf xylem saturated pressure at predawn [MPa]

    Returns:
    --------
    kcost1: float
        hydraulic downregulation [unitless] at Pxpd

    kcost 2: float
        hydraulic downregulation [unitless] at Pmid, the midpoint
        between Pxpd and P50

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    return f(Pxpd, b, c), f(0.5 * (Pxpd - p.P50), b, c)


def dcost_dpsi(p, P):

    """
    Calculates an increasing cost of carbon assimilation emulated by
    the derivative of a positive parabolic function, following
    Anderegg et al. (2018).


    Note: the Beta here is taken in opposite sign from the equation, so
          as to get a positive parameter value. When reported, the units
          should be taken care of by also reporting the minus sign.
          (e.g., -Beta of calibrated value)

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    Returns:
    --------
    The unitless normalized cost of carbon assimilation that avoids a
    unit increase in xylem tension.

    """

    return p.Alpha * np.abs(P) + p.Beta


def hydraulic_cost(p, P):

    """
    Calculates a hydraulic cost function that reflects the increasing
    damage from cavitation and greater difficulty of moving up the
    transpiration stream with decreasing values of the hydraulic
    conductance, following Sperry et al. (2017).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    Returns:
    --------
    cost: array
        hydraulic cost from krcmax @ Ps to kcrit @ Pcrit [unitless]

    f(P, b, c): array
        vulnerability curves of the the plant [unitless]

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless
    vc = f(P, b, c)

    return (f(p.Ps, b, c) - vc) / (f(p.Ps, b, c) - p.ratiocrit), vc


def fPLC(p, P):

    """
    Normalizes the percentage loss of conductivity (PLC) between 0 - 1,
    thus approximating it to a hydraulic cost, following
    Lu et al. (2020).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    Returns:
    --------
    An array of unitless PLCs corresponding to each P of the water
    potential array.

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    return 1. - f(P, b, c)


def kcost(p, P):

    """
    Calculates a hydraulic reduction function that reflects the
    increasing cost of stomatal aperture and greater difficulty of
    moving up the transpiration stream with decreasing values of the
    hydraulic conductance, following Eller et al. (2018).

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    Returns:
    --------
    An array of unitless hydraulic reduction factors corresponding to
    each P of the water potential array.

    """

    # Weibull parameters setting the shape of the vulnerability curve
    b, c = Weibull_params(p)  # MPa, unitless

    return (f(P, b, c) - p.ratiocrit) / (1. - p.ratiocrit)


def phiLWP(P, Pcrit):

    """
    Calculates a linear hydraulic reduction factor that reflects the
    damage from cavitation and the greater difficulty of assimilating
    carbon with decreasing values of the hydraulic conductance,
    following Dewar et al. (2017).

    Arguments:
    ----------
    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    Pcrit: float
        critical leaf water potential [MPa] below which there can be no
        photosynthesis. This sets the Pcrit of the water potential
        array if specified.

    Returns:
    --------
    An array of unitless reduction factors corresponding to each P of
    the water potential array.

    """

    return np.maximum(0., np.minimum(1., 1. - P / Pcrit))
