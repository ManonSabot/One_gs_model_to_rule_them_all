# -*- coding: utf-8 -*-

"""
Functions related to atmospheric processes relevant at the canopy level.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Abramowitz, G., Pouyanné, L., & Ajami, H. (2012). On the information
  content of surface meteorology for downward atmospheric long‐wave
  radiation synthesis. Geophysical Research Letters, 39(4).
* Norman, J. M., & Campbell, G. S. (1998). An introduction to
  environmental biophysics. Springer, New York.
* Monteith, J. L., & Unsworth, M. H. (1990). Principles of environmental
  physics. Arnold. SE, London, UK.

"""

__title__ = "Canopy atmospheric processes"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (10.09.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants


# ======================================================================

def vpsat(T):

    """
    Calculates the saturation vapour pressure at a specific temperature
    T as given in Monteith & Unsworth, 1990.

    Arguments:
    ----------
    T: array or float
        temperature [degC]

    Returns:
    --------
    The saturation vapour pressure [kPa] at T.

    """

    return 0.61078 * np.exp(17.27 * T / (T + 237.3))


def slope_vpsat(p):

    """
    Calculates the slope of saturation vapour pressure of water at air
    temperature.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The slope of saturation vapour pressure of water [kPa degC-1].

    """

    return (vpsat(p.Tair + 0.1) - vpsat(p.Tair)) / 0.1


def LH_water_vapour(p):

    """
    Calculates the latent heat of water vapor at air temperature as
    given by eq A5 of Medlyn et al., 2007.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The latent heat of water vapor [J mol-1].

    """

    return (cst.LH2O - 2.365e3 * p.Tair) * cst.MH2O * conv.FROM_MILI


def psychometric(p):

    """
    Calculates the atmospheric psychrometric constant.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The psychrometric constant [kPa degC-1].

    """

    return cst.Cp * p.Patm / LH_water_vapour(p)


def emissivity(p):

    """
    Calculates the emissivity of the atmosphere by deriving it from the
    empirical long-wave down estimate proposed by Abramowitz et al.
    (2012): LWdown = 0.031 * ea + 2.84 * T - 522.5 (W m-2)

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The apparent emissivity at air temperature [unitless].

    """

    # unit conversion
    TairK = p.Tair + conv.C_2_K  # degK

    # actual vapour pressure
    ea = (vpsat(p.Tair) - p.VPD) * conv.MILI  # Pa

    return (0.031 * ea + 2.84 * TairK - 522.5) / (cst.sigma * TairK ** 4.)


def net_radiation(p):

    """
    Calculates net isothermal radiation, i.e. the net radiation that
    would be recieved if object and air temperature were the same
    (Eq 11.14 of Campbell and Norman, 1998.). Having a dependency on
    VPD through the emissivity partly accounts for clouds.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    The leaf net isothermal radiation [W m-2].

    """

    # unit conversions
    TairK = p.Tair + conv.C_2_K  # degK

    # incoming short and long wave radiation
    Rsw = (1. - p.albedo_l) * p.PPFD * conv.PAR_2_SW  # W m-2
    Rlw = emissivity(p) * cst.sigma * TairK ** 4.  # W m-2

    return Rsw + Rlw - p.eps_l * cst.sigma * TairK ** 4.
