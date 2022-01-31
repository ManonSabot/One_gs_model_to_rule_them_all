# -*- coding: utf-8 -*-

"""
Translates volumetric soil water content to water potential.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

Reference:
-----------
* Clapp, R. B., & Hornberger, G. M. (1978). Empirical equations for some
  soil hydraulic properties. Water resources research, 14(4), 601-604.
* Cosby, B. J., Hornberger, G. M., Clapp, R. B., & Ginn, T. (1984). A
  statistical exploration of the relationships of soil moisture
  characteristics to the physical properties of soils. Water resources
  research, 20(6), 682-690.

"""

__title__ = "Soil water potential formulation"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (12.07.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import cst  # general constants


# ======================================================================

def water_potential(p, sw):

    """
    Calculates the soil water potential [MPa]. The parameters bch and
    Psie are estimated using the Cosby et al. (1984) regression
    relationships from the soil sand/silt/clay fractions.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    sw: float
        volumetric soil water content [m3 m-3]

    Returns:
    --------
    The soil water potential [MPa], Ps, using Clapp and Hornberger
    eq (1978)

    """

    if (sw is not None) and (sw >= cst.zero):
        return p.Psie * (sw / p.theta_sat) ** (- p.bch)

    elif ((sw is not None) and
          np.isclose(abs(p.Ps), 0., rtol=cst.zero, atol=cst.zero)):
        return p.Psie

    else:
        return p.theta_sat * (p.Ps / p.Psie) ** (-1. / p.bch)
