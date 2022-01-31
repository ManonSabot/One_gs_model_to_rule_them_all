# -*- coding: utf-8 -*-

"""
Functions related to plant hydraulics, used to calculate xylem
vulnerability to cavitation, hydraulic conductance, and transpiration.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Neufeld et al. (1992). Genotypic variability in vulnerability of leaf
  xylem to cavitation in water-stressed and well-irrigated sugarcane.
  Plant Physiology, 100(2), 1020-1028.
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.

"""

__title__ = "Plant hydraulics module"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (29.11.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators
from scipy.integrate import quad  # integrate on a range

# own modules
from TractLSM import conv, cst  # unit converter & general constants


# ======================================================================

def Weibull_params(p):

    """
    Calculates the two Weibull parameters b and c, using two different
    leaf water potentials that each cause a decline of x% in hydraulic
    conductance. The function looks for two different x% values of P in
    the parameters, with x1 < x2. Such that:
        at x1% ln(1) - ln(1 - x1) = -ln(1 - x1) = ln(2)
        and x2% ln(1) - ln(1 - x2) = -ln(1 - x2) = -ln(1 / 50) = ln(50)

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Returns:
    --------
    b: float
        one of two Weibull parameters [MPa], it is P at k / kmax = 0.37

    c: float
        one of two Weibull parameters [unitless], it controls whether
        the curve has a threshold sigmoidal form or non-threshold
        sigmoidal form

    """

    x = []  # empty array to save the two x%

    try:
        plist = list(p.dtype.names)  # numpy recarray

    except TypeError:
        try:
            plist = list(p.index)  # pandas series

        except TypeError:
            plist = list(p.__dict__.keys())  # python class

    # find the x% and associated Px values
    for param in plist:

        if str(param)[0] == 'P':

            try:
                if (int(str(param)[1:3])) and (len(str(param)) == 3):
                    x += [float(str(param)[1:3])]

            except ValueError:
                pass

    x1 = min(x)  # x1 < x2, at this stage, still a %
    x2 = max(x)
    Px1 = p['P%d' % (x1)]  # get the % value in the class / series
    Px2 = p['P%d' % (x2)]
    x1 /= 100.  # normalise between 0-1
    x2 /= 100.

    try:  # c is derived from both expressions of b
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))

    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))

    # b = Px2 / ((- np.log(1 - x2)) ** (1. / c)), or
    b = Px1 / ((- np.log(1 - x1)) ** (1. / c))

    return b, c


def f(P, b, c):

    """
    Calculates soil and xylem vulnerability curves based on a
    two-parameter Weibull function (Neufeld et al., 1992). In turn, this
    is used to calculate hydraulic conductance and transpiration at
    steady-state.

    Arguments:
    ----------
    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    b: float
        one of two Weibull parameters [MPa], it is P at k / kmax = 0.37

    c: float
        one of two Weibull parameters [unitless], it controls whether
        the curve has a threshold sigmoidal form or non-threshold
        sigmoidal form

    Returns:
    --------
    The vulnerability curves of the the plant [unitless] describing the
    response of stomatal conductance to various leaf water potential
    drop and thus, how likely the plant is to cavitate for a specific
    value of P.

    """

    return np.maximum(cst.zero, np.exp(-(-P / b) ** c))


def transpiration(P, kmax, b, c):

    """
    Calculates the transpiration at steady-state by integrating over all
    vulnerability curves in the soil-plant system, i.e. over the full
    range of water potentials P. The err associated with the quad
    integral function is relatively small and this method is equivalent
    to using a gamma function.

    Arguments:
    ----------
    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    kmax: float
        maximum hydraulic conductance [mmol s-1 m-2 MPa-1] of the plant

    b: float
        one of two Weibull parameters [MPa], it is P at k / kmax = 0.37

    c: float
        one of two Weibull parameters [unitless], it controls whether
        the curve has a threshold sigmoidal form or non-threshold
        sigmoidal form

    Returns:
    --------
    trans: array
        transpiration rate [mol.m-2.s-1]

    """

    trans = np.empty_like(P)  # empty numpy array of right length

    for i in range(len(P)):  # at Ps, trans=0; at Pcrit, trans=transcrit

        trans[i], __ = quad(f, P[i], P[0], args=(b, c))

    trans[trans > cst.zero] *= kmax * conv.FROM_MILI  # mol.s-1.m-2

    return np.maximum(cst.zero, trans)


def hydraulics(p, res='low', Kirchhoff=True, kmax=None, Pcrit=None):

    """
    Calculates the hydraulics used to solve the leaf energy balance.
    These are the pressure drop range from Ps to Pcrit and the
    transpiration. Pcrit is defined as the critical point related to the
    critical conductance below which embolism occurs:
    Pcrit = - b * np.log(kmax / kcrit) ** (1. / c)
        with kcrit a percentage of kmax: kcrit = p.ratiocrit * kmax
    hence:
        Pcrit = - b * np.log(kmax / (p.ratiocrit * kmax)) ** (1. / c)

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    res: string
        either 'low' (default), 'med', or 'high' for the transpiration
        stream

    Kirchhoff: boolean
        is this solved on the full transpiration stream using the
        associated integral / Kirchhoff transform?

    kmax: float
        maximum hydraulic conductance [mmol s-1 m-2 MPa-1] of the plant

    Pcrit: float
        critical leaf water potential [MPa] below which there can be no
        photosynthesis. This sets the Pcrit of the water potential
        array if specified.

    Returns:
    --------
    P: array
        leaf or soil (upstream) water potential [MPa], an array of
        values from the soil water potential Ps to the critical water
        potential Pcrit for which cavitation of the xylem occurs

    trans: array
        transpiration rate [mol m-2 s-1]

    """

    # two Weibull parameters setting the shape of the vuln curves
    b, c = Weibull_params(p)  # MPa, unitless

    if Pcrit is None:  # get the leaf water potentials P
        Pcrit = - b * np.log(1. / p.ratiocrit) ** (1. / c)  # MPa

    else:
        Pcrit = np.minimum(- b * np.log(1. / p.ratiocrit) ** (1. / c), Pcrit)

    if p.Ps <= Pcrit:  # plants cavitate, the optimisation cannot happen
        raise IndexError('critical cavitation, no optimisation possible')

    if res == 'low':
        P = np.linspace(p.Ps, Pcrit, 200)  # MPa, between Ps and Pcrit

    if res == 'med':
        P = np.linspace(p.Ps, Pcrit, 400)  # MPa

    if res == 'high':
        P = np.linspace(p.Ps, Pcrit, 600)  # MPa

    if Kirchhoff:  # integrate the VC on P to get E
        if kmax is not None:
            trans = transpiration(P, kmax, b, c)  # mol m-2 s-1

        else:
            trans = transpiration(P, p.kmax, b, c)  # mol m-2 s-1

        return P, trans

    else:

        return P
