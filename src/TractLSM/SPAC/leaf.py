# -*- coding: utf-8 -*-

"""
Functions related to leaf processes: used to calculate conductances,
leaf temperature, and photosynthesis.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* Collatz et al. (1991). Regulation of stomatal conductance and
  transpiration: a physiological model of canopy processes. Agric. For.
  Meteorol, 54, 107-136.
* De Pury, D. G. G., & Farquhar, G. D. (1997). Simple scaling of
  photosynthesis from leaves to canopies without the errors of big‐leaf
  models. Plant, Cell & Environment, 20(5), 537-557.
* Farquhar, G. D., von Caemmerer, S. V., & Berry, J. A. (1980). A
  biochemical model of photosynthetic CO2 assimilation in leaves of C3
  species. Planta, 149(1), 78-90.
* Jones, H. G. (2013). Plants and microclimate: a quantitative approach
  to environmental plant physiology. Cambridge university press.
* Kirschbaum, M. U. F., & Farquhar, G. D. (1984). Temperature dependence
  of whole-leaf photosynthesis in Eucalyptus pauciflora Sieb. ex Spreng.
  Functional Plant Biology, 11(6), 519-538.
* Leuning, R. (1990). Modelling stomatal behaviour and photosynthesis of
  Eucalyptus grandis. Functional Plant Biology, 17(2), 159-175.
* Norman, J. M., & Campbell, G. S. (1998). An introduction to
  environmental biophysics. Springer, New York.
* Medlyn et al. (2002). Temperature response of parameters of a
  biochemically based model of photosynthesis. II. A review of
  experimental data. Plant, Cell & Environment, 25(9), 1167-1179.
* Medlyn et al. (2007). Linking leaf and tree water use with an
  individual-tree model. Tree Physiology, 27(12), 1687-1699.
* Monteith, J. L., & Unsworth, M. H. (1990). Principles of environmental
  physics. Arnold. SE, London, UK.
* Slatyer, R. O. (1967). Plant-water relationships. Academic Press; New
  York; San Frncisco; London.

"""

__title__ = "Leaf level photosynthetic processes"
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (29.11.2019)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import vpsat, slope_vpsat, LH_water_vapour


# ======================================================================

def conductances(p, Tleaf=None, gs=None, inf_gb=False):

    """
    Both forced and free convection (gHa) contribute to exchange of heat
    and mass through leaf boundary layers at the wind speeds typically
    encountered within plant canopies (< 0-5 m.s-1). Total leaf
    conductance to heat and total leaf conductance to water vapour (or
    simply boundary layer conductance to water vapour) are needed for
    the energy balance. The leaf LAI is used in gHf to adjust for the
    2-leaf model.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Tleaf: array or float
        leaf temperature [degC]

    gs: float
        stomatal conductance to water vapour [mol m-2 s-1]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    gw: float
        total leaf conductance to water vapour [mol m-2 s-1]

    gH: float
        total leaf conductance to heat [mol m-2 s-1]

    gb: float
        boundary layer conductance to water vapour [mol m-2 s-1]

    gr: float
        radiative conductance [mol m-2 s-1]

    """

    # unit conversions
    TairK = p.Tair + conv.C_2_K  # degK
    cmolar = p.Patm * conv.MILI / (cst.R * TairK)  # air molar density

    # Sutherland Eq for dynamic viscosity
    mu = 1.458e-6 * TairK ** 1.5 / (TairK + 110.4)  # Pa s

    # kinematic viscosity
    nu = mu * cst.R * TairK / (p.Patm * cst.Mair)  # m2 s-1
    prandtl = nu / cst.DH  # unitless

    # boundary layer cond to forced convect. (Campbell & Norman, 1998)
    d = 0.72 * p.max_leaf_width  # leaf width, m
    reynolds = p.u * d / nu  # unitless
    gHa = (0.664 * cmolar * cst.DH * (reynolds ** 0.5) * (prandtl ** (1. / 3.))
           / d)

    if Tleaf is None:
        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf

        # arbitrarily force a 1 deg difference
        except (IndexError, AttributeError, ValueError):
            grashof = ((cst.g0 * (1. / TairK) * (d ** 3.)) / (nu ** 2.))

    if Tleaf is not None:
        grashof = ((cst.g0 * (1. / TairK) * abs(Tleaf - p.Tair) * (d ** 3.)) /
                   (nu ** 2.))  # unitless

    else:  # forcing a 1 deg difference
        grashof = ((cst.g0 * (1. / TairK) * (d ** 3.)) / (nu ** 2.))

    # boundary layer cond to free convect. (Campbell & Norman, 1998)
    gHf = (p.LAI * (0.54 * cmolar * cst.DH * ((grashof * prandtl) ** 0.25)) /
           d)
    gHa = np.maximum(cst.zero, 2. * (gHa + gHf))  # units heat

    # boundary layer conductance to water vapour
    if inf_gb:
        gb = conv.MILI  # 1.e3 mol H2O m-2 s-1, large so disappears
        gHa = conv.MILI  # 1.e3 mol heat m-2 s-1, large so disappears

    else:
        try:  # is gb one of the input fields?
            gb = p.gb  # mol H2O m-2 s-1, 1-sided in LICOR
            gHa = gb * 2. * conv.GbhvGb  # mol heat m-2 s-1, 2-sided

        except (IndexError, AttributeError, ValueError):  # calc. gb
            gb = np.maximum(cst.zero, gHa * conv.GbvGbh)

            if np.isclose(gHa, cst.zero, rtol=cst.zero, atol=cst.zero):
                gb = cst.zero  # mol m-2 s-1

    # radiative conductance (Jones, 2013; Medlyn et al., 2007)
    gr = np.maximum(cst.zero, 4. * p.eps_l * cst.sigma * TairK ** 3. / cst.Cp)

    # total two-sided leaf conductance to heat (Medlyn et al., 2007)
    gH = gHa + 2. * gr  # hypostomatous leaf (2-sided)

    if (np.isclose(gHa, cst.zero, rtol=cst.zero, atol=cst.zero) and
       np.isclose(gr, cst.zero, rtol=cst.zero, atol=cst.zero)):
        gH = cst.zero

    if gs is None:

        return gH, gb, gr

    if gs is not None:  # total cond to water vap (Medlyn et al., 2007)
        gw = np.maximum(cst.zero, (gb * gs) / (gb + gs))

        if ((gs < -cst.zero) or np.isclose(gb, cst.zero, rtol=cst.zero,
            atol=cst.zero) or np.isclose(gs, cst.zero, rtol=cst.zero,
           atol=cst.zero)):
            gw = cst.zero

        return gw, gH, gb, gr


def leaf_temperature(p, trans, Tleaf=None, inf_gb=False):

    """
    Calculates the leaf temperature for each supply function, i.e. over
    the transpiration stream. A factor 2 is introduced in the
    denominator because gHa and gr because leaves are two-sided.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array or float
        transpiration rate [mol m-2 s-1]

    Tleaf: array or float
        leaf temperature [degC]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    Tleaf: array or float
        leaf temperature [degC]

    gb: float
        boundary layer conductance to water vapour [mol m-2 s-1]

    """

    # unit conversion
    TairK = p.Tair + conv.C_2_K  # degK

    # get conductances, mol m-2 s-1
    gH, gb, __ = conductances(p, Tleaf=Tleaf, inf_gb=inf_gb)

    # latent heat of water vapor
    Lambda = LH_water_vapour(p)  # J mol-1

    # slope of saturation vapour pressure of water
    slp = slope_vpsat(p)  # kPa degK-1

    # canopy / leaf sensible heat flux
    H = p.Rnet - Lambda * trans  # W m-2

    # simplified Tleaf (gb for gw), eq 14.6 of Campbell & Norman, 1998
    if np.isclose(abs(p.Tair), 0., rtol=cst.zero, atol=cst.zero):
        Tleaf = (p.Tair + H / (cst.Cp * gH * TairK / cst.zero +
                               Lambda * slp * gb / p.Patm))  # degC
    else:
        Tleaf = (p.Tair + H / (cst.Cp * gH * TairK / p.Tair +
                               Lambda * slp * gb / p.Patm))  # degC

    return Tleaf, gb


def leaf_energy_balance(p, trans, Tleaf=None, inf_gb=False):

    """
    Calculates the CO2 diffusive conductance of leaf gc using the
    saturation vapour pressure deficit of water (vpsat) and the stomatal
    conductance to water vapour using the differential between
    the water content of saturated air at leaf temperature and at air
    temperature.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array or float
        transpiration rate [mol m-2 s-1]

    Tleaf: array or float
        leaf temperature [degC]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    gc: array
        leaf CO2 diffusive leaf conductance [mol s-1 m-2]

    gs: array
        stomatal conductance to water vapour [mol m-2 s-1]

    gb: float
        boundary layer conductance to water vapour [mol m-2 s-1]

    ww: array
        plant-air saturated H2O content differential
        [moles(H2O) mole-1(air)]

    """

    if Tleaf is None:
        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf
            __, gb = leaf_temperature(p, trans, Tleaf=Tleaf, inf_gb=inf_gb)

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            Tleaf, gb = leaf_temperature(p, trans, inf_gb=inf_gb)

    else:
        __, gb = leaf_temperature(p, trans, Tleaf=Tleaf, inf_gb=inf_gb)

    try:
        if np.isclose(Tleaf, p.Tleaf):
            Dleaf = p.VPD

        else:
            esat_l = vpsat(Tleaf)  # saturation vapour pressure of water
            esat_a = vpsat(p.Tair)  # saturation vapour pressure of air
            Dleaf = (esat_l - (esat_a - p.VPD))  # leaf-air vpd, kPa

    except (IndexError, AttributeError, ValueError):  # calc. Dleaf
        esat_l = vpsat(Tleaf)  # saturation vapour pressure of water
        esat_a = vpsat(p.Tair)  # saturation vapour pressure of air
        Dleaf = (esat_l - (esat_a - p.VPD))  # leaf-air vpd, kPa

    # total leaf vapour diff. cond, mol H2O s-1 m-2
    gw = p.Patm * trans / Dleaf
    gw[np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero)] = cst.zero

    # gs, stomatal conductance to water vapour
    gs = gb * gw / (gb - gw)  # mol H2O s-1 m-2
    gs[gs < 0.] = cst.zero
    gs[np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero)] = cst.zero

    if inf_gb:  # total leaf diff. cond to CO2, mol CO2 s-1 m-2
        gc = np.maximum(cst.zero, gw * conv.GcvGw)

    else:
        gc = np.maximum(cst.zero,
                        gs * gb / (gb * conv.GwvGc + gs * conv.GbvGbc))

    gc[np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero)] = cst.zero

    # leaf-air H2O vap diff (Slatyer, 1967), moles(H2O) mole-1(air)
    ww = Dleaf / p.Patm

    return gc, gs, gb, ww


def arrhen(v25, Ea, Tref, Tleaf, deltaS=None, Hd=None):

    """
    Calculates the temperature dependence of a kinetic variable using an
    Arrhenius function which transforms the variable at 25 degC given
    its energy of activation and the leaf temperature (Medlyn et al.,
    2002). Providing deltaS and Hd returns a peaked Arrhenius function
    which accounts for the rate of inhibition at higher temperatures.

    Arguments:
    ----------
    v25: float
        kinetic variable at Tref degC [varies]

    Ea: float
        energy of activation of the variable [J mol-1]

    Tref: float
        reference temperature [degC] at which the kinetic and
        activation energy variables are defined, typically 25 deg C

    Tleaf: array or float
        leaf temperature [degC]

    deltaS: float
        entropy factor [J mol-1 K-1]

    Hd: float
        rate of decrease about the optimum temperature [J mol-1]

    Returns:
    --------
    The temperature-dependent kinetic variable [varies].

    """

    # unit conversion
    Tl_K = Tleaf + conv.C_2_K  # degK
    arrhenius = v25 * np.exp(Ea * (Tl_K - Tref) / (Tref * cst.R * Tl_K))

    if (deltaS is None) or (Hd is None):
        return arrhenius

    else:
        arg2 = 1. + np.exp((deltaS * Tref - Hd) / (cst.R * Tref))
        arg3 = 1. + np.exp((deltaS * Tl_K - Hd) / (cst.R * Tl_K))

        return arrhenius * arg2 / arg3


def adjust_low_T(var, Tleaf, lower_bound=0., upper_bound=10.):

    """
    Function linearly forcing a variable to zero at low temperature

    Arguments:
    ----------
    var: float or array
        kinetic variable [varies]

    Tleaf: array or float
        leaf temperature [degC]

    lower_bound: float
        lowest possible leaf temperature [degC]

    upper_bound: float
        upper "lower" leaf temperature [degC]

    Returns:
    --------
    The temperature-dependent kinetic variable [varies].

    """

    if 'float' in str(type(Tleaf)):
        if Tleaf < lower_bound:
            var = 0.

        elif Tleaf < upper_bound:
            var *= (Tleaf - lower_bound) / (upper_bound - lower_bound)

    else:
        if np.any(Tleaf < lower_bound):
            var[Tleaf < lower_bound] = 0.

        if np.any(Tleaf < upper_bound):
            low = np.logical_and(Tleaf < upper_bound, Tleaf > lower_bound)

            try:
                var[np.where(low)] *= ((Tleaf[np.where(low)] - lower_bound) /
                                       (upper_bound - lower_bound))

            except TypeError:
                var[low] *= ((Tleaf[low] - lower_bound) /
                             (upper_bound - lower_bound))

    return var


def quad(a, b, c, large_root=True):

    """
    Calculates the square root given by the quadratic formula,
        with a, b, and c from ax2 + bx + c = 0.

    Arguments:
    ----------
    a, b, c: float
        coefficients of the equation to solve

    large_root: boolean
        if True, the largest root is returned

    Returns:
    --------
    Either one of the large or small roots given by the quadratic
    formula.

    """

    if large_root:
        return 0.5 * (-b + (b ** 2. - 4. * a * c) ** 0.5) / a

    else:
        return 0.5 * (-b - (b ** 2. - 4. * a * c) ** 0.5) / a


def quad_solve_Ci(p, Cs, gs_over_A, Rleaf, gamstar, v1, v2):

    """
    Solves for Ci starting from Cs, according to the standard quadratic
    way of solving for Ci as described in Leuning, 1990.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Cs: float
        leaf surface CO2 concentration [Pa]

    gs_over_A: float
        gs/A as predicted by the USO (Medlyn, 2011) model

    Rleaf: float
        leaf day respiration [μmol m-2 s-1]

    gamstar: float
        CO2 compensation point [Pa]

    v1: float
        Vmax or J

    v2: float
        Km or 2 * gamstar

    Returns:
    --------
    The intercellular CO2 concentration Ci [Pa].

    """

    # unit conversions, from Pa to μmol mol-1
    Csi = Cs * conv.MILI / p.Patm
    gammastar = gamstar * conv.MILI / p.Patm
    V2 = v2 * conv.MILI / p.Patm

    g0 = 1.e-9  # removing g0 introduces a solving error

    a = g0 + gs_over_A * (v1 - Rleaf)
    b = ((1. - Csi * gs_over_A) * (v1 - Rleaf) + g0 * (V2 - Csi) - gs_over_A *
         (v1 * gammastar + V2 * Rleaf))
    c = - ((1. - Csi * gs_over_A) * (v1 * gammastar + V2 * Rleaf) + g0 * V2 *
           Csi)

    ref_root = quad(a, b, c) * p.Patm * conv.FROM_MILI

    if (ref_root > Cs) or (ref_root < cst.zero):
        return quad(a, b, c, large_root=False) * p.Patm * conv.FROM_MILI

    else:
        return ref_root


def calc_colim_Ci(p, Cs, Tleaf, photo):

    """
    Solves for Ci by assuming colimitation of Aj and Ac.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Cs: float
        leaf surface CO2 concentration [Pa]

    Tleaf: float
        leaf temperature [degC]

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    The intercellular CO2 concentration Ci [Pa].

    """

    # gamstar, Vmax, Kc and Ko are known at Tref, get their T dependency
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC

    # CO2 compensation point
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    # max carbohylation and electron transport rates
    Vmax = arrhen(p.Vmax25, p.Ev, Tref, Tleaf, deltaS=p.deltaSv, Hd=p.Hdv)
    Jmax = arrhen(p.JV * p.Vmax25, p.Ej, Tref, Tleaf, deltaS=p.deltaSj,
                  Hd=p.Hdj)

    # adjust at low temperatures
    Vmax = adjust_low_T(Vmax, Tleaf)
    Jmax = adjust_low_T(Jmax, Tleaf)

    # Michaelis-Menten constants
    Kc = arrhen(p.Kc25, p.Ec, Tref, Tleaf)  # cst for carboxylation, Pa
    Ko = arrhen(p.Ko25, p.Eo, Tref, Tleaf)  # cst for oxygenation, kPa
    Ko = np.maximum(cst.zero, Ko)  # we don't want zeros in Km div

    # Michaelis-Menten constant for O2/CO2
    Km = Kc * (1. + p.O2 / Ko)

    # no scaling from single leaf to canopy, scale2can set to 1
    # Vmax *= p.scale2can
    # Jmax *= p.scale2can

    # electron transport-limited photosynthesis rate
    if photo == 'Farquhar':  # De Pury & Farquhar, 1997
        J = quad(p.c1, -((1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD +
                 Jmax), (1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD * Jmax,
                 large_root=False)  # μmol m-2 s-1
        J *= 0.25  # account for RuBP regeneration rate

    else:  # Collatz et al. 1991
        J = (1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD

    return (2. * Vmax * gamstar - J * Km) / (J - Vmax)


def calc_photosynthesis(p, trans, Ci_s, photo, smooth=True, Tleaf=None,
                        Rleaf=None, gs_over_A=None, gsc=None, Vmax25=None,
                        inf_gb=False):

    """
    Calculates the assimilation rate given the internal leaf CO2
    concentration following either the classic Farquhar photosynthesis
    model (with smoothed solve) or the Collatz model. The non-smoothed
    alternative would be An = min(Aj, Ac) - Rleaf.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    trans: array or float
        transpiration rate [mol m-2 s-1]

    Ci_s: array or float
        intercellular CO2 concentration [Pa] (the leaf surface CO2
        concentration [Pa] can be parsed instead with gs_over_A as well
        to solve for Ci)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    smooth: boolean
        for the Sperry model to accurately solve for Ci, the transition
        point between Aj and Ac must be smoothed. True is the default

    Tleaf: float
        leaf temperature [degC]

    Rleaf: float
        leaf day respiration [μmol m-2 s-1]

    gs_over_A: float
        gs/A as predicted by USO (Medlyn, 2011) model. Used for the
        quadratic solving of Ci

    gsc: float
        stomatal conductance to carbon [μmol m-2 s-1]

    Vmax25: array or float
        maximum carboxylation rate at 25 degrees [μmol m-2 s-1]

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    A: array or float
        net C assimilation rate [μmol m-2 s-1]

    Aj: array or float
        electron transport-limited photosynthesis rate [μmol m-2 s-1]

    Ac: array or float
        rubisco-limited photosynthesis rate [μmol m-2 s-1]

    Ci: array of float
        intercellular CO2 concentration [Pa]

    """

    # initialise Ci
    Ci = Ci_s

    # should we account for gs in Ac and Aj calculations?
    given_gs = False

    if gsc is not None:
        if gsc > 0.:
            given_gs = True  # for the empirical SOX model

    if Tleaf is None:
        try:  # is Tleaf one of the input fields?
            Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            Tleaf, __ = leaf_temperature(p, trans, inf_gb=inf_gb)

    # gamstar, Vmax, Kc and Ko are known at Tref, get their T dependency
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC

    # CO2 compensation point
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    if Vmax25 is None:  # max carbohylation and electron transport rates
        Vmax = arrhen(p.Vmax25, p.Ev, Tref, Tleaf, deltaS=p.deltaSv, Hd=p.Hdv)
        Jmax = arrhen(p.JV * p.Vmax25, p.Ej, Tref, Tleaf, deltaS=p.deltaSj,
                      Hd=p.Hdj)

    else:
        Vmax = arrhen(Vmax25, p.Ev, Tref, Tleaf, deltaS=p.deltaSv, Hd=p.Hdv)
        Jmax = arrhen(p.JV * Vmax25, p.Ej, Tref, Tleaf, deltaS=p.deltaSj,
                      Hd=p.Hdj)

    # adjust for low temperatures
    Vmax = adjust_low_T(Vmax, Tleaf)
    Jmax = adjust_low_T(Jmax, Tleaf)

    if Rleaf is None:  # leaf day respiration
        Rleaf = Vmax * 0.015  # μmol m-2 s-1

    # Michaelis-Menten constants
    Kc = arrhen(p.Kc25, p.Ec, Tref, Tleaf)  # cst for carboxylation, Pa
    Ko = arrhen(p.Ko25, p.Eo, Tref, Tleaf)  # cst for oxygenation, kPa
    Ko = np.maximum(cst.zero, Ko)  # we don't want zeros in Km div

    # Michaelis-Menten constant for O2/CO2
    Km = Kc * (1. + p.O2 / Ko)

    # no scaling from single leaf to canopy, scale2can set to 1
    # Rleaf *= p.scale2can
    # Vmax *= p.scale2can
    # Jmax *= p.scale2can

    # rubisco-limited photosynthesis rate (De Pury & Farquhar, 1997)
    if gs_over_A is not None:
        Ci = quad_solve_Ci(p, Ci_s, gs_over_A, Rleaf, gamstar, Vmax, Km)
        Ci_c = Ci  # track Ci to know which one was used

    if given_gs:
        Ac = quad(1., Rleaf - Vmax - (Ci + Km) * gsc / (p.Patm * conv.MILI),
                  (Vmax * (Ci - gamstar) - (Ci + Km) * Rleaf) * gsc / (p.Patm *
                  conv.MILI), large_root=False) + Rleaf  # μmol m-2 s-1

    else:
        try:
            if (Ci <= cst.zero) or (Ci > Ci_s) or np.isnan(Ci):
                Ac = 0.
                Ci_c = 0.

            else:
                Ac = Vmax * (Ci - gamstar) / (Ci + Km)  # μmol m-2 s-1

        except ValueError:
            Ac = Vmax * (Ci - gamstar) / (Ci + Km)  # μmol m-2 s-1

    # electron transport-limited photosynthesis rate
    if photo == 'Farquhar':  # De Pury & Farquhar, 1997
        J = quad(p.c1, -((1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD +
                 Jmax), (1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD * Jmax,
                 large_root=False)  # μmol m-2 s-1
        J *= 0.25  # account for RuBP regeneration rate

    else:  # Collatz et al. 1991
        J = (1. - p.tau_l - p.albedo_l) * p.alpha * p.PPFD

    if gs_over_A is not None:
        Ci = quad_solve_Ci(p, Ci_s, gs_over_A, Rleaf, gamstar, J, 2. * gamstar)

    if given_gs:
        Aj = (quad(1., Rleaf - J - (Ci + 2. * gamstar) * gsc / (p.Patm *
                   conv.MILI), (J * (Ci - gamstar) - (Ci + 2. * gamstar) *
                   Rleaf) * gsc / (p.Patm * conv.MILI), large_root=False) +
              Rleaf)

    else:
        Aj = J * (Ci - gamstar) / (Ci + 2. * gamstar)  # μmol m-2 s-1

        try:  # below light compensation point?
            if (Ci - gamstar <= cst.zero) or (Ci > Ci_s) or np.isnan(Ci):
                if gs_over_A is not None:
                    Ci = Ci_s  # reinitialise Ci and solve using Cs
                    Aj = J * (Ci - gamstar) / (Ci + 2. * gamstar)

                else:
                    Aj = 0.

        except ValueError:
            pass

    if smooth:  # smooth transition point (Kirschbaum & Farquhar, 1984)
        if photo == 'Farquhar':
            An = quad(p.c2, -(Aj + Ac), Aj * Ac, large_root=False) - Rleaf

        else:
            An = quad(p.c4, -(Aj + Ac), Aj * Ac, large_root=False) - Rleaf

    else:  # non-smoothed transition pt between Aj and Ac
        An = min(Aj, Ac) - Rleaf

    if gs_over_A is not None:
        Rublim = rubisco_limit(Aj, Ac)

        if Rublim == 'True':
            Ci = Ci_c  # the system is Rubisco-limited, this Ci

        return An, Aj, Ac, Ci

    else:

        return An, Aj, Ac


def rubisco_limit(Aj, Ac):

    """
    Tests whether the standard model for photosynthesis is rubisco
    limited or not, in which case it is limited by electron transport.

    Arguments:
    ----------
    Aj: float
        electron transport-limited photosynthesis rate [μmol m-2 s-1]

    Ac: float
        rubisco-limited photosynthesis rate [μmol m-2 s-1]

    Returns:
    --------
    'True' if the C assimilation is rubisco limited, 'False' otherwise.

    """

    if (np.minimum(Ac, Aj) > 0.) and np.isclose(np.minimum(Ac, Aj), Ac):
        return str(bool(1))

    elif (np.minimum(Ac, Aj) > 0.) and np.isclose(np.minimum(Ac, Aj), Aj):
        return str(bool(0))

    else:
        return 0.
