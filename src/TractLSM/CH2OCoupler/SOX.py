# -*- coding: utf-8 -*-

"""
Both versions of the SOX model (optimisation version and numerical
approximation), adapted from the hydraulic-limited stomatal
optimization model of Eller et al. (2018) and Eller et al. (2020).

This file is part of the TractLSM model.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Eller, C. B., Rowland, L., Oliveira, R. S., Bittencourt, P. R.,
  Barros, F. V., Da Costa, A. C., ... & Cox, P. (2018).
  Modelling tropical forest responses to drought and El Niño with a
  stomatal optimization model based on xylem hydraulics. Philosophical
  Transactions of the Royal Society B: Biological Sciences, 373(1760),
  20170315.
* Eller, C. B., Rowland, L., Mencuccini, M., Rosas, T., Williams, K.,
  Harper, A., ... & Cox, P. M. (2020). Stomatal optimization based on
  xylem hydraulics (SOX) improves land surface model simulation of
  vegetation responses to climate. New Phytologist, 226(6), 1622-1637.

"""

__title__ = "Stomatal optimization based on xylem hydraulics algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (27.04.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, dkcost, kcost
from TractLSM.SPAC import leaf_temperature, calc_photosynthesis
from TractLSM.SPAC import calc_colim_Ci, rubisco_limit
from TractLSM.SPAC.leaf import arrhen
from TractLSM.CH2OCoupler import calc_trans


# ======================================================================

def Ci_stream(p, Cs, Tleaf, res):

    """
    Creates arrays of possible Ci over which to solve the optimization
    criterion.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    Cs: float
        CO2 concentration at the leaf surface [Pa]

    Tleaf: float
        leaf temperature [degC]

    res: string
        either 'low' (default), 'med', or 'high' to solve for Ci

    Returns:
    --------
    An array of all potential Ci [Pa] values (Ci values can be anywhere
    between a lower bound and Cs) at a resolution chosen by the user.

    """

    # CO2 compensation point
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    # declare all potential Ci values
    if res == 'low':
        NCis = 500

    if res == 'med':
        NCis = 2000

    if res == 'high':
        NCis = 8000

    return np.linspace(gamstar, Cs, NCis, endpoint=False)


def supply_max(p, photo='Farquhar', case=1, res='low', iter_max=40,
               threshold_conv=0.1, inf_gb=False, deriv=False):

    """
    Finds the instateneous stomatal optimization based on xylem
    hydraulics, following the optimization criterion for which, at each
    instant in time, the stomata regulate canopy gas exchange and
    pressure to achieve the maximum photosynthesis downregulated by a
    hydraulic cost function.

    Arguments:
    ----------
    p: recarray object or pandas series or class containing the data
        time step's met data & params

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    case: int
        version of the model to run, '1' runs the analytical
        approximation, '2' runs the full optimization model

    res: string
        either 'low' (default), 'med', or 'high' to solve for Ci

    iter_max: int
        maximum number of iterations allowed on the leaf temperature
        before reaching the conclusion that the system is not energy
        balanced

    threshold_conv: float
        convergence threshold for the new leaf temperature to be in
        energy balance

    inf_gb: bool
        if True, gb is prescrived and very large

    deriv: bool
        if True, uses the derivative form of the optimality criterion

    Returns:
    --------
    An: float
        net photosynthetic assimilation rate [umol m-2 s-1]

    Ci: float
        intercellular CO2 concentration [Pa]

    rublim: bool
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    trans: float
        transpiration rate [mmol m-2 s-1]

    gs: float
        stomatal conductance [mol m-2 s-1]

    gb: float
        leaf boundary layer conductance [mol m-2 s-1]

    new_Tleaf: float
        leaf temperature [degC]

    Pleaf: float
        leaf water potential [MPa]

    """

    # initial state
    Cs = p.CO2  # Pa
    Dleaf = p.VPD  # kPa

    try:   # is Tleaf one of the input fields?
        Tleaf = p.Tleaf

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf = p.Tair  # deg C

    if case == 1:  # hydraulics
        Pleaf_sat = p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA
        P, E = hydraulics(p, res=res, kmax=p.kmaxS1)

    else:
        Pref, Eref = hydraulics(p, res=res, kmax=p.kmaxS2)

    # iter on the solution until it is stable enough
    iter = 0

    while True:

        if case == 1:  # analytical solution, assuming colimitation
            Cicol = calc_colim_Ci(p, Cs, Tleaf, photo)
            dCi = Cs - Cicol  # Pa

            # calculate dA, μmol m-2 s-1
            As, __, __ = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                             gsc=0.)
            Acol, __, __ = calc_photosynthesis(p, 0., Cicol, photo,
                                               Tleaf=Tleaf, gsc=0.)
            dA = As - Acol  # ambient - colimitation

            # dAdCi (in mol H2O) is needed to calculate gs, mmol m-2 s-1
            dAdCi = dA * conv.GwvGc * p.Patm / dCi

            # dcostdP is needed to calculate gs
            cost_pd, cost_mid = dkcost(p, Pleaf_sat)  # unitless
            dcost = cost_pd - cost_mid
            dP = 0.5 * (Pleaf_sat + p.P50)  # MPa, /!\ sign of P50

            # xi, loss of xylem cost of stomatal opening, mmol m-2 s-1
            dq = Dleaf / p.Patm  # unitless, equivalent to D / Patm
            Xi = 2. * p.kmaxS1 * (cost_pd ** 2.) * dP / (dq * dcost)

            # calculate gs at the co-limitation point, mmol m-2 s-1
            gscol = Acol * conv.GwvGc * p.Patm / dCi

            # calculate gs, mol m-2 s-1
            if dAdCi <= 0.:
                gs = gscol * conv.FROM_MILI

            else:
                gs = 0.5 * dAdCi * conv.FROM_MILI * (((1. + 4. * Xi / dAdCi)
                                                      ** 0.5) - 1.)

            # calculate An, Ci
            An, Aj, Ac = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                             gsc=conv.U * conv.GcvGw * gs)
            Ci = Cs - p.Patm * conv.FROM_MILI * An / (gs * conv.GcvGw)  # Pa

        else:  # optimization model, on a stream of possible Ci values
            Cis = Ci_stream(p, Cs, Tleaf, res)

            # rate of photosynthesis, μmol m-2 s-1
            A, __, __ = calc_photosynthesis(p, 0., Cis, photo, Tleaf=Tleaf)

            # gb?
            __, gb = leaf_temperature(p, 0., Tleaf=Tleaf, inf_gb=inf_gb)

            if inf_gb or (iter < 1):  # gas-exchange trans, mol m-2 s-1
                E = A * conv.FROM_MILI * conv.GwvGc * Dleaf / (p.CO2 - Cis)

            else:
                E = (A * conv.FROM_MILI * (gb * conv.GwvGc + gs * conv.GbvGbc)
                     / (gs + gb) * Dleaf / (p.CO2 - Cis))

            # Pleaf, kcost
            P = np.array([Pref[np.nanargmin(np.abs(e - Eref))] for e in E])
            mask = np.logical_and(E > cst.zero, P >= Pref[-1])
            cost = kcost(p, P[mask])

            # optimal point
            iopt = np.argmax(cost * A[mask])

            if deriv:
                iopt = np.argmin(np.abs(np.gradient(cost * A[mask],
                                                    Cis[mask])))

            Ci = Cis[mask][iopt]

            # get net rate of photosynthesis at optimum, μmol m-2 s-1
            An, Aj, Ac = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf)

            # get associated gc and gs
            gc = p.Patm * conv.FROM_MILI * An / (p.CO2 - Ci)

            if inf_gb:
                gs = gc * conv.GwvGc

            else:
                gs = np.maximum(cst.zero,
                                gc * gb * conv.GwvGc / (gb - conv.GbvGbc * gc))

        # calculate new trans, gw, gb, mol.m-2.s-1
        trans, real_zero, gw, gb, Dleaf = calc_trans(p, Tleaf, gs,
                                                     inf_gb=inf_gb)

        try:  # is Tleaf one of the input fields?
            new_Tleaf = p.Tleaf

        except (IndexError, AttributeError, ValueError):  # calc. Tleaf
            new_Tleaf, __ = leaf_temperature(p, trans, Tleaf=Tleaf,
                                             inf_gb=inf_gb)

        # update Cs (Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb)
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))

        if (np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) or
            np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero) or
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero)):
            Dleaf = p.VPD  # kPa

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter >= 1) and
            real_zero and (abs(Tleaf - new_Tleaf) <= threshold_conv) and not
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate on leaf temperature
        Tleaf = new_Tleaf
        iter += 1

    if case == 1:  # calc. leaf water potential
        Pleaf = P[np.nanargmin(np.abs(E - trans))]

    else:
        Pleaf = P[mask][iopt]

    # rubisco- or electron transport-limitation?
    rublim = rubisco_limit(Aj, Ac)

    if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
        (An > 0.)) or np.isclose(Ci, 0., rtol=cst.zero, atol=cst.zero) or
        (Ci < 0.) or np.isclose(Ci, p.CO2, rtol=cst.zero, atol=cst.zero) or
        (Ci > p.CO2) or (real_zero is None) or (not real_zero) or
       any(np.isnan([An, Ci, trans, gs, new_Tleaf, Pleaf]))):
        An, Ci, trans, gs, gb, new_Tleaf, Pleaf = (9999.,) * 7

    elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
        trans *= conv.MILI  # mmol.m-2.s-1

    return An, Ci, rublim, trans, gs, gb, new_Tleaf, Pleaf
