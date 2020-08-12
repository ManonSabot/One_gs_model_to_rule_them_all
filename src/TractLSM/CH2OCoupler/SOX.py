# -*- coding: utf-8 -*-

"""
The profit maximisation algorithm (between carbon gain and hydraulic
cost), adapted from Sperry et al. (2017)'s hydraulic-limited stomatal
optimization model.

This file is part of the TractLSM model.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
----------
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.

"""

__title__ = "Profit maximisation algorithm"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (29.11.2017)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import sys  # check for version on the system
import numpy as np  # array manipulations, math operators
import bottleneck as bn  # faster C-compiled np for all nan operations

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC.leaf import arrhen
from TractLSM.SPAC import kcost, leaf_temperature, vpsat
from TractLSM.SPAC import calc_colim_Ci, calc_photosynthesis, rubisco_limit
from TractLSM.CH2OCoupler import calc_trans


# ======================================================================

def Ci_stream(p, Cs, Tleaf, res):

    # CO2 compensation point
    Tref = p.Tref + conv.C_2_K  # degk, Tref set to 25 degC
    gamstar = arrhen(p.gamstar25, p.Egamstar, Tref, Tleaf)

    # declare all potential Ci values
    if res == 'low':
        NCis = 500

    if res == 'med':
        NCis = 8000

    if res == 'high':
        NCis = 50000

    return np.linspace(gamstar, Cs, NCis, endpoint=False)


def supply_max(p, photo='Farquhar', case=1, res='low', threshold_conv=0.1,
               iter_max=40, inf_gb=False):

    """
    Finds the instateneous profit maximization, following the
    optmization criterion for which, at each instant in time, the
    stomata regulate canopy gas exchange and pressure to achieve the
    maximum profit, which is the maximum difference between the
    normalized photosynthetic gain (gain) and the hydraulic cost
    function (cost). That is when d(gain)/dP = d(cost)/dP.

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

    window: float
        solving window around the last point of optimisation

    onopt: boolean
        if True, the optimisation is performed. If Fall, fall back on
        previously performed optimisation for the value of the maximum
        profit.

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    E_can: float
        transpiration [mmol m-2 s-1] at maximum profit across leaves

    gs_can: float
        stomatal conductance [mol m-2 s-1] at maximum profit across
        leaves

    An_can: float
        net photosynthetic assimilation rate [umol m-2 s-1] at maximum
        profit across leaves

    Ci_can: float
        intercellular CO2 concentration [Pa] at maximum profit across
        leaves

    rublim_can: string
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    """

    # initial state
    Cs = p.CO2  # Pa
    Tleaf = p.Tair  # deg C
    Dleaf = p.VPD  # kPa

    # saturation vapour pressure of water at Tair, kPa
    esat_a = vpsat(p.Tair)

    # hydraulics
    Pleaf_pd = p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA

    # iter on the solution until it is stable enough
    iter = 0

    while True:

        if case ==1:  # assuming colimitation
            Cicol = calc_colim_Ci(p, Cs, Tleaf, photo)
            dCi = Cs - Cicol  # Pa

            # calculate dA, μmol m-2 s-1
            As, __, __ = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                             gsc=0.)
            Acol, __, __ = calc_photosynthesis(p, 0., Cicol, photo, Tleaf=Tleaf,
                                               gsc=0.)
            dA = As - Acol  # ambient - colimitation

            # dAdCi (in mol H2O) is needed to calculate gs, mmol-1 m-2 s-1
            dAdCi = dA * conv.GwvGc / (dCi * conv.FROM_kPa)

            # kcost, unitless
            cost_pd, __ = kcost(p, Pleaf_pd, Pleaf_pd)
            cost_mid, __ = kcost(p, -p.P50, Pleaf_pd)
            dkcost = cost_pd - cost_mid

            # dkcostdP is needed to calculate gs
            dP = 0.5 * (Pleaf_pd + p.P50)  # MPa, /!\ sign of P50
            dkcostdP = dkcost / dP * 1. / cost_pd  # MPa-1

            # xi, the loss of xylem cost of stomatal opening, mmol m-2 s-1
            dq = Dleaf * conv.FROM_kPa  # unitless, equivalent to D / Patm
            Xi = 2. * p.kmaxS1 * (cost_pd ** 2.) * dP / (dq * dkcost)

            # calculate gs at the co-limitation point, mmol m-2 s-1
            gscol = Acol * conv.GwvGc * p.Patm / dCi

            # calculate gs, mol m-2 s-1
            if dAdCi <= 0.:
                gs = gscol * conv.FROM_MILI

            else:
                gs = 0.5 * dAdCi * conv.FROM_MILI * (((1. + 4. * Xi / dAdCi)
                                                      ** 0.5) - 1.)

        else:  # retrieve the Ci stream of possible Ci values
            Cis = Ci_stream(p, Cs, Tleaf, res)

            # gross rate of photosynthesis, μmol m-2 s-1
            A, __, __ = calc_photosynthesis(p, 0., Cis, photo, Tleaf=Tleaf,
                                            Rleaf=0.)

            # trans (perfect coupling) for the hydraulic cost, mmol m-2 s-1
            E = A * conv.GwvGc * Dleaf / (p.CO2 - Cis)

            # kcost, Pleaf
            cost, P = kcost(p, Pleaf_pd - E / p.ksc_prev, Pleaf_pd)

            # optimal point
            A = A[:len(cost)] # shortening avoids multiple kcost = 0
            iopt = np.argmax(cost * A)
            Ci = Cis[iopt]

            # get net rate of photosynthesis at optimum, μmol m-2 s-1
            An, Aj, Ac = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf)

            # get associated gc, gb, gs (mol m-2 s-1)
            gc = p.Patm * conv.FROM_MILI * An / (p.CO2 - Ci)
            __, gb = leaf_temperature(p, 0., Tleaf=Tleaf, inf_gb=inf_gb)
            gs = np.maximum(cst.zero,
                            gb * conv.GwvGc * gc / (gb - conv.GwvGc * gc))

        # new associated transpiration (Penman-Monteith), Tleaf
        trans, real_zero, gw, gb = calc_trans(p, Tleaf, gs, inf_gb=inf_gb)
        new_Tleaf, __ = leaf_temperature(p, trans, Tleaf=Tleaf, inf_gb=inf_gb)
        
        if case == 1:  # calculate gc, An, Ci
            gc = np.maximum(cst.zero, conv.GcvGw * (gb * gs) / (gb + gs))
            An, Aj, Ac = calc_photosynthesis(p, 0., Cs, photo, Tleaf=p.Tair,
                                             gsc=conv.U * conv.GcvGw * gs)
            Ci = Cs - p.Patm * conv.FROM_MILI * An / (conv.GcvGw * gs)  # Pa

        # new Cs (in Pa)
        boundary_CO2 = (conv.ref_kPa * conv.FROM_MILI * An /
                        (gb * conv.GbcvGb + gs * conv.GcvGw))
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))

        if (np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) or
            np.isclose(gc, cst.zero, rtol=cst.zero, atol=cst.zero) or
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero)):
            Dleaf = p.VPD  # kPa

        else:
            esat_l = vpsat(new_Tleaf)  # vpsat at new Tleaf, kPa
            Dleaf = (esat_l - (esat_a - p.VPD))  # leaf-air vpd, kPa

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter > 1) and
            real_zero and (abs(Tleaf - new_Tleaf) <= threshold_conv) and not
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate on leaf temperature
        Tleaf = new_Tleaf
        iter += 1

    if case == 1:  # infer leaf water potential, MPa
        Pleaf = Pleaf_pd - trans * conv.MILI / (p.kmaxS1 * cost_pd)

    else:
        Pleaf = P[iopt]
        ksc_prev = p.kmaxS2 * cost[iopt]

    rublim = rubisco_limit(Aj, Ac)  # lim?

    if ((np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero) and
        (An > 0.)) or np.isclose(Ci, 0., rtol=cst.zero, atol=cst.zero) or
        (Ci < 0.) or np.isclose(Ci, p.CO2, rtol=cst.zero, atol=cst.zero) or
        (Ci > p.CO2) or (real_zero is None) or (not real_zero) or
       any(np.isnan([An, Ci, trans, gs, new_Tleaf, Pleaf]))):
        An, Ci, trans, gs, gb, new_Tleaf, Pleaf = (9999.,) * 7

    elif not np.isclose(trans, cst.zero, rtol=cst.zero, atol=cst.zero):
        trans *= conv.MILI  # mmol.m-2.s-1

    if case == 1:

        return An, Ci, rublim, trans, gs, gb, new_Tleaf, Pleaf

    else:

        return An, Ci, rublim, trans, gs, gb, new_Tleaf, Pleaf, ksc_prev
