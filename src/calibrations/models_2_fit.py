# -*- coding: utf-8 -*-

"""
Stripped version of all the stomatal optimization models and the
empirical stomatal models compared. Only the bare minimum need to return
gs is kept, and the different models are all assembled under the same
functions, with switches used to navigate from one to the other.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
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
* Medlyn et al. (2011). Reconciling the optimal and empirical approaches
  to modelling stomatal conductance. Global Change Biology, 17(6),
  2134-2144.
* Prentice et al. (2014). Balancing the costs of carbon gain and water
  transport: testing a new theoretical framework for plant functional
  ecology. Ecology letters, 17(1), 82-91.
* Sperry et al. (2017). Predicting stomatal responses to the environment
  from the optimization of photosynthetic gain and hydraulic cost.
  Plant, cell & environment, 40(6), 816-830.
* Tuzet et al. (2003). A coupled model of stomatal conductance,
  photosynthesis and transpiration. Plant, Cell & Environment, 26(7),
  1097-1116.
* Wang et al. (2020). A theoretical and empirical assessment of stomatal
  optimization modeling. New Phytologist, 227(2), 311-325.
* Wolf et al. (2016). Optimal stomatal behavior with competition for
  water and risk of hydraulic impairment. Proceedings of the National
  Academy of Sciences, 113(46), E7222-E7230.

"""

__title__ = "Minimal (calibration) version of the gs models"
__author__ = "Manon E. B. Sabot"
__version__ = "8.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import hydraulics, leaf_energy_balance, leaf_temperature
from TractLSM.SPAC import calc_colim_Ci, calc_photosynthesis
from TractLSM.SPAC import fwWP, fLWP, phiLWP
from TractLSM.SPAC import fPLC, hydraulic_cost, dkcost, kcost, dcost_dpsi
from TractLSM.CH2OCoupler import Ci_sup_dem, calc_trans, A_trans
from TractLSM.CH2OCoupler.SOX import Ci_stream
from TractLSM.CH2OCoupler.ProfitMax import photo_gain
from TractLSM.CH2OCoupler.LeastCost import Vmax_T

import warnings  # ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def floop(p, model, photo='Farquhar', inf_gb=True):

    """
    Returns gs depending on the model called and a set of forcing
    conditions.

    Arguments:
    ----------
    p: recarray object or pandas series
        time step's met data & params

    model: string
        model to run. The choices are: 'Eller' (i.e., SOX empirical),
        'Medlyn', 'Tuzet', or 'SOX-OPT' (i.e., the optimization version
        of SOX)

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    gs: float
        stomatal conductance [mmol m-2 s-1]

    Pleaf: float
        leaf water potential [MPa], only for the Tuzet model which
        iterates over Pleaf a few times.

    """

    # initialize the system
    Dleaf = p.VPD  # kPa
    Cs = p.CO2  # Pa

    if model == 'Tuzet':
        iter_min = 2  # needed to update the fw with the LWP

    else:
        iter_min = 1

    try:  # is Tleaf one of the input fields?
        Tleaf = p.Tleaf
        iter_max = 0

        if model == 'Tuzet':
            iter_max = 2  # needed to update the LWP

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf = p.Tair  # deg C
        iter_max = 40

    if model == 'Medlyn':
        Dleaf = np.maximum(0.05, Dleaf)  # gs model not valid < 0.05

        if iter_max > 0:
            fw = 1.  # no moisture stress

        else:
            fw = fwWP(p, p.Ps)  # moisture stress function

    elif model == 'Tuzet':
        fw = fLWP(p, p.LWP_ini)  # stress factor
        P, trans = hydraulics(p, kmax=p.kmaxT)  # hydraulics

    elif model == 'Eller':
        Pleaf_sat = p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA

    else:  # SOX-OPT model
        P, trans = hydraulics(p, kmax=p.kmaxS2)  # hydraulics

    if (model == 'Medlyn') or (model == 'Tuzet'):  # init. gs over A
        g0 = 1.e-9  # g0 ~ 0, removing it entirely introduces errors
        Cs_umol_mol = Cs * conv.MILI / p.Patm  # umol mol-1

        if model == 'Medlyn':
            gsoA = g0 + (1. + p.g1 * fw / (Dleaf ** 0.5)) / Cs_umol_mol

        else:  # Tuzet
            gsoA = g0 + p.g1T * fw * Cs_umol_mol

    # iter on the solution until it is stable enough
    iter = 0

    while True:

        if model == 'Eller':
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
            dP = 0.5 * (Pleaf_sat + p.P50)  # MPa,  /!\ sign of P50

            # xi, loss of xylem cost of stomatal opening, mmol m-2 s-1
            dq = Dleaf / p.Patm  # mol mol-1, equivalent to D / Patm
            Xi = 2. * p.kmaxS1 * (cost_pd ** 2.) * dP / (dq * dcost)

            # calculate gs at the co-limitation point, mmol m-2 s-1
            gscol = Acol * conv.GwvGc * p.Patm / dCi

            # calculate gs, mol m-2 s-1
            if dAdCi <= 0.:  # from SOX code, but it should never happen
                gs = gscol * conv.FROM_MILI

            else:
                gs = 0.5 * dAdCi * conv.FROM_MILI * (((1. + 4. * Xi / dAdCi)
                                                      ** 0.5) - 1.)

        elif model == 'SOX-OPT':  # retrieve all potential Ci values
            Cis = Ci_stream(p, Cs, Tleaf, 'low')

            # rate of photosynthesis, μmol m-2 s-1
            A, __, __ = calc_photosynthesis(p, 0., Cis, photo, Tleaf=Tleaf)

            # gb?
            __, gb = leaf_temperature(p, 0., Tleaf=Tleaf, inf_gb=inf_gb)

            if inf_gb or (iter < 1):  # gas-exchange trans, mol m-2 s-1
                E = A * conv.FROM_MILI * conv.GwvGc * Dleaf / (p.CO2 - Cis)

            else:
                E = (A * conv.FROM_MILI * (gb * conv.GwvGc + gs * conv.GbvGbc)
                     / (gs + gb) * Dleaf / (p.CO2 - Cis))

            # Pleaf, cost
            Pleaf = np.array([P[np.nanargmin(np.abs(e - trans))] for e in E])
            mask = np.logical_and(E > cst.zero, Pleaf >= P[-1])
            cost = kcost(p, Pleaf[mask])

            try:  # optimal point
                iopt = np.argmax(cost * A[mask])
                Ci = Cis[mask][iopt]

            except Exception:

                return 9999. * 1000.

            # get net rate of photosynthesis at optimum, μmol m-2 s-1
            An, __, __ = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf)

            # get associated gc and gs
            gc = p.Patm * conv.FROM_MILI * An / (p.CO2 - Ci)

            if inf_gb:
                gs = gc * conv.GwvGc

            else:
                gs = np.maximum(cst.zero,
                                gc * gb * conv.GwvGc / (gb - conv.GbvGbc * gc))

        else:
            An, __, __, __ = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                                 gs_over_A=gsoA)
            gs = np.maximum(cst.zero, conv.GwvGc * gsoA * An)

        # calculate new trans, gw, gb, Tleaf
        E, real_zero, gw, gb, Dleaf = calc_trans(p, Tleaf, gs, inf_gb=inf_gb)
        new_Tleaf, __ = leaf_temperature(p, E, Tleaf=Tleaf, inf_gb=inf_gb)

        if model == 'Eller':  # calculate An
            An, __, __ = calc_photosynthesis(p, 0., Cs, photo, Tleaf=Tleaf,
                                             gsc=conv.U * conv.GcvGw * gs)

        # update Cs (Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb)
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))
        Cs_umol_mol = Cs * conv.MILI / p.Patm

        if model == 'Tuzet':  # update Pleaf and fw
            Pleaf = P[np.nanargmin(np.abs(trans - E))]  # Tuzet model

            if np.abs(fw - fLWP(p, Pleaf)) < 0.5:  # is fw stable?
                fw = fLWP(p, Pleaf)  # update

            # update gsoA
            gsoA = g0 + p.g1T * fw / Cs_umol_mol

        else:  # update the leaf-to-air VPD
            if (np.isclose(E, cst.zero, rtol=cst.zero, atol=cst.zero) or
                np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero) or
               np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero)):
                Dleaf = p.VPD  # kPa

            if model == 'Medlyn':
                Dleaf = np.maximum(0.05, Dleaf)  # model invalid < 0.05

                # update gs over A
                gsoA = g0 + (1. + p.g1 * fw / (Dleaf ** 0.5)) / Cs_umol_mol

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter > iter_min) and
            real_zero and (abs(Tleaf - new_Tleaf) <= 0.1) and not
           np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate
        Tleaf = new_Tleaf
        iter += 1

        if iter_max < 5:  # no "real" iteration if Tleaf is prescribed
            Cs = p.CO2
            Tleaf = p.Tleaf

    if model == 'Tuzet':

        return gs * 1000., Pleaf

    else:

        return gs * 1000.  # mmol m-2 s-1


def fmtx(p, model, photo='Farquhar', inf_gb=True):

    """
    Returns the optimized gs depending on the model called and a set of
    forcing conditions.

    Arguments:
    ----------
    p: recarray object or pandas series
        time step's met data & params

    model: string
        model to run. The choices are: 'CAP', 'CGain', 'Cmax',
        'LeastCost', 'MES', 'ProfitMax', 'ProfitMax2', 'WUE-LWP'

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    gs: float
        stomatal conductance [mmol m-2 s-1]

    """

    # hydraulics
    if (model == 'CAP') or (model == 'MES'):
        if model == 'CAP':
            P, trans = hydraulics(p, kmax=p.krlC, Pcrit=p.PcritC)

        else:
            P, trans = hydraulics(p, kmax=p.krlM, Pcrit=p.PcritM)

    elif model == 'LeastCost':
        P, trans = hydraulics(p, kmax=p.kmaxLC)

    else:
        P, trans = hydraulics(p)

    # expressions of optimisation
    if model == 'ProfitMax':  # look for the most net profit
        cost, __ = hydraulic_cost(p, P)
        gain, Ci, mask = photo_gain(p, trans, photo, 'low', inf_gb=inf_gb)
        expr = gain - cost[mask]

    elif (model != 'CAP') and (model != 'MES'):
        Ci, mask = Ci_sup_dem(p, trans, photo=photo, inf_gb=inf_gb)

    if model == 'ProfitMax2':
        expr = A_trans(p, trans[mask], Ci, inf_gb=inf_gb) * (1. - trans[mask] /
                                                             trans[-1])

    if model == 'CGain':
        expr = (A_trans(p, trans[mask], Ci, inf_gb=inf_gb) -
                p.Kappa * fPLC(p, P[mask]))

    if model == 'LeastCost':
        expr = ((p.Eta * conv.MILI * trans[mask] +
                 Vmax_T(p, trans[mask], inf_gb=inf_gb)) /
                A_trans(p, trans[mask], Ci, inf_gb=inf_gb))

    # leaf energy balance
    gc, gs, gb, ww = leaf_energy_balance(p, trans, inf_gb=inf_gb)

    if model == 'CMax':

        try:
            expr = np.abs(np.gradient(A_trans(p, trans[mask], Ci,
                                              inf_gb=inf_gb), P[mask]) -
                          dcost_dpsi(p, P[mask]))

        except Exception:

            return 9999. * 1000.  # returning NaNs causes trouble

    if model == 'WUE-LWP':
        expr = (A_trans(p, trans[mask], Ci, inf_gb=inf_gb) -
                p.Lambda * conv.MILI * trans[mask])

    if model == 'CAP':
        cost = phiLWP(P, p.PcritC)
        Ci, mask = Ci_sup_dem(p, trans, photo=photo, Vmax25=p.Vmax25 * cost,
                              inf_gb=inf_gb)
        An, __, __ = calc_photosynthesis(p, trans[mask], Ci, photo,
                                         Vmax25=p.Vmax25 * cost[mask],
                                         inf_gb=inf_gb)

        try:
            expr = An

        except Exception:

            return 9999. * 1000.  # returning NaNs causes trouble

    if model == 'MES':  # Ci is Cc
        cost = phiLWP(P, p.PcritM)
        Cc, mask = Ci_sup_dem(p, trans, photo=photo, phi=cost, inf_gb=inf_gb)
        An, __, __ = calc_photosynthesis(p, trans[mask], Cc, photo,
                                         inf_gb=inf_gb)

        try:
            expr = An

        except Exception:

            return 9999. * 1000.  # returning NaNs causes trouble

    if inf_gb:  # deal with edge cases by rebounding the solution
        check = expr[gc[mask] > cst.zero]

    else:  # accounting for gb
        check = expr[np.logical_and(gc[mask] > cst.zero, gs[mask] < 1.5 * gb)]

    try:
        if (model == 'LeastCost') or (model == 'CMax'):
            idx = np.isclose(expr, min(check))

        else:
            idx = np.isclose(expr, max(check))

        idx = [list(idx).index(e) for e in idx if e]

        if len(idx) >= 1:  # opt values

            return gs[mask][idx[0]] * 1000.  # mmol/m2/s

        else:

            return 9999. * 1000.  # returning NaNs causes trouble

    except ValueError:  # expr function is empty

        return 9999. * 1000.


def fres(params, model, inputs, target, inf_gb):

    """
    Minimizing function between any model and the observations.

    Arguments:
    ----------
    params: object
        parameters to calibrate, including their initialisation values

    model: string
        model to run. The choices are: 'CAP', 'CGain', 'Cmax', 'Eller'
        (i.e., SOX empirical), 'LeastCost', 'MES', 'Medlyn',
        'ProfitMax', 'ProfitMax2', 'Tuzet', or 'SOX-OPT' (i.e., the
        optimization version of SOX), 'WUE-LWP'

    inputs: recarray object or pandas series
        time step's met data & params

    target: recarray object or pandas series
        target observations to calibrate against

    inf_gb: bool
        if True, gb is prescrived and very large

    Returns:
    --------
    Residuals between the simulations and the observations.

    """

    if model == 'Tuzet':  # LWP from Ps predawn
        inputs['LWP_ini'] = (inputs['Ps_pd'] - inputs.loc[0, 'height'] *
                             cst.rho * cst.g0 * conv.MEGA)

    # copy and transform pandas to recarray object for speed
    inputs = inputs.copy().to_records(index=False)

    for pname in params.items():  # update the model's specific param.s

        try:
            inputs[:][pname[0]] = params[pname[0]].value

        except ValueError:  # do not account for ancillary params
            pass

    if model == 'Tuzet':
        ymodel = np.zeros(len(inputs))

        for step in range(len(inputs)):

            ymodel[step], oo = floop(inputs[step].copy(), model, inf_gb=inf_gb)

            try:  # iterate to update and stabilise the ini LWP
                if inputs[step + 1].doy == inputs[step].doy:
                    inputs[step + 1].LWP_ini = oo

            except IndexError:
                pass

    elif model in ['Medlyn', 'Eller', 'SOX-OPT']:
        ymodel = np.asarray([floop(inputs[step].copy(), model, inf_gb=inf_gb)
                             for step in range(len(inputs))], dtype=np.float64)

    else:
        ymodel = np.asarray([fmtx(inputs[step].copy(), model, inf_gb=inf_gb)
                             for step in range(len(inputs))], dtype=np.float64)

    if not inf_gb:  # replace 9999. by actual NaNs
        if len(np.isclose(ymodel, 9999.)) <= round(0.1 * len(target)):
            target[np.isclose(ymodel, 9999.)] = np.nan
            ymodel[np.isclose(ymodel, 9999.)] = np.nan

    return ymodel - target
