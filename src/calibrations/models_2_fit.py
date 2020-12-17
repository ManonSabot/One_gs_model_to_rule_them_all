# -*- coding: utf-8 -*-

"""
Apologies, this is completely hardwired right now... Will get it fixed soonish!

"""

__title__ = ""
__author__ = "[Manon Sabot]"
__version__ = "1.0 (16.01.2019)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

import warnings # ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# general modules
import numpy as np # array manipulations, math operators

# own modules
from TractLSM import conv, cst  # unit converter & general constants
from TractLSM.SPAC import f, Weibull_params, hydraulics, leaf_energy_balance
from TractLSM.SPAC import leaf_temperature, calc_colim_Ci, calc_photosynthesis
from TractLSM.SPAC import fwLWPpd, fLWP, phiLWP, fPLC, hydraulic_cost, kcost
from TractLSM.SPAC import dcost_dpsi
from TractLSM.CH2OCoupler import calc_trans
from TractLSM.CH2OCoupler import Ci_sup_dem, dAdgs, A_trans
from TractLSM.CH2OCoupler.SOX import Ci_stream
from TractLSM.CH2OCoupler.ProfitMax import photo_gain
from TractLSM.CH2OCoupler.WUE_LWP import dEdgs
from TractLSM.CH2OCoupler.LeastCost import dEoAdXi, dVmaxoAdXi


#==============================================================================

def floop(p, model, photo='Farquhar', threshold_conv=0.1, inf_gb=True):

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
            iter_max = 2  # needed to update the fw with the LWP

    except (IndexError, AttributeError, ValueError):  # calc. Tleaf
        Tleaf = p.Tair  # deg C
        iter_max = 40

    if model != 'Tuzet':  # energy balance requirements
        Pleaf_pd = p.Ps_pd - p.height * cst.rho * cst.g0 * conv.MEGA

        if model == 'Medlyn-LWP':
            Dleaf = np.maximum(0.05, Dleaf)  # gs model not valid < 0.05
            fw = fwLWPpd(p, Pleaf_pd)  # moisture stress function

    else:  # Tuzet model
        P, trans = hydraulics(p, kmax=p.kmaxT)  # hydraulics
        fw = p.fLWP_ini  # init. stress factor

    if (model == 'Medlyn-LWP') or (model == 'Tuzet'): # init. gs over A
        g0 = 1.e-9  # g0 ~ 0, removing it entirely introduces errors
        Cs_umol_mol = Cs * conv.MILI / p.Patm  # umol mol-1

        if model == 'Medlyn-LWP':
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
            Acol, __, __ = calc_photosynthesis(p, 0., Cicol, photo, Tleaf=Tleaf,
                                               gsc=0.)
            dA = As - Acol  # ambient - colimitation

            # dAdCi (in mol H2O) is needed to calculate gs, mmol m-2 s-1
            dAdCi = dA * conv.GwvGc * p.Patm / dCi

            # kcost, unitless
            cost_pd, __ = kcost(p, Pleaf_pd, Pleaf_pd)
            cost_mid, __ = kcost(p, -p.P50, Pleaf_pd)
            dkcost = cost_pd - cost_mid

            # dP is needed to calculate gs
            dP = 0.5 * (Pleaf_pd + p.P50)  # MPa,  /!\ sign of P50

            # xi, the loss of xylem cost of stomatal opening, mmol m-2 s-1
            dq = Dleaf / p.Patm  # mol mol-1, equivalent to D / Patm
            Xi = 2. * p.kmaxS1 * (cost_pd ** 2.) * dP / (dq * dkcost)

            # calculate gs at the co-limitation point, mmol m-2 s-1
            gscol = Acol * conv.GwvGc * p.Patm / dCi

            # calculate gs, mol m-2 s-1
            if dAdCi <= 0.:  # cp from SOX code, ??? it should never happen!
                gs = gscol * conv.FROM_MILI

            else:
                gs = 0.5 * dAdCi * conv.FROM_MILI * (((1. + 4. * Xi / dAdCi)
                                                      ** 0.5) - 1.)

        elif model == 'SOX-OPT': # retrieve all potential Ci values
            Cis = Ci_stream(p, Cs, Tleaf, 'low')

            # gross rate of photosynthesis, μmol m-2 s-1
            A, __, __ = calc_photosynthesis(p, 0., Cis, photo, Tleaf=Tleaf,
                                            Rleaf=0.)

            # trans used for the hydraulic cost, mmol m-2 s-1
            E = A * conv.GwvGc * Dleaf / (p.CO2 - Cis)

            # cost, Pleaf
            cost, __ = kcost(p, Pleaf_pd - E / p.ksc_prev, Pleaf_pd)

            # optimal point
            A = A[:len(cost)] # shortening avoids multiple kcost = 0
            iopt = np.argmax(cost * A)
            Ci = Cis[iopt]

            # get net rate of photosynthesis at optimum, μmol m-2 s-1
            An, __, __ = calc_photosynthesis(p, 0., Ci, photo, Tleaf=Tleaf)

            # get associated gc, gb, gs (mol m-2 s-1)
            gc = p.Patm * conv.FROM_MILI * An / (p.CO2 - Ci)
            __, gb = leaf_temperature(p, 0., Tleaf=Tleaf, inf_gb=inf_gb)
            gs = np.maximum(cst.zero,
                            gb * conv.GwvGc * gc / (gb - conv.GwvGc * gc))

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

        # new Cs (in Pa)
        boundary_CO2 = p.Patm * conv.FROM_MILI * An / (gb * conv.GbcvGb +
                                                       gs * conv.GcvGw)
        Cs = np.maximum(cst.zero, np.minimum(p.CO2, p.CO2 - boundary_CO2))
        Cs_umol_mol = Cs * conv.MILI / p.Patm

        if model == 'Tuzet':  # update stress function
            Pleaf = P[np.nanargmin(np.abs(trans - E))]

            if np.abs(fw - fLWP(p, Pleaf)) < 0.25:  # is fw stable?
                fw = fLWP(p, Pleaf)  # update

            gsoA = g0 + p.g1T * fw / Cs_umol_mol

        else:  # update the leaf-to-air VPD
            if (np.isclose(E, cst.zero, rtol=cst.zero, atol=cst.zero) or
                np.isclose(gw, cst.zero, rtol=cst.zero, atol=cst.zero) or
                np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero)):
                Dleaf = p.VPD  # kPa

            if model == 'Medlyn-LWP':
                Dleaf = np.maximum(0.05, Dleaf)  # gs model not valid < 0.05

                # update gs over A
                gsoA = g0 + (1. + p.g1 * fw / (Dleaf ** 0.5)) / Cs_umol_mol

        # force stop when atm. conditions yield E < 0. (non-physical)
        if (iter < 1) and (not real_zero):
            real_zero = None

        # check for convergence
        if ((real_zero is None) or (iter >= iter_max) or ((iter > iter_min) and
            real_zero and (abs(Tleaf - new_Tleaf) <= threshold_conv) and not
            np.isclose(gs, cst.zero, rtol=cst.zero, atol=cst.zero))):
            break

        # no convergence, iterate on leaf temperature
        Tleaf = new_Tleaf
        iter += 1

        if iter_max < 5:  # no iteration if Tleaf is prescribed
            Cs = p.CO2
            Tleaf = p.Tleaf

    if model == 'Tuzet':

        return gs * 1000., fw

    elif model == 'SOX-OPT':

        return gs * 1000., p.kmaxS2 * cost[iopt]

    else:

        return gs * 1000.  # mmol/m2/s


def fmtx(p, model, photo='Farquhar', inf_gb=True):

    # hydraulics
    if (model != 'CAP') and (model != 'MES'):
        if model == 'LeastCost':
            P, trans = hydraulics(p, kmax=p.kmaxLC)

        else:
            P, trans = hydraulics(p)

    if (model == 'CAP') or (model == 'MES'):
        if model == 'CAP':
            P = hydraulics(p, Kirchhoff=False, Pcrit=p.PcritC)
            ksr = p.ksrmaxC * (p.Psie / p.Ps) ** (2. + 3. / p.bch)
            ksl = 1. / (1. / ksr + 1. / p.krlC)  # soil-leaf conductance

        if model == 'MES':
            P = hydraulics(p, Kirchhoff=False, Pcrit=p.PcritM)
            ksr = p.ksrmaxM * (p.Psie / p.Ps) ** (2. + 3. / p.bch)
            ksl = 1. / (1. / ksr + 1. / p.krlM)  # soil-leaf conductance

        trans = ksl * (p.Ps - P) * conv.FROM_MILI  # mol.s-1.m-2

    # expressions of optimisation
    if model == 'ProfitMax': # look for the most net profit
        cost, __ = hydraulic_cost(p, P)
        gain, Ci, mask = photo_gain(p, trans, photo, 'low', inf_gb=inf_gb)
        expr = gain - cost[mask]

    elif (model != 'CAP') and (model != 'MES'):
        Ci, mask = Ci_sup_dem(p, trans, photo=photo, inf_gb=inf_gb)

    if model == 'ProfitMax2':
        expr =  np.abs(np.gradient(A_trans(p, trans[mask], Ci, inf_gb=inf_gb),
                                   trans[mask]) * (trans[-1] - trans[mask]) -
                       A_trans(p, trans[mask], Ci, inf_gb=inf_gb))

    if model == 'CGain':
        cost = fPLC(p, P[mask])
        expr = A_trans(p, trans[mask], Ci, inf_gb=inf_gb) - p.Kappa * cost

    if model == 'LeastCost':
        expr = np.abs(dEoAdXi(p, trans[mask], Ci, inf_gb=inf_gb) + p.Eta *
                      dVmaxoAdXi(p, trans[mask], Ci, inf_gb=inf_gb))

    # leaf energy balance
    gc, gs, gb, ww = leaf_energy_balance(p, trans, inf_gb=inf_gb)

    if (model == 'CMax') or (model == 'WUE-LWP'):
        if model == 'CMax':
            cost = dcost_dpsi(p, P[mask], gs[mask])

        if model == 'WUE-LWP':
            try:
                cost = p.Lambda * dEdgs(gs[mask], gb, ww[mask])

            except IndexError:  # only one Tleaf
                cost = p.Lambda * dEdgs(gs[mask], gb, ww)

        expr = np.abs(dAdgs(p, gs[mask], gb, Ci) - cost)

    if model == 'CAP':
        cost = phiLWP(P, p.PcritC)
        sVmax25 = p.Vmax25 * cost
        Ci, mask = Ci_sup_dem(p, trans, photo=photo, Vmax25=sVmax25,
                              inf_gb=inf_gb)
        An, __, __ = calc_photosynthesis(p, trans[mask], Ci, photo,
                                         Vmax25=sVmax25[mask], inf_gb=inf_gb)

        try:
            expr = np.abs(np.gradient(An, gs[mask]))

        except Exception:

            return 9999. * 1000.  # returning an actual NaN causes trouble

    if model == 'MES':  # Ci is Cc
        cost = phiLWP(P, p.PcritM)
        Cc, mask = Ci_sup_dem(p, trans, photo=photo, phi=cost, inf_gb=inf_gb)
        expr = np.abs(dAdgs(p, gs[mask], gb, Cc))

    if inf_gb:  # deal with edge cases by rebounding the solution
        check = expr[gc[mask] > cst.zero]

        # erratic model behaviour at low VPD and low-ish PAR
        if (model == 'ProfitMax2') and ((p.VPD < 1.) or (p.PPFD < 500.)):
            check = check[:np.minimum(np.argmax(np.diff(check) > 0.) +
                                      int(len(P) / 20), len(check) - 1)]

    else:
        check = expr[np.logical_and(gc[mask] > cst.zero, gs[mask] < 1.5 * gb)]

    if model == 'ProfitMax2':   # expr can have 'fake' zeros passed opt pt
        check = check[check > 1.e-9]

    try:
        if (model == 'ProfitMax') or (model == 'CGain'):
            idx = np.isclose(expr, max(check))

        else:
            idx = np.isclose(expr, min(check))

        idx = [list(idx).index(e) for e in idx if e]

        if len(idx) >= 1:  # opt values

            return gs[mask][idx[0]] * 1000.  # mmol/m2/s

        else:

            return 9999. * 1000.  # returning an actual NaN causes trouble

    except ValueError:  # expr function is empty

        return 9999. * 1000.


def fres(params, model, inputs, target, inf_gb):

    if model == 'Tuzet':  # dummy init. of stress factor
        inputs['fLWP_ini'] = fLWP(inputs.loc[0], inputs['Ps_pd'] -
                                  inputs.loc[0, 'height'] * cst.rho * cst.g0 *
                                  conv.MEGA)

    # copy and transform pandas to recarray object for speed
    inputs = inputs.copy().to_records(index=False)

    for pname in params.items():  # update the model's specific param.s

        try:
            inputs[:][pname[0]] = params[pname[0]].value

        except ValueError:  # do not account for ancillary params
            pass

    if model in ['Tuzet', 'SOX-OPT']:
        ymodel = np.zeros(len(inputs))

        for step in range(len(inputs)):

            ymodel[step], oo = floop(inputs[step].copy(), model, inf_gb=inf_gb)

            try:

                if model == 'Tuzet':
                    if inputs[step + 1].doy == inputs[step].doy:
                        inputs[step + 1].fLWP_ini = oo

                else:
                    inputs[step + 1:].ksc_prev = oo

            except IndexError:
                pass

    elif model in ['Medlyn-LWP', 'Tuzet', 'Eller']:
        ymodel = np.asarray([floop(inputs[step].copy(), model, inf_gb=inf_gb)
                             for step in range(len(inputs))], dtype=np.float64)

    else:
        ymodel = np.asarray([fmtx(inputs[step].copy(), model, inf_gb=inf_gb)
                             for step in range(len(inputs))], dtype=np.float64)

    return ymodel - target
