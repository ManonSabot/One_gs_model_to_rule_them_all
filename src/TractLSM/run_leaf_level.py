# -*- coding: utf-8 -*-

"""
Runs the LSM with prescribed soil moiture and at the leaf-level.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Run a tractable LSM at the leaf-level"
__author__ = "Manon E. B. Sabot"
__version__ = "8.0 (29.08.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import collections  # ordered dictionaries
import numpy as np  # maths operations

# own modules
from TractLSM import cst, conv  # general constants & unit conversions
from TractLSM.SPAC import net_radiation  # radiative conditions
from TractLSM.CH2OCoupler import solve_std  # USO model (Medlyn)
from TractLSM.CH2OCoupler import Tuzet  # Tuzet model (Tuzet)
from TractLSM.CH2OCoupler import supply_max  # SOX (Eller) and SOX opt
from TractLSM.CH2OCoupler import WUE_gs  # WUE-hydraulics (Wolf)
from TractLSM.CH2OCoupler import profit_psi  # Profit max (Sperry)
from TractLSM.CH2OCoupler import profit_AE  # Profit max (Wang)
from TractLSM.CH2OCoupler import Cgain_plc  # Carbon gain net (Lu)
from TractLSM.CH2OCoupler import Cmax_gs  # Carbon max (Wolf)
from TractLSM.CH2OCoupler import least_cost  # Least cost (Prentice)
from TractLSM.CH2OCoupler import CAP  # CAP optimisation (Dewar)
from TractLSM.CH2OCoupler import MES  # MES optimisation (Dewar)

try:  # support functions
    from run_utils import find_model_cases, write_csv

except (ImportError, ModuleNotFoundError):
    from TractLSM.run_utils import find_model_cases, write_csv


# ======================================================================

def over_time(idata, step, Nsteps, dic, photo, resolution, inf_gb, temporal,
              deriv):

    """
    Optimization wrapper at each time step that updates the soil
    moisture and soil water potential for each of the models before
    running them in turn: (i) the Medlyn model (solve_std), (ii)
    the Profit maximisation (profit_psi). None of these are run for
    timesteps when PPFD = 0.

    Arguments:
    ----------
    idata: recarray object
        dataframe containing all input data & params

    step: int
        current time step

    Nsteps: int
        total number of steps. This is necessary to know whether unit
        conversion must be based on half-hourly time steps or longer
        time steps!

    dic: dictionary
        initially empty upon input, this dictionary allows to return the
        outputs in a trackable manner. From a time-step to another, it
        also keeps in store the soil moisture and transpiration relative
        to each model, in order to accurately update the soil water
        bucket.

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    inf_gb: bool
        if True, gb is prescrived and very large

    temporal: bool
        if True, the run is progressing temporally and the Tuzet model's
        stability routine for leaf water potential can be initialised
        from the leaf water potential of the previous timestep

    deriv: bool
        if True, uses the derivative form of the optimality criterion

    Returns:
    --------
    Outputs a tuple of variables depending on the input dic structure.
    When PPFD is zero, a tuple of zero values is returned. If the models
    behave in a non-physical manner, zeros are returned too. Overall,
    the following variables are returned:

    A(model): float
        net photosynthetic assimilation rate [umol m-2 s-1]

    E(model): float
        transpiration rate [mmol m-2 s-1]

    gs(model): float
        stomatal conductance to water vapour [mol m-2 s-1]

    gb(model): float
        leaf boundary layer conductance to water vapour [mol m-2 s-1]

    Ci(model): float
        intercellular CO2 concentration [Pa]

    Tleaf(model): float
        leaf temperature [degC]

    Pleaf(model): float
        leaf water potential [MPa]

    Rublim(model): float
        'True' if the C assimilation is rubisco limited, 'False'
        otherwise

    Ps(model): float
        soil water potential [MPa]

    """

    # parameters & met data
    p = idata[step].copy()

    # tuple of return values
    tpl_return = ()

    # if SOX case number >= 1
    Scases = [key for key in dic.keys() if 'sox' in key]

    if len(Scases) >= 1:  # dic to store each SOX case values
        s_cases = [sub.replace('sox', '') for sub in Scases]

    for key in dic.keys():  # own Ps to account for trans feedback
        dic[key]['Ps'] = p.Ps

    # night time
    if p.PPFD <= 50.:  # min threshold for photosynthesis

        for key in dic.keys():

            dic[key]['A'], dic[key]['Ci'], dic[key]['Rublim'], dic[key]['E'], \
                dic[key]['gs'], dic[key]['gb'], dic[key]['Tleaf'], \
                dic[key]['Pleaf'] = (0.,) * 8

    else:  # day time
        p.Rnet = net_radiation(p)  # radiative conditions

        if 'std' in dic.keys():  # standard model (Medlyn)
            try:
                dic['std']['A'], dic['std']['Ci'], dic['std']['Rublim'], \
                    dic['std']['E'], dic['std']['gs'], dic['std']['gb'], \
                    dic['std']['Tleaf'], dic['std']['Pleaf'] = \
                    solve_std(p, p.sw, photo=photo, res=resolution,
                              inf_gb=inf_gb)

            except (IndexError, ValueError):  # no solve
                dic['std']['A'], dic['std']['Ci'], dic['std']['Rublim'], \
                    dic['std']['E'], dic['std']['gs'], dic['std']['gb'], \
                    dic['std']['Tleaf'], dic['std']['Pleaf'] = (9999.,) * 8

        if 'tuz' in dic.keys():  # Tuzet
            try:
                dic['tuz']['A'], dic['tuz']['Ci'], dic['tuz']['Rublim'], \
                    dic['tuz']['E'], dic['tuz']['gs'], dic['tuz']['gb'], \
                    dic['tuz']['Tleaf'], dic['tuz']['Pleaf'] = \
                    Tuzet(p, photo=photo, res=resolution, inf_gb=inf_gb)

                if temporal and step < Nsteps - 1:
                    if idata[step + 1].doy == idata[step].doy:
                        idata[step + 1].LWP_ini = dic['tuz']['Pleaf']

            except (IndexError, ValueError):  # no solve
                dic['tuz']['A'], dic['tuz']['Ci'], dic['tuz']['Rublim'], \
                    dic['tuz']['E'], dic['tuz']['gs'], dic['tuz']['gb'], \
                    dic['tuz']['Tleaf'], dic['tuz']['Pleaf'] = (9999.,) * 8

        if len(Scases) >= 1:  # SOX model

            for icase in range(len(s_cases)):

                SOX = 'sox%s' % (s_cases[icase])
                this_case = int(s_cases[icase])

                try:
                    dic[SOX]['A'], dic[SOX]['Ci'], dic[SOX]['Rublim'], \
                        dic[SOX]['E'], dic[SOX]['gs'], dic[SOX]['gb'], \
                        dic[SOX]['Tleaf'], dic[SOX]['Pleaf'] = \
                        supply_max(p, photo=photo, res=resolution,
                                   case=this_case, inf_gb=inf_gb)

                except (IndexError, ValueError):  # no solve
                    dic[SOX]['A'], dic[SOX]['Ci'], dic[SOX]['Rublim'], \
                        dic[SOX]['E'], dic[SOX]['gs'], dic[SOX]['gb'], \
                        dic[SOX]['Tleaf'], dic[SOX]['Pleaf'] = (9999.,) * 8

        if 'wue' in dic.keys():
            try:
                dic['wue']['A'], dic['wue']['Ci'], dic['wue']['Rublim'], \
                    dic['wue']['E'], dic['wue']['gs'], dic['wue']['gb'], \
                    dic['wue']['Tleaf'], dic['wue']['Pleaf'] = \
                    WUE_gs(p, photo=photo, res=resolution, inf_gb=inf_gb,
                           deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['wue']['A'], dic['wue']['Ci'], dic['wue']['Rublim'], \
                    dic['wue']['E'], dic['wue']['gs'], dic['wue']['gb'], \
                    dic['wue']['Tleaf'], dic['wue']['Pleaf'] = (9999.,) * 8

        if 'cmax' in dic.keys():
            try:
                dic['cmax']['A'], dic['cmax']['Ci'], dic['cmax']['Rublim'], \
                    dic['cmax']['E'], dic['cmax']['gs'], dic['cmax']['gb'], \
                    dic['cmax']['Tleaf'], dic['cmax']['Pleaf'] = \
                    Cmax_gs(p, photo=photo, res=resolution, inf_gb=inf_gb)

            except (IndexError, ValueError):  # no solve
                dic['cmax']['A'], dic['cmax']['Ci'], dic['cmax']['Rublim'], \
                    dic['cmax']['E'], dic['cmax']['gs'], dic['cmax']['gb'], \
                    dic['cmax']['Tleaf'], dic['cmax']['Pleaf'] = (9999.,) * 8

        if 'pmax' in dic.keys():  # ProfitMax
            try:
                dic['pmax']['A'], dic['pmax']['Ci'], dic['pmax']['Rublim'], \
                    dic['pmax']['E'], dic['pmax']['gs'], dic['pmax']['gb'], \
                    dic['pmax']['Tleaf'], dic['pmax']['Pleaf'] = \
                    profit_psi(p, photo=photo, res=resolution, inf_gb=inf_gb,
                               deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['pmax']['A'], dic['pmax']['Ci'], dic['pmax']['Rublim'], \
                    dic['pmax']['E'], dic['pmax']['gs'], dic['pmax']['gb'], \
                    dic['pmax']['Tleaf'], dic['pmax']['Pleaf'] = (9999.,) * 8

        if 'cgn' in dic.keys():  # CGain
            try:
                dic['cgn']['A'], dic['cgn']['Ci'], dic['cgn']['Rublim'], \
                    dic['cgn']['E'], dic['cgn']['gs'], dic['cgn']['gb'], \
                    dic['cgn']['Tleaf'], dic['cgn']['Pleaf'] = \
                    Cgain_plc(p, photo=photo, res=resolution, inf_gb=inf_gb,
                              deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['cgn']['A'], dic['cgn']['Ci'], dic['cgn']['Rublim'], \
                    dic['cgn']['E'], dic['cgn']['gs'], dic['cgn']['gb'], \
                    dic['cgn']['Tleaf'], dic['cgn']['Pleaf'] = (9999.,) * 8

        if 'pmax2' in dic.keys():  # ProfitMax2
            try:
                dic['pmax2']['A'], dic['pmax2']['Ci'], \
                    dic['pmax2']['Rublim'], dic['pmax2']['E'], \
                    dic['pmax2']['gs'], dic['pmax2']['gb'], \
                    dic['pmax2']['Tleaf'], dic['pmax2']['Pleaf'] = \
                    profit_AE(p, photo=photo, res=resolution, inf_gb=inf_gb,
                              deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['pmax2']['A'], dic['pmax2']['Ci'], \
                    dic['pmax2']['Rublim'], dic['pmax2']['E'], \
                    dic['pmax2']['gs'], dic['pmax2']['gb'], \
                    dic['pmax2']['Tleaf'], dic['pmax2']['Pleaf'] = (9999.,) * 8

        if 'lcst' in dic.keys():  # LeastCost
            try:
                dic['lcst']['A'], dic['lcst']['Ci'], dic['lcst']['Rublim'], \
                    dic['lcst']['E'], dic['lcst']['gs'], dic['lcst']['gb'], \
                    dic['lcst']['Tleaf'], dic['lcst']['Pleaf'] = \
                    least_cost(p, photo=photo, res=resolution, inf_gb=inf_gb,
                               deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['lcst']['A'], dic['lcst']['Ci'], dic['lcst']['Rublim'], \
                    dic['lcst']['E'], dic['lcst']['gs'], dic['lcst']['gb'], \
                    dic['lcst']['Tleaf'], dic['lcst']['Pleaf'] = (9999.,) * 8

        if 'cap' in dic.keys():  # CAP (Dewar)
            try:
                dic['cap']['A'], dic['cap']['Ci'], dic['cap']['Rublim'], \
                    dic['cap']['E'], dic['cap']['gs'], dic['cap']['gb'], \
                    dic['cap']['Tleaf'], dic['cap']['Pleaf'] = \
                    CAP(p, photo=photo, res=resolution, inf_gb=inf_gb,
                        deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['cap']['A'], dic['cap']['Ci'], dic['cap']['Rublim'], \
                    dic['cap']['E'], dic['cap']['gs'], dic['cap']['gb'], \
                    dic['cap']['Tleaf'], dic['cap']['Pleaf'] = (9999.,) * 8

        if 'mes' in dic.keys():  # MES (Dewar)
            try:
                dic['mes']['A'], dic['mes']['Ci'], dic['mes']['Rublim'], \
                    dic['mes']['E'], dic['mes']['gs'], dic['mes']['gb'], \
                    dic['mes']['Tleaf'], dic['mes']['Pleaf'] = \
                    MES(p, photo=photo, res=resolution, inf_gb=inf_gb,
                        deriv=deriv)

            except (IndexError, ValueError):  # no solve
                dic['mes']['A'], dic['mes']['Ci'], dic['mes']['Rublim'], \
                    dic['mes']['E'], dic['mes']['gs'], dic['mes']['gb'], \
                    dic['mes']['Tleaf'], dic['mes']['Pleaf'] = (9999.,) * 8

    # output must be in same order than output dic
    for key in dic.keys():

        tpl_return += (dic[key]['A'], dic[key]['E'], dic[key]['gs'],
                       dic[key]['gb'], dic[key]['Ci'], dic[key]['Tleaf'],
                       dic[key]['Pleaf'], dic[key]['Rublim'], dic[key]['Ps'],)

    return tpl_return


def run(fname, df, Nsteps, photo, models=['Medlyn', 'ProfitMax'],
        resolution=None, inf_gb=False, temporal=True, deriv=False):

    """
    Runs the profit maximisation algorithm within a simplified LSM,
    alongside the Medlyn model which follows traditional photosynthesis
    and transpiration coupling schemes.

    Arguments:
    ----------
    fname: string
        output filename

    df: pandas dataframe
        dataframe containing all input data & params

    Nsteps: int
        total number of time steps over which the models will be run

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    models: list of strings
        names of the models to call. Calling both SOX cases can
        be done via the 'SOX12' string or by listing them
        individually. 'SOX' runs the default case = 1.

    resolution: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    inf_gb: bool
        if True, gb is prescrived and very large

    temporal: bool
        if True, the run is progressing temporally and the Tuzet model's
        stability routine for leaf water potential can be initialised
        from the leaf water potential of the previous timestep

    deriv: bool
        if True, uses the derivative form of the optimality criterion

    Returns:
    --------
    df2: pandas dataframe
        dataframe of the outputs:
            A(model), E(model), gs(model), gb(model), Ci(model),
            Tleaf(model), Pleaf(model), Rublim(model), Ps(model)

    """

    # two empty dics, to structure the run setup and retrieve the output
    dic = {}  # appropriately run the models
    output_dic = collections.OrderedDict()  # unpack the output in order

    # sub-dic structures
    dic_keys = ['A', 'E', 'gs', 'gb', 'Ci', 'Tleaf', 'Pleaf', 'Rublim', 'Ps']
    subdic = {key: None for key in dic_keys}

    # for the output dic, the order of the keys matters!
    subdic2 = collections.OrderedDict([(key, None) for key in dic_keys])

    # Medlyn model
    if ('Medlyn' in models) or ('Medlyn'.lower() in models):
        dic['std'] = subdic.copy()
        output_dic['std'] = subdic2.copy()

    # Tuzet model
    if ('Tuzet' in models) or ('Tuzet'.lower() in models):
        dic['tuz'] = subdic.copy()
        output_dic['tuz'] = subdic2.copy()

    # specific SOX cases (i.e. SOX analytical and SOX optimal)?
    Scases = find_model_cases(models, 'SOX')

    # if cases aren't specified, then set to default (analytical) SOX
    if (len(Scases) < 1) and (('SOX' in models) or ('SOX'.lower() in models)):
        dic['sox1'] = subdic.copy()
        output_dic['sox1'] = subdic2.copy()

    # if several SOX cases
    if len(Scases) >= 1:

        for case in Scases:

            dic['sox%d' % (case)] = subdic.copy()
            output_dic['sox%d' % (case)] = subdic2.copy()

    # WUE-hydraulics model
    if ('WUE' in models) or ('WUE'.lower() in models):
        dic['wue'] = subdic.copy()
        output_dic['wue'] = subdic2.copy()

    # CMax model
    if ('CMax' in models) or ('CMax'.lower() in models):
        dic['cmax'] = subdic.copy()
        output_dic['cmax'] = subdic2.copy()

    # ProfitMax model
    if ('ProfitMax' in models) or ('ProfitMax'.lower() in models):
        dic['pmax'] = subdic.copy()
        output_dic['pmax'] = subdic2.copy()

    # CGain model
    if ('CGain' in models) or ('CGain'.lower() in models):
        dic['cgn'] = subdic.copy()
        output_dic['cgn'] = subdic2.copy()

    # 'New' ProfitMax model
    if ('ProfitMax2' in models) or ('ProfitMax2'.lower() in models):
        dic['pmax2'] = subdic.copy()
        output_dic['pmax2'] = subdic2.copy()

    # LeastCost model
    if ('LeastCost' in models) or ('LeastCost'.lower() in models):
        dic['lcst'] = subdic.copy()
        output_dic['lcst'] = subdic2.copy()

    # CAP model
    if ('CAP' in models) or ('CAP'.lower() in models):
        dic['cap'] = subdic.copy()
        output_dic['cap'] = subdic2.copy()

    # MES model
    if ('MES' in models) or ('MES'.lower() in models):
        dic['mes'] = subdic.copy()
        output_dic['mes'] = subdic2.copy()

    # how to run the optimisation?
    if resolution is None:
        resolution = 'low'

    if 'Ps_pd' not in df.columns:  # diurnal pre-dawn water potential
        df['Ps_pd'] = df['Ps'].copy()
        df['Ps_pd'].where(df['PPFD'] <= 50., np.nan, inplace=True)

    # big-leaf without scaling: internal scaling factor is 1
    df['scale2can'] = 1.
    df['Rnet'] = np.nan  # empty Rnet data column

    if 'Tuzet' in models:
        df['LWP_ini'] = (df['Ps_pd'] - df.iloc[0, df.columns.get_loc('height')]
                         * cst.rho * cst.g0 * conv.MEGA)

    # non time-sensitive: last valid value propagated until next valid
    df.fillna(method='ffill', inplace=True)

    # from pandas to recarray object for execution speed
    force = df.to_records(index=False)

    # run the model(s) over the range of timesteps / the timeseries
    tpl_out = list(zip(*[over_time(force, step, Nsteps, dic, photo, resolution,
                                   inf_gb, temporal, deriv)
                         for step in range(Nsteps)]))

    # unpack the output tuple 7 by 7 (A, E, Ci, Rublim, gs, Pleaf, Ps)
    track = 0  # initialize

    for key in output_dic.keys():

        output_dic[key]['A'] = tpl_out[track]
        output_dic[key]['E'] = tpl_out[track + 1]
        output_dic[key]['gs'] = tpl_out[track + 2]
        output_dic[key]['gb'] = tpl_out[track + 3]
        output_dic[key]['Ci'] = tpl_out[track + 4]
        output_dic[key]['Tleaf'] = tpl_out[track + 5]
        output_dic[key]['Pleaf'] = tpl_out[track + 6]
        output_dic[key]['Rublim'] = tpl_out[track + 7]
        output_dic[key]['Ps'] = tpl_out[track + 8]
        track += 9

    # save the outputs to a csv file and get the corresponding dataframe
    df2 = write_csv(fname, df, output_dic)

    return df2
