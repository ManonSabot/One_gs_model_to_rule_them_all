#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apologies, this is completely hardwired right now... Will get it fixed soonish!

DE: a stochastic population based method that is useful for global optimization
    problems because it does not use the gradient of the problem being
    optimized, which means there is no need for the optimization problem to be
    differentiable.
    https://en.wikipedia.org/wiki/Differential_evolution
** Storn, R and Price, K, Differential Evolution - a Simple and Efficient
   Heuristic for Global Optimization over Continuous Spaces, Journal of Global
   Optimization, 1997, 11, 341 - 359.

Basin-Hopping: a two-phase method that combines a global stepping algorithm
               with local minimization at each step.
** Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and the
   Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110
   Atoms. Journal of Physical Chemistry A, 1997, 101, 5111.

Nelder-Mead: uses the Simplex algorithm.
** Gao, F. and Han, L. Implementing the Nelder-Mead simplex algorithm with
   adaptive parameters. 2012. Computational Optimization and Applications.
   51:1, pp. 259-277

Powell: a conjugate direction method. It performs sequential one-dimensional
        minimizations along each vector of the directions set, which is updated
        at each iteration of the main minimization loop. The function need not
        be differentiable, and no derivatives are taken.
** Press, William H., et al. Numerical recipes. Vol. 3. Cambridge:
   Cambridge University Press, 1989.

COBYLA: uses the Constrained Optimization BY Linear Approximation (COBYLA)
        method. The algorithm is based on linear approximations to the
        objective function and each constraint.
** Powell, M J D. A direct search optimization method that models the objective
   and constraint functions by linear interpolation. 1994. Advances in
   Optimization and Numerical Analysis, eds. S. Gomez and J-P Hennart, Kluwer
   Academic (Dordrecht), 51-67.10
** Powell M J D. Direct search algorithms for optimization calculations. 1998.
   Acta Numerica 7: 287-336.11
** Powell M J D. A view of algorithms for optimization without derivatives.
   2007. Cambridge University Technical Report DAMTP 2007/NA03

DA: a stochastic approach which combines the generalization of CSA
    (Classical Simulated Annealing) and FSA (Fast Simulated Annealing) coupled
    to a strategy for applying a local search on accepted locations.
** Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated Annealing Algorithm
   and Its Application to the Thomson Model. Physics Letters A, 233, 216-220
   (1997).
** Xiang Y, Gong XG. Efficiency of Generalized Simulated Annealing. Physical
   Review E, 62, 4473 (2000).

AMPGO: Adaptive Memory Programming for Global Optimization
** Lasdon, Leon, et al. "Adaptive memory programming for constrained global
   optimization." Computers & operations research 37.8 (2010): 1500-1509.
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
import os  # check for files, paths
import sys  # check for files, paths
import shutil  # move files
import random  # pick a random day for the forcings to be generated
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from itertools import groupby

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import InForcings  # generate met data & read default params
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules
from TractLSM.SPAC import net_radiation  # preset it for model training
from TractLSM import hrun  # run the reference gs model
from fit_models import NLMFIT  # training functions

#==============================================================================

def soil_water(df, profile):

    sw = np.full(len(df), df['theta_sat'][0])
    track = 1

    if profile == 'wet':
        start = sw[0]
        rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (sw[0] + df['fc'][0]) / 2.25

    if profile == 'inter':
        start = 0.9 * sw[0]
        rate = -5. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (df['fc'][0] + df['pwp'][0]) / 2.

        # alternative
        #start = sw[0]
        #rate = -8. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))

    for i in range(len(df)):

        if i == 0:
            sw[i] = start

        else:
            sw[i] = sw[i-1]

        if df['PPFD'].iloc[i] > 50.:
            sw[i] = np.maximum(start / (1. - rate * track), sw_min)
            track += 1

    # now get the soil water potentials matching the soil moisture profile
    Ps = np.asarray([water_potential(df.iloc[0], sw[i])
                     for i in range(len(sw))])

    return sw, Ps


def check_X_Y(swaters):

    base_dir = get_main_dir()

    # check that the 4 week forcing file exists
    fname1 = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                          'input'), 'calibrations'), 'idealised'),
                          'training_x.csv')

    if not os.path.isfile(fname1):  # create file if it doesn't exist
        params = InForcings().defparams
        params.doy = random.randrange(92, 275)  # random day within GS
        InForcings().run(fname1, params, Ndays=7*4)

    for profile in swaters:

        # check that the output file from the reference model exists
        fname2 = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                              'input'), 'calibrations'), 'idealised'),
                              'training_%s_y.csv' % (profile))

        if not os.path.isfile(fname2):
            df1, __ = read_csv(fname1)

            # add the soil moisture profile to the input data
            df1['sw'], df1['Ps'] = soil_water(df1, profile)
            df1['Ps_pd'] = df1['Ps'].copy()  # pre-dawn soil water potentials
            df1['Ps_pd'].where(df1['PPFD'] <= 50., np.nan, inplace=True)

            # fixed value for the wind speed
            df1['u'] = df1['u'].iloc[0]

            # non time-sensitive: last valid value propagated until next valid
            df1.fillna(method='ffill', inplace=True)

            __ = hrun(fname2, df1, len(df1.index), 'Farquhar',
                      models=['Medlyn1'], inf_gb=True)

    return


def subsample(training, target, sample):

    base_dir = get_main_dir()

    # randomly subsample the data and store the subsampling distribution
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'input'), 'calibrations'), 'idealised'),
                         'subsample_%d.npy' % (sample))

    if not os.path.isfile(fname):
        size = 7
        doys = np.unique(training['doy'])
        sub = np.unique(np.random.randint(doys[0], doys[-1] + 1, size=size))

        while len(sub) < size:

            sub = np.unique(np.append(sub, np.unique(np.random.randint(doys[0],
                            doys[-1] + 1, size=size - len(sub)))))

        ssub = np.asarray(training.loc[training['doy'].isin(sub)].index)

        """
        # there are small differences in day time hours... same sizes?
        if sample > 1:  # the first distribution is the reference size
            ref = np.load(fname.replace('%d.npy' % (sample), '1.npy'))
            diff = len(ref) - len(ssub)

            while diff > 0:  # random extra data points from any one day

                sub = np.unique(np.append(sub,
                                np.unique(np.random.randint(doys[0],
                                doys[-1] + 1, size=1))))

                ssub = np.asarray(training.loc[training['doy'].isin(sub)].index)
                diff = len(ref) - len(ssub)

            if diff < 0:  # randomly remove the excess
                rm = np.random.randint(0, len(ssub), size=abs(diff))
                ssub = np.delete(ssub, rm)
        """

        np.save(fname, ssub)

    else:
        ssub = np.load(fname)

    # now accordingly subsample input and output
    Y = target[ssub]
    X = training.iloc[ssub]
    X.reset_index(inplace=True, drop=True)

    return X, Y


def prep_training_N_target(profile, sub=None):

    base_dir = get_main_dir()

    # path to input data
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'input'), 'calibrations'), 'idealised'),
                         'training_x.csv')
    df1, __ = read_csv(fname)

    # path to output data from the reference model
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'input'), 'calibrations'), 'idealised'),
                         'training_%s_y.csv' % (profile))
    df2, __ = read_csv(fname)

    # add the soil moisture profile to the input data
    df1['sw'], df1['Ps'] = soil_water(df1, profile)
    df1['Ps_pd'] = df1['Ps'].copy()  # daily pre-dawn soil water potentials
    df1['Ps_pd'].where(df1['PPFD'] <= 50., np.nan, inplace=True)

    # fix the wind speed
    df1['u'] = df1['u'].iloc[0]

    # non time-sensitive: last valid value propagated until next valid
    df1.fillna(method='ffill', inplace=True)

    # drop everything below min threshold for photosynthesis and reindex
    Y = np.asarray(df2['gs(std1)'][df1['PPFD'] > 50.]) * 1000.  # mmol m-2 s-1
    X = df1[df1['PPFD'] > 50.]
    X.reset_index(inplace=True, drop=True)

    # add Rnet to the input (no ET or soil albedo feedbacks, this can be done)
    X['Rnet'] = net_radiation(X)
    X['scale2can'] = 1.

    if sub is not None:  # randomly subsample one week out of the data
        X, Y = subsample(X, Y, sub)

    return X, Y


#==============================================================================

to_fit = False
sample = None # None, 1, 2, or 3

swaters = ['wet', 'inter']  # two different soil moisture profiles

# declare empty dataframe which will be used to analyse the calibrations
odf = pd.DataFrame(columns=['Model', 'training', 'solver', 'BIC', 'Rank', 'p1',
                            'v1', 'p2', 'v2'])

# where should the fitting solvers' outputs be stored?
base_dir = get_main_dir()  # working paths

check_X_Y(swaters)  # check that the training calibration files exist

if to_fit:

    for swater in swaters:  # loop over the training soil moisture profiles

        X, Y = prep_training_N_target(swater, sub=sample)

        # where should the calibration output be stored?
        opath = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                             'calibrations'), 'idealised')

        if sample is not None:  # move files to the relevant sub-dir
            opath = os.path.join(opath, '%s_subsample_%d' % (swater, sample))

        else:
            opath = os.path.join(opath, swater)

        if not os.path.isdir(opath):  # make new dirs if they don't exist
            os.makedirs(opath)

        # use a non-linear least square minimiser to train the models
        for test in ['differential_evolution', 'basinhopping', 'nelder',
                     'powell', 'cobyla', 'dual_annealing', 'ampgo']:  # 'emcee',

            XX = X.copy()
            nlmfit = NLMFIT(method=test, store=opath)

            if swater == 'inter':  # no point in calibrating this when wet
                __ = nlmfit.run(XX, Y, 'Medlyn-LWP')

            __ = nlmfit.run(XX, Y, 'Eller')
            __ = nlmfit.run(XX, Y, 'SOX-OPT')
            __ = nlmfit.run(XX, Y, 'CAP')
            __ = nlmfit.run(XX, Y, 'MES')
            __ = nlmfit.run(XX, Y, 'LeastCost')
            __ = nlmfit.run(XX, Y, 'ProfitMax2')
            fkmax = nlmfit.run(XX, Y, 'ProfitMax')

            # these models come after ProfitMax as they use its kmax
            XX['kmax'] = fkmax['kmax']
            __ = nlmfit.run(XX, Y, 'Tuzet')
            __ = nlmfit.run(XX, Y, 'WUE-LWP')
            __ = nlmfit.run(XX, Y, 'CMax')
            __ = nlmfit.run(XX, Y, 'CGain')

        exit(1)

else:  # read over the calibration files and analyse these outputs

    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'idealised'),
                         'overview_of_fits.csv')

    if not os.path.isfile(fname):

        for swater in swaters:  # loop over the training soil moisture profiles

            for sample in [None, 1, 2, 3]:

                if sample is None:
                    opath = os.path.join(os.path.join(os.path.join(os.path.join(
                                         base_dir, 'output'), 'calibrations'),
                                         'idealised'), '%s' % (swater))
                    sample = 0  # int written in output file

                else:
                    opath = os.path.join(os.path.join(os.path.join(os.path.join(
                                         base_dir, 'output'), 'calibrations'),
                                         'idealised'), '%s_subsample_%d'
                                         % (swater, sample))

                for file in os.listdir(opath):

                    if file.endswith('.txt'):
                        f = open(os.path.join(opath, file), 'r')
                        model = file.split('.txt')[0]
                        lines = f.readlines()

                        # info to keep
                        k1 = 'fitting method'
                        k2 = 'function evals'
                        k3 = 'data points'
                        k4 = 'Bayesian info crit'
                        k5 = '%) '  # calibrated parameters
                        k6 = '(init'  # calibrated parameters
                        k7 = '+/-'  # calibrated parameters
                        k8 = '=='  # calibrated parameters
                        info = [e.split('=') if (k1 in e) else
                                [e.split('=')[1]] if ((k2 in e) or (k3 in e) or
                                (k4 in e)) else
                                e.split(k5)[0].split(k7)[0].split(':')
                                if (k5 in e)
                                else e.split(k6)[0].split(k7)[0].split(':')
                                if (k6 in e) else
                                e.split(k8)[0].split(':') if (k8 in e) else
                                [''] for e in lines]
                        info = [e.strip('\n') for sub in info for e in sub
                                if e != '']
                        info = [e.replace(' ', '') if (':' in e) else e.strip()
                                for e in info ]

                        # split into sublists containing each solver's info
                        by_solver = [list(sub) for e, sub in
                                     groupby(info, lambda x: k1 not in x) if e]

                        # put that info in a dataframe
                        for solver in by_solver:

                            # deal with ampgo's long name
                            solver[0] = solver[0].split(',')[0]

                            # append the df row
                            dic = {'Model': model, 'training': swater,
                                   'sub-sample': sample, 'solver': solver[0],
                                   'Ntotal': float(solver[1]) *
                                             float(solver[2]),
                                   'BIC': float(solver[3]), 'p1': solver[4],
                                   'v1': float(solver[5])}

                            if len(solver) > 6:
                                if model == 'SOX-OPT':  # deal with the 'factor'
                                    dic['p2'] = solver[8]
                                    dic['v2'] = float(solver[9])

                                else:
                                    dic['p2'] = solver[6]
                                    dic['v2'] = float(solver[7])

                            odf = odf.append(dic, ignore_index=True)

            # add the median param info to rerank the models
            by = ['Model', 'training', 'sub-sample']
            odf['med1'] = (odf['v1'] / odf.groupby(by)['v1'].transform('median')
                           - 1.).abs()
            odf['med2'] = (odf['v2'] / odf.groupby(by)['v2'].transform('median')
                           - 1.).abs()
            odf['med'] = odf[['med1', 'med2']].mean(axis=1)

            # rank the solvers (absolute rankings)
            odf['Rank'] = (odf.sort_values(['BIC', 'med', 'Ntotal'])
                              .groupby(by)[['BIC', 'med']].rank(method='first')
                              .astype(int))

        # change param name for ProfitMax2 to allow differentiation
        odf['p1'].loc[odf['Model'] == 'ProfitMax2'] = 'kmax2'

        # column order
        columns = ['Model', 'training', 'sub-sample', 'solver', 'Rank', 'BIC',
                   'Ntotal', 'p1', 'v1', 'p2', 'v2']

        # save the overview file
        odf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    else:
        odf = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    # best three solvers
    subw = (odf[odf['training'] == 'wet'].groupby('solver')['Rank'].mean()
               .nsmallest(n=3).index.tolist())
    subi = (odf[odf['training'] == 'inter'].groupby('solver')['Rank']
               .mean().nsmallest(n=3).index.tolist())
    subset = odf.groupby('solver')['Rank'].mean().nsmallest(n=3).index.tolist()

    # are the three best solvers the same regardless of training?
    if set(subset) == set(subw) == set(subi):
        print('All top 3 solvers are the same:', subset)
    # if not, are the three overall best in each training's four best?
    else:
        subw = (odf[odf['training'] == 'wet'].groupby('solver')['Rank']
                   .mean().nsmallest(n=4).index.tolist())
        subi = (odf[odf['training'] == 'inter'].groupby('solver')['Rank']
                   .mean().nsmallest(n=4).index.tolist())

        if (set(subw).issuperset(set(subset)) and
            set(subi).issuperset(set(subset))):
            print('Overall top 3 solvers are in each top 4 solvers:', subset)

        else:  # the 'best solvers' are too different, rethink the method
            msg = 'Abort: the best solvers change with the type of training!'
            raise ValueError(msg)

    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'idealised'),
                         'top_3_fits.csv')

    if not os.path.isfile(fname):

        """
        odf = (odf[np.logical_and(odf['solver'].isin(subset),
                                  odf['sub-sample'] == 0)]
                  .drop(['sub-sample'], axis=1))

        # re-rank the solvers
        odf['Rank'] = (odf.groupby(['Model', 'training'])['BIC'].rank()
                          .astype(int))
        """

        # check that there are no duplicated ranks within a group
        eq = (odf.groupby(['Model', 'training'])['Rank'].nunique()
                 .le(len(odf['solver'].unique()) - 1))

        """
        eq_models = eq[eq == True].index.get_level_values(0)
        eq_trainings = eq[eq == True].index.get_level_values(1)

        for i in range(len(eq_models)):

            where = np.logical_and(odf['Model'] == eq_models[i],
                                   odf['training'] == eq_trainings[i])
            sub = odf[where]

            for j in range(len(sub['sub-sample'].unique())):

                subsub = sub[sub['sub-sample'] == float(j)]
                med = subsub['v1'].median()
                Rmin = subsub['Rank'].min()
                dup = subsub['Rank'].duplicated()

            # if rank duplicated, rerank by median param
            if len(sub[sub['Rank'] == sub['Rank'].min()]) > 1:
                odf['Rank'][where] = 3  # deal with duplicated Rank = 1
                idx = sub[sub['v1'] == sub['v1'].median()].index

                if len(idx) > 1:  # if params are still equal, pick fastest
                    sub = sub.loc[idx]
                    idx = sub[sub['Ntotal'] == sub['Ntotal'].min()].index

                odf.loc[idx, 'Rank'] = 1
        """

        if not any(eq):  # no duplicates, things working properly

            keep = []

            for m in odf['Model'].unique():

                sub = odf[odf['Model'] == m]
                subset = (sub.groupby(['solver'])['Rank'].mean().nsmallest(n=3)
                             .index.tolist())
                isub = sub[sub['solver'].isin(subset)].index.tolist()
                keep += [isub]

            keep = pd.Index([e for sublist in keep for e in sublist])
            sdf = odf.loc[keep]
            sdf.reset_index(inplace=True)

            # column order
            columns = ['Model', 'training', 'sub-sample', 'solver', 'Rank',
                       'BIC', 'Ntotal', 'p1', 'v1', 'p2', 'v2']

            # within best 3
            sdf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    else:
        sdf = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'idealised'),
                         'best_fit.csv')

    if not os.path.isfile(fname):  # pick best param

        # full timeseries
        sdf = sdf[sdf['sub-sample'] == 0.]
        odf = odf[odf['sub-sample'] == 0.]

        # min rank is the best param
        sdf['Rank'] = (sdf.groupby(['Model', 'training'])['Rank'].rank()
                           .astype(int))
        sdf = sdf[sdf['Rank'] == 1].drop(['Rank'], axis=1)

        # add params to Tuzet, WUE-LWP, CGain, CMax
        sdf['p3'] = np.nan  # own kmax
        sdf['v3'] = np.nan

        # specific param names on a per model basis
        sdf['p2'].loc[sdf['Model'] == 'WUE-LWP'] = 'kmaxWUE'
        sdf['p2'].loc[sdf['Model'] == 'CGain'] = 'kmaxCN'
        sdf['p3'].loc[sdf['Model'] == 'Tuzet'] = 'kmaxT'
        sdf['p3'].loc[sdf['Model'] == 'CMax'] = 'kmaxCM'

        for training in sdf['training'].unique():  # add the param values

            sub1 = sdf[sdf['training'] == training]
            sub2 = odf[odf['training'] == training]

            for m in ['WUE-LWP', 'CGain', 'Tuzet', 'CMax']:

                solver = sub1[sub1['Model'] == m].solver.values[0]
                kval = (sub2[np.logical_and(sub2['solver'] == solver,
                        sub2['Model'] == 'ProfitMax')]).v1  # own kmax
                idx = sub1[np.logical_and(sub1['Model'] == m,
                                          sub1['solver'] == solver)].index

                if m in ['WUE-LWP', 'CGain']:
                    sdf.loc[idx, 'v2'] = float(kval)

                else:
                    sdf.loc[idx, 'v3'] = float(kval)

        # column order
        columns = ['Model', 'training', 'solver', 'BIC', 'Ntotal', 'p1',
                   'v1', 'p2', 'v2', 'p3', 'v3']

        # best calibrations
        sdf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    exit(1)
