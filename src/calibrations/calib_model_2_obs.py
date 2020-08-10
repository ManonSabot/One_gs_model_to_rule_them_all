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

Powell: a conjugate direction method. It performs sequential one-dimensional
        minimizations along each vector of the directions set, which is updated
        at each iteration of the main minimization loop. The function need not
        be differentiable, and no derivatives are taken.
** Press, William H., et al. Numerical recipes. Vol. 3. Cambridge:
   Cambridge University Press, 1989.

DA: a stochastic approach which combines the generalization of CSA
    (Classical Simulated Annealing) and FSA (Fast Simulated Annealing) coupled
    to a strategy for applying a local search on accepted locations.
** Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated Annealing Algorithm
   and Its Application to the Thomson Model. Physics Letters A, 233, 216-220
   (1997).
** Xiang Y, Gong XG. Efficiency of Generalized Simulated Annealing. Physical
   Review E, 62, 4473 (2000).
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
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import net_radiation  # preset it for model training
from fit_models import NLMFIT  # training functions

#==============================================================================

def prep_training_N_target(ifile, ofile):

    # read in the input and output data
    df1, __ = read_csv(ifile)
    df2, __ = read_csv(ofile)

    # add the predawn soil moisture profile to the input data
    df1['Ps_pd'] = df1['Ps'].copy()  # daily pre-dawn soil water potentials

    # non time-sensitive: last valid value propagated until next valid
    df1.fillna(method='ffill', inplace=True)

    # add Rnet to the input (no ET or soil albedo feedbacks, this can be done)
    df1['Rnet'] = net_radiation(df1)
    df1['scale2can'] = 1.

    # exclude Rnet < 0. and reindex
    Y = df2['gs'][df1['Rnet'] > 0.] * 1000.  # mmol m-2 s-1
    X = df1[df1['Rnet'] > 0.]
    X.reset_index(inplace=True, drop=True)
    Y.reset_index(inplace=True, drop=True)

    return X, Y


#==============================================================================

to_fit = False

# declare empty dataframe which will be used to analyse the calibrations
odf = pd.DataFrame(columns=['Model', 'training', 'solver', 'BIC', 'Rank', 'p1',
                            'v1', 'p2', 'v2'])

# where should the fitting solvers' outputs be stored?
base_dir = get_main_dir()  # working paths

if to_fit:

    ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                         'calibrations'), 'obs_driven')
    opath = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                         'calibrations'), 'obs_driven')
    xfiles = sorted([e for e in os.listdir(ipath) if e.endswith('_x.csv')])
    yfiles = sorted([e for e in os.listdir(ipath) if e.endswith('_y.csv')])

    xfiles = xfiles[8:9]
    yfiles = yfiles[8:9]

    for ifile, ofile in zip(xfiles, yfiles):  # loop over the files

        X, Y = prep_training_N_target(os.path.join(ipath, ifile),
                                      os.path.join(ipath, ofile))

        # where should the calibration output be stored?
        out = os.path.join(opath, ofile.split('_y')[0])

        if not os.path.isdir(out):  # make new dirs if they don't exist
            os.makedirs(out)

        # use a non-linear least square minimiser to train the models
        for test in ['differential_evolution', 'powell', 'dual_annealing']:

            XX = X.copy()
            nlmfit = NLMFIT(method=test, store=out, inf_gb=False)

            __ = nlmfit.run(XX, Y, 'Medlyn-LWP', g1=True)
            __ = nlmfit.run(XX, Y, 'SOX')
            __ = nlmfit.run(XX, Y, 'SOX-OPT')
            __ = nlmfit.run(XX, Y, 'CAP')
            __ = nlmfit.run(XX, Y, 'MES')
            __ = nlmfit.run(XX, Y, 'LeastCost')
            fkmax = nlmfit.run(XX, Y, 'ProfitMax')

            # these models come after ProfitMax as they use its kmax
            XX['kmax'] = fkmax['kmax']
            __ = nlmfit.run(XX, Y, 'Tuzet')
            __ = nlmfit.run(XX, Y, 'WUE-LWP')
            __ = nlmfit.run(XX, Y, 'CGainNet')
            __ = nlmfit.run(XX, Y, 'CMax')

    exit(1)

else:  # read over the calibration files and analyse these outputs
    opath = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                         'calibrations'), 'obs_driven')
    fname = os.path.join(opath, 'overview_of_fits.csv')

    if not os.path.isfile(fname):

        site_spp = [e[1] for e in os.walk(opath)][0]  # directories

        for training in site_spp:  # loop over the site x spp combinations

            for file in os.listdir(os.path.join(opath, training)):

                if file.endswith('.txt') and not file.endswith('2.txt'):
                    f = open(os.path.join(os.path.join(opath, training), file),
                             'r')
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

                        # append the df row
                        dic = {'Model': model, 'training': training,
                               'solver': solver[0], 'Ntotal': float(solver[1]) *
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

            # rank the solvers (absolute ranking)
            odf['Rank'] = (odf.groupby(['Model', 'training'])['BIC'].rank()
                              .astype(int))

        # column order
        columns = ['Model', 'training', 'solver', 'Rank', 'BIC', 'Ntotal', 'p1',
                   'v1', 'p2', 'v2']

        # save the overview file
        odf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    else:
        odf = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    fname = os.path.join(opath, 'best_fit.csv')

    if not os.path.isfile(fname):  # pick best param

        # where are there several equal best ranks within a group?
        eq = odf.groupby(['Model', 'training'])['Rank'].nunique().le(2)
        eq_models = eq[eq == True].index.get_level_values(0)
        eq_trainings = eq[eq == True].index.get_level_values(1)

        for i in range(len(eq_models)):

            where = np.logical_and(odf['Model'] == eq_models[i],
                                   odf['training'] == eq_trainings[i])
            sub = odf[where]

            # if min rank duplicated, assign 1 to median params
            if len(sub[sub['Rank'] == sub['Rank'].min()]) > 1:
                odf['Rank'][where] = 3  # deal with duplicated Rank = 1
                idx = sub[sub['v1'] == sub['v1'].median()].index
                print(sub, idx)

                if len(idx) > 1:  # if params are equal, pick fastest
                    sub = sub.loc[idx]
                    idx = sub[sub['Ntotal'] == sub['Ntotal'].min()].index
                    print(sub, idx)

                odf.loc[idx, 'Rank'] = 1

        # add params to Tuzet, WUE-LWP, CGainNet, CMax
        odf['p3'] = np.nan  # own kmax
        odf['v3'] = np.nan

        # specific param names on a per model basis
        odf['p2'].loc[odf['Model'] == 'WUE-LWP'] = 'kmaxWUE'
        odf['p2'].loc[odf['Model'] == 'CGainNet'] = 'kmaxCN'
        odf['p3'].loc[odf['Model'] == 'Tuzet'] = 'kmaxT'
        odf['p3'].loc[odf['Model'] == 'CMax'] = 'kmaxCM'

        for training in odf['training'].unique():  # add the param values

            sub = odf[odf['training'] == training]

            for solver in sub['solver'].unique():

                # own kmax
                value = (sub[np.logical_and(sub['solver'] == solver,
                         sub['Model'] == 'ProfitMax')]).v1

                idx = (sub[np.logical_and(sub['solver'] == solver,
                                         sub['Model'].isin(['WUE-LWP',
                                                            'CGainNet']))]
                          .index)
                odf.loc[idx, 'v2'] = float(value)

                idx = (sub[np.logical_and(sub['solver'] == solver,
                                         sub['Model'].isin(['Tuzet', 'CMax']))]
                          .index)
                odf.loc[idx, 'v3'] = float(value)

        # Rank = 1 is assumed to be the best param
        odf = odf[odf['Rank'] == 1].drop(['Rank'], axis=1)

        # column order
        columns = ['Model', 'training', 'solver', 'BIC', 'Ntotal', 'p1',
                   'v1', 'p2', 'v2', 'p3', 'v3']

        # best calibrations
        odf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    exit(1)
