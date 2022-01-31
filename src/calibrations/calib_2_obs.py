#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a simplified version of the script used to calibrate the 12 gs
models to observations of leaf gas exchange. Specifically, in this
version of the script, there is no parallelisation of the code when
performing the calibrations, so everything is written in a loop. This is
obviously sub-optimal, and anyone who wishes to recreate our results or
reuse the method may want to parallelise this process.

Information on the minimizers used is available from the documentation
of the lmfit package, at https://lmfit.github.io/lmfit-py/

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Calibration of the gs models"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import argparse  # read in the user input
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import warnings  # ignore warnings

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files

from calibrations import prep_training_N_target  # prepare the data
from calibrations import extract_calib_info  # process text calib files
from calib_utils import NLMFIT  # training functions

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# ======================================================================

def main(calib=False, solvers=['differential_evolution']):

    """
    Main function: either calibrates 11 models to reference gs model
                   outputs, for two distinct soil water profiles X N
                   subsets of data (any number could be created), using
                   seven minimizers, or analyses the parameter
                   calibrations.

    Arguments:
    ----------
    calib: bool
        if True, calibrates the models. Otherwise analyses the outputs

    solvers: list
        minimizing method.s to use

    Returns:
    --------
    Text files that contain all the calibration information, organised
    in a tree of directories under 'output/calibrations/obs_driven',
    plus summary files: 'overview_of_fits.csv' and 'best_fit.csv'.

    """

    base_dir = get_main_dir()  # working paths

    # path to input files
    ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                         'calibrations'), 'obs_driven')

    # driving + target files
    xfiles = sorted([e for e in os.listdir(ipath) if e.endswith('_x.csv')])
    yfiles = sorted([e for e in os.listdir(ipath) if e.endswith('_y.csv')])

    if calib:

        calibrate(ipath, xfiles, yfiles, solvers)

    else:

        analyse_calib(ipath, xfiles)

    return


def calibrate(ipath, xfiles, yfiles, solvers):

    """
    Calls the NLMFIT wrapper to calibrate the models.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    xfiles: list
        names of files that contains the input met & parameter data

    yfiles: list
        names of files that contain the gs data to calibrate against

    solvers: list
        minimizing method.s to use

    Returns:
    --------
    Text files that contain all the calibration information, organised
    in a tree of directories under 'output/calibrations/obs_driven/'.

    """

    for ifile, ofile in zip(xfiles, yfiles):  # loop over the files

        X, Y = prep_training_N_target(os.path.join(ipath, ifile),
                                      os.path.join(ipath, ofile))

        # where should the calibration output be stored?
        opath = ipath.replace('input', 'output')
        out = os.path.join(opath, ofile.split('_y')[0])

        if not os.path.isdir(out):  # make new dirs if they don't exist
            os.makedirs(out)

        # use a non-linear least square minimizer to train the models
        for minimizer in solvers:

            XX = X.copy()
            nlmfit = NLMFIT(method=minimizer, store=out, inf_gb=False)

            __ = nlmfit.run(XX, Y, 'Medlyn')
            __ = nlmfit.run(XX, Y, 'Tuzet')
            __ = nlmfit.run(XX, Y, 'Eller')

            __ = nlmfit.run(XX, Y, 'SOX-OPT')
            __ = nlmfit.run(XX, Y, 'CAP')
            __ = nlmfit.run(XX, Y, 'MES')
            __ = nlmfit.run(XX, Y, 'LeastCost')
            __ = nlmfit.run(XX, Y, 'ProfitMax2')

            # use ProfitMax's kmax
            fkmax = nlmfit.run(XX, Y, 'ProfitMax')
            XX['kmax'] = fkmax['kmax']

            __ = nlmfit.run(XX, Y, 'WUE-LWP')
            __ = nlmfit.run(XX, Y, 'CMax')
            __ = nlmfit.run(XX, Y, 'CGain')

    return


def analyse_calib(ipath, xfiles):

    """
    Combines all the calibration outputs and organises them into .csv
    files.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    xfiles: list
        names of files that contains the input met & parameter data

    Returns:
    --------
    Summary .csv files in 'output/calibrations/idealised/':
        * 'overview_of_fits.csv' contains all the important information
          from all the calibrations performed
        * 'best_fit.csv' contains information obtained by the best
          calibration method for each gs model at each site X species
          (i.e., for each dataset)

    """

    # declare empty dataframe, used to analyse the calibs
    columns = ['Model', 'training', 'solver', 'Rank', 'BIC', 'Ntotal', 'p1',
               'v1', 'ci1', 'p2', 'v2', 'ci2']
    odf = pd.DataFrame(columns=columns)

    # where to store this odf
    opath = ipath.replace('input', 'output')
    fname = os.path.join(opath, 'overview_of_fits.csv')

    # read over the calibration files and analyse these outputs
    if not os.path.isfile(fname):

        site_spp = [e[1] for e in os.walk(opath)][0]  # directories

        for training in site_spp:  # loop over the site x spp combinations

            for file in os.listdir(os.path.join(opath, training)):

                if file.endswith('.txt'):
                    model = file.split('.txt')[0]
                    info = extract_calib_info(os.path.join(os.path.join(opath,
                                              training), file))

                    # put that info in a dataframe
                    for solver in info:

                        # append the df row
                        dic = {'Model': model, 'training': training,
                               'solver': solver[0],
                               'Ntotal': float(solver[1]) * float(solver[2]),
                               'BIC': float(solver[3]), 'p1': solver[4],
                               'v1': float(solver[5]), 'ci1': float(solver[6])}

                        if len(solver) > 7:
                            dic['p2'] = solver[7]
                            dic['v2'] = float(solver[8])
                            dic['ci2'] = float(solver[9])

                        odf = odf.append(dic, ignore_index=True)

            # add the median param info to rerank the models
            by = ['Model', 'training']
            odf['med1'] = ((odf['v1'] /
                            odf.groupby(by)['v1'].transform('median') - 1.)
                           .abs())
            odf['med2'] = ((odf['v2'] /
                            odf.groupby(by)['v2'].transform('median') - 1.)
                           .abs())
            odf['med'] = odf[['med1', 'med2']].mean(axis=1)

            # rank the solvers (absolute ranking)
            odf['Rank'] = (odf.sort_values(['BIC', 'med'])
                              .groupby(by)[['BIC', 'med']].rank(method='first')
                              .astype(int))

        # change param name for ProfitMax2 to allow differentiation
        odf.loc[odf['Model'] == 'ProfitMax2', 'p1'] = 'kmax2'

        # save the overview file
        odf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    else:
        odf = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    fname = os.path.join(opath, 'best_fit.csv')

    if not os.path.isfile(fname):  # pick best param

        # check whether the calibrated parameters are stuck at boundary
        stuck = []
        nlmfit = NLMFIT()

        for ifile in xfiles:

            df, __ = read_csv(os.path.join(ipath, ifile))  # ref params
            sub = odf[odf['training'] == ifile.split('_x.csv')[0]]

            for e in np.append(sub['p1'].unique(),
                               sub['p2'].dropna().unique()):

                sub1 = sub[sub['p1'] == e]
                sub2 = sub[sub['p2'] == e]
                min, max = nlmfit.param_space(e, P88=df.loc[0, 'P88'])
                min += 0.01 * min  # above min is not stuck at bound
                max -= 0.01 * max  # below max is not stuck at bound

                if len(sub1) > 0:
                    lims = np.logical_or(sub1['v1'] < min, sub1['v1'] > max)

                    if any(lims):
                        stuck += sub1.loc[lims, 'v1'].index.to_list()

                if len(sub2) > 0:
                    lims = np.logical_or(sub2['v2'] < min, sub2['v2'] > max)

                    if any(lims):
                        stuck += sub2.loc[lims, 'v2'].index.to_list()

        # boundary params
        if len(stuck) > 0:
            sub = odf[odf.index.isin(stuck)]
            eq = sub.groupby(['Model', 'training']).size().le(3)
            eq_models = eq[eq == True].index.get_level_values(0)
            eq_trainings = eq[eq == True].index.get_level_values(1)

            for i in range(len(eq_models)):

                where = np.logical_and(odf['Model'] == eq_models[i],
                                       odf['training'] == eq_trainings[i])
                sub = odf[where]

                # which of these values are at the boundary?
                if not all(sub.loc[sub.index.isin(stuck), 'Rank'].values >
                           sub.loc[~sub.index.isin(stuck), 'Rank'].values):
                    odf.loc[sub[sub.index.isin(stuck)].index, 'Rank'] = 3

                    while all(sub.loc[~sub.index.isin(stuck), 'Rank'] > 1):
                        odf.loc[sub[~sub.index.isin(stuck)].index, 'Rank'] -= 1
                        sub.loc[~sub.index.isin(stuck), 'Rank'] -= 1

        # are there still several equal best ranks within a group?
        eq = odf.groupby(['Model', 'training'])['Rank'].nunique().le(2)
        eq_models = eq[eq == True].index.get_level_values(0)
        eq_trainings = eq[eq == True].index.get_level_values(1)

        for i in range(len(eq_models)):

            where = np.logical_and(odf['Model'] == eq_models[i],
                                   odf['training'] == eq_trainings[i])
            sub = odf[where]

            # if min rank duplicated, assign 1 to median params
            if len(sub[sub['Rank'] == sub['Rank'].min()]) > 1:
                odf.loc[where, 'Rank'] = 3  # deal with duplicated Rank = 1
                idx = sub[sub['v1'] == sub['v1'].median()].index

                if len(idx) > 1:  # if params are equal, pick fastest
                    sub = sub.loc[idx]
                    idx = sub[sub['Ntotal'] == sub['Ntotal'].min()].index

                odf.loc[idx, 'Rank'] = 1

        # add params to Tuzet, WUE-LWP, CGain, CMax
        odf['p3'] = np.nan  # own kmax
        odf['v3'] = np.nan
        odf['ci3'] = np.nan

        # specific param names on a per model basis
        odf.loc[odf['Model'] == 'WUE-LWP', 'p2'] = 'kmaxWUE'
        odf.loc[odf['Model'] == 'CGain', 'p2'] = 'kmaxCN'
        odf.loc[odf['Model'] == 'CMax', 'p3'] = 'kmaxCM'

        for training in odf['training'].unique():  # add the param values

            sub = odf[odf['training'] == training]

            for solver in sub['solver'].unique():

                # own kmax
                v1 = (sub[np.logical_and(sub['solver'] == solver,
                                         sub['Model'] == 'ProfitMax')]).v1
                ci1 = (sub[np.logical_and(sub['solver'] == solver,
                                          sub['Model'] == 'ProfitMax')]).ci1

                idx = (sub[np.logical_and(sub['solver'] == solver,
                                          sub['Model'].isin(['WUE-LWP',
                                                             'CGain']))]).index
                odf.loc[idx, 'v2'] = float(v1)
                odf.loc[idx, 'ci2'] = float(ci1)

                idx = (sub[np.logical_and(sub['solver'] == solver,
                                          sub['Model'].isin(['CMax']))]).index
                odf.loc[idx, 'v3'] = float(v1)
                odf.loc[idx, 'ci3'] = float(ci1)

        # Rank = 1 is assumed to be the best param
        odf = odf[odf['Rank'] == 1].drop(['Rank'], axis=1)

        # column order
        columns = ['Model', 'training', 'solver', 'BIC', 'Ntotal', 'p1',
                   'v1', 'ci1', 'p2', 'v2', 'ci2', 'p3', 'v3', 'ci3']

        # best calibrations
        odf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = "perform calibrations or analyse calibration outputs?"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--calibrate', action='store_true',
                        help='calibrate the models')
    args = parser.parse_args()

    # user input
    solvers = ['differential_evolution', 'basinhopping', 'powell',
               'dual_annealing']

    main(calib=args.calibrate, solvers=solvers)
