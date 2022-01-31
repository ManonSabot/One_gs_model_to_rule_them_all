#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a simplified version of the script used to calibrate 11 gs
models to synthetic gs outputs. Specifically, in this version of the
script, there is no parallelisation of the code when performing the
calibrations, so everything is written in a loop. This is obviously
sub-optimal, and anyone who wishes to recreate our results or reuse the
method may want to parallelise this process.

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
from calibrations import check_idealised_files  # training files?
from calibrations import prep_training_N_target  # prepare the data
from calibrations import extract_calib_info  # process text calib files
from calib_utils import NLMFIT  # training functions

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(calib=False, sample=None, swaters=['wet'],
         solvers=['differential_evolution']):

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

    sample: int
        if an int is given, a random (but numbered and stored)
        subsampling will be applied to the data, drawing a 7-day period
        for calibration

    swaters: list
        the soil moisture profiles to calibrate over, e.g., 'wet' and
        'inter'

    solvers: list
        minimizing method.s to use

    Returns:
    --------
    Text files that contain all the calibration information, organised
    in a tree of directories under 'output/calibrations/idealised',
    plus summary files: 'overview_of_fits.csv', 'top_fits.csv', and
                        'best_fit.csv'.

    """

    base_dir = get_main_dir()  # working paths

    # path to input files
    ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                         'calibrations'), 'idealised')

    # driving + target files
    xfile = ifile = 'training_x.csv'
    yfiles = ['training_%s_y.csv' % (swater) for swater in swaters]

    if calib:

        for yfile in yfiles:  # check for the idealised training files

            check_idealised_files(os.path.join(ipath, ifile),
                                  os.path.join(ipath, yfile))

        calibrate(ipath, xfile, yfiles, solvers, sample=sample)

    else:

        analyse_calib(ipath, swaters)

    return


def subsample(training, target, sfname):

    """
    Randomly subsamples the datasets into 7-day long datasets.

    Arguments:
    ----------
    training: pandas dataframe
        dataframe containing all input data & params

    target: pandas series
        stomatal conductance [mmol m-2 s-1]

    sfname: string
        subsample reference file name, including path

    Returns:
    --------
    X: pandas dataframe
        dataframe containing the (sampled) input data & all params

    Y: pandas series
        stomatal conductance [mmol m-2 s-1]

    """

    if not os.path.isfile(sfname):  # randomly subsample
        size = 7
        doys = np.unique(training['doy'])
        sub = np.unique(np.random.randint(doys[0], doys[-1] + 1, size=size))

        while len(sub) < size:

            sub = np.unique(np.append(sub, np.unique(np.random.randint(doys[0],
                            doys[-1] + 1, size=size - len(sub)))))

        ssub = np.asarray(training.loc[training['doy'].isin(sub)].index)

        # store the subsampling distribution for reference if re-used
        np.save(sfname, ssub)

    else:  # draw from existing ref. file
        ssub = np.load(sfname)

    # now accordingly subsample input and output
    Y = target.iloc[ssub]
    X = training.iloc[ssub]
    X.reset_index(inplace=True, drop=True)

    return X, Y


def calibrate(ipath, ifile, ofiles, solvers, sample=None):

    """
    Calls the NLMFIT wrapper to calibrate the models.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    ifile: string
        name of file that contains the input met & parameter data

    ofiles: list
        names of files that contain the gs data to calibrate against

    solvers: list
        minimizing method.s to use

    sample: int
        if an int is given, a random (but numbered and stored)
        subsampling will be applied to the data, drawing a 7-day period
        for calibration

    Returns:
    --------
    Text files that contain all the calibration information, organised
    in a tree of directories under 'output/calibrations/idealised/'.

    """

    for ofile in ofiles:  # loop over the soil moisture profiles

        X, Y = prep_training_N_target(os.path.join(ipath, ifile),
                                      os.path.join(ipath, ofile),
                                      profile=ofile.split('_')[1])

        if sample is not None:  # randomly subsample a week of data
            sfname = os.path.join(ipath, 'subsample_%d.npy' % (sample))
            X, Y = subsample(X, Y, sfname)

        # where should the calibration output be stored?
        opath = ipath.replace('input', 'output')

        if sample is not None:  # move files to the relevant sub-dir
            opath = os.path.join(opath, '%s_subsample_%d' %
                                        (ofile.split('_')[1], sample))

        else:
            opath = os.path.join(opath, ofile.split('_')[1])

        if not os.path.isdir(opath):  # make dirs if they don't exist
            os.makedirs(opath)

        # use a non-linear least square minimiser to train the models
        for minimizer in solvers:

            XX = X.copy()
            nlmfit = NLMFIT(method=minimizer, store=opath)

            __ = nlmfit.run(XX, Y, 'Eller')

            __ = nlmfit.run(XX, Y, 'SOX-OPT')
            __ = nlmfit.run(XX, Y, 'CAP')
            __ = nlmfit.run(XX, Y, 'MES')
            __ = nlmfit.run(XX, Y, 'LeastCost')
            __ = nlmfit.run(XX, Y, 'ProfitMax2')

            # use ProfitMax's kmax
            fkmax = nlmfit.run(XX, Y, 'ProfitMax')
            XX['kmax'] = fkmax['kmax']

            __ = nlmfit.run(XX, Y, 'Tuzet')
            __ = nlmfit.run(XX, Y, 'WUE-LWP')
            __ = nlmfit.run(XX, Y, 'CMax')
            __ = nlmfit.run(XX, Y, 'CGain')

    return


def analyse_calib(ipath, swaters):

    """
    Combines all the calibration outputs and organises them into .csv
    files.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    swaters: list
        the soil moisture profiles to calibrate over, e.g., 'wet' and
        'inter'

    Returns:
    --------
    Summary .csv files in 'output/calibrations/idealised/':
        * 'overview_of_fits.csv' contains all the important information
          from all the calibrations performed
        * 'top_fits.csv' contains information from the fits obtained
          from the three/four overall best calibration methods
        * 'best_fit.csv' contains information obtained by the best
          calibration method for each gs model under each different set
          of conditions (i.e., for each dataset)

    """

    # declare empty dataframe, used to analyse the calibrations
    columns = ['Model', 'training', 'sub-sample', 'solver', 'Rank', 'BIC',
               'Ntotal', 'p1', 'v1', 'ci1', 'p2', 'v2', 'ci2']
    odf = pd.DataFrame(columns=columns)

    # where to store this odf
    opath = ipath.replace('input', 'output')
    fname = os.path.join(opath, 'overview_of_fits.csv')

    # read over the calibration files and analyse these outputs
    if not os.path.isfile(fname):

        for swater in swaters:  # loop over the training soil moisture profiles

            for sample in [None] + [int(e.split('_')[1].split('.')[0])
                                    for e in os.listdir(ipath)
                                    if '.npy' in e]:

                if sample is None:
                    oopath = os.path.join(opath, swater)
                    sample = 0  # int written in output file

                else:
                    oopath = os.path.join(opath, '%s_subsample_%d'
                                          % (swater, sample))

                for file in os.listdir(oopath):

                    if file.endswith('.txt'):
                        model = file.split('.txt')[0]
                        info = extract_calib_info(os.path.join(oopath, file))

                        # put that info in a dataframe
                        for solver in info:

                            # deal with ampgo's long name
                            solver[0] = solver[0].split(',')[0]

                            # append the df row
                            dic = {'Model': model, 'training': swater,
                                   'sub-sample': sample, 'solver': solver[0],
                                   'Ntotal': float(solver[1]) *
                                   float(solver[2]), 'BIC': float(solver[3]),
                                   'p1': solver[4], 'v1': float(solver[5]),
                                   'ci1': float(solver[6])}

                            if len(solver) > 7:
                                dic['p2'] = solver[7]
                                dic['v2'] = float(solver[8])
                                dic['ci2'] = float(solver[9])

                            odf = odf.append(dic, ignore_index=True)

            # add the median param info to rerank the models
            by = ['Model', 'training', 'sub-sample']
            odf['med1'] = ((odf['v1'] /
                            odf.groupby(by)['v1'].transform('median') - 1.)
                           .abs())
            odf['med2'] = ((odf['v2'] /
                            odf.groupby(by)['v2'].transform('median') - 1.)
                           .abs())
            odf['med'] = odf[['med1', 'med2']].mean(axis=1)

            # rank the solvers (absolute rankings)
            odf['Rank'] = (odf.sort_values(['BIC', 'med', 'Ntotal'])
                              .groupby(by)[['BIC', 'med']].rank(method='first')
                              .astype(int))

        # change param name for ProfitMax2 to allow differentiation
        odf.loc[odf['Model'] == 'ProfitMax2', 'p1'] = 'kmax2'

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
        print('Same overall top 3 solvers regardless of soil moisture:',
              subset)

    else:  # are the three overall best in each training's four best?
        subw = (odf[odf['training'] == 'wet'].groupby('solver')['Rank']
                .mean().nsmallest(n=4).index.tolist())
        subi = (odf[odf['training'] == 'inter'].groupby('solver')['Rank']
                .mean().nsmallest(n=4).index.tolist())

        if (set(subw).issuperset(set(subset)) and
           set(subi).issuperset(set(subset))):
            print('Overall top 3 solvers are in each soil moisture top 4:',
                  subset)

        else:  # there are no equivocal three best solvers, test 4?
            subset = (odf.groupby('solver')['Rank'].mean().nsmallest(n=4)
                         .index.tolist())
            subset = [[e, (subset + subw + subi).count(e)]
                      for e in set(subset + subw + subi)]
            subset = [e[0] for e in subset if e[1] > 1]

            if len(subset) > 3:
                print('The overall top 4 solvers are:', subset)

            else:
                msg = 'Abort: there are no better solvers!'
                raise ValueError(msg)

    fname = os.path.join(opath, 'top_fits.csv')

    if not os.path.isfile(fname):

        # check that there are no duplicated ranks within a group
        eq = (odf.groupby(['Model', 'training'])['Rank'].nunique()
                 .le(len(odf['solver'].unique()) - 1))

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
                       'BIC', 'Ntotal', 'p1', 'v1', 'ci1', 'p2', 'v2', 'ci2']

            # within best 3
            sdf[columns].to_csv(fname, index=False, na_rep='',
                                encoding='utf-8')

    else:
        sdf = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    fname = os.path.join(opath, 'best_fit.csv')

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
        sdf['ci3'] = np.nan

        # specific param names on a per model basis
        sdf.loc[sdf['Model'] == 'WUE-LWP', 'p2'] = 'kmaxWUE'
        sdf.loc[sdf['Model'] == 'CGain', 'p2'] = 'kmaxCN'
        sdf.loc[sdf['Model'] == 'Tuzet', 'p3'] = 'kmaxT'
        sdf.loc[sdf['Model'] == 'CMax', 'p3'] = 'kmaxCM'

        for training in sdf['training'].unique():  # add the param values

            sub1 = sdf[sdf['training'] == training]
            sub2 = odf[odf['training'] == training]

            for m in ['WUE-LWP', 'CGain', 'Tuzet', 'CMax']:

                solver = sub1[sub1['Model'] == m].solver.values[0]

                # own kmax
                kval = (sub2[np.logical_and(sub2['solver'] == solver,
                        sub2['Model'] == 'ProfitMax')]).v1
                cival = (sub2[np.logical_and(sub2['solver'] == solver,
                         sub2['Model'] == 'ProfitMax')]).ci1
                idx = sub1[np.logical_and(sub1['Model'] == m,
                                          sub1['solver'] == solver)].index

                if m in ['WUE-LWP', 'CGain']:
                    sdf.loc[idx, 'v2'] = float(kval)
                    sdf.loc[idx, 'ci2'] = float(cival)

                else:
                    sdf.loc[idx, 'v3'] = float(kval)
                    sdf.loc[idx, 'ci3'] = float(cival)

        # column order
        columns = ['Model', 'training', 'solver', 'BIC', 'Ntotal', 'p1',
                   'v1', 'ci1', 'p2', 'v2', 'ci2', 'p3', 'v3', 'ci3']

        # best calibrations
        sdf[columns].to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = "perform calibrations or analyse calibration outputs?"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--calibrate', action='store_true',
                        help='calibrate the models')
    parser.add_argument('-s', '--sample', type=int, default=None,
                        help='sample on which to calibrate')
    args = parser.parse_args()

    # user input
    swaters = ['wet', 'inter']
    solvers = ['differential_evolution', 'basinhopping', 'nelder', 'powell',
               'cobyla', 'dual_annealing', 'ampgo']

    main(calib=args.calibrate, sample=args.sample, swaters=swaters,
         solvers=solvers)
