#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code that runs the idealised model simulations where the models are
calibrated to one another, and generates summary files of the change in
each model's outputs relative to the Medlyn model's outputs.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Harmonized idealised model simulations"
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
from TractLSM import conv  # unit converter
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from calibrations.calib_utils import soil_water  # soil moist. profiles
from sim_utils import build_calibrated_forcing  # forcing files
from TractLSM import hrun  # run the models

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# ======================================================================

def main(runsim=False, swaters=['wet'], models=['Medlyn', 'ProfitMax']):

    """
    Main function: either runs the 12 harmonized models, for two
                   distinct soil water profiles, or analyses the model
                   outputs.

    Arguments:
    ----------
    runsim: bool
        if True, runs the models. Otherwise analyses the model outputs

    swaters: list
        the soil moisture profiles for which to run the models, e.g.,
        'wet' and 'inter'

    models: list
        the models to run

    Returns:
    --------
    .csv files that contain all the simulations, plus analyses files,
    all stored under 'output/simulations/idealised/insample/'.

    """

    base_dir = get_main_dir()  # working paths

    # path to input files
    ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                         'simulations'), 'idealised')

    if not os.path.isdir(ipath):  # make dir
        os.makedirs(ipath)

    # driving + model output file names
    xfiles = ['%s_calibrated.csv' % (swater) for swater in swaters]
    yfiles = ['%s.csv' % (swater) for swater in swaters]

    # check whether the input/output files exist
    if (sum([e in os.listdir(ipath) for e in xfiles]) != len(xfiles) or
        sum([e in os.listdir(os.path.join(ipath.replace('input', 'output'),
             'insample')) for e in yfiles]) != len(yfiles)):
        run_simulations(ipath, xfiles, yfiles, models)

    if not runsim:
        analyse_simulations(ipath, xfiles, yfiles)

    return


def run_simulations(ipath, xfiles, yfiles, models):

    """
    Calls TractLSM to run the models.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    xfiles: list
        names of files that contains the input met & parameter data

    yfiles: list
        names under which to save the model output files

    models: list
        models to run

    Returns:
    --------
    .csv files that contain all the simulations, under
    'output/simulations/idealised/insample/'.

    """

    for ifile, ofile in zip(xfiles, yfiles):  # loop over the files

        if not os.path.isfile(os.path.join(ipath, ifile)):
            build_calibrated_forcing(ipath, ofile.split('.')[0])

        # output dir paths
        opath = os.path.join(ipath.replace('input', 'output'), 'insample')

        if not os.path.isdir(opath):  # create dir
            os.makedirs(opath)

        # output file name
        fname = os.path.join(opath, ofile)

        if not os.path.isfile(fname):  # run model and save output
            df, __ = read_csv(os.path.join(ipath, ifile))  # input

            # add soil moisture profile
            df['sw'], df['Ps'] = soil_water(df, ofile.split('.')[0])
            df.fillna(method='ffill', inplace=True)

            # run models
            __ = hrun(fname, df, len(df.index), 'Farquhar', models=models,
                      resolution='med', inf_gb=True)

    return


def combine_dfs(df1, df2, profile):

    """
    Combines input and output data during daytime hours.

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing all input data & params

    df2: pandas dataframe
        dataframe containing all output data

    profile: string
        soil moisture profile, 'wet' or 'inter'

    Returns:
    --------
    df2: pandas dataframe
        dataframe containing all input & output data but no params

    """

    # merge the inputs / outputs from all the simulations
    df2 = df2.merge(df1, on=['doy', 'hod'])

    # only keep the variables, i.e. remove parameters
    df2 = df2.iloc[:, :df2.columns.get_loc('u')]
    df2 = df2.drop(df2.filter(like='Ps(').columns, axis=1)
    df2['CO2'] = df1['CO2'].iloc[0]  # keep Ca

    # restrict to the hod when photosynthesis happens in the models
    df2 = df2[df2[df2.filter(like='gs(').columns].sum(axis=1) > 0.]

    # add the info about the simulation
    df2['soil'] = profile  # which sim?

    return df2


def relative_changes(df, which='mean'):

    """
    Calculates the amount of change in gs, A, E, and the WUE (A/E) of
    each model, for the soil moisture profiles, at various times of the
    day, all relative to the Medlyn model.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input & output data

    which: string
        the kind of relative changes to calculate, either 'mean', 'max',
        or 'cumul' (the latter of which computes weekly cumulative
        changes)

    Returns:
    --------
    A dataframe that contains information on change in the gas exchange
    variables relative to the Medlyn model.

    """

    # deal with NaNs
    df.replace(9999., np.nan, inplace=True)
    df.replace('9999.0', np.nan, inplace=True)

    # variables to consider
    gs = df.filter(like='gs(').columns.to_list()
    A = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()

    if which == 'cumul':
        Nweeks = len(df['doy'].unique()) / 7
        df = df.groupby('soil').sum() / Nweeks
        df[gs] *= conv.SEC_2_HLFHR
        df[A] *= conv.umolCpm2ps_2_gCpm2phlfhr
        df[E] *= conv.mmolH2Opm2ps_2_mmphlfhr

    # add the water use efficiency
    for mod in [e.split(')')[0].split('(')[1] for e in gs]:

        df['WUE(%s)' % (mod)] = df['A(%s)' % (mod)] / df['E(%s)' % (mod)]

    # variables to consider
    WUE = df.filter(like='WUE(').columns.to_list()

    for e in gs[1:]:  # relative to the Medlyn model outputs

        df[e] = (df[e] - df[gs[0]]) / df[gs[0]]

    for e in A[1:]:

        df[e] = (df[e] - df[A[0]]) / df[A[0]]

    for e in E[1:]:

        df[e] = (df[e] - df[E[0]]) / df[E[0]]

    for e in WUE[1:]:

        df[e] = (df[e] - df[WUE[0]]) / df[WUE[0]]

    df[gs + A + E + WUE] *= 100.

    if which == 'cumul':  # cumulative % change
        df['change'] = which
        df.reset_index(inplace=True)

        return df[['soil', 'change'] + gs[1:] + A[1:] + E[1:] + WUE[1:]]

    elif which == 'mean':  # mean % change
        all = df.groupby('soil')[gs[1:] + A[1:] + E[1:] + WUE[1:]].mean()
        low_light = np.logical_or(df['hod'] < 10., df['hod'] > 17.)
        morn = (df[low_light].groupby('soil')[gs[1:] + A[1:] + E[1:] + WUE[1:]]
                .mean())
        arvo = df[~low_light].groupby('soil')[gs[1:] + A[1:] + E[1:] +
                                              WUE[1:]].mean()

    elif which == 'max':  # max % change
        all = (df.groupby(['soil', 'doy'])[gs[1:] + A[1:] + E[1:] + WUE[1:]]
               .mean().groupby(level=0).max())
        low_light = np.logical_or(df['hod'] < 10., df['hod'] > 17.)
        morn = (df[low_light].groupby(['soil', 'doy'])[gs[1:] + A[1:] + E[1:] +
                                                       WUE[1:]]
                .mean().groupby(level=0).max())
        arvo = (df[~low_light].groupby(['soil', 'doy'])[gs[1:] + A[1:] + E[1:]
                                                        + WUE[1:]]
                .mean().groupby(level=0).max())

    for sub in [all, morn, arvo]:

        sub['change'] = which
        sub.reset_index(inplace=True)

    return pd.concat([all, morn, arvo], keys=['day', 'mornNeve', 'arvo'])


def compute_relative_changes(df, fname):

    """
    Wrapper around the relative_changes function that computes each type
    of change (mean, max, cumul) and stores the output in a .csv file.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input & output data

    fname: string
        name under which to save the analysis file

    Returns:
    --------
    A .csv file that contains information on change in the gas exchange
    variables relative to the Medlyn model.

    """

    if not os.path.isfile(fname):
        df2 = relative_changes(df.copy(), which='mean')
        df2 = df2.append(relative_changes(df.copy(), which='max'),
                         ignore_index=True)
        df2 = df2.append(relative_changes(df.copy(), which='cumul'),
                         ignore_index=True)

        # save outputs
        columns = df2.columns.to_list()  # modify column order
        columns.remove('change')  # modify column order
        df2[['change'] + columns].to_csv(fname, index=False, na_rep='',
                                         encoding='utf-8')

    return


def analyse_simulations(ipath, xfiles, yfiles):

    """
    Combines all the daytime simulation outputs and analyses the changes
    in gas exchange variables in each model relative to the Medlyn
    model.

    Arguments:
    ----------
    ipath: string
        path to the folder where the input files are

    xfiles: list
        names of files that contains the input met & parameter data

    yfiles: list
        names of files that contains the model outputs

    Returns:
    --------
    Summary .csv files in 'output/simulations/idealised/insample/':
        * 'all_daytime_simulations.csv' contains all the model outputs
          during daytime hours (i.e., when photosynthesis is not null)
        * 'relative_changes_from_Medlyn.csv' contains information on the
          amount of change in gs, A, E, and the WUE (A/E) of each model,
          for the two soil moisture profiles, at various times of the,
          relative to the Medlyn model

    """

    # all run info
    opath = os.path.join(ipath.replace('input', 'output'), 'insample')
    fname = os.path.join(opath, 'all_daytime_simulations.csv')

    if not os.path.isfile(fname):  # create file if it doesn't exist

        for ifile, ofile in zip(xfiles, yfiles):  # loop over the files

            # load input data into a dataframe
            df, __ = read_csv(os.path.join(ipath, ifile))

            # add soil moisture profile
            df['sw'], df['Ps'] = soil_water(df, ofile.split('.')[0])
            df.fillna(method='ffill', inplace=True)

            # read output file
            df2, __ = read_csv(os.path.join(opath, ofile))

            # combine the dataframes
            df = combine_dfs(df, df2, (ofile.split('.')[0]))

            try:  # append new combined df to previously combined dfs
                dfs = dfs.append(df, ignore_index=True)

            except NameError:  # first run
                dfs = df.copy()

        # remove duplicates and save
        dfs.drop_duplicates(inplace=True)
        columns = dfs.columns.to_list()  # modify column order
        columns.remove('soil')  # modify column order
        dfs[['soil'] + columns].to_csv(fname, index=False, na_rep='',
                                       encoding='utf-8')

    else:
        dfs = (pd.read_csv(fname).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    # compute relative changes
    compute_relative_changes(dfs,
                             fname.replace('all_daytime_simulations',
                                           'relative_changes_from_Medlyn'))

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = "run simulations or analyse simulation outputs?"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--runsim', action='store_true',
                        help='run the models')
    args = parser.parse_args()

    # user input
    swaters = ['wet', 'inter']
    models = ['Medlyn', 'Tuzet', 'SOX12', 'WUE', 'CMax', 'ProfitMax', 'CGain',
              'ProfitMax2', 'LeastCost', 'CAP', 'MES']

    main(runsim=args.runsim, swaters=swaters, models=models)
