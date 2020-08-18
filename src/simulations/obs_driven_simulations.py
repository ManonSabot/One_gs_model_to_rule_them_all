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
import os  # check for files, paths
import sys  # check for files, paths
import itertools
import random  # pick a random day for the forcings to be generated
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import InForcings  # generate met data & read default params
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential, hydraulics  # soil
from TractLSM import hrun  # run the models


#==============================================================================

def build_calibrated_forcing(training):

    base_dir = get_main_dir()  # working paths

    # forcing file used to calibrate the models
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'input'), 'calibrations'), 'obs_driven'),
                         '%s_x.csv' % (training))
    df1, columns = read_csv(fname)

    # file containing the best calibrated params
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'obs_driven'),
                         'best_fit.csv')
    df2 = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())
    df2 = df2[df2['training'] == training]

    # attribute the first (and second and third) parameter(s)
    for i in df2.index:

        df1.loc[0, df2.loc[i, 'p1']] = df2.loc[i, 'v1']

        if not pd.isnull(df2.loc[i, 'v2']):
            df1.loc[0, df2.loc[i, 'p2']] = df2.loc[i, 'v2']

        if not pd.isnull(df2.loc[i, 'v3']):
            df1.loc[0, df2.loc[i, 'p3']] = df2.loc[i, 'v3']

    # save the forcing file containing the calibrated params
    df1.columns = columns  # original columns
    df1.to_csv(os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
               'input'), 'simulations'), 'obs_driven'),
               '%s_calibrated.csv' % (training)), index=False, na_rep='',
               encoding='utf-8')

    return


def calc_rmse(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        rmse = ((df1['%s' % (e)] - df1['%s' % (var)]) ** 2.).mean() ** 0.5

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = rmse

    return


def calc_nse(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        nse = 1. - (((df1['%s' % (e)] - df1['%s' % (var)]) ** 2.).sum() /
              ((df1['%s' % (var)] - df1['%s' % (var)].mean()) ** 2.).sum())

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = nse

    return


def calc_mape(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        mask = df1['%s' % (var)] > 0.
        mape = (((df1['%s' % (var)][mask] - df1['%s' % (e)][mask]) /
                 df1['%s' % (var)][mask]).abs()).mean()

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = mape

    return


def calc_smape(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        smape = ((df1['%s' % (var)] - df1['%s' % (e)]).abs().sum() /
                 (df1['%s' % (var)] + df1['%s' % (e)]).sum())

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = smape

    return


def calc_mase(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        mase = ((df1['%s' % (var)] - df1['%s' % (e)]).abs().mean() /
                df1['%s' % (var)].diff()[1:].abs().mean())

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = mase

    return


def calc_logratio(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        mask = df1['%s' % (var)] > 0.
        perf = np.log((df1['%s' % (e)][mask] / df1['%s' % (var)][mask]).mean())

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['site_spp'].iloc[0]] = perf

    return


def model_performance(df, which='RMSE'):

    columns = list(df['site_spp'].unique()) + ['model', 'variable']
    perf = pd.DataFrame(columns=columns,
                        index=np.arange(3 * len(df.filter(like='gs(').columns)))

    if which == 'NSE':
        df.groupby('site_spp').apply(calc_nse, df2=perf)
        df.groupby('site_spp').apply(calc_nse, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_nse, df2=perf, var='A')

    elif which == 'MAPE':
        df.groupby('site_spp').apply(calc_mape, df2=perf)
        df.groupby('site_spp').apply(calc_mape, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_mape, df2=perf, var='A')

    elif which == 'SMAPE':
        df.groupby('site_spp').apply(calc_smape, df2=perf)
        df.groupby('site_spp').apply(calc_smape, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_smape, df2=perf, var='A')

    elif which == 'MASE':
        df.groupby('site_spp').apply(calc_mase, df2=perf)
        df.groupby('site_spp').apply(calc_mase, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_mase, df2=perf, var='A')

    elif which == 'LOG_Ratio':
        df.groupby('site_spp').apply(calc_logratio, df2=perf)
        df.groupby('site_spp').apply(calc_logratio, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_logratio, df2=perf, var='A')

    else:
        df.groupby('site_spp').apply(calc_rmse, df2=perf)
        df.groupby('site_spp').apply(calc_rmse, df2=perf, var='E')
        df.groupby('site_spp').apply(calc_rmse, df2=perf, var='A')

    return perf


###############################################################################

# working paths
base_dir = get_main_dir()

# do the 'calibrated' input files already exist?
ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'obs_driven')

if not os.path.isdir(ipath):  # make new dirs if they don't exist
    os.makedirs(ipath)

opath = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'calibrations'), 'obs_driven')
site_spp = [e[1] for e in os.walk(opath)][0]  # directories

for training in site_spp:  # loop over the site x spp combinations

    if not os.path.isfile(os.path.join(ipath,
                                       '%s_calibrated.csv' % (training))):
        build_calibrated_forcing(training)

for file in os.listdir(ipath):  # loop over all the possibilities

    # load input data into a dataframe
    df1, __ = read_csv(os.path.join(ipath, file))
    df1.fillna(method='ffill', inplace=True)

    # output dir paths
    ofdir = os.path.join(ipath.replace('input', 'output'), 'all_site_spp')

    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)

    # add the necessary extra variables
    df1['Ps_pd'] = df1['Ps'].copy()  # pre-dawn Ps
    df1['sw'] = 0.  # add sw (not used) or it won't run
    df1['scale2can'] = 1.

    # run the models
    fname = os.path.join(ofdir, '%s.csv' % (file.split('_calibrated')[0]))

    if not os.path.isfile(fname):  # create file if it doesn't exist

        df2 = hrun(fname, df1, len(df1.index), 'Farquhar',
                   models=['Medlyn2', 'Tuzet', 'SOX12', 'WUE', 'CGainNet',
                           'ProfitMax', 'CMax', 'LeastCost', 'CAP', 'MES'],
                   resolution='low')
        df2.columns = df2.columns.droplevel(level=1)

    else:
        df2, __ = read_csv(fname)

    # overall df containing all the outputs / inputs from the simulations
    df2 = df2.merge(df1, left_index=True, right_index=True)
    df2 = df2.drop('year', axis=1)

    # add in the leaf-level flux observations
    fname = os.path.join(ipath.replace('simulations', 'calibrations'),
                         '%s_y.csv' % (file.split('_calibrated')[0]))
    df3, __ = read_csv(fname)
    df3 = df3.drop('year', axis=1)
    df2 = df2.merge(df3, left_index=True, right_index=True)

    # only keep the variables, i.e. remove parameters
    df2 = df2[[c for c in list(df2) if ((len(df2[c].unique()) > 1) or
              (c == 'u') or ('Rublim' in c) or ('gb' in c))]]
    df2 = df2.drop(df2.filter(like='Ps(').columns, axis=1)

    # restrict to the hod when photosynthesis happens in the models
    df2 = df2[df2[df2.filter(like='gs(').columns].sum(axis=1) > 0.]

    # add the info about the site x spp combination
    df2['site_spp'] = file.split('_calibrated')[0]

    try:  # append the simulations to one another
        dfs = dfs.append(df2, ignore_index=True)

    except NameError:
        dfs = df2.copy()  # first combination

fname = os.path.join(os.path.dirname(ofdir), 'all_site_spp_simulations.csv')

if not os.path.isfile(fname):
    dfs.drop_duplicates(inplace=True)  # remove exactly duplicated lines
    columns = dfs.columns.to_list()  # modify column order
    columns.remove('site_spp')  # modify column order
    dfs[['site_spp'] + columns].to_csv(fname, index=False, na_rep='',
                                       encoding='utf-8')

else:
    dfs = (pd.read_csv(fname).dropna(axis=0, how='all')
                             .dropna(axis=1, how='all').squeeze())

fname = os.path.join(os.path.dirname(ofdir), 'all_RMSEs.csv')

if not os.path.isfile(fname):
    rmses = model_performance(dfs)
    rmses.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_NSEs.csv')

if not os.path.isfile(fname):
    nses = model_performance(dfs, which='NSE')
    nses.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_MAPEs.csv')

if not os.path.isfile(fname):
    mapes = model_performance(dfs, which='MAPE')
    mapes.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_MASEs.csv')

if not os.path.isfile(fname):
    mases = model_performance(dfs, which='MASE')
    mases.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_SMAPEs.csv')

if not os.path.isfile(fname):
    smapes = model_performance(dfs, which='SMAPE')
    smapes.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_logs.csv')

if not os.path.isfile(fname):
    logs = model_performance(dfs, which='LOG_Ratio')
    logs.to_csv(fname, index=False, na_rep='', encoding='utf-8')
