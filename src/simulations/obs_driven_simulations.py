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
from scipy import stats

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
    df1.drop([('Tleaf', '[deg C]')], axis=1, inplace=True)  # drop Tleaf
    df1.to_csv(os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
               'input'), 'simulations'), 'obs_driven'),
               '%s_calibrated.csv' % (training)), index=False, na_rep='',
               encoding='utf-8')

    return

def calc_rank(sub):

    return np.array([stats.percentileofscore(sub, a, 'strict') / 100.
                     for a in sub])


def calc_perf(df1, df2, var='gs', metric='NSE'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    pts = len(df1)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        # mask invalid data
        df3 = df1.copy()[~np.isclose(df1['%s' % (e)], 9999.)]

        if metric == 'NaNs':
            perf = ((pts - len(df3) + (len(df3) - len(df3['%s' % (e)].isnull()))
                     + len(df3[df3['%s' % (e)] / df3['%s' % (var)] < 1.e-9]))
                    / pts)

        elif metric == 'NSE':
            perf = 1. - (((df3['%s' % (e)] - df3['%s' % (var)]) ** 2.).sum() /
                         ((df3['%s' % (var)] - df3['%s' % (var)].mean()) ** 2.)
                          .sum())

        elif metric == 'KGE':
            r, __ = stats.kendalltau(df3['%s' % (var)], df3['%s' % (e)])
            alpha = (1. - 0.5 * np.sum(np.abs(df3['%s' % (e)].sort_values() /
                     df3['%s' % (e)].mean() - df3['%s' % (var)].sort_values() /
                     df3['%s' % (var)].mean())) / len(df3))
            perf = 1. - ((r - 1.) ** 2. + (alpha - 1.) ** 2. +
                         (df3['%s' % (e)].mean() / df3['%s' % (var)].mean()
                          - 1.) ** 2.) ** 0.5

        elif (metric == 'BIC') or (metric == 'RBIC'):
            N = 1

            if (('std' in e) or ('sox2' in e) or ('wue' in e) or ('cgn' in e)
                or ('lcst' in e) or ('cap' in e) or ('mes' in e)):
                N = 2

            elif 'cmax' in e:
                N = 3

            elif 'tuz' in e:
                N = 4

            coef = 1.  # avoid skewing the results due to units

            if var == 'gs':  # limit logs <<<< 0 by computing in mmol
                coef = 1.e3

            rss = ((coef * (df3['%s' % (e)] - df3['%s' % (var)])) ** 2.).sum()
            perf = len(df3) * np.log(rss / len(df3)) + N * np.log(len(df3))

        elif metric == 'RMSE':
            perf = ((df3['%s' % (e)] - df3['%s' % (var)]) ** 2.).mean() ** 0.5

        elif metric == 'NMSE':
            perf = ((((df3['%s' % (e)] - df3['%s' % (var)]) ** 2.).mean()
                     ** 0.5) / (df3['%s' % (var)].quantile(0.75) -
                     df3['%s' % (var)].quantile(0.25)))

        elif metric == 'NSAE':
            perf = ((df3['%s' % (e)] - df3['%s' % (var)]).abs().mean() /
                    (df3['%s' % (var)].quantile(0.75) -
                     df3['%s' % (var)].quantile(0.25)))

        elif metric == 'MASE':
            perf = ((df3['%s' % (var)] - df3['%s' % (e)]) /
                     df3['%s' % (var)].diff()[1:].abs().mean()).abs().mean()

        elif metric == 'WAPE':
            df3 = df3[df3['%s' % (var)] != 0.]
            perf = ((df3['%s' % (var)] - df3['%s' % (e)]).abs() /
                     df3['%s' % (var)].abs()).mean()

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df3['site_spp'].iloc[0]] = perf

    if metric == 'RBIC':
        sub = df2.loc[idx1:idx1 + i, df1['site_spp'].iloc[0]]
        rank = calc_rank(sub)
        df2.loc[idx1:idx1 + i, df1['site_spp'].iloc[0]] = rank

    return


def model_performance(df, which='NSE'):

    cols = ['model', 'variable']

    if which == 'NaNs':
        cols = ['mean'] + cols

    elif which != 'RMSE':
        cols = ['mean', 'median'] + cols

    perf = pd.DataFrame(columns=list(df['site_spp'].unique()) + cols,
                        index=np.arange(3 * len(df.filter(like='gs(').columns)))

    # calculate perf by site x species
    df.groupby('site_spp').apply(calc_perf, df2=perf, metric=which)
    df.groupby('site_spp').apply(calc_perf, df2=perf, var='E', metric=which)
    df.groupby('site_spp').apply(calc_perf, df2=perf, var='A', metric=which)

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

    # output dir paths
    ofdir = os.path.join(ipath.replace('input', 'output'), 'all_site_spp')

    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)

    # load input data into a dataframe
    df1, __ = read_csv(os.path.join(ipath, file))
    df1.fillna(method='ffill', inplace=True)

    # add the necessary extra variables
    df1['Ps_pd'] = df1['Ps'].copy()
    df1['sw'] = 0.  # add sw (not used) or it won't run
    df1['scale2can'] = 1.

    # run the models
    fname = os.path.join(ofdir, '%s.csv' % (file.split('_calibrated')[0]))

    if not os.path.isfile(fname):  # create file if it doesn't exist
        df2 = hrun(fname, df1, len(df1.index), 'Farquhar',
                   models=['Medlyn', 'Tuzet', 'SOX12', 'WUE', 'CMax',
                           'ProfitMax', 'CGain', 'ProfitMax2', 'LeastCost',
                           'CAP', 'MES'], resolution='high')
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

fname = os.path.join(os.path.dirname(ofdir), 'all_NSEs.csv')

if not os.path.isfile(fname):
    nses = model_performance(dfs)
    nses['mean'] = (nses.iloc[:, :nses.columns.get_loc('mean')].sum(axis=1) -
                    nses.iloc[:, :nses.columns.get_loc('mean')].max(axis=1) -
                    nses.iloc[:, :nses.columns.get_loc('mean')].min(axis=1)) / (nses.columns.get_loc('mean') - 2.)
    nses['mean'] = nses.iloc[:, :nses.columns.get_loc('mean')].mean(axis=1)
    nses['median'] = nses.iloc[:, :nses.columns.get_loc('mean')].median(axis=1)
    nses['rank'] = nses.groupby(['variable'])['mean'].rank(ascending=False)
    nses.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_KGEs.csv')

if not os.path.isfile(fname):
    kges = model_performance(dfs, which='KGE')
    kges['mean'] = (kges.iloc[:, :kges.columns.get_loc('mean')].sum(axis=1) -
                    kges.iloc[:, :kges.columns.get_loc('mean')].max(axis=1) -
                    kges.iloc[:, :kges.columns.get_loc('mean')].min(axis=1)) / (kges.columns.get_loc('mean') - 2.)
    kges['mean'] = kges.iloc[:, :kges.columns.get_loc('mean')].mean(axis=1)
    kges['median'] = kges.iloc[:, :kges.columns.get_loc('mean')].median(axis=1)
    kges['rank'] = kges.groupby(['variable'])['mean'].rank(ascending=False)
    kges.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_RBICs.csv')

if not os.path.isfile(fname):
    rbics = model_performance(dfs, which='RBIC')
    rbics['mean'] = (rbics.iloc[:, :rbics.columns.get_loc('mean')].sum(axis=1) -
                rbics.iloc[:, :rbics.columns.get_loc('mean')].max(axis=1) -
                rbics.iloc[:, :rbics.columns.get_loc('mean')].min(axis=1)) / (rbics.columns.get_loc('mean') - 2.)
    rbics['mean'] = rbics.iloc[:, :rbics.columns.get_loc('mean')].mean(axis=1)
    rbics['median'] = (rbics.iloc[:, :rbics.columns.get_loc('mean')]
                            .median(axis=1))
    rbics['rank'] = rbics.groupby(['variable'])['mean'].rank()
    rbics.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_NMSEs.csv')

if not os.path.isfile(fname):
    nmses = model_performance(dfs, which='NMSE')
    nmses['mean'] = (nmses.iloc[:, :nmses.columns.get_loc('mean')].sum(axis=1) -
            nmses.iloc[:, :nmses.columns.get_loc('mean')].max(axis=1) -
            nmses.iloc[:, :nmses.columns.get_loc('mean')].min(axis=1)) / (nmses.columns.get_loc('mean') - 2.)
    nmses['mean'] = nmses.iloc[:, :nmses.columns.get_loc('mean')].mean(axis=1)
    nmses['median'] = (nmses.iloc[:, :nmses.columns.get_loc('mean')]
                            .median(axis=1))
    nmses['rank'] = nmses.groupby(['variable'])['mean'].rank()
    nmses.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_MASEs.csv')

if not os.path.isfile(fname):
    mases = model_performance(dfs, which='MASE')
    mases['mean'] = (mases.iloc[:, :mases.columns.get_loc('mean')].sum(axis=1) -
            mases.iloc[:, :mases.columns.get_loc('mean')].max(axis=1) -
            mases.iloc[:, :mases.columns.get_loc('mean')].min(axis=1)) / (mases.columns.get_loc('mean') - 2.)
    mases['mean'] = mases.iloc[:, :mases.columns.get_loc('mean')].mean(axis=1)
    mases['median'] = (mases.iloc[:, :mases.columns.get_loc('mean')]
                            .median(axis=1))
    mases['rank'] = mases.groupby(['variable'])['median'].rank()
    mases.to_csv(fname, index=False, na_rep='', encoding='utf-8')

fname = os.path.join(os.path.dirname(ofdir), 'all_NaNs.csv')

if not os.path.isfile(fname):
    nans = model_performance(dfs, which='NaNs')
    nans['mean'] = nans.iloc[:, :nans.columns.get_loc('mean')].mean(axis=1)
    nans.to_csv(fname, index=False, na_rep='', encoding='utf-8')
