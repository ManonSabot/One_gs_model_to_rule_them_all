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
from TractLSM import conv
from TractLSM import InForcings  # generate met data & read default params
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules
from TractLSM import hrun  # run the models

#==============================================================================


def build_calibrated_forcing(training):

    base_dir = get_main_dir()  # working paths

    # forcing file used to calibrate the models
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'input'), 'calibrations'), 'idealised'),
                         'training_x.csv')
    df1, columns = read_csv(fname)

    # file containing the best calibrated params
    fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'idealised'),
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
               'input'), 'simulations'), 'idealised'),
               '%s_calibration.csv' % (training)), index=False, na_rep='',
               encoding='utf-8')

    return


def soil_water(df, profile):

    sw = np.full(len(df), df['theta_sat'][0])
    track = 1

    if profile == 'wet':
        rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (sw[0] + df['fc'][0]) / 2.25

    if profile == 'inter':
        rate = -8. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (df['fc'][0] + df['pwp'][0]) / 2.

    if profile == 'dry':
        rate = -16. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = df['pwp'][0] / 2.

    for i in range(len(df)):

        sw[i] = sw[i-1]

        if df['PPFD'].iloc[i] > 50.:
            sw[i] = np.maximum(sw[0] / (1. - rate * track), sw_min)
            track += 1

    # now get the soil water potentials matching the soil moisture profile
    Ps = np.asarray([water_potential(df.iloc[0], sw[i])
                     for i in range(len(sw))])

    return sw, Ps


def combine_dfs(df1, df2, xpe):

    # combine df1 and df2
    df2 = df2.merge(df1, on=['doy', 'hod'])

    # only keep the variables, i.e. remove parameters
    df2 = df2[[c for c in list(df2) if len(df2[c].unique()) > 1]]
    df2 = df2.drop(df2.filter(like='Ps(').columns, axis=1)
    df2['CO2'] = df1['CO2'].iloc[0]  # keep Ca

    # restrict to the hod when photosynthesis happens
    df2 = df2[df2[df2.filter(like='gs(').columns].sum(axis=1) > 0.]

    # add the info about the simulation
    df2['xpe'] = xpe  # which simu?

    return df2


def calc_rmse(df1, df2, var='gs'):

    if var == 'gs':
        idx1 = 0

    elif var == 'E':
        idx1 = len(df1.filter(like='%s(' % (var)).columns)

    else:
        idx1 = 2 * len(df1.filter(like='%s(' % (var)).columns)

    for i, e in enumerate(df1.filter(like='%s(' % (var)).columns):

        rmse = (((df1['%s' % (e)] - df1['%s(std1)' % (var)]) ** 2.).mean()
                ** 0.5)

        if df2.loc[idx1 + i].isnull().all().all():  # columns all empty
            df2.loc[idx1 + i, 'model'] = e.split('%s(' % (var))[1].split(')')[0]
            df2.loc[idx1 + i, 'variable'] = var

        df2.loc[idx1 + i, df1['xpe'].iloc[0]] = rmse

    return


def model_performance(df, which='RMSE'):

    columns = list(np.unique(df['xpe'])) + ['model', 'variable']
    perf = pd.DataFrame(columns=columns,
                        index=np.arange(3 * len(df.filter(like='gs(').columns)))

    if which == 'MAPE':
        df.groupby('xpe').apply(calc_mape, df2=perf)
        df.groupby('xpe').apply(calc_mape, df2=perf, var='E')
        df.groupby('xpe').apply(calc_mape, df2=perf, var='A')

    else:
        df.groupby('xpe').apply(calc_rmse, df2=perf)
        df.groupby('xpe').apply(calc_rmse, df2=perf, var='E')
        df.groupby('xpe').apply(calc_rmse, df2=perf, var='A')

    return perf

###############################################################################

trainings = ['wet', 'inter']
soils = ['wet', 'inter', 'dry']
atms = ['insample', 'highD', 'highCa']
combis = list(itertools.product(*[trainings, soils, atms]))  # possibilities
univar_xpes = ['%s_%s_%s' % (e[2], e[1], e[0]) for e in combis if
               ((e[0] == e[1]) or (e[2] == 'insample'))]

# working paths
base_dir = get_main_dir()

# do the 'calibrated' input files already exist?
ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'idealised')

for training in trainings:

    if not os.path.isfile(os.path.join(ipath,
                                       '%s_calibration.csv' % (training))):
        build_calibrated_forcing(training)

for combi in combis:  # loop over all the possibilities

    xpe = '%s_%s_%s' % (combi[2], combi[1], combi[0])

    # load input data into a dataframe
    df, __ = read_csv(os.path.join(ipath, '%s_calibration.csv' % (combi[0])))

    # output dir paths
    ofdir = os.path.join(ipath.replace('input', 'output'), 'multivar_change')

    if xpe in univar_xpes:
        ofdir = ofdir.replace('multivar_change', 'univar_change')

    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)

    # how should the atm forcing data change?
    df1 = df.copy()  # reset the df so as to not keep previous changes

    if combi[2] == 'highD':
        df1['VPD'] *= 2.

    elif combi[2] == 'highCa':
        df1['CO2'] *= 2.

    # soil moisture profile
    df1['sw'] = df1['theta_sat']
    df1.fillna(method='ffill', inplace=True)
    df1['sw'], df1['Ps'] = soil_water(df1, combi[1])

    # run the models
    models = ['Medlyn12', 'Tuzet', 'SOX12', 'ProfitMax', 'CGainNet', 'WUE',
              'CMax', 'LeastCost', 'CAP', 'MES']

    fname = os.path.join(ofdir, '%s.csv' % (xpe))

    if not os.path.isfile(fname):  # create file if it doesn't exist
        df2 = hrun(fname, df1, len(df1.index), 'Farquhar', models=models,
                   resolution='low', inf_gb=True)
        df2.columns = df2.columns.droplevel(level=1)

    else:
        df2, __ = read_csv(fname)

    # all run info
    fname = os.path.join(os.path.dirname(ofdir),
                         'all_idealised_simulations.csv')

    if not os.path.isfile(fname):  # create file if it doesn't exist
        df = combine_dfs(df1, df2, xpe)

        try:  # append the new combined df to previously combined dfs
            dfs = dfs.append(df, ignore_index=True)

        except NameError:  # first run
            dfs = df.copy()

    if (combi[0] == combi[1]) and (combi[2] == 'insample'):
        fname = os.path.join(ofdir, 'vargb_%s.csv' % (xpe))

        if not os.path.isfile(fname):  # create file if it doesn't exist
            df3 = hrun(fname, df1, len(df1.index), 'Farquhar', models=models,
                       resolution='low')
            df3.columns = df3.columns.droplevel(level=1)

        else:
            df3, __ = read_csv(fname)

        # append the run to other runs
        fname = os.path.join(os.path.dirname(ofdir),
                             'all_idealised_simulations.csv')

        if not os.path.isfile(fname):  # create file if it doesn't exist
            df = combine_dfs(df1, df3, 'vargb_%s' % (xpe))
            dfs = dfs.append(df, ignore_index=True)

if not os.path.isfile(fname):
    dfs.drop_duplicates(inplace=True)  # remove duplicated lines
    dfs.to_csv(fname, index=False, na_rep='', encoding='utf-8')

else:
    dfs = (pd.read_csv(fname).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

# create an impact df
fname = fname.replace('idealised_simulations', 'cumulative_impacts')

if not os.path.isfile(fname):
    dfs.replace(9999., 0., inplace=True)  # NaNs don't matter for cumuls

    # cumulative fluxes
    GPP = dfs.filter(like='A(').columns.to_list()
    E = dfs.filter(like='E(').columns.to_list()
    impacts = dfs.groupby('xpe')[GPP + E].sum()

    # upscale from 4 weeks to 1 year
    weeks_p_y = 52.1429 / 4.
    impacts[GPP] *= conv.umolCpm2ps_2_gCpm2phlfhr * weeks_p_y
    impacts[E] *= conv.mmolH2Opm2ps_2_mmphlfhr * weeks_p_y

    # save impact df
    impacts.to_csv(fname, na_rep='', encoding='utf-8')

# create a RMSE df
fname = fname.replace('cumulative_impacts', 'RMSEs')

if not os.path.isfile(fname):

    # the var gb RMSE will need to be corrected to account for inf v varying
    rmses = model_performance(dfs)
    rmses.to_csv(fname, index=False, na_rep='', encoding='utf-8')
