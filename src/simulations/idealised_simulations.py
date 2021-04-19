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

    if profile == 'dry':
        start = 0.8 * sw[0]
        rate = -12. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = df['pwp'][0] / 2.

        # alternative
        #start = sw[0]
        #rate = -16. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))

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


def combine_dfs(df1, df2, xpe):

    # combine df1 and df2
    df2 = df2.merge(df1, on=['doy', 'hod'])

    # only keep the variables, i.e. remove parameters
    #df2 = df2[[c for c in list(df2) if len(df2[c].unique()) > 1]]
    df2 = df2.iloc[:, :df2.columns.get_loc('u')]
    df2 = df2.drop(df2.filter(like='Ps(').columns, axis=1)
    df2['CO2'] = df1['CO2'].iloc[0]  # keep Ca

    # restrict to the hod when photosynthesis happens
    df2 = df2[df2[df2.filter(like='gs(').columns].sum(axis=1) > 0.]

    # add the info about the simulation
    df2['xpe'] = xpe  # which simu?

    return df2


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

combis = [e for e in combis if ((e[0] == e[1]) or (e[2] == 'insample'))]

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
    models = ['Medlyn', 'Tuzet', 'SOX12', 'WUE', 'CMax', 'ProfitMax', 'CGain',
              'ProfitMax2', 'LeastCost', 'CAP', 'MES']

    fname = os.path.join(ofdir, '%s.csv' % (xpe))

    if not os.path.isfile(fname):  # create file if it doesn't exist
        df2 = hrun(fname, df1, len(df1.index), 'Farquhar', models=models,
                   resolution='med', inf_gb=True)
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
fname = fname.replace('idealised_simulations', 'relative_impacts')

if not os.path.isfile(fname):
    df = dfs.copy()
    df.replace(9999., np.nan, inplace=True)

    # fluxes relative to Medlyn
    gs = df.filter(like='gs(').columns.to_list()[1:]
    A = df.filter(like='A(').columns.to_list()[1:]
    E = df.filter(like='E(').columns.to_list()[1:]

    for e in gs[1:]:

        df[e] = (df[e] - df['gs(std)']) / df['gs(std)']

    for e in A:

        df[e] = (df[e] - df['A(std)']) / df['A(std)']

    for e in E:

        df[e] = (df[e] - df['E(std)']) / df['E(std)']

    # relative fluxes
    all = df.groupby('xpe')[gs + A + E].mean() * 100.
    low_light = np.logical_or(df['hod'] < 10., df['hod'] > 17.)
    morn = df[low_light].groupby('xpe')[gs + A + E].mean() * 100.
    arvo = df[~low_light].groupby('xpe')[gs + A + E].mean() * 100.

    # save relative change df
    (pd.concat([all, morn, arvo], keys=['day', 'mornNeve', 'arvo'])
       .to_csv(fname, na_rep='', encoding='utf-8'))


# create an impact df
fname = fname.replace('relative_impacts', 'cumulative_impacts')

if not os.path.isfile(fname):
    df = dfs.copy()
    df.replace(9999., 0., inplace=True)  # NaNs don't matter for cumuls
    df.replace('9999.0', 0., inplace=True)  # NaNs don't matter for cumuls

    # cumulative fluxes
    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()
    impacts = df.groupby('xpe')[GPP + E].sum()
    impacts[GPP] *= conv.umolCpm2ps_2_gCpm2phlfhr / 4.
    impacts[E] *= conv.mmolH2Opm2ps_2_mmphlfhr / 4.

    # add the water use efficiency
    for mod in [e.split(')')[0].split('(')[1] for e in GPP]:

        impacts['WUE(%s)' % (mod)] = (impacts['A(%s)' % (mod)] /
                                      impacts['E(%s)' % (mod)])

    # relative to Medlyn
    WUE = impacts.filter(like='WUE(').columns.to_list()
    relMed = impacts.copy()

    for e in GPP[1:]:

        relMed[e] = (relMed[e] - relMed['A(std)']) / relMed['A(std)'] * 100.

    relMed['A(std)'] = 0.

    for e in E[1:]:

        relMed[e] = (relMed[e] - relMed['E(std)']) / relMed['E(std)'] * 100.

    relMed['E(std)'] = 0.

    for e in WUE[1:]:

        relMed[e] = (relMed[e] - relMed['WUE(std)']) / relMed['WUE(std)'] * 100.

    relMed['WUE(std)'] = 0.

    # relative to its ref calibration
    relself = impacts.copy()
    wet = [e for e in relself.index.to_list() if ((e.split('_')[-1] == 'wet') and not ('insample_wet' in e))]
    inter = [e for e in relself.index.to_list() if ((e.split('_')[-1] == 'inter') and not ('insample_inter' in e))]

    relself.loc[wet] = (relself.loc[wet] - relself.loc['insample_wet_wet']) / relself.loc['insample_wet_wet']
    relself.loc[inter] = (relself.loc[inter] - relself.loc['insample_inter_inter']) / relself.loc['insample_inter_inter']

    relself.loc[[e for e in relself.index.to_list() if 'insample_wet_wet' in e]] = 0.
    relself.loc[[e for e in relself.index.to_list() if 'insample_inter_inter' in e]] = 0.

    # save impact df
    (pd.concat([impacts, relMed, relself], keys=['actual', 'relMedlyn', 'relSelf'])
       .to_csv(fname, na_rep='', encoding='utf-8'))
