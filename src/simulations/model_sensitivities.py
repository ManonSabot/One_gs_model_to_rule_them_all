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
import random  # pick a random day for the forcings to be generated
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from itertools import permutations

# sensitivity modules
from SALib.sample import saltelli  # generate orthogonal matrix
from SALib.analyze import sobol  # analyse the sensitivities

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules
from TractLSM import hrun  # run the models


###############################################################################

def unique_combinations(array, N=2):

    combinations = np.asarray(list(permutations(array, N)))
    keep = np.arange(1, len(combinations) + 1)

    for i in range(len(array)):

        keep[i * (len(array) - 1):] = \
            keep[i * (len(array) - 1):] % (len(array) - i)

    combinations = combinations[keep > 0]

    return combinations


def chunks(a, N):

    """
    Splits a list or array into N-roughly equal parts.

    Arguments:
    ----------
    a: list or array
        list/array to be split, can contain any data type
    N: int
        number of sub-lists/arrays the input must be split into

    Returns:
    --------
    A list of N-roughly equal parts.
    """

    integ = int(len(a) / N)
    remain = int(len(a) % N)

    splitted = [a[i * integ + min(i, remain):(i + 1) * integ +
                  min(i + 1, remain)] for i in range(N)]

    return splitted


###############################################################################

base_dir = get_main_dir()

# path to input data
fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'idealised'), 'sensitivity_mtx.csv')

# create the pbm for the Sobol sensitivity analysis
fname1 = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                      'input'), 'calibrations'), 'idealised'), 'training_x.csv')
df1, columns = read_csv(fname1)  # load training data

# get the bounds for the different variables to look at
variables = ['PPFD', 'Tair', 'VPD', 'CO2', 'Ps']
PPFD = [50., 2500.]  # umol m-2 s-1
Tair = [2., 40.]  # degC
VPD = [0.1, 10.]  # kPa
CO2 = [250. * 101.325 / 1000., 900. * 101.325 / 1000.]  # Pa, 250 - 900 ppm
Ps = [-3., df1['Psie'].iloc[0]]
bounds = [PPFD, Tair, VPD, CO2, Ps]

# define the sensitivity problem
problem = {'num_vars': len(variables), 'names': variables, 'bounds': bounds}

if not os.path.isfile(fname):  # generate the sensitivity inputs
    N = 84000  # Saltelli mtx obtained via cross-sampling method
    var_values = saltelli.sample(problem, N)  # size = 2N(Nparams + 1)

    # put the variable values in a new df
    df2 = pd.DataFrame(index=np.arange(len(var_values)))

    for i, var in enumerate(variables):

        df2[var] = var_values[:, i]

    # now calculate the sw series associated with the Ps
    df2['sw'] = (df1.loc[0, 'theta_sat'] * (var_values[:, -1] /
                 df1.loc[0, 'Psie'])  ** (-1. / df1.loc[0, 'bch']))
    df2['Ps_pd'] = df2['Ps'].copy()  # equate to Ps at predawn

    # add all other missing columns compared with df1
    df2['doy'] = np.arange(len(df2))  # use the doy as an index
    df2['hod'] = 12.  # hod doesn't matter but used to build output file
    df2['u'] = 0.  # set wind speed to 0 as we're using an inf. gb
    add = [e for e in df1.columns if e not in df2.columns]

    for col in add:

        df2[col] = df1.loc[0, col]

    # replace the parameter values with the best wet calibrated params
    fname2 = os.path.join(os.path.join(os.path.join(os.path.join(base_dir,
                         'output'), 'calibrations'), 'idealised'),
                         'best_fit.csv')
    df3 = (pd.read_csv(fname2, header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())
    df3 = df3[df3['training'] == 'wet']

    # attribute the first (and second and third) parameter(s)
    for i in df3.index:

        df2[df3.loc[i, 'p1']] = df3.loc[i, 'v1']

        if not pd.isnull(df3.loc[i, 'v2']):
            df2[df3.loc[i, 'p2']] = df3.loc[i, 'v2']

        if not pd.isnull(df3.loc[i, 'v3']):
            df2[df3.loc[i, 'p3']] = df3.loc[i, 'v3']

    # save the sensitivity input file
    df2 = df2[ df1.columns.to_list() + ['sw', 'Ps_pd']]  # df1 column order
    columns = columns.insert(len(columns), ('sw', '[m3 m-3]'))  # at the end
    columns = columns.insert(len(columns), ('Ps_pd', '[MPa]'))  # at the end
    df2.columns = pd.MultiIndex.from_tuples(columns)
    df2.to_csv(fname, index=False, na_rep='', encoding='utf-8')
    df2.columns = df2.columns.droplevel(level=1)  # drop units for runs

else:
    df2, __ = read_csv(fname)  # load sensitivity input data

"""
# run this in 36 different chunks to 'speed' things up
dfs = chunks(df2, 36)

for i in range(len(dfs)):

    i = 36

    fname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                         'Sensitivities'),
                         'model_sensitivities_%d.csv' % (i + 1))

    if not os.path.isfile(fname):  # generate the sensitivity outputs

        df3 = hrun(fname, dfs[i], len(dfs[i]), 'Farquhar',
                   models=['Medlyn', 'Tuzet', 'SOX12', 'WUE', 'CGain',
                           'ProfitMax', 'CMax', 'LeastCost', 'CAP', 'MES'],
                   inf_gb=True)  # run the models
        #df3.columns = df3.columns.droplevel(level=1)  # drop the units

    exit(1)
"""

fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'idealised'), 'model_sensitivity_mtx.csv')
df3, __ = read_csv(fname)  # load outputs

fname = os.path.join(os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'idealised'),
                     'overview_of_sensitivities.csv')

if not os.path.isfile(fname):

    # only keep the output that will be used for the Sobol indices
    df3 = df3[df3.filter(like='gs(').columns.to_list() +
              df3.filter(like='Ci(').columns.to_list() +
              df3.filter(like='Pleaf(').columns.to_list()]

    # adjust the limits for each individual model
    df3.loc[df2.index[df2['Ps'] <= -1.5], df3.filter(like='std1').columns] = 0.
    df3.loc[df2.index[df2['Ps'] <= df2['PrefT']],
            df3.filter(like='tuz').columns] = 0.
    df3.loc[df2.index[df2['Ps'] <= df2['PcritC']],
            df3.filter(like='cap').columns] = 0.
    df3.loc[df2.index[df2['Ps'] <= df2['PcritM']],
            df3.filter(like='mes').columns] = 0.

    # where 9999., the output is effectively 0.
    df3.replace(9999., 0., inplace=True)  # all NaNs to zero

    # create a summary of outputs df
    summarise = df3.columns.to_list()
    covar = unique_combinations(variables)
    idxs = np.arange(len(summarise) * (len(variables) +  len(covar)))
    df4 = pd.DataFrame(index=idxs, columns=['output', 'driver', 'S1', 'S1_conf',
                                            'ST', 'ST_conf', 'S2', 'S2_conf'])

    iter = 0

    for output in summarise:

        Si = sobol.analyze(problem, df3[output].values)

        for i, variable in enumerate(variables):

            df4.loc[iter, 'output'] = output
            df4.loc[iter, 'driver'] = variable
            df4.loc[iter, 'S1'] = Si['S1'][i]
            df4.loc[iter, 'S1_conf'] = Si['S1_conf'][i]
            df4.loc[iter, 'ST'] = Si['ST'][i]
            df4.loc[iter, 'ST_conf'] = Si['ST_conf'][i]
            iter += 1

        iter1 = 0
        iter2 = 1

        for i, cov in enumerate(covar):

            if i > 0:
                if covar[i][0] != covar[i - 1][0]:
                    iter1 += 1
                    iter2 = 1 + iter1

            df4.loc[iter, 'output'] = output
            df4.loc[iter, 'driver'] = '%s-%s' % (cov[0], cov[1])
            df4.loc[iter, 'S2'] = Si['S2'][iter1, iter2]
            df4.loc[iter, 'S2_conf'] = Si['S2_conf'][iter1, iter2]
            iter2 += 1
            iter += 1

    df4.to_csv(fname, index=False, na_rep='', encoding='utf-8')
