#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a simplified version of the script used to run a Sobol'
sensitivity experiment for the 12 gs models. Specifically, in this
version of the script, there is no parallelisation of the code when
performing the simulations, so everything is written as a single
simulation performed for >1 million independent inputs. This is
obviously sub-optimal, and anyone who wishes to recreate our results or
reuse the method may want to parallelise this process.

Information on the setup for the sensitivity analysis is available from
the documentation of the salib package, at https://salib.readthedocs.io/

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Global sensitivity analysis"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from itertools import permutations  # infer permutations of variables
import warnings  # ignore warnings

# sensitivity modules
from SALib.sample import saltelli  # generate orthogonal matrix
from SALib.analyze import sobol  # analyse the sensitivities

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM import hrun  # run the models

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# ======================================================================

def main(variables, bounds, N, models=['Medlyn', 'ProfitMax']):

    """
    Main function: either runs the 12 harmonized models, for two
                   distinct soil water profiles, or analyses the model
                   outputs.

    Arguments:
    ----------
    variables: list
        name of the variables for which to run the sensitivity analysis

    bounds: list
        nested lists of [min, max] values for each of the variables

    N: int
        sets the size of the cross-sampling matrix over which we yield
        inputs/outputs

    models: list
        the models to run

    Returns:
    --------
    .csv files that contain sensitivity simulations, plus total-,
    first-, and second-order sensitivity indices for different model
    variables, as per the definition given by Sobol'. The files are
    stored in 'output/simulations/idealised/insample/'.

    """

    base_dir = get_main_dir()  # working paths

    # paths to input/output files
    ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                         'simulations'), 'idealised')
    opath = os.path.join(ipath.replace('input', 'output'), 'sensitivities')

    if not os.path.isdir(ipath):  # make dir
        os.makedirs(ipath)

    if not os.path.isdir(opath):  # make dir
        os.makedirs(opath)

    # input matrix file
    ifname = os.path.join(ipath, 'sensitivity_mtx.csv')

    # associated output matrix file
    ofname = os.path.join(opath, 'model_sensitivities.csv')

    # associated sensitivity index file
    sfname = os.path.join(opath, 'overview_of_sensitivities.csv')

    # ancillary file that contains the reference forcing data
    afname1 = os.path.join(ipath.replace('simulations', 'calibrations'),
                           'training_x.csv')

    # ancillary file that contains the best parameter values
    afname2 = os.path.join(os.path.basename(opath).replace('simulations',
                           'calibrations'), 'best_fit.csv')

    # define the sensitivity problem for the Sobol analysis
    problem = {'num_vars': len(variables), 'names': variables,
               'bounds': bounds}

    if not os.path.isfile(ofname):  # generate >1M outputs
        output_matrix(ifname, ofname, afname1, afname2, problem, N, models)

    if not os.path.isfile(sfname):  # sensitivity analysis
        df, __ = read_csv(ofname)  # read output matrix
        compute_sensitivity_indices(sfname, df, problem)

    return


def generate_input_matrix(problem, N, ref_input, ref_params):

    """
    Generates an input matrix of size 2N(Nparams + 1) which varies the
    variables defined in the problem dictionary.

    Arguments:
    ----------
    problem: dictionary
        contains information on the variables to vary (names and bounds)

    N: int
        sets the size of the cross-sampling matrix

    ref_input: pandas dataframe
        dataframe containing reference input data to build from

    ref_params: pandas dataframe
        dataframe containing reference parameter values

    Returns:
    --------
    df: pandas dataframe
        the input matrix to use for the sensitivity analysis

    """

    # Saltelli mtx obtained via the cross-sampling method
    var_values = saltelli.sample(problem, N)

    # put the variables in a new df
    df = pd.DataFrame(index=np.arange(len(var_values)))

    for i, var in enumerate(problem['names']):

        df[var] = var_values[:, i]

    # now calculate the sw series associated with the Ps
    df['sw'] = (ref_input.loc[0, 'theta_sat'] * (var_values[:, -1] /
                ref_input.loc[0, 'Psie']) ** (-1. / ref_input.loc[0, 'bch']))
    df['Ps_pd'] = df['Ps'].copy()  # equate to Ps at predawn

    # add all other missing columns compared with ref_input
    df['doy'] = np.arange(len(df))  # use the doy as an index
    df['hod'] = 12.  # hod doesn't matter but used to build output file
    df['u'] = 0.  # set wind speed to 0 as we're using an inf. gb
    add = [e for e in ref_input.columns if e not in df.columns]

    for col in add:

        df[col] = ref_input.loc[0, col]

    # attribute the first, second, and third parameter(s)
    for i in ref_params.index:

        df[ref_params.loc[i, 'p1']] = ref_params.loc[i, 'v1']

        if not pd.isnull(ref_params.loc[i, 'v2']):
            df[ref_params.loc[i, 'p2']] = ref_params.loc[i, 'v2']

        if not pd.isnull(ref_params.loc[i, 'v3']):
            df[ref_params.loc[i, 'p3']] = ref_params.loc[i, 'v3']

    return df


def output_matrix(ifile, ofile, ffile, pfile, problem, N, models):

    """
    Generates an output matrix of size 2N(Nparams + 1) which contains
    all the outputs corresponding to the variations as defined in the
    problem dictionary.

    Arguments:
    ----------
    ifile: string
        input sensitivity matrix file name

    ofile: string
        output sensitivity matrix file name

    ffile: string
        name of file that contains the reference forcing data

    pfile: string
        name of file that contains the reference parameter values

    problem: dictionary
        contains information on the variables to vary (names and bounds)

    N: int
        sets the size of the cross-sampling matrix

    models: list
        models for which to generate sensivity outputs

    Returns:
    --------
    A .csv file that contains the model sensitivity simulations, i.e.,
    > 1M independent outputs if N is taken to be 84000.

    """

    if not os.path.isfile(ifile):  # generate the input matrix

        # read default forcing data
        df1, columns = read_csv(ffile)

        # ref parameter values from the best wet calibrated params
        df2 = (pd.read_csv(pfile, header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())
        df2 = df2[df2['training'] == 'wet']

        # input matrix
        df = generate_input_matrix(problem, N, df1.copy(), df2.copy())

        # save the mtx of inputs
        df = df[df1.columns.to_list() + ['sw', 'Ps_pd']]  # df1 cols

        # add soil water info at the end of the columns
        columns = columns.insert(len(columns), ('sw', '[m3 m-3]'))
        columns = columns.insert(len(columns), ('Ps_pd', '[MPa]'))
        df.columns = pd.MultiIndex.from_tuples(columns)
        df.to_csv(ifile, index=False, na_rep='', encoding='utf-8')
        df.columns = df.columns.droplevel(level=1)  # drop units

    else:  # read the input matrix
        df, __ = read_csv(ifile)

    # run the models
    __ = hrun(ofile, df, len(df), 'Farquhar', models=models, resolution='low',
              inf_gb=True, temporal=False)

    return


def unique_combinations(array, N=2):

    """
    Generates unique combinations of the elements present in the array,
    where elements are paired N by N.

    Arguments:
    ----------
    array: list
        the array from which to generate unique element
        combinations/permutations

    N: int
        sets the number of elements that are paired together

    Returns:
    --------
    combinations: list
        unique element combinations/permutations

    """

    combinations = np.asarray(list(permutations(array, N)))
    keep = np.arange(1, len(combinations) + 1)

    for i in range(len(array)):

        keep[i * (len(array) - 1):] = \
            keep[i * (len(array) - 1):] % (len(array) - i)

    combinations = combinations[keep > 0]

    return combinations


def compute_sensitivity_indices(fname, df, problem):

    """
    Calculates total-, first-, and second-order sensitivity indices for
    different model variables, as per the definition given by Sobol',
    and using the method of Saltelli.

    Arguments:
    ----------
    fname: string
        name of the files that contains the sensitivity indices

    df: pandas dataframe
        the output matrix that contains the model sensitivity
        simulations

    problem: dictionary
        contains information on the variables to vary (names and bounds)

    Returns:
    --------
    A .csv file that contains the sensitivity indices.

    """

    # only keep the output that will be used for the Sobol indices
    df = df[df.filter(like='gs(').columns.to_list() +
            df.filter(like='Ci(').columns.to_list() +
            df.filter(like='Pleaf(').columns.to_list()]

    # where 9999., the output is effectively 0.
    df.replace(9999., 0., inplace=True)  # all NaNs to zero

    # create a summary of outputs df
    summarise = df.columns.to_list()
    covar = unique_combinations(variables)
    idxs = np.arange(len(summarise) * (len(variables) + len(covar)))
    sdf = pd.DataFrame(index=idxs, columns=['output', 'driver', 'S1',
                                            'S1_conf', 'ST', 'ST_conf', 'S2',
                                            'S2_conf'])

    iter = 0

    for output in summarise:

        Si = sobol.analyze(problem, df[output].values)

        for i, variable in enumerate(variables):

            sdf.loc[iter, 'output'] = output
            sdf.loc[iter, 'driver'] = variable
            sdf.loc[iter, 'S1'] = Si['S1'][i]
            sdf.loc[iter, 'S1_conf'] = Si['S1_conf'][i]
            sdf.loc[iter, 'ST'] = Si['ST'][i]
            sdf.loc[iter, 'ST_conf'] = Si['ST_conf'][i]
            iter += 1

        iter1 = 0
        iter2 = 1

        for i, cov in enumerate(covar):

            if i > 0:
                if covar[i][0] != covar[i - 1][0]:
                    iter1 += 1
                    iter2 = 1 + iter1

            sdf.loc[iter, 'output'] = output
            sdf.loc[iter, 'driver'] = '%s-%s' % (cov[0], cov[1])
            sdf.loc[iter, 'S2'] = Si['S2'][iter1, iter2]
            sdf.loc[iter, 'S2_conf'] = Si['S2_conf'][iter1, iter2]
            iter2 += 1
            iter += 1

    sdf.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


# ======================================================================

if __name__ == "__main__":

    # user input
    variables = ['PPFD', 'Tair', 'VPD', 'CO2', 'Ps']  # to vary
    PPFD = [50., 2500.]  # set bounds, umol m-2 s-1
    Tair = [2., 40.]  # set bounds, degC
    VPD = [0.1, 10.]  # set bounds, kPa
    CO2 = [250. * 101.325 / 1000., 900. * 101.325 / 1000.]  # Pa
    Ps = [-1.5, -0.0008]  # set bounds, MPa
    bounds = [PPFD, Tair, VPD, CO2, Ps]
    N = 84000  # sets N output size, which will be 2N(Nparams + 1)
    models = ['Medlyn', 'Tuzet', 'SOX12', 'WUE', 'CMax', 'ProfitMax',
              'ProfitMax2', 'CGain', 'LeastCost', 'CAP', 'MES']

    main(variables, bounds, N, models=models)
