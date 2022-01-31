# -*- coding: utf-8 -*-

"""
Support function used to prepare the forcing to run the models.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Useful ancillary formatting function"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (09.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# general modules
import os  # check for paths
import pandas as pd  # read/write dataframes, csv files

# own modules
from TractLSM.Utils import read_csv  # read in files


# ======================================================================

def build_calibrated_forcing(ipath, training):

    """
    Generates an input file that contains calibrated parameter values.

    Arguments:
    ----------
    ipath: string
        path where to store the calibrated input file

    training: string
        either a site_species, or a soil moisture profile ('wet' or
        'inter')

    Returns:
    --------
    A calibrated input file in ipath.

    """

    # forcing file used to calibrate the models
    if (training == 'wet') or (training == 'inter'):
        fname = 'training_x.csv'

    else:
        fname = '%s_x.csv' % (training)

    df1, columns = read_csv(os.path.join(ipath.replace('simulations',
                                                       'calibrations'), fname))

    # file containing the best calibrated params
    fname = os.path.join(ipath.replace('simulations', 'calibrations')
                              .replace('input', 'output'), 'best_fit.csv')
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
    df1.columns = pd.MultiIndex.from_tuples(columns)

    try:  # drop the Tleaf obs if applicable
        df1.drop([('Tleaf', '[deg C]')], axis=1, inplace=True)

    except KeyError:
        pass

    df1.to_csv(os.path.join(ipath, '%s_calibrated.csv' % (training)),
               index=False, na_rep='', encoding='utf-8')

    return
