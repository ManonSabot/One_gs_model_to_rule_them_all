# -*- coding: utf-8 -*-

"""
General support functions

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "useful general support functions"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (18.08.2020)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# import general modules
import os  # check for paths
import sys  # check for files, version on the system
import pandas as pd  # read/write dataframes, csv files


# ======================================================================

def get_script_dir():

    """
    Returns a script's directory.

    """

    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_main_dir():

    """
    Returns the parent directory of a script's directory

    """

    dir = get_script_dir()

    while 'src' in dir:
        dir = os.path.dirname(dir)

    return dir


def read_csv(fname, drop_units=True):

    """
    Reads csv file with two headers, one for the variables, one for
    their units.

    Arguments:
    ----------
    fname: string
        input filename (with path)

    drop_units: boolean
        if True, drops the second row of the columns, i.e. the units

    Returns:
    --------
    df: pandas dataframe
        dataframe containing all the csv data, units header dropped

    columns: array
        original columns names and units present in csv file

    """

    df = (pd.read_csv(fname, header=[0, 1]).dropna(axis=0, how='all')
          .dropna(axis=1, how='all').squeeze())
    columns = df.columns

    if drop_units:  # drop the units (in second header) for readability
        df.columns = df.columns.droplevel(level=1)

        return df, columns

    else:
        return df
