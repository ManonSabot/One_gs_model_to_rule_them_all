#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots 1:1 "goodness-of-fits" of the simulations to
observations.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Sim:obs goodness-of-fit"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (28.05.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
from scipy import stats  # compute regressions
import warnings  # ignore warnings

# plotting modules
import matplotlib as mpl  # general matplotlib libraries
import matplotlib.pyplot as plt  # plotting interface

# own modules
from plot_utils import default_plt_setup
from plot_utils import model_order, which_model

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # get the project's directory

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(identity):

    """
    Main function: plots studentized simulated variables to studentized
                   observed variables for all the gs models at the
                   site x species specified by 'identity'

    Arguments:
    ----------
    identity: string
        site_species combination to plot

    Returns:
    --------
    'goodness_of_fit.jpg' in output/plots

    """

    base_dir = get_main_dir()  # working paths

    # data to plot
    df = pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                     base_dir, 'output'), 'simulations'), 'obs_driven'),
                     'all_site_spp_simulations.csv'))

    # figure
    fname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                         'plots'), 'goodness_of_fit.jpg')

    if not os.path.isfile(fname):
        plt_setup()  # figure specs
        one_2_one(df, fname, identity)  # plot data

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self, colours=None):

        default_plt_setup(ticks=True)  # default setup
        plt.rcParams['lines.linewidth'] = 1.75  # adjust lines
        plt.rcParams['legend.borderpad'] = 0.5  # adjust legend


def studentize(df, vars=['gs', 'Ci', 'E', 'A', 'Pleaf']):

    """
    Studentizes the variables selected by 'vars' from a dataframe

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to studentize

    vars: list
        names of the variables to studentize

    Returns:
    --------
    df: pandas dataframe
        dataframe containing studentized data

    """

    # calc mean and standard deviation
    mean = df.mean(axis=0)
    std = df.std(axis=0)

    # variables to consider
    columns = [c for c in df.columns if any(var in c for var in vars)]

    if 'gs' in vars:  # give each model the avg obs attributes
        mean.loc[[e for e in columns if 'gs(' in e]] = mean.gs
        std.loc[[e for e in columns if 'gs(' in e]] = std.gs

    if 'Ci' in vars:
        mean.loc[[e for e in columns if 'Ci(' in e]] = mean.Ci
        std.loc[[e for e in columns if 'Ci(' in e]] = std.Ci

    if 'E' in vars:
        mean.loc[[e for e in columns if 'E(' in e]] = mean.E
        std.loc[[e for e in columns if 'E(' in e]] = std.E

    if 'A' in vars:
        mean.loc[[e for e in columns if 'A(' in e]] = mean.A
        std.loc[[e for e in columns if 'A(' in e]] = std.A

    if 'Pleaf' in vars:
        mean.loc[[e for e in columns if 'Pleaf(' in e]] = mean.Pleaf
        std.loc[[e for e in columns if 'Pleaf(' in e]] = std.Pleaf

    # now studentize each model's variables and the obs
    icol = [df.columns.get_loc(e) for e in columns]
    df.iloc[:, icol] = (df.iloc[:, icol].subtract(mean, axis=1)
                          .div(std, axis=1))

    return df


def one_2_one(df, figname, identity):

    """
    Generates 1:1 plots of studentized simulation to observation data,
    organized in subplots for each model considered

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to plot

    figname: string
        name of the figure to produce, including path

    identity: string
        site_species combination to plot

    Returns:
    --------
    'goodness_of_fit.jpg'

    """

    # decide on how many columns and rows
    Nrows = 0
    Ncols = 0

    while Nrows * Ncols < len(model_order()):

        Nrows += 1

        if Nrows * Ncols < len(model_order()):
            Ncols += 1

    # declare figure
    fig, axes = plt.subplots(Nrows, Ncols, figsize=(Ncols + 2, Nrows + 2.25),
                             sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.1, wspace=0.05)
    axes = axes.flat  # flatten axes

    # select site x species data
    df = df[df['site_spp'] == identity].copy()
    df.reset_index(drop=True, inplace=True)

    # limit to roughly midday LWP
    not_midday = df[np.logical_or(df['hod'] < 12., df['hod'] > 14.)].index
    df.loc[not_midday, 'Pleaf'] = np.nan

    # studentize the data
    df = studentize(df)  # rescale across all
    columns = [c for c in df.columns if any(var in c for var in
               ['gs', 'Ci', 'E', 'A', 'Pleaf'])]
    df[columns] = df[columns].where(df[columns] > -1000.)
    df[columns] = df[columns].where(df[columns] < 1000.)

    for i, mod in enumerate(model_order()):

        x = df[['gs', 'E', 'A', 'Ci']].values.flatten()
        y = df[['gs(%s)' % (mod), 'E(%s)' % (mod), 'A(%s)' % (mod),
                'Ci(%s)' % (mod)]].values.flatten()

        axes[i].scatter(df['gs'], df['gs(%s)' % (mod)], alpha=0.5,
                        label=r'$g_{s}$')
        axes[i].scatter(df['Ci'], df['Ci(%s)' % (mod)], alpha=0.5,
                        label=r'$C_{i}$')
        axes[i].scatter(df['A'], df['A(%s)' % (mod)], alpha=0.5,
                        label=r'$A_{n}$')
        axes[i].scatter(df['E'], df['E(%s)' % (mod)], alpha=0.5, label=r'$E$')

        if mod != 'std':
            axes[i].scatter(df['Pleaf'], df['Pleaf(%s)' % (mod)],
                            label=r'$\Psi$$_{l, midday}$')

        mask = np.logical_and(np.isfinite(x), np.isfinite(y))
        slope, intercept, r, p, sem = stats.linregress(y[mask], x[mask])

        axes[i].plot([-2, 2], [-2, 2], c='orange', ls='--', lw=2.)
        axes[i].plot([-2.75, 2], [intercept - 2. * slope,
                                  intercept + 2. * slope], c='orange', lw=2.)

        axes[i].text(0.05, 0.8, '%s:\nslope=%s, $r^{2}$=%s' %
                                (which_model(mod), round(slope, 2),
                                 round(r ** 2., 2)), ha='left',
                     transform=axes[i].transAxes, weight='bold')

        if i % Ncols == 0:
            axes[i].set_ylabel('Simulated variables')

        if i >= (Nrows - 1) * Ncols:
            axes[i].set_xlabel('Observed variables')

    # format axes ticks
    axes[-1].set_xlim(-2.9, 2.9)
    axes[-1].set_ylim(-2.9, 2.9)
    axes[-1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axes[-1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    # add legend
    axes[-1].legend(bbox_to_anchor=(1., 0.75))

    fig.savefig(figname)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # user input
    site_spp = 'Richmond_Eucalyptus_cladocalyx'

    main(site_spp)
