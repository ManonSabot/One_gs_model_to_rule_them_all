#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the synthetic forcing + target variables from the
idealised experiments, and associated model outputs.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Idealised model xpe setup and associated simulations"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (13.01.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import argparse  # read in the user input
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import warnings  # ignore warnings

# plotting modules
import matplotlib as mpl  # general matplotlib libraries
import matplotlib.pyplot as plt  # plotting interface
import scipy.signal as signal  # smooth lines
import statsmodels.api as sm  # smooth lines
import string   # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import model_order, which_model
from plot_utils import render_xlabels, render_ylabels

from plot_utils import get_Px  # vulnerability curve

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from calibrations.calib_utils import soil_water  # soil moist. profiles

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(xpe=False):

    """
    Main function: either plots the forcing and target outputs for the
                   idealised experiment setup or plots the idealised
                   outputs from these experiments

    Arguments:
    ----------
    xpe: bool
        if True, plots the synthetic forcing and target outputs

    Returns:
    --------
    'training_forcing_soil_moisture.jpg' and 'training_targets.jpg' or
    'harmonized_models_wet.jpg' and 'harmonized_models_dry.jpg' in
    output/plots

    """

    base_dir = get_main_dir()  # working paths
    fig_dir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    plt_setup()  # figure specs

    # meteorological drivers
    df, __ = read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                      base_dir, 'input'), 'simulations'), 'idealised'),
                      'wet_calibrated.csv'))

    if xpe:  # plot the synthetic forcing and target outputs

        # meteorological drivers figure
        fname = os.path.join(fig_dir, 'training_forcing.jpg')

        if not os.path.isfile(fname):
            plot_forcings(df, fname)

        # reference model / target gs figure
        fname = os.path.join(fig_dir, 'training_targets.jpg')

        if not os.path.isfile(fname):

            # target data
            fname2 = os.path.join(os.path.join(os.path.join(os.path.join(
                                  base_dir, 'input'), 'calibrations'),
                                  'idealised'), 'training_wet_y.csv')
            df2, __ = read_csv(fname2)
            df3, __ = read_csv(fname2.replace('wet', 'inter'))
            plot_target(df2, df3, fname)

    for swater in ['wet', 'inter']:

        fname = os.path.join(fig_dir, 'harmonized_models_%s.jpg' % (swater))

        if not os.path.isfile(fname):
            df2, __ = read_csv(os.path.join(os.path.join(os.path.join(
                               os.path.join(os.path.join(base_dir, 'output'),
                                            'simulations'), 'idealised'),
                                            'insample'), '%s.csv' % (swater)))
            plot_simulations(df2, fname, Ca=df.loc[0, 'CO2'],
                             P50=-df.loc[0, 'P50'], P88=-df.loc[0, 'P88'])

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self):

        default_plt_setup(ticks=True)  # default setup
        plt.rcParams['legend.borderpad'] = 0.5  # adjust legend


def weekly(df):

    """
    Returns weekly versions of a dataframe

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to modify

    Returns:
    --------
    df1: pandas dataframe
        dataframe containing average weekly data

    df2: pandas dataframe
        dataframe containing maximum weekly data

    df3: pandas dataframe
        dataframe containing minimum weekly data

    """

    # add info on unique weeks
    df['week'] = 1
    days = 7.  # new week every 7 days, regardless of starting doy

    for __ in range(3):  # deal with the three other weeks

        df.loc[df['doy'] >= df.loc[0, 'doy'] + days, 'week'] += 1
        days += 7.

    # groupby unique week
    df1 = df.groupby([df.week, df.hod]).mean()
    df2 = df.groupby([df.week, df.hod]).max()
    df3 = df.groupby([df.week, df.hod]).min()

    # reset the indexes
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df3 = df3.reset_index()

    return df1, df2, df3


def smooth_soil_water(df):

    """
    Returns smooth soil moisture profiles by applying a frequency filter

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to modify

    Returns:
    --------
    sw_wet: array
        wet soil moisture profile

    sw_inter: array
        intermediately dry soil moisture profile

    """

    # retrieve the soil moisture profiles
    sw_wet, __ = soil_water(df, 'wet')
    sw_inter, __ = soil_water(df, 'inter')

    # smooth the decay of sm when plotting it
    N = 3  # filter order
    Wn = 0.0245  # cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    sw_wet = signal.filtfilt(B, A, sw_wet)
    sw_inter = signal.filtfilt(B, A, sw_inter)

    return sw_wet, sw_inter


def weekly_ticks(ax, N):

    """
    Distributes the x-ticks on a weekly basis depending on the length N

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    N: int
        number of datapoints in the plot

    Returns:
    --------
    Ticks at the right position on the x-axis

    """

    ax.set_xlim([0, N])

    # is the data weekly or half-hourly?
    ticks = [48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5]

    if N / 48 > 10:
        ticks = [e * 7 for e in ticks]

    ax.set_xticks(ticks)

    return


def plot_forcings(df, figname):

    """
    Generates plots of the synthetic environmental drivers used in the
    idealised model experiments

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to plot

    figname: string
        name of the figure to produce, including path

    Returns:
    --------
    'training_forcing.jpg'

    """

    # declare figure
    fig, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.3)
    axes = axes.flat  # flatten axes

    # first, plot the average atmospheric drivers
    wavg, wmax, wmin = weekly(df.copy())
    axes[0].plot(wavg['PPFD'], color='k')
    axes[1].plot(wavg['Tair'], color='k')
    axes[2].plot(wavg['VPD'], color='k')

    # and the uncertainties
    axes[0].fill_between(wmax['PPFD'].index, wmin['PPFD'], wmax['PPFD'],
                         facecolor='lightgrey', edgecolor='none')
    axes[1].fill_between(wmax['Tair'].index, wmin['Tair'], wmax['Tair'],
                         facecolor='lightgrey', edgecolor='none')
    axes[2].fill_between(wmax['VPD'].index, wmin['VPD'], wmax['VPD'],
                         facecolor='lightgrey', edgecolor='none')

    # plot the soil moisture profiles
    sw_wet, sw_inter = smooth_soil_water(df)

    axes[3].plot(sw_wet, color='k')
    axes[3].plot(sw_inter, color='k', ls='--')

    # plot reference soil moisture levels
    axes[3].plot(df['theta_sat'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.99 * len(df), df['theta_sat'][0] - 0.02115, 'saturation',
                 ha='right')
    axes[3].plot(df['fc'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.99 * len(df), df['fc'][0] + 0.008, 'field capacity',
                 ha='right')
    axes[3].plot(df['pwp'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.92 * len(df), df['pwp'][0] + 0.008, 'wilting point',
                 ha='right')

    for i, ax in enumerate(axes):  # format axes ticks

        if i < 3:  # weekly diurnal forcings
            weekly_ticks(ax, len(wavg))

        else:
            weekly_ticks(ax, len(df))

        if i < 2:
            ax.set_xticklabels(['', ] * 4)

        else:
            ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

        if i < 2:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))

        else:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        # subplot labelling
        t = ax.text(0.025, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                        alpha=0.8))

    # axes labels
    pad = 5.
    render_ylabels(axes[0], 'PPFD', r'$\mu$mol photon m$^{-2}$ s$^{-1}$',
                   pad=pad)
    render_ylabels(axes[1], 'Air temperature', r'$^\circ$C', pad=pad)
    render_ylabels(axes[2], 'Vapour pressure deficit', 'kPa', pad=pad)
    render_ylabels(axes[3], 'Soil water content', r'm$^{3}$ m$^{-3}$', pad=pad)

    fig.savefig(figname)
    plt.close()

    return


def plot_target(df1, df2, figname):

    """
    Generates plots of the target outputs from the idealised experiment,
    i.e., the Medlyn model gs estimates

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe containing the data to plot for the wet soil moisture
        profile

    df2: pandas dataframe
        dataframe containing the data to plot for the intermediately dry
        soil moisture profile

    figname: string
        name of the figure to produce, including path

    Returns:
    --------
    'training_targets.jpg'

    """

    # declare figure
    fig, axes = plt.subplots(figsize=(5.5, 2), ncols=2, sharey=True)
    plt.subplots_adjust(wspace=0.05)

    # gs under wet conditions
    wavg, wmax, wmin = weekly(df1.copy())
    axes[0].plot(wavg['gs(std)'], color='k')

    # add the weekly diurnal uncertainties
    axes[0].fill_between(wmax['gs(std)'].index, wmin['gs(std)'],
                         wmax['gs(std)'], facecolor='lightgrey',
                         edgecolor='none')

    # gs under intermediately dry conditions
    wavg, wmax, wmin = weekly(df2.copy())
    axes[1].plot(wavg['gs(std)'], color='k')

    # add the weekly diurnal uncertainties
    axes[1].fill_between(wmax['gs(std)'].index, wmin['gs(std)'],
                         wmax['gs(std)'], facecolor='lightgrey',
                         edgecolor='none')

    for i, ax in enumerate(axes):

        # format axes ticks
        weekly_ticks(ax, len(wavg))
        ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        ax.set_ylim(0.02, 0.085)

        # subplot labelling
        t = ax.text(0.015, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                        alpha=0.8))

    # axes labels
    render_ylabels(axes[0], '$g_{s}$', r'mol m$^{-2}$ s$^{-1}$', pad=5.)

    fig.savefig(figname)
    plt.close()

    return


def plot_simulations(df, figname, Ca=40., P50=None, P88=None):

    """
    Generates plots of the models' key variables from the idealised
    simulations

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to plot

    figname: string
        name of the figure to produce, including path

    Ca: float
        atmospheric CO2 [Pa]

    P50: float
        leaf water potential [MPa] at which there is 50% loss of
        hydraulic conductance

    P88: float
        leaf water potential [MPa] at which there is 88% loss of
        hydraulic conductance

    Returns:
    --------
    either 'harmonized_models_wet.jpg' or 'harmonized_models_dry.jpg'

    """

    # declare figure
    fig, axes = plt.subplots(figsize=(6., 6), nrows=3, ncols=2, sharex=True)
    plt.subplots_adjust(hspace=0.15, wspace=0.35)
    axes = axes.flat  # flatten axes

    # ignore the axis on the first row, second column
    axes[1].axis('off')
    axes = np.delete(axes, 1)

    # extract info from the vulnerability curve, marked on the LWP plot
    if (P50 is not None) and (P88 is not None):
        P12 = get_Px(P50, P88, 50., 88., 12.)
        axes[2].axhline(P12, linestyle=':', linewidth=1.)

    # deal with the nans
    davg = df.select_dtypes(exclude=['object']).copy()
    davg.where(davg < 9999., inplace=True)
    (davg.loc[:, davg.filter(like='gs(').columns]
     .where(davg.loc[:, davg.filter(like='gs(').columns] > 1.e-9,
            inplace=True))
    davg = davg[~np.isclose(davg, 0.)].groupby(['hod']).mean()

    # remove unnecessary hod
    davg = davg[davg[davg.filter(like='gs(').columns].sum(axis=1) != 0.]

    # smooth the lines
    lowess = sm.nonparametric.lowess

    for col in davg.columns.to_list():

        where = davg[np.logical_and(davg.index > 6.5, davg.index < 17.5)].index
        davg.loc[where, col] = lowess(davg.loc[where, col], where, frac=0.3,
                                      return_sorted=False)

    # inset axis for the really low LWP
    if any(davg[davg.filter(like='Pleaf(').columns] < P50):
        iax = axes[2].inset_axes([0.69, 0.69, 0.3, 0.3])
        iax.spines['bottom'].set_color('grey')
        iax.spines['left'].set_color('grey')
        iax.spines['right'].set_visible(False)
        iax.spines['top'].set_visible(False)
        iax.axhline(P12, linestyle=':', linewidth=1.)
        iax.axhline(P50, linestyle=':', linewidth=1.)
        iax.axhline(P88, linestyle=':', linewidth=1.)

    for mod in model_order():  # plot each model's simulated outputs

        if mod == 'std':
            lw = 4.
            alpha = 1.

        else:
            lw = plt.rcParams['lines.linewidth']
            alpha = 0.8

        axes[0].plot(davg['gs(%s)' % (mod)], linewidth=lw, alpha=alpha,
                     label=which_model(mod))
        axes[1].plot(davg['Ci(%s)' % (mod)] / Ca, linewidth=lw,
                     alpha=alpha)

        if (mod != 'std') and any(davg['Pleaf(%s)' % (mod)] < P50):
            iax.plot(davg['Pleaf(%s)' % (mod)],
                     linewidth=lw, alpha=alpha)
            next(axes[2]._get_lines.prop_cycler)  # skip this colour

        elif mod != 'std':
            axes[2].plot(davg['Pleaf(%s)' % (mod)],
                         linewidth=lw, alpha=alpha)
            next(iax._get_lines.prop_cycler)  # skip this colour

        else:  # skip colours
            next(axes[2]._get_lines.prop_cycler)
            next(iax._get_lines.prop_cycler)

        axes[3].plot(davg['A(%s)' % (mod)],
                     linewidth=lw, alpha=alpha)
        axes[4].plot(davg['E(%s)' % (mod)],
                     linewidth=lw, alpha=alpha)

    # adjust LWP axis limits
    bottom, __ = axes[2].get_ylim()
    axes[2].set_ylim(bottom, 0.175)

    for i, ax in enumerate(axes):  # format axes ticks

        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

        if (ax == axes[-1]) or (ax == axes[-2]):
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.d'))

            if ax == axes[-1]:
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))

        else:
            ax.set_xticklabels([])

        if (ax == axes[0]) or (ax == axes[1]):  # gs and Ci:Ca
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        else:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        # subplot labelling
        t = ax.text(0.025, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                        alpha=0.8))

    # format the inset axis' ticks
    iax.xaxis.set_major_locator(plt.NullLocator())
    iax.set_yticks([-1., round(P50), round(P88)])
    iax.tick_params(axis='y', colors='grey')

    # axes labels
    pad = 5.
    render_xlabels(axes[-1], 'hour of day', 'h', pad=pad)
    render_xlabels(axes[-2], 'hour of day', 'h', pad=pad)
    render_ylabels(axes[0], r'$g_s$', r'mol m$^{-2}$ s$^{-1}$', pad=pad)
    render_ylabels(axes[1], r'$C_i$ : $C_{a}$', '-', pad=pad)
    render_ylabels(axes[2], r'$\Psi$$_l$', 'MPa', pad=pad)
    render_ylabels(axes[3], r'$A_n$', r'$\mu$mol m$^{-2}$ s$^{-1}$', pad=pad)
    render_ylabels(axes[4], r'$E$', r'mmol m$^{-2}$ s$^{-1}$', pad=pad)

    # split the legend in several parts
    axes[0].legend(loc=2, ncol=2, bbox_to_anchor=(1.35, 0.75),
                   handleheight=1.5, labelspacing=0.1)
    fig.savefig(figname)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # define the argparse settings to read run set up file
    description = "plot the experimental setup or the outputs?"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-xpe', '--xpe', action='store_true',
                        help='plot the experimental setup')
    args = parser.parse_args()

    main(xpe=args.xpe)
