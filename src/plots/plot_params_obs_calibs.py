#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the gs(Pleaf) observation-based functional shapes used
to fit some parameters of the Tuzet model, and compares the 'observed'
and calibrated kmax.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = ""
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (28.05.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import warnings  # ignore warnings

# plotting modules
import matplotlib as mpl  # general matplotlib libraries
import matplotlib.pyplot as plt  # plotting interface
from scipy.ndimage import gaussian_filter  # signal processing
from scipy.optimize import curve_fit  # fit the functional shapes
from scipy.integrate import quad  # integrate on a range
import string   # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import model_order, which_model
from plot_utils import site_spp_order, missing_var
from plot_utils import render_xlabels, render_ylabels

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import f, Weibull_params  # used to calc. kmax

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(site_spp=site_spp_order()):

    """
    Main function: plots the gs(Pleaf) observations used to fit the
                   empirical gs downregulation factor of the Tuzet
                   model. Also plots 'observed' (back-inferred)
                   distributions of kmax compared to all the calibrated
                   kmax parameter values

    Arguments:
    ----------
    site_spp: list
        site_species combinations to plot

    Returns:
    --------
    'obs_data_calibs.jpg' in output/plots

    """

    base_dir = get_main_dir()  # working paths

    # figure
    figname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                           'plots'), 'obs_data_calibs.jpg')

    if not os.path.isfile(figname):
        plt_setup()  # figure specs

        # data to plot
        df1 = pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                          base_dir, 'output'), 'simulations'), 'obs_driven'),
                          'all_site_spp_simulations.csv'))
        df2 = pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                          base_dir, 'output'), 'calibrations'), 'obs_driven'),
                          'overview_of_fits.csv'))

        # organise the dataframes
        df1 = sorted_data(df1, 'site_spp', site_spp)
        df2 = sorted_data(df2, 'training', df1['site_spp'].unique().to_list())

        # plot data
        obs_calibs(df1, df2, figname)

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self, colours=None):

        # default setup
        default_plt_setup(ticks=True)

        # lines and markers
        plt.rcParams['lines.color'] = '#d3d3d3'
        plt.rcParams['scatter.edgecolors'] = 'k'

        # legend
        plt.rcParams['legend.columnspacing'] = 0.8


def sorted_data(df, key, site_spp):

    """
    Ensures the dataframe is ordered from the most mesic site_spp to
    the most xeric_spp, as per the order given in site_spp

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to order

    key: string
        what to order, e.g., site_spp, training

    site_spp: list
        list in order

    Returns:
    --------
    df: pandas dataframe
        appropriately ordered dataframe

    """

    # only keep the relevant site x species data
    df = df.loc[df[key].isin(site_spp)].copy()

    # order by site x species
    df[key] = df[key].astype('category')
    df[key].cat.set_categories(site_spp, inplace=True)
    df = df.sort_values(key)

    try:  # exclude the site x species where there are no observed LWP

        df = df[~df[key].isin(missing_var(df, 'Pleaf', key))]

    except KeyError:
        pass

    return df


def plot_obs(ax, x, y, which='gs'):

    """
    Draws all the observational data on the figure, including scatters
    of gs to Pleaf, vertical markers, and violin plots the data

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot the data

    x: array
        data

    y: array
        data

    which: string
        variable being plotted, e.g., gs, kmax

    Returns:
    --------
    The plots

    """

    if which == 'gs':
        ax.scatter(x, y, marker='+', s=25., edgecolor='none', alpha=0.5)

    else:
        vp = ax.violinplot(y, showextrema=False, positions=[x], widths=0.8)
        plt.setp(vp['bodies'], facecolor='w', edgecolor='#d3d3d3', alpha=1)
        ax.vlines(x, np.amin(y), np.amax(y), zorder=1)
        ax.vlines(x, np.percentile(y, 25), np.percentile(y, 75), lw=6,
                  zorder=2)
        ax.scatter(x, np.percentile(y, 50), marker='_', color='w', zorder=3)

    return


def fLWP(Pleaf, sref, Pref):

    """
    Calculates an empirical logistic function used to describe the
    sensitivity of the stomates to leaf water potential, following
    Tuzet et al. (2003)

    Arguments:
    ----------
    Pleaf: float
        leaf water potential [MPa]

    sref: float
        sensitivity of the stomates to Pleaf [unitless]

    Pref: float
        reference water potential [MPa]

    Returns:
    --------
    The unitless empirical stomatal conductance's response to leaf water
    potential.

    """

    return (1. + np.exp(sref * Pref)) / (1. + np.exp(sref * (Pref - Pleaf)))


def envelope(x, y):

    """
    Finds the upper envelope of the y to x relationship

    Arguments:
    ----------
    x: array
        data

    y: array
        data

    Returns:
    --------
    ux: array
        upper envelope x data

    uy: array
        upper envelope y data

    """

    # declare the first values
    u_x = [x[0], ]
    u_y = [y[0], ]
    lastPeak = 0

    # detect peaks and mark their location
    for i in range(1, len(y) - 1):

        if ((y[i] - y[i - 1]) > 0. and ((y[i] - y[i + 1]) > 0.) and
           ((i - lastPeak) > 0.)):
            u_x.append(x[i])
            u_y.append(y[i])
            lastPeak = i

    # append the last values
    u_x.append(x[-1])
    u_y.append(y[-1])

    return u_x, u_y


def fit_Tuzet(df):

    """
    Wrapper that fits the upper envelope of fLWP to observations of
    gs and Pleaf using the curve_fit package

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the observational data

    Returns:
    --------
    x0: float
        upper bound possible value for the Tuzet model's Pref

    x1: float
        lower bound possible value for the Tuzet model's Pref

    obs_popt: list
        the fitted sref and Pref

    """

    # smooth out noise
    smoothed = gaussian_filter(df['gs'], df['gs'].std())

    # point where the signal goes above the background noise
    base = 0.3  # background noise is +/- 15% of max gs
    supp = (df.loc[df['Pleaf'] < -df['Pleaf'].std(), 'gs'] - base).std()
    m = smoothed < (base - df['gs'].std() * supp)
    x0 = np.maximum(df.loc[m, 'Pleaf'].max(), df.loc[np.isclose(df['gs'], 1.),
                                                     'Pleaf'])
    x1 = df.loc[m, 'Pleaf'].min()

    # now sort df by LWP
    df.sort_values(by=['Pleaf'], ascending=False, inplace=True)
    LWP, gs = envelope(df['Pleaf'].to_numpy(), df['gs'].to_numpy())

    try:
        obs_popt, __ = curve_fit(fLWP, LWP, gs, p0=[2., (x0+x1) / 2.],
                                 bounds=([0.01, df['Pleaf'].min()],
                                         [10, df['Pleaf'].max()]))

        return x0, x1, obs_popt

    except Exception:

        return (0.,) * 3


def get_calib_kmax(df):

    """
    Extracts all the models' calibrated kmax parameter values from df

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the parameter values

    Returns:
    --------
    params: list
        kmax parameter values

    models: list
        the models the parameter values correspond to, in the same order

    """

    params = []
    models = []

    for what in df['training'].unique().dropna():

        sub = df.copy()[df['training'] == what]
        sub['v3'] = sub['v1']
        sub.loc[sub['Model'] == 'Tuzet', 'v3'] = sub['v2']
        keep = np.logical_or(sub['p2'].str.contains('kmaxT').fillna(False),
                             np.logical_or(sub['p1'].str.contains('kmax'),
                                           sub['p1'].str.contains('krl')))

        # set model order
        sub['order'] = sub['Model'].replace({'Tuzet': 0, 'Eller': 1,
                                             'ProfitMax': 2, 'SOX-OPT': 3,
                                             'ProfitMax2': 4, 'LeastCost': 5,
                                             'CAP': 6, 'MES': 7})

        sub = sub[keep].sort_values(by=['solver', 'order'])
        sub.reset_index(inplace=True)
        params += [np.log(sub['v3'].values)]
        models += [sub['Model'].values]

    return params, models


def obs_calibs(df1, df2, figname):

    """
    Generates plots of the observed relations in gs to Pleaf, as well as
    of 'observed' kmax distributions with calibrated kmax overlaid

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    df2: pandas dataframe
        dataframe that contains information on all the parameter
        calibrations

    figname: string
        name of the figure to produce, including path

    Returns:
    --------
    'obs_data_calibs.jpg'

    """

    # declare figure
    fig = plt.figure(figsize=(6.5, 8.))
    gs = fig.add_gridspec(nrows=96, ncols=16, hspace=0.3, wspace=0.2)
    ax2 = fig.add_subplot(gs[52:, 6:])  # conductance data

    ipath = os.path.join(os.path.join(os.path.join(get_main_dir(),
                         'input'), 'simulations'), 'obs_driven')

    labels = []

    for i, what in enumerate(df1['site_spp'].unique().dropna()):

        if i < 13:
            nrow = int(i / 4) * 16
            ncol = (i % 4) * 4
            ax1 = fig.add_subplot(gs[nrow:nrow + 16, ncol:ncol + 4])

        else:
            nrow += 16
            ax1 = fig.add_subplot(gs[nrow:nrow + 16, :4])

        sub = (df1.loc[df1['site_spp'] == what]
               .select_dtypes(exclude=['object', 'category']).copy())
        sub = sub[sub['Pleaf'] > -9999.]
        sub['gs'] /= sub['gs'].max()

        for day in sub['doy'].unique():

            mask = sub['doy'] == day
            plot_obs(ax1, sub.loc[mask, 'Pleaf'], sub.loc[mask, 'gs'])

        x0, x1, obs_popt = fit_Tuzet(sub)
        x = np.linspace(sub['Pleaf'].max(), sub['Pleaf'].min(), 500)
        ax1.plot(x, fLWP(x, obs_popt[0], obs_popt[1]), 'k', zorder=30)
        ax1.vlines(x0, 0., 1., linestyle=':')
        ax1.vlines(x1, 0., 1., linestyle=':')

        # get the integrated VC given by the obs and site params
        ref, __ = read_csv(os.path.join(ipath, '%s_calibrated.csv' % (what)))
        b, c = Weibull_params(ref.iloc[0])
        int_VC = np.zeros(len(sub))

        for j in range(len(sub)):

            int_VC[j], __ = quad(f, sub['Pleaf'].iloc[j], sub['Ps'].iloc[j],
                                 args=(b, c))

        plot_obs(ax2, i, np.log(sub['E'] / int_VC), which='kmax')

        # subplot titles (including labelling)
        what = what.split('_')
        species = r'\textit{%s %s}' % (what[-2], what[-1])
        labels += [r'\textit{%s. %s}' % (what[-2][0], what[-1])]

        if 'Quercus' in what:
            species += ' (%s)' % (what[0][0])
            labels[-1] += ' (%s)' % (what[0][0])

        txt = ax1.annotate(r'\textbf{(%s)} %s' % (string.ascii_lowercase[i],
                                                  species),
                           xy=(0.025, 0.98), xycoords='axes fraction',
                           ha='left', va='top')
        txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                     alpha=0.8))

        # format axes ticks
        ax1.xaxis.set_major_locator(mpl.ticker.NullLocator())

        if ((i == len(df1['site_spp'].unique().dropna()) - 1) or
           ((ncol > 0) and (nrow == 32))):
            render_xlabels(ax1, r'$\Psi_{l}$', 'MPa')

        if ncol == 0:
            ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax1.yaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))
            ax1.set_ylabel(r'$g_{s, norm}$')

        else:
            ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax1.set_yticklabels([])

    ax2.annotate(r'\textbf{(%s)}' % (string.ascii_lowercase[i + 1]),
                 xy=(0.05, 0.98), xycoords='axes fraction', ha='right',
                 va='top')

    # add max conductance parameter values
    params, models = get_calib_kmax(df2)
    params = np.asarray(params)
    locs = np.arange(len(df1['site_spp'].unique()))

    # update colour list
    models2consider = [e.split('-')[0] for e in
                       models[0][:len(np.unique(models))]]
    ref_models = [which_model(e).split('-')[0].split('$')[0] for e in
                  model_order()]
    ref_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colours = ([ref_colours[ref_models.index(e)] for e in models2consider] *
               len(params))

    for i in range(params.shape[1]):

        if i < 8:
            ax2.scatter(locs, params[:, i], s=50, linewidths=0.25,
                        c=colours[i], alpha=0.9, label=models[0][i],
                        zorder=4)

        else:
            ax2.scatter(locs, params[:, i], s=50, linewidths=0.25,
                        c=colours[i], alpha=0.9, zorder=4)

    # tighten the subplot
    ax2.set_xlim(locs[0] - 0.8, locs[-1] + 0.8)
    ax2.set_ylim(np.log(0.025) - 0.1, np.log(80.))

    # ticks
    ax2.set_xticks(locs + 0.5)
    ax2.set_xticklabels(labels, ha='right', rotation=40)
    ax2.xaxis.set_tick_params(length=0.)

    yticks = [0.025, 0.25, 1, 5, 25, 75]
    ax2.set_yticks([np.log(e) for e in yticks])
    ax2.set_yticklabels(yticks)
    render_ylabels(ax2, r'k$_{max}$', 'mmol m$^{-2}$ s$^{-1}$ MPa$^{-1}$')

    handles, labels = ax2.get_legend_handles_labels()
    labels[3] = r'SOX$_\mathrm{\mathsf{opt}}$'
    ax2.legend(handles, labels, ncol=3, labelspacing=1. / 3.,
               columnspacing=0.5, loc=3)

    # save
    fig.savefig(figname)
    plt.close()


# ======================================================================

if __name__ == "__main__":

    main()
