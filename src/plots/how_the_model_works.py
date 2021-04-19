#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions used to plot the ProfitMax algorith behaviour on both
instantaneous and longer timescales. This is a useful tool which can be
played with to better understand model behaviour.

This file is part of the TractLSM project.

Copyright (c) 2019 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "explore the ProfitMax algorithm behaviour"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (08.10.2018)"
__email__ = "m.e.b.sabot@gmail.com"


#=======================================================================

import warnings  # ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# import general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# plotting modules
import matplotlib.pyplot as plt
import string  # automate subplot lettering

# first make sure that modules can be loaded from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import dparams  # model parameters
from TractLSM.SPAC import hydraulics, Weibull_params  # hydraulics
from TractLSM.SPAC import net_radiation  # big leaf canopy radiation
from TractLSM.CH2OCoupler.ProfitMax import hydraulic_cost, photo_gain


#=======================================================================

def main():

    """
    Main: Explores the ProfitMax algorith behaviour on both
          instantaneous and longer timescales of optimisation. Plots and
          saves outputs.

    Returns:
    --------
    files in output/figures/final_4_paper/ and in
    output/figures/not_shown_in_paper/

    """

    # how the model works
    rcrit_model_behaviour()

    return


#=======================================================================

# ~~~ Other functions are defined here ~~~

def declared_params():

    """
    Sets up the model to run, by making the parameter class a pandas
    series and adding missing forcings to it.

    Returns:
    --------
    p: pandas series
        met forcings & params necessary to run the model

    """

    # make the default param class a pandas series
    p = dparams
    attrs = vars(p)
    p = {item[0]: item[1] for item in attrs.items()}
    p = pd.Series(p)

    # add missing params, radiation is arbritrary (model not sensitive)
    p = p.append(pd.Series([25., 1., 1.],
                           index=['Tair', 'VPD', 'scale2can']))
    p = p.append(pd.Series([1800.], index=['PPFD']))
    p = p.append(pd.Series([net_radiation(p)], index=['Rnet']))

    return p


def opt_stream(p, res, photo):

    """
    Finds the instateneous profit maximization, following the
    optmization criterion for which, at each instant in time, the
    stomata regulate canopy gas exchange and pressure to achieve the
    maximum profit, which is the maximum difference between the
    normalized photosynthetic gain (gain) and the hydraulic cost
    function (cost). That is when d(gain)/dP = d(cost)/dP.

    Arguments:
    ----------
    p: pandas series
        met forcings & params

    res: string
        either 'low' (default), 'med', or 'high' to run the optimising
        solver

    photo: string
        either the Farquhar model for photosynthesis, or the Collatz
        model

    Returns:
    --------
    P: array
        leaf water potential [MPa], from the soil water potential Ps to
        the critical water potential Pcrit for which cavitation of the
        xylem occurs

    cost: array
        hydraulic cost [unitless], depending on the plant vulnerability
        curve

    gain: array
        unitless instantaneous photosynthetic gains for possible values
        of Ci minimized over the hydraulic stream

    """

    # hydraulics
    P, trans = hydraulics(p, res=res)
    cost, __ = hydraulic_cost(p, P)

    # assimilation
    gain, __, __ = photo_gain(p, trans, photo, res)

    return P[1:], cost[1:], gain


def Px(p, x):

    """
    Finds the leaf water potential associated with a specific x%
    decrease in hydraulic conductance, using the plant vulnerability
    curve.

    Arguments:
    ----------
    p: pandas series
        met forcings & params

    x: float
        percentage loss in hydraulic conductance

    Returns:
    --------
    Px: float
        leaf water potential [MPa] at which x% decrease in hydraulic
        conductance is observed

    """

    b, c = Weibull_params(p)  # MPa, unitless
    Px = -b * ((- np.log(1 - float(x) / 100.)) ** (1. / c))

    return Px


def plot_model(ax, P, gain, cost, colours, label=None, alpha=None,
               linestyle='-', zorder=None):

    """
    Plots each of the gain, cost, and net profit functions, along the
    water potential stream.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot

    P: array
        leaf water potential [MPa], from the soil water potential Ps to
        the critical water potential Pcrit for which cavitation of the
        xylem occurs

    gain: array
        unitless instantaneous photosynthetic gains for possible values
        of Ci minimized over the hydraulic stream

    cost: array
        hydraulic cost [unitless], depending on the plant vulnerability
        curve

    colours: array
        color used to plot gain, cost, and profit

    label: string
        label associated with the specific run mode

    alpha: float
        transparency

    linestyle: string
        standard linestyles

    zorder: int
        zorder of the plot

    Returns:
    --------
    Plots the relevant data on the axis.

    """

    ax.plot(-P, gain, linestyle, color=colours[2], label=label, alpha=alpha,
            linewidth=1.5, zorder=zorder)
    ax.plot(-P, cost, linestyle, color=colours[0], alpha=alpha,
            linewidth=1.5, zorder=zorder)
    ax.plot(-P, gain - cost, linestyle, color=colours[1], alpha=alpha,
            linewidth=2.5, zorder=zorder)

    return


def behavioural_markers(ax, P, profits, costs):

    """
    Plots the maximum profit marks.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot

    P: array
        leaf water potential [MPa], from the soil water potential Ps to
        the critical water potential Pcrit for which cavitation of the
        xylem occurs

    profits: array
        net profit, i.e. gain - cost for all three strategies

    costs: array
        hydraulic cost [unitless], depending on the plant vulnerability
        curve, for all three strategies

    Returns:
    --------
    Highlights the profit marks.

    """

    # most profit?
    Popts = [-P[0][np.argmax(profits[0])], -P[1][np.argmax(profits[1])],
             -P[2][np.argmax(profits[2])]]

    # plot the arrow
    ax.scatter(Popts, np.amax(profits, axis=1))

    ax.text(Popts[-1] - 0.8, np.amax(profits[-1]) + 0.05,
            r'${\rm \Delta}{\rm \Psi}_{\rm leaf,opt}$ = ' +
            r'{0:.1f} MPa'.format(Popts[2] - Popts[1]), fontsize=9.,
            ha='left', va='center')

    return


def annotate_model(ax, colours):

    """
    Annotates the modelled curves.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot

    colours: array
        color used to plot gain, cost, and profit

    Returns:
    --------
    Plots the relevant data on the axis.

    """

    ax.text(0.35, 0.88, 'Carbon\ngain', color=colours[2], fontsize=12.,
            va='center', ha='center', transform=ax.transAxes)
    ax.text(0.7, 0.75, 'Hydraulic\ncost', color=colours[0], fontsize=12.,
            va='center', ha='center', transform=ax.transAxes)
    ax.text(0.6, 0.315, 'Net\nprofit', color=colours[1], fontsize=12.,
            va='center', ha='center', transform=ax.transAxes)

    return


def stream_ticks(p, P, Pcrit=False):

    """
    Sets up the ticks and tick labels on the x axis, following commonly
    referred to water potentials on the hydraulic stream.

    Arguments:
    ----------
    p: pandas series
        met forcings & params

    P: array
        leaf water potential [MPa], from the soil water potential Ps to
        the critical water potential Pcrit for which cavitation of the
        xylem occurs

    Pcrit: boolean
        if True, the 'Pcrit' is added to the tick list

    Returns:
    --------
    Renders the x axis accordingly.

    """

    P12 = Px(p, 12)
    iP12 = np.argmin(P - P12 >= 0.)

    Pticks = [-np.amax(P), -P[iP12], p.P50, p.P88]
    Ptick_labels = [r'${\rm \Psi}_{\rm sat}$', r'${\rm \Psi}_{\rm 12}$',
                    r'${\rm \Psi}_{\rm 50}$', r'${\rm \Psi}_{\rm 88}$']

    if Pcrit:
        Pticks += [-P[-1]]
        Ptick_labels += [r'${\rm \Psi}_{\rm 95}$']

    return Pticks, Ptick_labels


def format_model_axes(ax):

    """
    Formats the x and y axes, for better rendering.

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot

    Returns:
    --------
    Renders the plot accordingly.

    """

    # make sure both x and y axes start at 0.
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_ylim(0., 1.)
    ax.set_xlabel(r'${\rm \Psi}_{\rm leaf}$ (-MPa)', fontsize=12.)
    ax.set_ylabel(r'Gain ${\vert}$ Cost ${\vert}$ Profit', fontsize=14.)

    return


def get_fig_dir():

    """
    Returns the figure directory in which to store the plots

    """

    basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

    while 'src' in basedir:
        basedir = os.path.dirname(basedir)

    fig_dir = os.path.join(os.path.join(basedir, 'output'), 'plots')

    if not os.path.isdir(os.path.dirname(fig_dir)):
        os.makedirs(os.path.dirname(fig_dir))

    return fig_dir


def rcrit_model_behaviour():

    """
    Explores the ProfitMax algorith behaviour on the instantaneous
    timescale of optimisation, following three potential ways in which
    hydraulic and photosynthetic traits could be coordinated on longer
    time scales.

    Returns:
    --------
    output/figures/final_4_paper/profitmax_behaviours.png

    """

    # figure size and ratio
    fig = plt.figure(figsize=(5., 4.))
    ax = plt.subplot(111)

    # retrieve the three potential "behavioural" kmax values
    p = declared_params()
    p.Ps = -4.5

    # no legacy, ref.
    P, cost, gain = opt_stream(p, res, photo)

    # large legacy
    p.ratiocrit = 0.12
    P12, cost12, gain12 = opt_stream(p, res, photo)

    # little legacy
    p.ratiocrit = 0.001
    P001, cost001, gain001 = opt_stream(p, res, photo)

    lines = ['-', ':', '--']  # line types for the plots
    labels = ['P95', 'P88', 'P99.9']

    # plot the three alternative instantaneous optimisation
    plot_model(ax, P, gain, cost, colours, linestyle=lines[0], label=labels[0])
    plot_model(ax, P12, gain12, cost12, colours, linestyle=lines[1],
               label=labels[1])
    plot_model(ax, P001, gain001, cost001, colours, linestyle=lines[2],
               label=labels[2])

    # annotate the change in Pleaf,opt with an arrow
    behavioural_markers(ax, [P, P12, P001], [gain - cost, gain12 - cost12,
                        gain001 - cost001,], [cost, cost12, cost001])

    # annotate the model curves
    annotate_model(ax, colours)

    # retrieve horizontal axes ticks and labels
    Pticks, Ptick_labels = stream_ticks(p, P, Pcrit=True)

    # set axes ticks and labels
    ax.set_xticks(Pticks)
    ax.set_xticklabels(Ptick_labels)
    ax.set_yticks([0., 0.5, 1.])  # cost|gain|profit ticks
    ax.set_yticklabels(['0.0', '0.5', '1.0'])

    # make sure both x and y axes start at 0.
    format_model_axes(ax)
    plt.tight_layout()

    # set the legend and modify it
    lgd = ax.legend(fontsize=10., frameon=False, numpoints=2, borderpad=0.,
                    handletextpad=0.2, loc=4)  #, bbox_to_anchor=(-0.01, 0.995))

    for handle in lgd.legendHandles:
        handle.set_linewidth(1.)
        handle.set_color('dimgrey')

    for label in lgd.get_texts():
        label.set_color('dimgrey')

    # save the figure
    namefig = os.path.join(get_fig_dir(), 'profitmax_rcrit')
    fig.savefig('%s.png' % (namefig), dpi=1000, bbox_inches='tight')
    plt.close()

    return


#=======================================================================

if __name__ == "__main__":

    # run mode
    res = 'high'
    photo = 'Farquhar'

    plt.style.use('seaborn-ticks')
    plt.rcParams['text.usetex'] = True  # use LaTeX
    preamble = [r'\usepackage[sfdefault,light]{roboto}',
                r'\usepackage{sansmath}', r'\sansmath']
    plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)
    plt.rcParams['font.weight'] = 'light'
    colours = ['#66ccee', '#228833', '#ccbb44', '#ee6677']

    main()
