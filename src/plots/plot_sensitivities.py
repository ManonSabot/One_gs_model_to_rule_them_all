#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the sensitivity of model variables to a suite of
environmental drivers.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Model sensitivities to environmental drivers"
__author__ = "Manon E. B. Sabot"
__version__ = "1.0 (05.12.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import warnings  # ignore warnings

# plotting modules
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

def main():

    """
    Main function: plots model sensitivities for a series of drivers and
                   output variables

    Returns:
    --------
    'model_sensitivities_SX.jpg' in output/plots

    """

    base_dir = get_main_dir()  # working paths

    # data to plot
    df = (pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                      os.path.join(base_dir, 'output'), 'simulations'),
                      'idealised'), 'sensitivities'),
                      'overview_of_sensitivities.csv'), header=[0])
          .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    for SI in ['ST', 'S1']:  # sensitivity indices to plot

        fname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                             'plots'), 'model_sensitivities_%s.jpg' % (SI))

        if not os.path.isfile(fname):
            plt_setup()  # figure specs
            plot_sensitivities(df, fname, which=SI)  # plot data

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self):

        # default setup
        default_plt_setup(colours=['#e2694e', '#415381', '#53e3d4'])

        # font sizes
        plt.rcParams['font.size'] = plt.rcParams['font.size'] + 2.
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size'] - 1.
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size'] - 1.

        # lines
        plt.rcParams['lines.linewidth'] = 1.25

        # legend
        plt.rcParams['legend.borderpad'] = 0.

        # ticks
        plt.rcParams['ytick.major.pad'] = -2.

        # spines and grid
        plt.rcParams['axes.edgecolor'] = '#aaaaaa'
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['grid.color'] = plt.rcParams['axes.edgecolor']
        plt.rcParams['grid.linewidth'] = plt.rcParams['axes.linewidth']

        # figure spacing
        plt.rcParams['figure.subplot.wspace'] = 0.6
        plt.rcParams['figure.subplot.hspace'] = 0.4


def plot_sensitivities(df, figname, which='ST'):

    """
    Generates circular plots of the sensitivity of a number of model
    outputs to the model drivers, organized in subplots for each model
    considered

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to plot

    figname: string
        name of the figure to produce, including path

    which: string
        sensivity index to plot

    Returns:
    --------
    'model_sensitivities_SX.jpg'

    """

    # declare figure
    fig = plt.figure(figsize=(7., 5.5))

    # plot sensitivity to these drivers
    drivers, idxs = ['VPD', 'Ps', 'CO2', 'PPFD', 'Tair'], [which, ] * 5

    iter = 1  # track subplots

    # calculate the sensitivities
    for mod in model_order():

        # initialise the axis for the plot
        ax = plt.subplot(3, 4, iter, polar=True)

        # plot in radar chart
        ax.set_theta_offset(0.5 * np.pi)
        ax.set_theta_direction(-1)

        # which are the five dominant features across all variables?
        sub = df[df['output'].str.contains('(%s)' % (mod))].copy()

        for var in ['gs', 'Pleaf', 'Ci']:

            sub = df[df['output'] == '%s(%s)' % (var, mod)].copy()
            sub = sub[sub['driver'].isin(drivers)]
            sub.fillna(0., inplace=True)  # fill NaNs will zeros

            # declare empty arrays
            values = []
            xlabels = []

            for driver, e in zip(drivers, idxs):

                values += [sub[sub['driver'] == driver][e].values[0]]
                xlabels += [driver]

            try:
                # update variable names for display
                xlabels = (','.join(xlabels).replace('CO2', '$C_{a}$')
                           .replace('Tair', '$T_{a}$')
                           .replace('Ps', r'$\varPsi_{s}$')
                           .replace('VPD', '$D_{a}$').split(','))

                # what should the angle of each axis be on the plot?
                angles = np.linspace(0, 2 * np.pi, len(xlabels),
                                     endpoint=False).tolist()

                # "complete the loop" by appending start value to end
                values += [values[0]]
                angles += [angles[0]]

                # draw lines angles and labels
                if which == 'ST':
                    ax.set_rgrids([0., 0.25, 0.5, 0.75], [])
                    ax.set_ylim(0., 1.)

                else:  # S1
                    ax.set_rgrids([0., 0.25, 0.5], [])
                    ax.set_ylim(0., 0.75)

                ax.spines['polar'].set_visible(False)
                ax.set_thetagrids(np.degrees(angles[:-1]), xlabels)

                # adjust label alignment based on position in the circle
                for label, angle in zip(ax.get_xticklabels(), angles):

                    if angle in [0., np.pi]:
                        label.set_horizontalalignment('center')
                        label.set_verticalalignment('bottom')

                    elif 0. < angle < np.pi:
                        label.set_horizontalalignment('left')

                        if 0. < angle < np.pi / 2.:
                            label.set_verticalalignment('bottom')

                        else:
                            label.set_verticalalignment('top')

                    else:
                        label.set_horizontalalignment('right')

                        if np.pi < angle < 3. * np.pi / 2.:
                            label.set_verticalalignment('top')

                        else:
                            label.set_verticalalignment('bottom')

                if var == 'gs':  # display nicely
                    var = '$g_s$'
                    zorder = 1

                if var == 'Pleaf':
                    var = r'$\varPsi_{l}$'
                    zorder = 2

                if var == 'Ci':
                    var = r'$C_i$'
                    zorder = 3

                # plot the data
                if mod == 'std' and var == r'$\varPsi_{l}$':
                    next(ax._get_lines.prop_cycler)

                else:
                    line = ax.plot(angles, values, zorder=zorder, label=var)
                    ax.fill(angles, values, ec=line[0].get_color(), alpha=0.25,
                            zorder=zorder)

                # which model?
                plt.title(which_model(mod), pad=10.)

            except IndexError:
                pass

        # format params
        ax.tick_params(axis='x', which='major', pad=-9.)
        iter += 1

    # add the legend
    ax.legend(loc=1, ncol=3, bbox_to_anchor=(-1.65, -0.425))
    ax = fig.add_axes([0.25, 0.01, 0.065, 0.065], projection='polar')
    ax.set_zorder(-1)

    # setup ticks and labels
    if which == 'ST':
        ax.set_rgrids([0., 0.25, 0.5, 0.75], ['0', '', '', '0.75'])
        ax.set_rmax(0.75)

    else:
        ax.set_rgrids([0., 0.25, 0.5], ['0', '', '0.5'])
        ax.set_rmax(0.5)

    # draw lines angles
    ax.set_thetagrids(np.degrees([0, -45]), [])
    ax.set_title('Reading Key:', fontsize=7., x=-1.4, y=0.05)

    fig.savefig(figname)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    main()
