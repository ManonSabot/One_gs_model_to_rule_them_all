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
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import itertools

# plotting
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.signal as signal
from pygam import LinearGAM  # fit the functional shapes
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# own modules
from TractLSM import conv, cst
from TractLSM import InForcings  # generate met data & read default params
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules

#==============================================================================

class plt_setup(object):

    def __init__(self):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.05  # padding to use
        plt.rcParams['savefig.jpeg_quality'] = 100
        plt.style.use('seaborn-ticks')

        # figure spacing
        plt.rcParams['figure.subplot.wspace'] = 0.6
        plt.rcParams['figure.subplot.hspace'] = 0.4

        # colors
        plt.rcParams['axes.prop_cycle'] = cycler(color=['#8da0cb', '#66c2a5',
                                                        '#fc8d62', '#a6d854',
                                                        '#e78ac3'])

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                               r'\usepackage{mathpazo}',
                                               r'\usepackage{amsmath}']
        plt.rcParams['axes.titlesize'] = 9.
        plt.rcParams['xtick.labelsize'] = 6.
        plt.rcParams['ytick.labelsize'] = 6.

        # lines
        plt.rcParams['lines.linewidth'] = 0.9

        # grid
        plt.rcParams['grid.color'] = 'grey'
        plt.rcParams['grid.linewidth'] = 0.25

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams['legend.edgecolor'] = 'w'


def dominant_features(df):

    df.reset_index(drop=True, inplace=True)

    # criteria 1: to be considered in the total, index must be > 0.05
    df[['S1', 'S1_conf']] = df[['S1', 'S1_conf']].where(df['S1'] > 0.05)
    df[['ST', 'ST_conf']] = df[['ST', 'ST_conf']].where(df['ST'] > 0.05)
    df[['S2', 'S2_conf']] = df[['S2', 'S2_conf']].where(df['S2'] > 0.05)
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria 2: the total effect index must be > S1 + 10%
    df[['ST', 'ST_conf']] = \
        df[['ST', 'ST_conf']].where(df['ST'] > 1.1 * df['S1'])
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria 3: first order effects cannot be << ST
    df[['S1', 'S1_conf']] = \
        df[['S1', 'S1_conf']].where(df['S1'] > 0.9 * df['ST'])
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria: confidence on at least one index
    #df = df[np.logical_or(np.logical_or(df['S1'] > 2. * df['S1_conf'],
    #                      df['ST'] > 2. * df['ST_conf']),
    #                      df['S2'] > 2. * df['S2_conf'])]

    # sum indices across variables
    for driver in df['driver'].unique():

        tot = df[df['driver'] == driver].sum(axis=0)
        total = pd.DataFrame({'output': 'total', 'driver': driver,
                              'S1': tot['S1'], 'S1_conf': 0., 'ST': tot['ST'],
                              'ST_conf': 0., 'S2': tot['S2'], 'S2_conf': 0.},
                             index=[0])
        df = df.append(total, ignore_index=True)

    # now select the 5 most dominant features
    total = df[df['output'] == 'total']
    dominant = (total[['driver', 'S1', 'ST', 'S2']]
                     .nlargest(5, ['S1', 'ST', 'S2']))

    # reapply criteria 2: the total effect index must be > S1 + 10%
    dominant['ST'] = \
        dominant['ST'].where(dominant['ST'] > 1.1 * dominant['S1'])
    dominant.fillna(0., inplace=True)  # fill NaNs will zeros

    # reapply criteria 3: first order effects cannot be << ST
    dominant['S1'] = \
        dominant['S1'].where(dominant['S1'] > 0.9 * dominant['ST'])
    dominant.fillna(0., inplace=True)  # fill NaNs will zeros

    # sort by order of dominant effect
    dominant['total'] = dominant.sum(axis=1)
    dominant = (dominant.sort_values('total', ascending=False)
                        .drop('total', axis=1))
    dominant['idxs'] = 'S1'
    dominant['idxs'][dominant['ST'] > 0.] = 'ST'
    dominant['idxs'][dominant['S2'] > 0.] = 'S2'

    return list(dominant['driver']), list(dominant['idxs'])


def update_var_names(array):


    array = [e.replace('CO2', '$C_{a}$') for e in array]
    array = [e.replace('Tair', '$T_{a}$') for e in array]
    array = [e.replace('Ps', r'$\varPsi_{s}$') for e in array]
    array = [e.replace('VPD', '$D_{a}$') for e in array]
    array = [e.replace('PPFD', '$PAR$') for e in array]
    array = [e.replace('u', '$u$') for e in array]

    return array


def which_model(short):

    if short == 'std1':
        lab = 'Reference'

    if short == 'tuz':
        lab = 'Tuzet'

    if short == 'sox1':
        lab = 'Eller'

    if short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

    if short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    if short == 'cgn':
        lab = 'CGainNet'

    if short == 'pmax':
        lab = 'ProfitMax'

    if short == 'cmax':
        lab = 'CMax'

    if short == 'lcst':
        lab = 'LeastCost'

    if short == 'cap':
        lab = 'CAP'

    if short == 'mes':
        lab = 'MES'

    return lab


def plot_sensitivities(df, figname):

    plt_setup()  # rendering
    fig = plt.figure(figsize=(6, 4.5))  # declaring the figure

    iter = 2

    # calculate the sensitivities
    for mod in ['std1', 'tuz', 'sox1', 'wue', 'cgn', 'pmax', 'cmax', 'lcst',
                'sox2', 'cap', 'mes']:

        # initialise the axis for the plot
        ax = plt.subplot(3, 4, iter, polar=True)

        # plot in radar chart
        ax.set_theta_offset(0.5 * np.pi)
        ax.set_theta_direction(-1)

        # which are the five dominant features across all variables?
        sub = df[df['output'].str.contains('(%s)' % (mod))].copy()
        drivers, idxs = dominant_features(sub)

        for var in ['gs', 'Pleaf', 'Ci', 'E', 'A']:

            sub = df[df['output'] == '%s(%s)' % (var, mod)].copy()
            sub = sub[sub['driver'].isin(drivers)]
            sub.fillna(0., inplace=True)  # fill NaNs will zeros

            # declare empty arrays
            values = []
            xlabels = []

            for driver, e in zip(drivers, idxs):

                values += [sub[sub['driver'] == driver][e].values[0]]
                xlabels += ['%s*' % (driver) if e == 'ST' else driver]

            try:
                xlabels = update_var_names(xlabels)

                # what should the angle of each axis be on the plot?
                angles = np.linspace(0, 2 * np.pi, len(xlabels),
                                     endpoint=False).tolist()

                # "complete the loop" by appending start value to the end
                values += [values[0]]
                angles += [angles[0]]

                # setup ticks and labels
                plt.xticks(angles[:-1], xlabels)
                plt.yticks([0.25, 0.5, 0.75], [])
                plt.ylim(0., 1.)

                # draw lines angles and labels
                ax.set_rlabel_position(0)  # y is radially centered around 0
                ax.set_thetagrids(np.degrees(angles), xlabels)
                ax.spines['polar'].set_color(plt.rcParams['grid.color'])
                ax.spines['polar'].set_linewidth(plt.rcParams['grid.linewidth'])

                # adjust label alignment based on where it is in the circle
                for label, angle in zip(ax.get_xticklabels(), angles):

                  if angle in (0, np.pi):
                    label.set_horizontalalignment('center')

                  elif 0 < angle < np.pi:
                    label.set_horizontalalignment('left')

                  else:
                    label.set_horizontalalignment('right')

                if var == 'gs':
                    var = '$g_{s}$'

                if var == 'Pleaf':
                    var = r'$\varPsi_{l}$'

                if var == 'E':
                    var = '$E$'

                if var == 'Ci':
                    var = '$C_{i}$'

                if var == 'A':
                    var = '$A_{n}$'

                # plot the data
                ax.plot(angles, values, label=var)

                if mod == 'gs2':
                    pad = 8.

                else:
                    pad = 11.

                plt.title(which_model(mod), pad=pad)

            except IndexError:
                pass

        # format params
        ax.tick_params(axis='x', which='major', pad=-9.)

        if iter == 2:
            ax.legend(loc=1, bbox_to_anchor=(-0.9, 1.25))

        iter += 1

    fig.savefig(figname)
    plt.close()

    return


###############################################################################

base_dir = get_main_dir()
figname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                       'figures'), 'model_sensitivities.png')

if not os.path.isfile(figname):
    fname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                         'Sensitivities'), 'overview_of_sensitivities.csv')
    df = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
            .dropna(axis=1, how='all').squeeze())
    plot_sensitivities(df, figname)
