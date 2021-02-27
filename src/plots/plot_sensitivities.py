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

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

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
        plt.rcParams['savefig.dpi'] = 600.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.05  # padding to use
        plt.rcParams['savefig.jpeg_quality'] = 100

        # figure spacing
        plt.rcParams['figure.subplot.wspace'] = 0.6
        plt.rcParams['figure.subplot.hspace'] = 0.4

        # colors
        plt.rcParams['axes.edgecolor'] = '#aaaaaa'
        plt.rcParams['axes.prop_cycle'] = cycler(color=['#e2694e', '#415381',
                                                        '#53e3d4'])

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        main_font = r'\usepackage[sfdefault,light]{merriweather}'
        plt.rcParams['text.latex.preamble'] = [main_font,
                                               r'\usepackage{mathpazo}'
                                               r'\usepackage{amsmath}']
        plt.rcParams['font.size'] = 8.
        plt.rcParams['axes.labelsize'] = 7.
        plt.rcParams['xtick.labelsize'] = 7.
        plt.rcParams['ytick.labelsize'] = 7.
        plt.rcParams['ytick.major.pad'] = -2.

        # lines
        plt.rcParams['lines.linewidth'] = 1.25

        # grid
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['grid.color'] = plt.rcParams['axes.edgecolor']
        plt.rcParams['grid.linewidth'] = plt.rcParams['axes.linewidth']

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams['legend.facecolor'] = 'none'
        plt.rcParams['legend.edgecolor'] = 'none'
        plt.rcParams['legend.borderpad'] = 0.


def dominant_features(df):

    df.reset_index(drop=True, inplace=True)

    # criteria 1: to be considered, index must be > 0.05
    df[['S1', 'S1_conf']] = df[['S1', 'S1_conf']].where(df['S1'] > 0.05)
    df[['ST', 'ST_conf']] = df[['ST', 'ST_conf']].where(df['ST'] > 0.05)
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria 2: the total effect index must be > S1 + 10%
    df[['ST', 'ST_conf']] = \
        df[['ST', 'ST_conf']].where(df['ST'] > 1.1 * df['S1'])
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria 3: first order effects cannot be << ST
    df[['S1', 'S1_conf']] = \
        df[['S1', 'S1_conf']].where(df['S1'] > 0.7 * df['ST'])
    df.fillna(0., inplace=True)  # fill NaNs will zeros

    # criteria: confidence on at least one index
    df = df[np.logical_or(df['S1'] > 2. * df['S1_conf'],
                          df['ST'] > 2. * df['ST_conf'])]

    # sum indices across variables
    for driver in df['driver'].unique():

        tot = df[df['driver'] == driver].sum(axis=0)
        total = pd.DataFrame({'output': 'total', 'driver': driver,
                              'S1': tot['S1'], 'S1_conf': 0., 'ST': tot['ST'],
                              'ST_conf': 0.}, index=[0])
        df = df.append(total, ignore_index=True)

    # now select the 5 most dominant features
    total = df[df['output'] == 'total']
    dominant = total[['driver', 'S1', 'ST']].nlargest(5, ['S1', 'ST'])

    # sort by order of dominant effect
    dominant['total'] = dominant.sum(axis=1)
    dominant = (dominant.sort_values('total', ascending=False)
                        .drop('total', axis=1))

    return list(dominant['driver']), ['ST',] * len(dominant)


def update_var_names(array):


    array = [e.replace('CO2', '$C_{a}$') for e in array]
    array = [e.replace('Tair', '$T_{a}$') for e in array]
    array = [e.replace('Ps', r'$\varPsi_{s}$') for e in array]
    array = [e.replace('VPD', '$D_{a}$') for e in array]

    return array


def which_model(short):

    if short == 'std1':
        lab = r'Medlyn-$\beta$'

    if short == 'tuz':
        lab = 'Tuzet'

    if short == 'sox1':
        lab = 'Eller'

    if short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    if short == 'cmax':
        lab = 'CMax'

    if short == 'pmax':
        lab = 'ProfitMax'

    if short == 'pmax2':
        lab = 'ProfitMax2'

    if short == 'cgn':
        lab = 'CGain'

    if short == 'lcst':
        lab = 'LeastCost'

    if short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

    if short == 'cap':
        lab = 'CAP'

    if short == 'mes':
        lab = 'MES'

    return lab


def plot_sensitivities(df, figname):

    plt_setup()  # rendering
    fig = plt.figure(figsize=(7., 5.5))  # declaring the figure

    iter = 1

    drivers, idxs = ['VPD', 'Ps', 'CO2', 'PPFD', 'Tair'], ['ST',] * 5 # dominant_features(df.copy())

    # calculate the sensitivities
    for mod in ['std1', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'sox2',
                'pmax2', 'lcst', 'cap', 'mes']:

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
                xlabels = update_var_names(xlabels)

                # what should the angle of each axis be on the plot?
                angles = np.linspace(0, 2 * np.pi, len(xlabels),
                                     endpoint=False).tolist()

                # "complete the loop" by appending start value to the end
                values += [values[0]]
                angles += [angles[0]]

                # draw lines angles and labels
                ax.set_rgrids([0., 0.25, 0.5, 0.75], [])
                ax.set_ylim(0., 0.95)
                ax.spines['polar'].set_visible(False)
                ax.set_thetagrids(np.degrees(angles[:-1]), xlabels)

                # adjust label alignment based on where it is in the circle
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

                if var == 'gs':
                    var = '$g_s$'
                    zorder = 1

                if var == 'Pleaf':
                    var = r'$\varPsi_{l}$'
                    zorder = 2

                if var == 'Ci':
                    var = r'$C_i$'
                    zorder = 3

                # plot the data
                if mod == 'std1' and var == r'$\varPsi_{l}$':
                    next(ax._get_lines.prop_cycler)

                else:
                    l = ax.plot(angles, values, zorder=zorder, label=var)
                    ax.fill(angles, values, ec=l[0].get_color(), alpha=0.25,
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
    ax.set_rgrids([0., 0.25, 0.5, 0.75], ['0', '', '', '0.75'])
    ax.set_rmax(0.75)

    # draw lines angles
    ax.set_thetagrids(np.degrees([0, -45]), [])
    ax.set_title('Reading Key:', fontsize=7., x=-1.4, y=0.05)

    fig.savefig(figname)
    plt.close()

    return


###############################################################################

base_dir = get_main_dir()
figname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                       'plots'), 'model_sensitivities_ST_1.5.jpg')

#if not os.path.isfile(figname):
fname = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(
                     base_dir, 'output'), 'simulations'), 'idealised'),
                     'sensitivities'), 'overview_of_sensitivities_1.5MPa.csv')
df = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
            .dropna(axis=1, how='all').squeeze())
plot_sensitivities(df, figname)
