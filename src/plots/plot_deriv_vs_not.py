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
import itertools
import random  # pick a random day for the forcings to be generated
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import conv
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules
from TractLSM import hrun  # run the models

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string   # automate subplot lettering

#==============================================================================


def soil_water(df):

    sw = np.full(len(df), df['theta_sat'][0])
    track = 1

    start = sw[0]
    rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
    sw_min = (sw[0] + df['fc'][0]) / 2.25

    for i in range(len(df)):

        if i == 0:
            sw[i] = start

        else:
            sw[i] = sw[i-1]

        if df['PPFD'].iloc[i] > 50.:
            sw[i] = np.maximum(start / (1. - rate * track), sw_min)
            track += 1

    # now get the soil water potentials matching the soil moisture profile
    Ps = np.asarray([water_potential(df.iloc[0], sw[i])
                     for i in range(len(sw))])

    return sw, Ps


def which_model(short):

    if short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    elif short == 'pmax':
        lab = 'ProfitMax'

    elif short == 'cgn':
        lab = 'CGain'

    elif short == 'pmax2':
        lab = 'ProfitMax2'

    elif short == 'lcst':
        lab = 'LeastCost'

    return lab


def render_ylabels(ax, name, unit, fs=8., pad=0.):

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit),
                  labelpad=pad)

    return


def plot_comparison(df1, df2, df3, df4, fname):

    fig, axes = plt.subplots(figsize=(4., 4), nrows=2, ncols=2, sharex=True,
                             sharey='row')
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = axes.flat

    colours = ['#197aff', '#009231', '#a6d96a', '#ff8e12', '#ffe020']

    for i, df in enumerate([df1, df2, df3, df4]):

        if i % 2 == 0:
            ls = '-'

        else:
            ls = ':'

        ax1 = axes[0]
        ax2 = axes[2]

        if i > 1:
            ax1 = axes[1]
            ax2 = axes[3]

        for j, mod in enumerate(['wue', 'pmax', 'cgn', 'pmax2', 'lcst']):

            if i < 1:
                ax1.plot(df['hod'], df['gs(%s)' % (mod)], c=colours[j], ls=ls,
                         label=which_model(mod))

            else:
                ax1.plot(df['hod'], df['gs(%s)' % (mod)], c=colours[j], ls=ls)

            ax2.plot(df['hod'], df['Pleaf(%s)' % (mod)], c=colours[j], ls=ls)

    bottom, __ = axes[0].get_ylim()
    axes[0].set_ylim(bottom, 0.25)

    iter = 0

    for ax in [axes[0], axes[1], axes[2], axes[3]]:  # format axes ticks

        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

        if (ax == axes[-1]) or (ax == axes[-2]):
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

        else:
            ax.set_xticklabels([])

        if (ax == axes[0]) or (ax == axes[1]):  # gs and Ci:Ca
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # subplot labelling
        t = ax.text(0.875, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[iter]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.8))
        iter += 1

    # axes labels
    render_ylabels(axes[0], r'$g_{s}$', r'mol m$^{-2}$ s$^{-1}$')
    render_ylabels(axes[-2], r'$\Psi$$_{l}$', 'MPa')
    axes[-1].set_xlabel('hour of day (h)')
    axes[-2].set_xlabel('hour of day (h)')

    # axes suptitles
    axes[0].set_title('Low resolution')
    axes[1].set_title('High resolution')

    # split the legend in several parts
    axes[0].legend(loc=2, bbox_to_anchor=(2.15, 0.225), handleheight=1.5,
                   labelspacing=0.1)
    fig.savefig(fname, dpi=600)
    plt.close()

    return


###############################################################################

# working paths
base_dir = get_main_dir()

ipath = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'idealised')
df, __ = read_csv(os.path.join(ipath, 'wet_calibration.csv'))

# soil moisture profile
df['sw'] = df['theta_sat']
df.fillna(method='ffill', inplace=True)
df['sw'], df['Ps'] = soil_water(df)

# week 1, at some point in the middle of the week
df = df[df['doy'] >= df['doy'].iloc[0] + 7 * 3]  #4]

# run the models
models = ['WUE', 'ProfitMax', 'CGain', 'ProfitMax2', 'LeastCost']

plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
plt.rcParams['text.usetex'] = True  # use LaTeX
preamble = [r'\usepackage[sfdefault,light]{merriweather}',
            r'\usepackage{mathpazo}', r'\usepackage{amsmath}']
plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)
plt.rcParams['font.size'] = 7.
plt.rcParams['axes.labelsize'] = 8.
plt.rcParams['xtick.labelsize'] = 6.
plt.rcParams['ytick.labelsize'] = 6.
plt.rcParams['legend.edgecolor'] = 'w'

# actual form
df1 = hrun(None, df, 48, 'Farquhar', models=models,
           resolution='low', inf_gb=True)
df1.columns = df1.columns.droplevel(level=1)

# derivative form
df2 = hrun(None, df, 48, 'Farquhar', models=models,
           resolution='low', inf_gb=True, deriv=True)
df2.columns = df2.columns.droplevel(level=1)

# actual form
df3 = hrun(None, df, 48, 'Farquhar', models=models,
           resolution='high', inf_gb=True)
df3.columns = df3.columns.droplevel(level=1)

# derivative form
df4 = hrun(None, df, 48, 'Farquhar', models=models,
           resolution='high', inf_gb=True, deriv=True)
df4.columns = df4.columns.droplevel(level=1)

plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
plt.rcParams['text.usetex'] = True  # use LaTeX
preamble = [r'\usepackage[sfdefault,light]{merriweather}',
            r'\usepackage{mathpazo}', r'\usepackage{amsmath}']
plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)
plt.rcParams['font.size'] = 7.
plt.rcParams['axes.labelsize'] = 8.
plt.rcParams['xtick.labelsize'] = 6.
plt.rcParams['ytick.labelsize'] = 6.
plt.rcParams['legend.edgecolor'] = 'w'
plot_comparison(df1, df2, df3, df4, 'deriv_vs_not.png')

exit(1)
