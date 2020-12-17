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
import string   # automate subplot lettering

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import conv, cst
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules

#==============================================================================


class plt_setup(object):

    def __init__(self, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # colors
        if colours is None:  # use the default colours
            colours = ['#1a1a1a', '#6f32c7', '#a182bf', '#1087e8', '#9be2fd',
                       '#086527', '#33b15d', '#a6d96a', '#a2a203', '#ecec3a',
                       '#a42565', '#f9aab7']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        main_font = r'\usepackage[sfdefault,light]{merriweather}'
        plt.rcParams['text.latex.preamble'] = [main_font,
                                               r'\usepackage{mathpazo}'
                                               r'\usepackage{amsmath}']
        plt.rcParams['font.size'] = 6.
        plt.rcParams['axes.labelsize'] = 7.
        plt.rcParams['xtick.labelsize'] = 6.
        plt.rcParams['ytick.labelsize'] = 6.

        # lines
        plt.rcParams['lines.linewidth'] = 1.5

        # patches (e.g. the shapes in the legend)
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['patch.edgecolor'] = 'k'
        plt.rcParams['patch.force_edgecolor'] = True  # ensure it's used

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams['legend.edgecolor'] = 'w'
        plt.rcParams['legend.borderpad'] = 0.5

        # grid
        plt.rcParams['grid.color'] = '#bdbdbd'
        plt.rcParams['grid.linewidth'] = 0.25

        # spines and ticks
        plt.rcParams['axes.linewidth'] = 0.65
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 1.5
        plt.rcParams['xtick.major.width'] = 0.75
        plt.rcParams['xtick.minor.width'] = 0.75
        plt.rcParams['ytick.major.size'] = plt.rcParams['xtick.major.size']
        plt.rcParams['ytick.minor.size'] = plt.rcParams['xtick.minor.size']
        plt.rcParams['ytick.major.width'] = plt.rcParams['xtick.major.width']
        plt.rcParams['ytick.minor.width'] = plt.rcParams['xtick.minor.width']


def figsize(nrows, ncols, width=4.5, height=3.25):

    x = width * ncols + (ncols - 1.) * plt.rcParams['figure.subplot.wspace']
    y = height * nrows + (nrows - 1.) * plt.rcParams['figure.subplot.hspace']

    if nrows == 1:
        y -= plt.rcParams['figure.subplot.hspace']

    return x, y


def soil_water(df, profile):

    sw = np.full(len(df), df['theta_sat'][0])
    track = 1

    if profile == 'wet':
        start = sw[0]
        rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (sw[0] + df['fc'][0]) / 2.25

    elif profile == 'inter':
        start = 0.9 * sw[0]
        rate = -5. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (df['fc'][0] + df['pwp'][0]) / 2.

    elif profile == 'dry':
        start = 0.8 * sw[0]
        rate = -12. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = df['pwp'][0] / 2.

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


def weekly(df, tidy=False):

    df0 = df.copy()  # avoids directly modifying the df

    # add the weeks
    df0['week'] = 1
    days = 7.

    for __ in range(3):  # deal with the three other weeks

        df0.loc[df0['doy'] >= df0.loc[0, 'doy'] + days, 'week'] += 1
        days += 7.

    df1 = df0.groupby([df0.week, df0.hod]).mean()
    df2 = df0.groupby([df0.week, df0.hod]).max()
    df3 = df0.groupby([df0.week, df0.hod]).min()

    # convert the hod in the index into a column once again
    df1 = df1.reset_index(level=['hod'])
    df2 = df2.reset_index(level=['hod'])
    df3 = df3.reset_index(level=['hod'])

    if tidy:  # remove the excess zeros for aesthetics when plotting
        df1 = df1[np.logical_and(df1['hod'] > 4., df1['hod'] < 20.)]
        df2 = df2[np.logical_and(df2['hod'] > 4., df2['hod'] < 20.)]
        df3 = df3[np.logical_and(df3['hod'] > 4., df3['hod'] < 20.)]

    # reset the indexes for good
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)

    return df1, df2, df3


def which_model(short):

    if short == 'std1':
        lab = r'Medlyn-$\beta$'

    elif short == 'std2':
        lab = r'Medlyn-$f_{\varPsi_{l,pd}}$'

    elif short == 'tuz':
        lab = 'Tuzet'

    elif short == 'sox1':
        lab = 'Eller'

    elif short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

    elif short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    elif short == 'cgn':
        lab = 'CGain'

    elif short == 'pmax':
        lab = 'ProfitMax'

    elif short == 'pmax2':
        lab = 'ProfitMax2'

    elif short == 'cmax':
        lab = 'CMax'

    elif short == 'lcst':
        lab = 'LeastCost'

    elif short == 'cap':
        lab = 'CAP'

    elif short == 'mes':
        lab = 'MES'

    else:
        lab = None

    return lab


def render_ylabels(ax, name, unit, fs=7.):

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit))

    return


def smooth_soil_water(df):

    # retrieve the soil moisture profiles
    sw_wet, __ = soil_water(df, 'wet')
    sw_inter, __ = soil_water(df, 'inter')
    sw_dry, __ = soil_water(df, 'dry')

    # smooth the decay of sm when plotting it
    N  = 3  # filter order
    Wn = 0.0245  # cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    sw_wet = signal.filtfilt(B, A, sw_wet)
    sw_inter = signal.filtfilt(B, A, sw_inter)
    sw_dry = signal.filtfilt(B, A, sw_dry)

    return sw_wet, sw_inter, sw_dry


def plot_forcings(df, fname):

    fig, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.3)
    axes = axes.flat

    # first, plot the atmospheric conditions
    wavg, wmax, wmin = weekly(df)
    axes[0].plot(wavg['PPFD'], color='k')
    axes[1].plot(wavg['Tair'], color='k')
    axes[2].plot(wavg['VPD'], color='k')

    # and the diurnal uncertainties
    axes[0].fill_between(wmax['PPFD'].index, wmin['PPFD'], wmax['PPFD'],
                         facecolor='lightgrey', edgecolor='none')
    axes[1].fill_between(wmax['Tair'].index, wmin['Tair'], wmax['Tair'],
                         facecolor='lightgrey', edgecolor='none')
    axes[2].fill_between(wmax['VPD'].index, wmin['VPD'], wmax['VPD'],
                         facecolor='lightgrey', edgecolor='none')

    # plot the soil moisture profiles
    sw_wet, sw_inter, sw_dry = smooth_soil_water(df)
    axes[3].plot(sw_wet, color='#fdcc8a')
    axes[3].plot(sw_inter, color='#fc8d59')
    axes[3].plot(sw_dry, color='#d7301f')

    # plot reference soil moisture levels
    axes[3].plot(df['theta_sat'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.99 * len(df), df['theta_sat'][0] - 0.02115, 'saturation',
                 ha='right')
    axes[3].plot(df['fc'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.99 * len(df), df['fc'][0] + 1.15e-2, 'field capacity',
                 ha='right')
    axes[3].plot(df['pwp'], ':k',
                 linewidth=plt.rcParams['lines.linewidth'] / 2., zorder=-1)
    axes[3].text(0.99 * len(df), df['pwp'][0] + 1.15e-2, 'wilting point',
                 ha='right')

    for i, ax in enumerate(axes):  # format axes

        if i < 3:  # weekly diurnal forcings
            ax.set_xlim([0, len(wavg)])
            ax.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
            ax.set_xticklabels(['',] * 4)

        else:  # soil moisture profiles
            ax.set_xlim([0, len(df)])
            ax.set_xticks([48. * 7. * 0.5, 48. * 7. * 1.5, 48. * 7. * 2.5,
                           48. * 7. * 3.5])

        if i > 1:
            ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # subplot labelling
        t = ax.text(0.025, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.8))

    # axes labels
    render_ylabels(axes[0], 'PPFD', r'$\mu$mol m$^{-2}$ s$^{-1}$')
    render_ylabels(axes[1], 'Air temperature', '$^\circ$C')
    render_ylabels(axes[2], 'Vapour pressure deficit', 'kPa')
    render_ylabels(axes[3], 'Soil water content', 'm$^{3}$ m$^{-3}$')

    fig.savefig(fname)
    plt.close()


def get_P12(Px1, Px2, x1, x2):


    """
    Finds the leaf water potential associated with a specific x% decrease in
    hydraulic conductance, using the plant vulnerability curve.

    Arguments:
    ----------
    Px: float
        leaf water potential [MPa] at which x% decrease in hydraulic conductance
        is observed

    x: float
        percentage loss in hydraulic conductance

    Returns:
    --------
    P88: float
        leaf water potential [MPa] at which 88% decrease in hydraulic
        conductance is observed
    """

    Px1 = np.abs(Px1)
    Px2 = np.abs(Px2)
    x1 /= 100. # normalise between 0-1
    x2 /= 100.

    try:  # c is derived from both expressions of b
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))

    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))

    b = Px1 / ((- np.log(1 - x1)) ** (1. / c))
    P12 = -b * ((- np.log(0.88)) ** (1. / c)) # MPa

    return P12


def plot_diag_target(df, fname, Ca=40., P50=None, P88=None):

    fig, axes = plt.subplots(figsize=(6., 6), nrows=3, ncols=2, sharex=True)
    plt.subplots_adjust(hspace=0.15, wspace=0.35)
    axes = axes.flat
    axes[1].axis('off')  # mask frame of first row second column
    axes = np.delete(axes, 1)

    # VC info
    if (P50 is not None) and (P88 is not None):
        P12 = get_P12(P50, P88, 50., 88.)
        axes[2].axhline(P12, linestyle=':', linewidth=1.)

    # deal with the nans
    davg = df.copy().select_dtypes(exclude=['object'])
    davg.where(davg < 9999., inplace=True)
    (davg[davg.filter(like='gs(').columns]
         .where(davg[davg.filter(like='gs(').columns] > 1.e-9, inplace=True))
    davg = davg[davg != 0].groupby(['hod']).mean()

    # remove hod
    davg = davg[davg[davg.filter(like='gs(').columns].sum(axis=1) > 0.]

    # inset axis for LWP
    if any(davg[davg.filter(like='Pleaf(').columns] < P50):
        iax = axes[2].inset_axes([0.69, 0.69, 0.3, 0.3])
        iax.spines['bottom'].set_color('grey')
        iax.spines['left'].set_color('grey')
        iax.spines['right'].set_visible(False)
        iax.spines['top'].set_visible(False)
        iax.axhline(P12, linestyle=':', linewidth=1.)
        iax.axhline(P50, linestyle=':', linewidth=1.)
        iax.axhline(P88, linestyle=':', linewidth=1.)

    # smooth the min-max diurnals
    B, A = signal.butter(3, 0.4)

    for mod in ['std1', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'pmax2', 'cgn',
                'lcst', 'sox2', 'cap', 'mes']:

        if mod == 'std1':
            lw = 4.
            alpha = 1.

        else:
            lw = plt.rcParams['lines.linewidth']
            alpha = 0.8

        axes[0].plot(davg['gs(%s)' % (mod)].rolling(window=2).mean(),
                     linewidth=lw, alpha=alpha, label=which_model(mod))
        axes[1].plot(davg['Ci(%s)' % (mod)].rolling(window=2).mean() / Ca,
                     linewidth=lw, alpha=alpha)

        if mod == 'tuz':
            iax.plot(davg['Pleaf(%s)' % (mod)].rolling(window=2).mean(),
                     linewidth=plt.rcParams['lines.linewidth'], alpha=alpha)
            axes[2].plot(davg['Pleaf(%s)' % (mod)].rolling(window=2).mean(),
                         linewidth=lw, alpha=alpha)

        elif any(davg['Pleaf(%s)' % (mod)] < P50):
            iax.plot(davg['Pleaf(%s)' % (mod)].rolling(window=2).mean(),
                     linewidth=lw, alpha=alpha)
            next(axes[2]._get_lines.prop_cycler)

        elif mod != 'std1':
            axes[2].plot(davg['Pleaf(%s)' % (mod)].rolling(window=2).mean(),
                         linewidth=lw, alpha=alpha)
            next(iax._get_lines.prop_cycler)

        else:
            next(axes[2]._get_lines.prop_cycler)
            next(iax._get_lines.prop_cycler)

        axes[3].plot(davg['A(%s)' % (mod)].rolling(window=2).mean(),
                     linewidth=lw, alpha=alpha)
        axes[4].plot(davg['E(%s)' % (mod)].rolling(window=2).mean(),
                     linewidth=lw, alpha=alpha)

    # axes labels
    render_ylabels(axes[0], r'$g_{s}$', r'mol m$^{-2}$ s$^{-1}$')
    render_ylabels(axes[1], r'$C_{i}$ : $C_{a}$', '-')
    render_ylabels(axes[2], r'$\Psi$$_{l}$', 'MPa')
    render_ylabels(axes[3], r'$A_n$', '$\mu$mol m$^{-2}$ s$^{-1}$')
    render_ylabels(axes[4], r'$E$', 'mmol m$^{-2}$ s$^{-1}$')
    axes[-1].set_xlabel('hour of day (h)')
    axes[-2].set_xlabel('hour of day (h)')

    for i, ax in enumerate(axes):  # format axes ticks

        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

        if (ax == axes[-1]) or (ax == axes[-2]):
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.d'))

            if ax == axes[-1]:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

        else:
            ax.set_xticklabels([])

        if (ax == axes[0]) or (ax == axes[-1]):  # gs and Ci:Ca
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # subplot labelling
        t = ax.text(0.025, 0.925,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                    transform=ax.transAxes, weight='bold')
        t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.8))

    iax.xaxis.set_major_locator(plt.NullLocator())
    iax.set_yticks([-0.5, round(P50)])
    iax.tick_params(axis='y', colors='grey')

    # split the legend in several parts
    axes[0].legend(loc=2, ncol=2, bbox_to_anchor=(1.35, 0.75),
                   handleheight=1.5, labelspacing=0.1)
    fig.savefig(fname)
    plt.close()

    return


def plot_impact_summary(df, fname):

    fig, axes = plt.subplots(figsize=(7, 4), nrows=2, ncols=2, sharex=True,
                             sharey='row')
    plt.subplots_adjust(hspace=0.05, wspace=0.065)
    axes = axes.flat

    mods = ['tuz', 'sox1', 'wue', 'cmax', 'pmax', 'pmax2', 'cgn', 'lcst',
            'sox2', 'cap', 'mes']

    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()
    twet = [e for e in df.index.to_list() if e.split('_')[-1] == 'wet']
    tinter = [e for e in df.index.to_list() if e.split('_')[-1] == 'inter']

    for i, ax in enumerate(axes):

        if i < 2:
            sub = df[GPP]

        else:
            sub = df[E]

        if i % 2 == 0:
            sub = sub.loc[twet]
            ax.axvspan(-0.1, 0.9, hatch='.' * 6, facecolor='none',
                       edgecolor='#2e7d9b', alpha=0.1)  # color calib.

        else:
            sub = sub.loc[tinter]
            ax.axvspan(0.9, 1.9, hatch='.' * 6, facecolor='none',
                       edgecolor='#fc8635', alpha=0.1)  # color calib.

        # we're plotting in reverse order
        sub = sub.iloc[::-1]
        sub.reset_index(inplace=True)
        pos = np.arange(float(len(sub)) + 1)

        # ref model
        ax.hlines(sub[sub.filter(like='std1').columns], pos[:-1] - 0.1,
                  pos[1:] - 0.1, linewidth=0.75, alpha=0.75, zorder=20,
                  label=which_model('std1'))
        pos += 0.0275  # necessary alignment when plotting

        for j, p in enumerate(pos[:-1]):

            next(ax._get_lines.prop_cycler)  # skip black used for Medlyn

            for mod in mods:

                ax.scatter(p, sub[sub.filter(like=mod).columns].values[j][0],
                           marker=r'$\diamondsuit$', s=45.,
                           c=next(ax._get_lines.prop_cycler)['color'],
                           label=which_model(mod))
                p += 0.075

        ax.set_xlim(-0.1, 4.9)

        if i > 1:
            ax.set_xticks(np.arange(0.4, 4.5))
            ax.set_xticklabels(['Wet', 'Inter.', 'Dry', '2 x $D_a$',
                                '1.5 x $C_a$'])

            if i < 3:
                render_ylabels(ax, r'$E$', r'mm y$^{-1}$')

        elif i < 1:
            render_ylabels(ax, 'GPP', r'gC y$^{-1}$')
            ax.set_title('Wet calibration')

        else:
            ax.set_title('Intermediate calibration')

        ax.set_xticks(np.arange(0.9, 5.), minor=True)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.set_yticklabels(ax.get_yticks())  # force LaTex
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.d'))

        # subplot labelling
        ax.text(0.025, 0.95, r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                transform=ax.transAxes, weight='bold')

        # remove spines and add grid
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(which='major', length=0)
        ax.tick_params(which='minor', length=0)
        #ax.grid(which='major', axis='y')
        ax.xaxis.grid(which='minor')
        ax.yaxis.grid(which='major')

    # legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(mods) + 1], labels[:len(mods) + 1],
              bbox_to_anchor=(1.025, 1. / 3.), loc=3)

    fig.savefig(fname)
    plt.close()


###############################################################################

# first, activate user defined rendering options
plt_setup()

base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'idealised')
ofdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'idealised')

# path to input data
fname1 = os.path.join(ifdir, 'wet_calibration.csv')
df1, __ = read_csv(fname1)

# initialise soil moisture forcings
df1['sw'] = df1['theta_sat']
df1.fillna(method='ffill', inplace=True)

# plot the atmospheric forcings
figdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')
figname = os.path.join(figdir, 'training_forcing_soil_moisture.png')

if not os.path.isfile(figname):
    plot_forcings(df1, figname)

for training in ['wet', 'inter']:

    ifdir = ifdir.replace('simulations', 'calibrations')
    opath = os.path.join(ofdir, 'univar_change')

    # now get the actual model outputs
    fname = os.path.join(opath, 'insample_%s_%s.csv' % (training, training))
    df2, __ = read_csv(fname)

    figname = os.path.join(figdir, 'calib_variables_%s.png' % (training))

    if not os.path.isfile(figname):
        plot_diag_target(df2, figname, Ca=df1.loc[0, 'CO2'],
                         P50=-df1.loc[0, 'P50'],
                         P88=-df1.loc[0, 'P88'])

# summary of the univariate experiments
trainings = ['wet', 'inter']
soils = ['wet', 'inter', 'dry']
atms = ['insample', 'highD', 'highCa']
combis = list(itertools.product(*[trainings, soils, atms]))  # possibilities
univar_xpes = ['%s_%s_%s' % (e[2], e[1], e[0]) for e in combis if
               ((e[0] == e[1]) or (e[2] == 'insample'))]

df = pd.read_csv(os.path.join(ofdir, 'all_cumulative_impacts.csv'))
df = df[df['xpe'].isin(univar_xpes)]
df.set_index('xpe', inplace=True)

figname = os.path.join(figdir, 'cummulative_impacts.png')

if not os.path.isfile(figname):
    plot_impact_summary(df, figname)
