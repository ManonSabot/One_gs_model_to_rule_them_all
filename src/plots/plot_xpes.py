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
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil modules

#==============================================================================


class plt_setup(object):

    def __init__(self, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.05  # padding to use
        plt.rcParams['savefig.jpeg_quality'] = 100

        # colors
        if colours is None:  # use the default colours
            colours = ['#1a1a1a', '#1a1a1a', '#984ea3', '#decbe4', '#0571b0',
                       '#92c5de', '#1a9641', '#a6d96a', '#ca0020', '#f4a582',
                       '#a6611a', '#dfc27d']

            colours = ['#984ea3', '#decbe4', '#0571b0',
                       '#92c5de', '#1a9641', '#a6d96a', '#ca0020', '#f4a582',
                       '#a6611a', '#dfc27d']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                               r'\usepackage{mathpazo}',
                                               r'\usepackage{amsmath}']
        plt.rcParams['font.size'] = 7.
        plt.rcParams['xtick.labelsize'] = 7.
        plt.rcParams['ytick.labelsize'] = 7.
        plt.rcParams['axes.labelsize'] = 7.

        # lines
        plt.rcParams['lines.linewidth'] = 1.5

        # legend
        plt.rcParams['legend.fontsize'] = 10.
        plt.rcParams['legend.framealpha'] = 1


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
        rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (sw[0] + df['fc'][0]) / 2.25

    if profile == 'inter':
        rate = -8. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = (df['fc'][0] + df['pwp'][0]) / 2.

    if profile == 'dry':
        rate = -16. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = df['pwp'][0] / 2.

    for i in range(len(df)):

        sw[i] = sw[i-1]

        if df['PPFD'].iloc[i] > 50.:
            sw[i] = np.maximum(sw[0] / (1. - rate * track), sw_min)
            track += 1

    # now get the soil water potentials matching the soil moisture profile
    Ps = np.asarray([water_potential(df.iloc[0], sw[i])
                     for i in range(len(sw))])

    return sw, Ps


def weekly(df, how='avg'):

    df0 = df.copy()  # avoids directly modifying the df

    # Which rubisco limitation?
    for col in df0.columns[df0.columns.str.contains('Rublim')]:

        df0[col].where(df0[col] != 'True', 2., inplace=True)
        df0[col].where(df0[col] != 'False', -2., inplace=True)
        df0[col].where(df0[col] != '0.0', 0., inplace=True)
        df0[col] = df0[col].astype('float64')

    # add the weeks
    df0['week'] = 1
    iter = 7.

    for __ in range(3):  # deal with the three other weeks

        df0.loc[df0['doy'] >= df0['doy'].iloc[0] + iter, 'week'] += 1
        iter += 7.

    if how == 'avg':
        df1 = df0.groupby([df0.week, df0.hod]).mean()

    if how == 'sum':
        df1 = df0.groupby(df0.week).sum()

    df2 = df0.groupby([df0.week, df0.hod]).max()
    df3 = df0.groupby([df0.week, df0.hod]).min()

    # convert the hod in the index into a column once again
    df1 = df1.reset_index(level=['hod'])
    df2 = df2.reset_index(level=['hod'])
    df3 = df3.reset_index(level=['hod'])

    # remove the excess zeros for aesthetics when plotting
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
        lab = 'CGainNet'

    elif short == 'pmax':
        lab = 'ProfitMax'

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


def render_ylabels(ax, name, unit, fs1=13., fs2=11.):

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs1, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (fs2, unit))

    return


def plot_forcings(df, fname, title=""):

    rows = 2
    cols = 2
    dunits = 0.7
    w, h = figsize(rows, cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(w + dunits, h))
    axes = [e for sub in axes for e in sub]
    axes += [axes[0].twinx()]  # double the axis to add units
    axes[3].axis('off')  # mask frame

    wavg, wmax, wmin = weekly(df)

    axes[0].plot(wavg['PPFD'] * conv.PAR_2_SW, linewidth=0.)
    axes[4].plot(wavg['PPFD'], color='k')
    axes[1].plot(wavg['Tair'], color='k')
    axes[2].plot(wavg['VPD'], color='k')

    axes[4].fill_between(wmax['PPFD'].index, wmin['PPFD'], wmax['PPFD'],
                         facecolor='lightgrey')
    axes[1].fill_between(wmax['Tair'].index, wmin['Tair'], wmax['Tair'],
                         facecolor='lightgrey')
    axes[2].fill_between(wmax['VPD'].index, wmin['VPD'], wmax['VPD'],
                         facecolor='lightgrey')

    # add info text
    info = ('Atmospheric pressure: %.3f kPa\n' % (wavg['Patm'][0]) +
            'Wind speed: %d m s$^{-1}$\n' % (wavg['u'][0]) +
            '[O$_2$]: %dx10$^3$ ppm\n' %
            (int(round(wavg['O2'][0] * conv.MILI * conv.FROM_kPa))) +
            '[CO$_2$]: %d ppm' %
            (int(round(wavg['CO2'][0] * conv.MILI * conv.FROM_kPa))))
    axes[3].text(0.15, 0.5, info, va='center', ha='left',
                 multialignment='left', linespacing=2.5)

    for ax in axes[:3]:

        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_xlim([0, len(wavg)])
        ax.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])

        if ax == axes[0]:
            ax.set_xticklabels(['',] * 4)

        else:
            ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

    # move additional axis to free space on the left
    axes[4].spines['left'].set_position(('axes', -0.2))
    axes[4].yaxis.set_label_position('left')
    axes[4].yaxis.set_ticks_position('left')
    axes[4].tick_params(direction='in')

    render_ylabels(axes[0], 'PAR', 'W m$^{-2}$')
    render_ylabels(axes[4], 'PAR', r'$\si{\micro}$mol m$^{-2}$ s$^{-1}$')
    render_ylabels(axes[1], 'Air temperature', '$^\circ$C')
    render_ylabels(axes[2], 'Vapour pressure deficit', 'kPa')
    #axes[0].set_ylabel(r'PAR (W m$^{-2}$)')
    #axes[4].set_ylabel(r'PAR (umol m$^{-2}$ s$^{-1}$')
    #axes[1].set_ylabel(r'Air temperature ($^\circ$C)')
    #axes[2].set_ylabel(r'Vapour pressure deficit (kPa)')

    plt.suptitle(title, y=1.)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=0.95)
    fig.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_soil_forcings(df, fname, title=""):

    rows = 2
    cols = 1
    w, h = figsize(rows, cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(w, h), sharex=True)

    for ax in axes:

        # plot reference soil moisture levels
        ax.plot(df['theta_sat'], ':', linewidth=1.)
        ax.text(0.99 * float(len(df)), df['theta_sat'][0] - 0.02115,
                'saturation', ha='right', fontsize=10.)
        ax.plot(df['fc'], ':k', linewidth=1.)
        ax.text(0.99 * float(len(df)), df['fc'][0] + 1.15e-2, 'field capacity',
                ha='right', fontsize=10.)
        ax.plot(df['pwp'], ':k', linewidth=1.)
        ax.text(0.99 * float(len(df)), df['pwp'][0] + 1.15e-2, 'wilting point',
                ha='right', fontsize=10.)

    # retrieve the soil moisture profiles
    sw_wet, __ = soil_water(df, 'wet')
    sw_interm, __ = soil_water(df, 'inter')

    # smooth the decay of sm when plotting it
    N  = 3  # filter order
    Wn = 0.0245  # cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    axes[0].plot(signal.filtfilt(B, A, sw_wet), color='k')
    axes[1].plot(signal.filtfilt(B, A, sw_interm), color='k')

    axes[1].set_xlim([0, len(df)])
    axes[1].set_xticks([48. * 3.5, 48. * 10.5, 48. * 17.5, 48. * 24.5])
    axes[1].set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

    for ax in axes:

        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        render_ylabels(ax, 'Soil water content', 'm$^{3}$ m$^{-3}$')

    plt.suptitle(title, y=1.)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=0.95)
    fig.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_targets(df1, df2, fname, title=""):

    rows = 2
    cols = 1
    dunits = 0.01
    w, h = figsize(rows, cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(w + dunits, h),
                             sharex=True)
    axes = list(axes) + [axes[0].twinx(), axes[1].twinx()]  # add units

    wavg, wmax, wmin = weekly(df1)
    axes[0].plot(wavg['gs(std1)'] * cst.MH2O * conv.MILI / cst.rho,
                 linewidth=0.)
    axes[2].plot(wavg['gs(std1)'], color='k')
    axes[2].fill_between(wmax['gs(std1)'].index, wmin['gs(std1)'],
                         wmax['gs(std1)'], facecolor='lightgrey')

    wavg, wmax, wmin = weekly(df2)
    axes[1].plot(wavg['gs(std1)'] * cst.MH2O * conv.MILI / cst.rho,
                 linewidth=0.)
    axes[3].plot(wavg['gs(std1)'], color='k')
    axes[3].fill_between(wmax['gs(std1)'].index, wmin['gs(std1)'],
                      wmax['gs(std1)'], facecolor='lightgrey')

    axes[1].set_xlim([0, len(wavg)])
    axes[1].set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    axes[1].set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

    for ax in axes:

        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        render_ylabels(ax, 'g$_\mathrm{s}$', 'm s$^{-1}$')

    # move additional axes to free space on the left
    for ax in axes[2:]:

        ax.spines['left'].set_position(('axes', -0.2))
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(direction='in')
        render_ylabels(ax, 'g$_\mathrm{s}$', 'mol m$^{-2}$ s$^{-1}$')

    plt.suptitle(title, y=1.)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=0.95)
    fig.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_all_perturbations(df, fname, title=""):

    rows = 1
    cols = 2
    xtext = 0.2
    w, h = figsize(rows, cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(w + xtext, h))

    # plot reference soil moisture levels
    axes[1].plot(df['theta_sat'], ':', linewidth=1.)
    axes[1].text(0.99 * float(len(df)), df['theta_sat'][0] - 0.02115,
                 'saturation', ha='right', fontsize=10.)
    axes[1].plot(df['fc'], ':k', linewidth=1.)
    axes[1].text(0.99 * float(len(df)), df['fc'][0] + 1.15e-2, 'field capacity',
                 ha='right', fontsize=10.)
    axes[1].plot(df['pwp'], ':k', linewidth=1.)
    axes[1].text(0.99 * float(len(df)), df['pwp'][0] + 1.15e-2, 'wilting point',
                 ha='right', fontsize=10.)

    # retrieve the soil moisture profiles
    sw_wet, __ = soil_water(df, 'wet')
    sw_interm, __ = soil_water(df, 'inter')
    sw_dry, __ = soil_water(df, 'dry')

    # smooth the decay of sm when plotting it
    N  = 3  # filter order
    Wn = 0.0245  # cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    axes[1].plot(signal.filtfilt(B, A, sw_wet), color='#fff7bc')
    axes[1].plot(signal.filtfilt(B, A, sw_interm), color='#fec44f')
    axes[1].plot(signal.filtfilt(B, A, sw_dry), color='#d95f0e')

    # double VPD
    df['VPD2'] = df['VPD'] * 2.
    wavg, wmax, wmin = weekly(df)

    axes[0].plot(wavg['VPD2'], color='k')
    axes[0].fill_between(wmax['VPD2'].index, wmin['VPD2'], wmax['VPD2'],
                         facecolor='lightgrey')

    # add info text
    info = '[CO$_2$]: %d ppm' % (int(round(wavg['CO2'][0] * 2. * conv.MILI *
                                           conv.FROM_kPa)))
    axes[1].text(1. + xtext, 0.5, info, va='center', ha='center',
                 transform=axes[1].transAxes)

    axes[0].set_xlim([0, len(wavg)])
    axes[0].set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    axes[1].set_xlim([0, len(df)])
    axes[1].set_xticks([48. * 3.5, 48. * 10.5, 48. * 17.5, 48. * 24.5])

    for ax in axes:
        ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    render_ylabels(axes[0], 'Vapour pressure deficit', 'kPa')
    render_ylabels(axes[1], 'Soil water content', 'm$^{3}$ m$^{-3}$')

    plt.suptitle(title, y=1.)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    fig.savefig(fname, bbox_inches='tight')
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

    # c is derived from both expressions of b
    try:
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))

    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))

    b = Px1 / ((- np.log(1 - x1)) ** (1. / c))
    P12 = -b * ((- np.log(0.88)) ** (1. / c)) # MPa

    return P12


def plot_diag_target(df, fname, title=None):

    fig = plt.figure(figsize=(6, 6))
    GS = fig.add_gridspec(15, 4, wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(GS[:6, :])
    ax2 = fig.add_subplot(GS[7:11, :2])
    ax3 = fig.add_subplot(GS[7:11, 2:])
    ax4 = fig.add_subplot(GS[11:, :2], sharex=ax2)
    ax5 = fig.add_subplot(GS[11:, 2:], sharex=ax3)

    # each week's diurnal gs
    wavg, __, __ = weekly(df.copy())

    # avg diurnal of Pleaf, Ci, etc wHen photosynthesis happens
    davg = df.groupby(['hod']).mean()
    davg = davg[davg[davg.filter(like='gs(').columns].sum(axis=1) > 0.]

    for mod in ['std1', 'std2', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn',
                'lcst', 'sox2', 'cap', 'mes']:

        if mod == 'std1':
            lw = 4.
            ax1.plot(wavg['gs(%s)' % (mod)], linewidth=lw,
                     label=which_model(mod))
            next(ax2._get_lines.prop_cycler)  # skip over Pleaf
            ax3.plot(davg['Ci(%s)' % (mod)], linewidth=lw)
            ax4.plot(davg['E(%s)' % (mod)], linewidth=lw)
            ax5.plot(davg['A(%s)' % (mod)], linewidth=lw)

        #elif mod == 'std2':

        elif mod != 'std2':
            ax1.plot(wavg['gs(%s)' % (mod)],
                     label=which_model(mod))
            ax2.plot(davg['Pleaf(%s)' % (mod)])
            ax3.plot(davg['Ci(%s)' % (mod)])
            ax4.plot(davg['E(%s)' % (mod)])
            ax5.plot(davg['A(%s)' % (mod)])

        else:

            for ax in [ax1, ax2, ax3, ax4, ax5]:

                next(ax._get_lines.prop_cycler)

    # add VC info
    P50 = -3.13
    P88 = -4.9
    P12 = get_P12(P50, P88, 50., 88.)
    ax2.axhline(P12, linestyle=':', linewidth=1.)
    ax2.text(davg.index[0] - 0.25, P12 - 0.75, 'P12')
    ax2.axhline(P50, linestyle=':', linewidth=1.)
    ax2.text(davg.index[0] - 0.25, P50 - 0.75, 'P50')
    ax2.axhline(P88, linestyle=':', linewidth=1.)
    ax2.text(davg.index[0] - 0.25, P88 - 0.75, 'P88')

    ax1.set_ylabel('g$_{s}$ (mol m$^{-2}$ s$^{-1}$)', fontsize=10.)
    ax1.set_xlim(-2., len(wavg) + 1.)
    ax1.set_xticks([len(wavg) / 8., len(wavg) * 3. / 8., len(wavg) * 5. / 8.,
                    len(wavg) * 7. / 8.])
    ax1.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])

    ax2.set_ylabel('$\Psi_{l}$ (MPa)')
    ax3.set_ylabel('$C_{i}$ (Pa)')
    ax4.set_ylabel('E (mmol m$^{-2}$ s$^{-1}$)')
    ax5.set_ylabel('A ($\mu$mol m$^{-2}$ s$^{-1}$)')
    ax4.set_xlabel('hod (h)')
    ax5.set_xlabel('hod (h)')

    for ax in [ax1, ax2, ax3, ax4, ax5]:

        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

        if (ax == ax4) or (ax == ax5):
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

    ax1.legend(bbox_to_anchor=(1.05, 1.05), loc=2, frameon=False)

    fig.savefig(fname, dpi=1200, bbox_inches='tight')
    plt.close()
    exit(1)


def plot_diagnostics(df, fname, title=None):

    wavg, wmax, wmin = weekly(df)

    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(nrows=3, ncols=2)

    # set the control aside
    mod = 'std1'
    ax1.plot(wavg['E(%s)' % (mod)], '-k', linewidth=1.5, label='Medlyn')
    ax1.plot(wmax['E(%s)' % (mod)], ':k', linewidth=1.)
    ax1.plot(wmin['E(%s)' % (mod)], ':k', linewidth=1.)

    ax2.plot(wavg['Pleaf(%s)' % (mod)][wavg['Pleaf(%s)' % (mod)] > -2.5], '-k',
             linewidth=1.5, label='Medlyn-SM')

    ax3.plot(wavg['gs(%s)' % (mod)], '-k', linewidth=1.5, label='Medlyn')
    ax3.plot(wmax['gs(%s)' % (mod)], ':k', linewidth=1.)
    ax3.plot(wmin['gs(%s)' % (mod)], ':k', linewidth=1.)

    ax4.plot(wavg['Ci(%s)' % (mod)], '-k', linewidth=1.5, label='Medlyn')
    ax4.plot(wmax['Ci(%s)' % (mod)], ':k', linewidth=1.)
    ax4.plot(wmin['Ci(%s)' % (mod)], ':k', linewidth=1.)

    ax5.plot(wavg['A(%s)' % (mod)], '-k', linewidth=1.5, label='Medlyn')
    ax5.plot(wmax['A(%s)' % (mod)], ':k', linewidth=1.)
    ax5.plot(wmin['A(%s)' % (mod)], ':k', linewidth=1.)

    ax6.plot(wavg['Rublim(%s)' % (mod)], '-k', linewidth=1.5, label='Medlyn')
    ax6.plot(wmax['Rublim(%s)' % (mod)], ':k', linewidth=1., label='Medlyn')
    ax6.plot(wmin['Rublim(%s)' % (mod)], ':k', linewidth=1., label='Medlyn')

    col = ['#969696', '#cccccc', '#bae4bc', '#7bccc4', '#43a2ca', '#0868ac',
           '#ffffb2', '#fecc5c', '#fd8d3c', '#c994c7', '#dd1c77']

    for i, mod in enumerate(['std2', 'tuz', 'sox1', 'wue', 'cgn', 'pmax',
                             'cmax', 'lcst', 'sox2', 'cap', 'mes']):

        lab = which_model(mod)
        ax1.plot(wavg['E(%s)' % (mod)], linewidth=1., color=col[i], label=lab)

        if mod == 'std2':
            ax2.plot(wavg['Pleaf(%s)' % (mod)][wavg['Pleaf(%s)' % (mod)] >
                          -2.5], linewidth=1., color=col[i], label=lab)

        else:
            ax2.plot(wavg['Pleaf(%s)' % (mod)], linewidth=1., color=col[i],
                     label=lab)

        ax3.plot(wavg['gs(%s)' % (mod)], linewidth=1., color=col[i], label=lab)
        ax4.plot(wavg['Ci(%s)' % (mod)], linewidth=1., color=col[i], label=lab)
        ax5.plot(wavg['A(%s)' % (mod)], linewidth=1., color=col[i], label=lab)
        ax6.plot(wavg['Rublim(%s)' % (mod)], linewidth=1., color=col[i],
                 label=lab)

    ax1.set_ylabel('E (mmol m-2 s-1)')
    ax2.set_ylabel('Pleaf (MPa)')
    ax3.set_ylabel('gs (mol m-2 s-1)')
    ax4.set_ylabel('Ci (Pa)')
    ax5.set_ylabel('A (umol m-2 s-1)')
    ax6.set_ylabel('Rubisco limited?')

    ax1.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax1.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax2.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax2.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax2.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax3.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax3.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax3.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax4.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax4.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax5.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax5.set_xticklabels(['W1', 'W2', 'W3', 'W4'])
    ax6.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax6.set_xticklabels(['W1', 'W2', 'W3', 'W4'])

    ax6.legend(bbox_to_anchor=(1.05, 3.5))

    plt.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.savefig(fname, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_perturb_target(df, fname, title=None):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))

    wavg, wmax, wmin = weekly(df)

    # set the control aside
    ax.plot(wavg['gs(std1)'], linewidth=3.5, label='Reference')

    for i, mod in enumerate(['std2', 'tuz', 'sox1', 'wue', 'cgn', 'pmax',
                             'cmax', 'lcst', 'sox2', 'cap', 'mes']):

        if (mod != 'std2'):
            #if mod == 'chi':
            #    wavg['gs(%s)' % (mod)] = np.convolve(wavg['gs(%s)' % (mod)],
            #                                         np.ones((4,)) / 4,
            #                                         mode='same')

            ax.plot(wavg['gs(%s)' % (mod)], linewidth=2.,
                    label=which_model(mod))

        else:
            next(ax._get_lines.prop_cycler)

    ax.set_ylabel('g$_{s}$ (mol m$^{-2}$ s$^{-1}$)', fontsize=14.)

    ax.set_xlim([0, len(wavg)])
    ax.set_xticks([48. * 0.5, 48. * 1.5, 48. * 2.5, 48. * 3.5])
    ax.set_xticklabels(['week 1', 'week 2', 'week 3', 'week 4'])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    ax.legend(bbox_to_anchor=(1.05, 0.), loc=3, frameon=False)

    plt.suptitle(title, fontsize=16.)
    fig.savefig(fname, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_behaviours(df, figname, title='', X='E', Y='A', Nbins=3):

    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(wspace=0.015)
    axes = fig.subplots(nrows=Nbins, ncols=Nbins, sharex=True, sharey=True)

    df0 = df.copy()  # avoids directly modifying the df

    # normalise the variables of interest
    Xmax = df0[df0.filter(like='%s(' % (X)).columns].max().max()
    Ymax = df0[df0.filter(like='%s(' % (Y)).columns].max().max()

    if X == 'Pleaf':
        Xmin = df0[df0.filter(like='%s(' % (X)).columns].min().min()
        df0[df0.filter(like='%s(' % (X)).columns] /= Xmin

    else:
        df0[df0.filter(like='%s(' % (X)).columns] /= Xmax

    if Y == 'Pleaf':
        Ymin = df0[df0.filter(like='%s(' % (Y)).columns].min().min()
        df0[df0.filter(like='%s(' % (Y)).columns] /= Ymin

    else:
        df0[df0.filter(like='%s(' % (Y)).columns] /= Ymax

    # bin the data by prescribed groups
    #Ps_bins = pd.IntervalIndex.from_tuples([(df0['Ps'].min(), -0.5),
    #                                        (-0.5, -0.05),
    #                                        (-0.05, df0['Ps'].max())])
    #df0['Ps_bins'] = pd.cut(df0['Ps'], Ps_bins)
    #VPD_bins = pd.IntervalIndex.from_tuples([(df0['VPD'].min(), 1.5),
    #                                         (1.5, 3.),
    #                                         (3., df0['VPD'].max())])
    #df0['VPD_bins'] = pd.cut(df0['VPD'], VPD_bins)

    # bin the data by equally-sized groups
    df0['Ps_bins'] = pd.qcut(df0['Ps'], Nbins)
    df0['VPD_bins'] = pd.qcut(df0['VPD'], Nbins)

    # sort the order of the values
    df0.sort_values(by=['Ps_bins', 'VPD_bins'], ascending=[False, True],
                    inplace=True)
    order = df0[['Ps_bins', 'VPD_bins']].drop_duplicates()
    order.reset_index(drop=True, inplace=True)
    order['order'] = order.index
    df0 = df0.merge(order, on=['Ps_bins', 'VPD_bins'])

    # split by unique group in order
    grouped = df0.groupby(['order'])
    dfs = [grouped.get_group(x) for x in grouped.groups]

    iter = 0  # loop over dfs

    for ax in axes.flat:

        df = dfs[iter]

        # loop over the different models
        for mod in ['std1', 'std2', 'tuz', 'sox1', 'wue', 'cgn', 'pmax', 'cmax',
                    'lcst', 'sox2', 'cap', 'mes']:

            xx = df['%s(%s)' % (X, mod)].values
            yy = df['%s(%s)' % (Y, mod)].values
            mask = np.logical_or(xx == 0., yy == 0.)

            # gam fit
            gam = (LinearGAM(n_splines=7, spline_order=5)
                            .gridsearch(xx[~mask].reshape(-1, 1), yy[~mask]))
            px = np.linspace(np.percentile(xx[~mask], 5),
                             np.percentile(xx[~mask], 95), num=500)
            py = gam.predict(px)

            ax.plot(px, py, linewidth=4., alpha=0.7)
            ax.scatter(xx[~mask], yy[~mask], c=ax.lines[-1].get_color(),
                       alpha=0.05, zorder=-1)

        Ps = np.unique(df['Ps_bins'])[0]
        VPD = np.unique(df['VPD_bins'])[0]

        if iter < 3:
            ax.set_title(r'D: %.1f to %.1f (kPa)' % (VPD.left, VPD.right))

        if iter % 3 == 0:
            ax.text(-1.25, 0.5, r'$\Psi_{leaf}$: %.4f to %.4f (MPa)'
                               % (Ps.right, Ps.left))
            ax.set_ylabel(r'%s$_{\rm norm}$ (-)' % (Y))

        if iter > 5:
            ax.set_xlabel(r'%s$_{\rm norm}$ (-)' % (X))

        iter += 1

    fig.tight_layout()
    fig.savefig(figname, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_impact_summary(fname, df):

    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(hspace=0.05, wspace=0.015)
    axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey='row')

    labels = ['Reference', 'Tuzet', 'SOX', r'WUE-$f_{\varPsi_l}$', 'CMax',
              'ProfitMax', 'CGainNet', 'LeastCost', 'SOX-Opt', 'CAP', 'MES']
    switch = ['tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
              'sox2', 'cap', 'mes']

    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()
    twet = [e for e in df.index.to_list() if e.split('_')[-1] == 'wet']
    tinter = [e for e in df.index.to_list() if e.split('_')[-1] == 'inter']

    iter = 0

    for ax in axes.flat:

        if iter < 2:
            sub = df[GPP]

        else:
            sub = df[E]
            ax.set_xticks(np.arange(0.4, 4.5))
            ax.set_xticklabels(['Wet', 'Inter.', 'Dry', '2 x D$_a$',
                                '2 x C$_a$'])

        if iter % 2 == 0:
            sub = sub.loc[twet]

            if iter < 2:
                ax.set_ylabel('GPP (gC y$^{-1}$)')
                ax.set_title('Wet calibration')

            else:
                ax.set_ylabel('E (mm y$^{-1}$)')

            # color the calibration
            ax.axvspan(0., 0.8, hatch='.' * 6, facecolor='none',
                       edgecolor='#2e7d9b', alpha=0.1)

        else:
            sub = sub.loc[tinter]

            if iter < 2:
                ax.set_title('Intermediate calibration')

            # color the calibration
            ax.axvspan(1., 1.8, hatch='.' * 6, facecolor='none',
                       edgecolor='#fc8635', alpha=0.1)

        # we're plotting in reverse order
        sub = sub.iloc[::-1]
        sub.reset_index(inplace=True)
        pos = np.arange(float(len(sub)) + 1)

        # ref model
        ax.hlines(sub[sub.filter(like='std1').columns], pos[:-1], pos[1:] - 0.2,
                  colors='#1a1a1a', linewidth=2., label=labels[0])

        for i, p in enumerate(pos[:-1]):

            for j, model in enumerate(switch):

                p += 0.075
                val1 = sub[sub.filter(like='std1').columns].values[i][0]
                val2 = sub[sub.filter(like=model).columns].values[i][0]

                if i == 0:
                    ax.plot([p, p], [np.minimum(val1, val2),
                            np.maximum(val1, val2)], linewidth=1.5,
                            label=labels[j + 1])

                else:
                    ax.plot([p, p], [np.minimum(val1, val2),
                            np.maximum(val1, val2)], linewidth=1.5)

        ax.set_xlim(-0.1, 4.9)
        iter += 1

    ax.legend(bbox_to_anchor=(1.05, 0.), loc=3, fontsize=6., frameon=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_impact_summary2(fname, df):

    fig = plt.figure(figsize=(6, 4))
    plt.subplots_adjust(hspace=0.05, wspace=0.015)
    axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey='row')

    labels = ['Tuzet', 'SOX', r'WUE-$f_{\varPsi_l}$', 'CMax',
              'ProfitMax', 'CGainNet', 'LeastCost', 'SOX-Opt', 'CAP', 'MES']
    switch = ['tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
              'sox2', 'cap', 'mes']

    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()
    twet = [e for e in df.index.to_list() if e.split('_')[-1] == 'wet']
    tinter = [e for e in df.index.to_list() if e.split('_')[-1] == 'inter']

    iter = 0

    for ax in axes.flat:

        if iter < 2:
            sub = df[GPP]

        else:
            sub = df[E]
            ax.set_xticks(np.arange(0.4, 4.5))
            ax.set_xticklabels(['Wet', 'Inter.', 'Dry', '2 x D$_a$',
                                '2 x C$_a$'])

        if iter % 2 == 0:
            sub = sub.loc[twet]

            if iter < 2:
                ax.set_ylabel('GPP (gC y$^{-1}$)')
                ax.set_title('Wet calibration')

            else:
                ax.set_ylabel('E (mm y$^{-1}$)')

            # color the calibration
            ax.axvspan(0., 0.8, hatch='.' * 6, facecolor='none',
                       edgecolor='#2e7d9b', alpha=0.1)

        else:
            sub = sub.loc[tinter]

            if iter < 2:
                ax.set_title('Intermediate calibration')

            # color the calibration
            ax.axvspan(1., 1.8, hatch='.' * 6, facecolor='none',
                       edgecolor='#fc8635', alpha=0.1)

        # remove ref. from other models
        sub = sub - sub[sub.filter(like='std1').columns].values

        # we're plotting in reverse order
        sub = sub.iloc[::-1]
        sub.reset_index(inplace=True)
        pos = np.arange(float(len(sub)))

        for i, p in enumerate(pos):

            for j, model in enumerate(switch):

                p += 0.075
                val = sub[sub.filter(like=model).columns].iloc[i]

                if i == 0:
                    ax.plot([p, p], [np.minimum(0., val), np.maximum(0., val)],
                            linewidth=1.5, label=labels[j])

                else:
                    ax.plot([p, p], [np.minimum(0., val), np.maximum(0., val)],
                            linewidth=1.5)

        ax.axhline(0., color='#1a1a1a')
        ax.set_xlim(-0.1, 4.9)
        iter += 1

    ax.legend(bbox_to_anchor=(1.05, 0.), loc=3, fontsize=6., frameon=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_impact_summary3(fname, df):

    fig = plt.figure(figsize=(5, 4))
    plt.subplots_adjust(hspace=0.05)
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    labels = ['Reference', 'Tuzet', 'SOX', r'WUE-$f_{\varPsi_l}$', 'CMax',
              'ProfitMax', 'CGainNet', 'LeastCost', 'SOX-Opt', 'CAP', 'MES']
    switch = ['tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
              'sox2', 'cap', 'mes']

    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()

    iter = 0

    for iter, ax in enumerate(axes.flat):

        if iter < 1:
            sub = df[GPP]
            ax.set_ylabel('GPP (gC y$^{-1}$)')

        else:
            sub = df[E]
            ax.set_xticks(np.arange(0.9, 10.5, 2.))
            ax.set_xticklabels(['Wet', 'Intermediate', 'Dry', '2 x D$_a$',
                                '2 x C$_a$'])
            ax.set_ylabel('E (mm y$^{-1}$)')

        # color by calibration type
        for p in np.arange(0., 10., 2.):  # wet

            if p == 0.:
                ax.axvspan(p, p + 0.8, hatch='.' * 6, facecolor='none',
                            edgecolor='#2e7d9b', alpha=0.1)

            else:
                ax.axvspan(p, p + 0.8, facecolor='none', edgecolor='#2e7d9b',
                           alpha=0.1)

        for p in np.arange(1., 11., 2.):  # inter

            if p == 3.:
                ax.axvspan(p, p + 0.8, hatch='.' * 6, facecolor='none',
                           edgecolor='#fc8635', alpha=0.1)

            ax.axvspan(p, p + 0.8, facecolor='none', edgecolor='#fc8635',
                       alpha=0.1)

        # we're plotting in reverse order
        sub = sub.iloc[::-1]
        sub.reset_index(inplace=True)
        pos = np.arange(float(len(sub)) + 1)

        # ref model
        ax.hlines(sub[sub.filter(like='std1').columns], pos[:-1], pos[1:] - 0.2,
                  colors='#1a1a1a', linewidth=2., label=labels[0])

        for i, p in enumerate(pos[:-1]):

            for j, model in enumerate(switch):

                p += 0.075
                val1 = sub[sub.filter(like='std1').columns].values[i][0]
                val2 = sub[sub.filter(like=model).columns].values[i][0]

                if i == 0:
                    ax.plot([p, p], [np.minimum(val1, val2),
                            np.maximum(val1, val2)], linewidth=1.5,
                            label=labels[j + 1])

                else:
                    ax.plot([p, p], [np.minimum(val1, val2),
                            np.maximum(val1, val2)], linewidth=1.5)

        ax.set_xlim(-0.1, 9.9)
        iter += 1

    ax.legend(bbox_to_anchor=(1.05, 0.), loc=3, fontsize=6., frameon=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_impact_summary4(fname, df):

    fig = plt.figure(figsize=(5, 4))
    plt.subplots_adjust(hspace=0.05)
    axes = fig.subplots(nrows=2, ncols=1, sharex=True)

    labels = ['Tuzet', 'SOX', r'WUE-$f_{\varPsi_l}$', 'CMax',
              'ProfitMax', 'CGainNet', 'LeastCost', 'SOX-Opt', 'CAP', 'MES']
    switch = ['tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
              'sox2', 'cap', 'mes']

    GPP = df.filter(like='A(').columns.to_list()
    E = df.filter(like='E(').columns.to_list()

    iter = 0

    for iter, ax in enumerate(axes.flat):

        if iter < 1:
            sub = df[GPP]
            ax.set_ylabel('GPP (gC y$^{-1}$)')

        else:
            sub = df[E]
            ax.set_xticks(np.arange(0.9, 10.5, 2.))
            ax.set_xticklabels(['Wet', 'Intermediate', 'Dry', '2 x D$_a$',
                                '2 x C$_a$'])
            ax.set_ylabel('E (mm y$^{-1}$)')

        # color by calibration type
        for p in np.arange(0., 10., 2.):  # wet

            if p == 0.:
                ax.axvspan(p, p + 0.8, hatch='.' * 6, facecolor='none',
                            edgecolor='#2e7d9b', alpha=0.1)

            else:
                ax.axvspan(p, p + 0.8, facecolor='none', edgecolor='#2e7d9b',
                           alpha=0.1)

        for p in np.arange(1., 11., 2.):  # inter

            if p == 3.:
                ax.axvspan(p, p + 0.8, hatch='.' * 6, facecolor='none',
                           edgecolor='#fc8635', alpha=0.1)

            ax.axvspan(p, p + 0.8, facecolor='none', edgecolor='#fc8635',
                       alpha=0.1)

        # remove ref. from other models
        sub = sub - sub[sub.filter(like='std1').columns].values

        # we're plotting in reverse order
        sub = sub.iloc[::-1]
        sub.reset_index(inplace=True)
        pos = np.arange(float(len(sub)) + 1)

        for i, p in enumerate(pos[:-1]):

            for j, model in enumerate(switch):

                p += 0.075
                val = sub[sub.filter(like=model).columns].iloc[i]

                if i == 0:
                    ax.plot([p, p], [np.minimum(0., val), np.maximum(0., val)],
                            linewidth=1.5, label=labels[j])

                else:
                    ax.plot([p, p], [np.minimum(0., val), np.maximum(0., val)],
                            linewidth=1.5)

        ax.axhline(0., color='#1a1a1a')
        ax.set_xlim(-0.1, 9.9)
        iter += 1

    ax.legend(bbox_to_anchor=(1.05, 0.), loc=3, fontsize=6., frameon=False)

    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


###############################################################################

# first, activate user defined rendering options
plt_setup()

trainings = ['wet', 'inter']
soils = ['wet', 'inter', 'dry']
atms = ['insample', 'highD', 'highCa']
combis = list(itertools.product(*[trainings, soils, atms]))  # possibilities

base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'input'),
                     'simulations'), 'idealised')
ofdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'idealised')

# univariate experiments
df, __ = read_csv(os.path.join(ofdir, 'all_cumulative_impacts.csv'))

xpe_names = ['%s_%s_%s' % (e[2], e[1], e[0]) for e in combis if
             ((e[0] == e[1]) or (e[-1] == 'insample'))]
df = df[df['xpe'].isin(xpe_names)]
df.set_index('xpe', inplace=True)

#plot_impact_summary('summary_impacts.png', df)
#plot_impact_summary2('summary_impacts_2.png', df)
plot_impact_summary3('summary_impacts_3.png', df)
plot_impact_summary4('summary_impacts_4.png', df)

exit(1)


# path to input data
fname1 = os.path.join(ifdir, 'wet_calibration.csv')
df1, __ = read_csv(fname1)

# initialise soil moisture forcings
df1['sw'] = df1['theta_sat']
df1.fillna(method='ffill', inplace=True)

# plot the atmospheric forcings
figdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

 # make new dirs if they don't exist
if not os.path.isdir(os.path.join(figdir, 'idealised_setup')):
    os.makedirs(os.path.join(figdir, 'idealised_setup'))

figname = os.path.join(os.path.join(figdir, 'idealised_setup'),
                       'training_atm_forcings.png')

if not os.path.isfile(figname):
    plot_forcings(df1, figname, title='Training atmospheric forcing')

# plot the two possible soil moisture profiles
figname = os.path.join(os.path.join(figdir, 'idealised_setup'),
                       'training_soil_forcings.png')

if not os.path.isfile(figname):
    plot_soil_forcings(df1, figname, title='Synthetic soil moisture profiles')

for combi in combis:  # loop over all the possibilities

    training = combi[0]
    soil = combi[1]
    atm = combi[2]

    if combi == combis[0]:  # reset ifdir to plot the calib. targets
        ifdir = ifdir.replace('simulations', 'calibrations')

    for gb in [False, True]:  # either account for variations in gb or not

        opath = os.path.join(ofdir, 'inf_gb')
        ofig = os.path.join(figdir, 'inf_gb')

        if gb:  # the calling order for gb matters!
            opath = opath.replace('inf_gb', 'var_gb')
            ofig = ofig.replace('inf_gb', 'var_gb')

        if not os.path.isdir(ofig):
            os.makedirs(ofig)

        if not os.path.isdir(os.path.join(ofig, training)):
            os.makedirs(os.path.join(ofig, training))

        # soil moisture profile used to train the models
        df1['sw'], df1['Ps'] = soil_water(df1, training)

        # control gs model training runs
        if training == trainings[0]:
            training2 = trainings[1]

        else:
            training2 = trainings[0]

        fname2 = os.path.join(ifdir, 'training_%s_y.csv' % (training))
        df2, __ = read_csv(fname2)
        fname3 = os.path.join(ifdir, 'training_%s_y.csv' % (training2))
        df3, __ = read_csv(fname3)

        # plot the target gs model for both trainings
        figname = os.path.join(os.path.join(os.path.dirname(ofig),
                               'idealised_setup'), 'training_targets.png')

        if not os.path.isfile(figname):
            figtitle = r'Reference (Medlyn-$\beta$) stomatal conductance'
            if training == trainings[0]:
                plot_targets(df2, df3, figname, title=figtitle)

            else:
                plot_targets(df3, df2, figname, title=figtitle)

        # plot the forcing data changes
        figname = os.path.join(os.path.join(os.path.dirname(ofig),
                               'idealised_setup'), 'perturb_forcing.png')

        if not os.path.isfile(figname):
            plot_all_perturbations(df1, figname, title='Perturbed forcing')

        # soil moisture forcing
        df1['sw'], df1['Ps'] = soil_water(df1, soil)

        # now get the actual model outputs
        fname3 = os.path.join(opath, '%s_%s_%s.csv' % (atm, soil, training))
        df3, __ = read_csv(fname3)

        if training == 'wet':
            if atm == 'highD':  # apply the atm perturbations as necessary
                df1['VPD'] *= 2.

            elif atm == 'highCa':
                df1['CO2'] *= 2.

            # reset the atmosphere to the default
            if atm == 'highD':
                df1['VPD'] /= 2.

            elif atm == 'highCa':
                df1['CO2'] /= 2.

        # plot the calibrated gs
        if (soil == training) and (atm == 'insample') and not gb:
            figname = os.path.join(os.path.join(ofig, training),
                                   'calib_variables.png')

            # now I need to merge those two functions into a single function!!!
            if not os.path.isfile(figname):
                plot_diag_target(df3, figname, title='Calibrated models')

        # diagnostic plots
        figname = os.path.join(os.path.join(ofig, training),
                               'diag_out_running_%s_%s.png' % (atm, soil))

        if not os.path.isfile(figname):
            plot_diagnostics(df3, figname,
                             title='%s atm, %s soil' % (atm, soil))

        # example impact on gs
        if (atm == 'insample') and (soil == 'dry'):
            figname = os.path.join(os.path.join(ofig, training),
                                   'perturb_gs_%s.png' % (soil))

            if not os.path.isfile(figname):
                plot_perturb_target(df3, figname,
                                    title='Severe drydown impacts')

else:
    dfs = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
                                         .dropna(axis=1, how='all').squeeze())

# model behaviours
for Ca in np.unique(dfs['CO2']):

    df = dfs[dfs['CO2'] == Ca]

    for x, y in zip(['E', 'gs', 'Pleaf', 'Ci', 'Pleaf', 'Ci'],
                    ['A', 'E', 'gs', 'gs', 'Ci', 'A']):

        figname = os.path.join(figdir, '%s(%s)_%sppm.png' % (y, x,
                               str(int(Ca * conv.MILI * conv.FROM_kPa))))

        if not os.path.isfile(figname):
            plot_behaviours(df, figname, X=x, Y=y)
