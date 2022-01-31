# -*- coding: utf-8 -*-

"""
Support function used for plotting.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Useful ancillary plotting function"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (09.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"

# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators

# plotting modules
import matplotlib.pyplot as plt
from cycler import cycler


# ======================================================================

class default_plt_setup(object):

    """
    Matplotlib configuration specifics shared by all the figures

    """

    def __init__(self, colours=None, ticks=False):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 600.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # text fonts
        plt.rcParams['text.usetex'] = True  # use LaTeX
        preamble = [r'\usepackage[sfdefault,light]{merriweather}',
                    r'\usepackage{mathpazo}', r'\usepackage{amsmath}']
        plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)

        # font sizes
        plt.rcParams['font.size'] = 6.
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size'] + 1
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size'] + 1

        # 12 default colours
        if colours is None:  # use default
            colours = ['#1a1a1a', '#6023b7', '#af97c5', '#197aff', '#9be2fd',
                       '#009231', '#a6d96a', '#6b3b07', '#ff8e12', '#ffe020',
                       '#f10c80', '#ffc2cd']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)

        # markers
        plt.rcParams['scatter.marker'] = '.'

        # patches (e.g. the shapes in the legend)
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['patch.edgecolor'] = 'k'
        plt.rcParams['patch.force_edgecolor'] = True  # ensure it's used

        # legend specs
        plt.rcParams['legend.facecolor'] = 'none'
        plt.rcParams['legend.edgecolor'] = 'none'

        if ticks:
            plt.rcParams['axes.linewidth'] = 0.65
            plt.rcParams['xtick.major.size'] = 3
            plt.rcParams['xtick.minor.size'] = 1.5
            plt.rcParams['xtick.major.width'] = 0.75
            plt.rcParams['xtick.minor.width'] = 0.75
            plt.rcParams['ytick.major.size'] = plt.rcParams['xtick.major.size']
            plt.rcParams['ytick.minor.size'] = plt.rcParams['xtick.minor.size']
            plt.rcParams['ytick.major.width'] = \
                plt.rcParams['xtick.major.width']
            plt.rcParams['ytick.minor.width'] = \
                plt.rcParams['xtick.minor.width']


def render_xlabels(ax, name, unit, fs=7., pad=0.):

    """
    Renders the plotting of the x-axis label such that the unit of a
    given variable is in a smaller font, which is a nicer display

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    name: string
        name of the variable for the x-axis

    unit: string
        unit of the variable for the x-axis

    fs: float
        font size

    pad: float
        distance between the rendered label and the plot axis frame

    Returns:
    --------
    Draws the axis label on the x-axis

    """

    ax.set_xlabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs,
                                                                unit),
                  labelpad=pad)

    return


def render_ylabels(ax, name, unit, fs=7., pad=0.):

    """
    Renders the plotting of the y-axis label such that the unit of a
    given variable is in a smaller font, which is a nicer display

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to apply the function

    name: string
        name of the variable for the y-axis

    unit: string
        unit of the variable for the y-axis

    fs: float
        font size

    pad: float
        distance between the rendered label and the plot axis frame

    Returns:
    --------
    Draws the axis label on the y-axis

    """

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs,
                                                                unit),
                  labelpad=pad)

    return


def model_order():

    """
    Sets the default model order to use in all plots

    """

    return ['std', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'sox2',
            'pmax2', 'lcst', 'cap', 'mes']


def which_model(short):

    """
    Associates a nice display of the model names to the model short
    handles given in model_order

    Arguments:
    ----------
    short: string
        short version of the model names that is used to differentiate
        the gs schemes in all model output files

    Returns:
    --------
    A nicer model name

    """

    if short == 'std':
        lab = r'Medlyn'

    elif short == 'tuz':
        lab = 'Tuzet'

    elif short == 'sox1':
        lab = 'Eller'

    elif short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    elif short == 'cmax':
        lab = 'CMax'

    elif short == 'pmax':
        lab = 'ProfitMax'

    elif short == 'cgn':
        lab = 'CGain'

    elif short == 'sox2':
        lab = r'SOX$_\mathrm{\mathsf{opt}}$'

    elif short == 'pmax2':
        lab = 'ProfitMax2'

    elif short == 'lcst':
        lab = 'LeastCost'

    elif short == 'cap':
        lab = 'CAP'

    elif short == 'mes':
        lab = 'MES'

    else:
        lab = 'Obs.'

    return lab


def site_spp_order():

    """
    Sets the default sites x species order to use in all plots

    """

    return ['San_Lorenzo_Carapa_guianensis',
            'San_Lorenzo_Tachigali_versicolor',
            'San_Lorenzo_Tocoyena_pittieri',
            'Parque_Natural_Metropolitano_Calycophyllum_candidissimum',
            'ManyPeaksRange_Alphitonia_excelsa',
            'ManyPeaksRange_Austromyrtus_bidwillii',
            'ManyPeaksRange_Brachychiton_australis',
            'ManyPeaksRange_Cochlospermum_gillivraei',
            'Richmond_Eucalyptus_dunnii', 'Richmond_Eucalyptus_saligna',
            'Richmond_Eucalyptus_cladocalyx', 'Puechabon_Quercus_ilex',
            'Vic_la_Gardiole_Quercus_ilex', 'Corrigin_Eucalyptus_capillosa',
            'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis']


def missing_var(df, var, key):

    """
    Finds the categories for which there are no data for a given
    variable

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to explore

    var: string
        variable to consider

    key: string
        name of the categories to explore, e.g., sites x species

    Returns:
    --------
    missing: list
        categories for which there are no data

    """

    missing = []

    for s in site_spp_order():  # go over the sites x species

        if len(df[var][df[key] == s].dropna()) < 1:
            missing += [s]

    return missing


def get_Px(Px1, Px2, x1, x2, x):

    """
    Finds the leaf water potential associated with a x% decrease in
    hydraulic conductance, using the plant vulnerability curve.

    Arguments:
    ----------
    Px1/Px2: float
        leaf water potential [MPa] at which x1/x2% decrease in hydraulic
        conductance is observed

    x1/x2: float
        percentage loss in hydraulic conductance

    x: float
        percentage loss in hydraulic conductance to return

    Returns:
    --------
    Px: float
        leaf water potential [MPa] at which there is x% loss of
        hydraulic conductance

    """

    Px1 = abs(Px1)
    Px2 = abs(Px2)
    x1 /= 100.  # normalise between 0-1
    x2 /= 100.
    x /= 100.

    try:  # c is derived from both expressions of b
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))

    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))

    b = Px1 / ((- np.log(1. - x1)) ** (1. / c))
    Px = -b * ((- np.log(1. - x)) ** (1. / c))  # MPa

    return Px
