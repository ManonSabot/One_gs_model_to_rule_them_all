#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the calibrated parameter values for the 11 gs schemes
in the idealised calibrations, as well as the solvers' performance.

from the idealised experiments, and the associated model outputs.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = ""
__author__ = "Manon E. B. Sabot"
__version__ = "3.0 (16.11.2020)"
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
from matplotlib.patches import Patch  # custom legend
from matplotlib.lines import Line2D  # custom legend

# own modules
from plot_utils import default_plt_setup

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # get the project's directory

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(swater='both', orientation='both'):

    """
    Main function: plots the spread in parameter values calibrated under
                   wet, or inter, or both wet and inter soil moisture
                   profiles for the 11 gs schemes. The solvers'
                   performance is also plotted in a separate figure

    Arguments:
    ----------
    swater: string
        calibration conditions, either wet, or inter, or both

    orientation: string
        either portrait, or landscape, or both

    Returns:
    --------
    'model_calibs_swater_orientation.jpg' and 'solver_performance.jpg'
    in output/plots

    """

    base_dir = get_main_dir()  # working paths
    dirname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                           'calibrations'), 'idealised')

    # read in the data to plot
    fname1 = 'overview_of_fits.csv'  # all the solvers' info
    fname2 = 'top_fits.csv'  # 3 best solvers
    fname3 = 'best_fit.csv'  # best solvers
    df1 = (pd.read_csv(os.path.join(dirname, fname1), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())
    df2 = (pd.read_csv(os.path.join(dirname, fname2), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())
    df3 = (pd.read_csv(os.path.join(dirname, fname3), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    if swater == 'both':
        colours = ['#53e3d4', '#e2694e', ] * 2 + ['#ecec3a']

    else:
        colours = ['#001a33', '#fffafa']

    if orientation == 'both':

        for orientation in ['landscape', 'portrait']:
            plt_setup(swater, orientation, colours=colours)  # specs
            calib_info_plot(df1.copy(), df2.copy(), df3.copy(), swater=swater,
                            orientation=orientation)

    else:
        plt_setup(swater, orientation, colours=colours)  # figure specs
        calib_info_plot(df1.copy(), df2.copy(), df3.copy(), swater=swater,
                        orientation=orientation)

    solver_info_plot(df1)

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self, swater, orientation, colours=None):

        # default setup
        default_plt_setup(colours=colours)

        # saving the figure
        plt.rcParams['savefig.orientation'] = orientation

        # font sizes
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size'] - 0.25
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size'] - 0.25

        if orientation == 'portrait':
            plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size'] - 0.25

        # lines, markers
        plt.rcParams['lines.linewidth'] = 2.
        plt.rcParams['lines.markersize'] = 8.
        plt.rcParams['lines.markeredgewidth'] = 0.5

        # legend
        plt.rcParams['legend.borderpad'] = 1.

        if orientation == 'portrait':
            plt.rcParams['legend.borderpad'] = 0.15

        # grid
        plt.rcParams['grid.color'] = '#bdbdbd'
        plt.rcParams['grid.linewidth'] = 0.25

        # spines
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['axes.spines.top'] = False


def normalise_params_by_group(df, by):

    """
    Normalizes the 'v1' and 'v2' parameter values by the median from the
    groups designed by 'by' in the dataframe

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to normalize

    by: list
        contains grouping criteria info, e.g., column names from which
        to use unique values for grouping

    Returns:
    --------
    df: pandas dataframe
        dataframe containing normalized data

    """

    # normalise
    df['norm_v1'] = df['v1'] / df.groupby(by)['v1'].transform('median')
    df['norm_v2'] = df['v2'] / df.groupby(by)['v2'].transform('median')

    return df


def percent_errors(df):

    """
    Calculates percentage uncertainties from confidence intervals around
    the 'v1' and 'v2' parameter values

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to analyse

    Returns:
    --------
    df: pandas dataframe
        dataframe containing percentage uncertainties

    """

    # confidence intervals to % uncertainties
    df['norm_ci1'] = 100. * np.abs(df['ci1'] / df['v1'])
    df['norm_ci2'] = 100. * np.abs(df['ci2'] / df['v2'])

    if len(df) > len(df['Model'].unique()):  # deal with nans
        df['norm_ci1'] = df['norm_ci1'].fillna(500.)
        df.loc[~df['v2'].isnull(), 'norm_ci2'] = (df.loc[~df['v2'].isnull(),
                                                  'norm_ci2'].fillna(500.))

    return df


def automate_model_order(df):

    """
    Orders the models from the dataframe from those with the smallest
    combined percentage errors on 'v1' and 'v2' to those with the
    largest percentate errors

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to analyse

    Returns:
    --------
    A list of the model names in order

    """

    # get the percentage errors
    df1 = percent_errors(df)

    # combine errors on 'v1' and 'v2'
    df1['div'] = np.nanmean(np.array([df1['norm_ci1'], df1['norm_ci2']]),
                            axis=0)

    # what is the 90th percentile error?
    df2 = df1.groupby('Model')['div'].quantile(0.9).sort_values()
    df2.where(df2 < 50., df1.groupby('Model')['div'].mean().sort_values(),
              inplace=True)  # cap the errors where they're too big
    df2.sort_values(inplace=True)

    # rename the models for consistency
    df2.rename(index={'SOX': 'Eller', 'CGainNet': 'CGain'}, inplace=True)

    return df2.index.to_list()


def update_model_names(df, models):

    """
    Renames the models for consistency and sets the model names to
    categories which then simplifies the data processing and analysis

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to analyse

    models: list
        model names

    Returns:
    --------
    df: pandas dataframe
        dataframe containing the updated categorical model names

    """

    # replace the model names
    df.replace({'Model': {'SOX': 'Eller', 'CGainNet': 'CGain'}}, inplace=True)
    df.Model = df.Model.astype('category')
    df.Model.cat.set_categories(models, inplace=True)

    return df


def sorted_data(df, norm_wet=None, norm_inter=None):

    """
    Organises the models in the order given by 'automate_model_order'
    and within those categories, organises the parameters by performance
    ranking

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to organise

    norm_wet: pandas dataframe
        'wet' subset of df

    norm_inter: pandas dataframe
        'inter' subset of df

    Returns:
    --------
    df: pandas dataframe
        dataframe in organised order

    w: pandas dataframe
        'wet' subset of df

    i: pandas dataframe
        'inter' subset of df

    """

    if norm_wet is None:  # normalise by sub-sample median value
        w = normalise_params_by_group(df[df['training'] == 'wet'].copy(),
                                      ['Model', 'sub-sample'])

    else:
        try:
            w = pd.merge(df.drop('Rank', axis=1), norm_wet,
                         on=['Model', 'training', 'sub-sample', 'solver'],
                         suffixes=('', '_y'))

        except KeyError:
            sort = 'Model'
            w = pd.merge(df, norm_wet[norm_wet['sub-sample'] == 0.],
                         on=['Model', 'training', 'solver'],
                         suffixes=('', '_y'))

        w.drop(w.filter(regex='_y$').columns.tolist(), axis=1,
               inplace=True)

    if norm_inter is None:
        i = normalise_params_by_group(df[df['training'] == 'inter'].copy(),
                                      ['Model', 'sub-sample'])
    else:
        try:
            i = pd.merge(df.drop('Rank', axis=1), norm_inter,
                         on=['Model', 'training', 'sub-sample', 'solver'],
                         suffixes=('', '_y'))

        except KeyError:
            sort = 'Model'
            i = pd.merge(df, norm_inter[norm_inter['sub-sample'] == 0.],
                         on=['Model', 'training', 'solver'],
                         suffixes=('', '_y'))

        i.drop(i.filter(regex='_y$').columns.tolist(), axis=1,
               inplace=True)

    if 'Rank' in df.columns:
        sort = ['Model', 'Rank']  # sorting order

    else:
        sort = ['Model']

    # data in order of model (and rank)
    df.sort_values(sort, inplace=True)
    df.reset_index(drop=True, inplace=True)
    w.sort_values(sort, inplace=True)
    w.reset_index(drop=True, inplace=True)
    i.sort_values(sort, inplace=True)
    i.reset_index(drop=True, inplace=True)

    return df, w, i


def scaled_data(data, sc=0.5):

    """
    Rescales the data to a reverse exponent scale which allows to plot
    of all the model parameter spreads on the same figure

    Arguments:
    ----------
    data: pandas dataframe
        dataframe containing the data to scale

    sc: float
        reference value on which to apply the parameter values as
        exponents

    Returns:
    --------
    datas: array
        scaled data from the dataframe

    """

    # sorted data in model order
    ordered = [e for __, e in data.groupby('Model')]
    datas = [[e['norm_v1'].values, e['norm_v2'].values] for e in ordered]
    datas = [-(sc ** e) for ee in datas for e in ee]

    if len(datas[0]) < 2:
        datas = np.asarray([e for ee in datas for e in ee])

    return datas


def parameter_names(df):

    """
    Updates the model parameter names for consistent display

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the parameters to rename

    Returns:
    --------
    params: list
        appropriate parameter names

    """

    pdf = df.copy().replace({'p1': {'kmax': r'$k_{max}$',
                                    'g1T': r'$g_{1,Tuz}$',
                                    'kmax2': r'$k_{max}$',
                                    'kmaxS1': r'$k_{max}$',
                                    'kmaxS2': r'$k_{max}$',
                                    'kmaxLC': r'$k_{max}$',
                                    'Lambda': r'$\lambda$',
                                    'Alpha': '$a$',
                                    'krlC': r'$k_{max}$',
                                    'krlM': r'$k_{max}$'}})
    params = [[pdf.loc[i, 'p1'], pdf.loc[i, 'p2']] for i in range(len(pdf))]
    params = [str(e) for ee in params for e in ee
              if str(e) not in ['nan', 'kmaxCN', 'kmaxWUE']]

    # deal with special characters and font case
    params[params.index('PrefT')] = r'$\varPsi_{ref}$'
    params[params.index('Beta')] = '$b$'
    params[params.index('Kappa')] = r'$\varpi$'
    params[params.index('Eta')] = r'$\eta$'
    params[params.index('PcritC')] = r'$\varPsi_{\varphi,lim}$'
    params[params.index('PcritM')] = r'$\varPsi_{\varphi,lim}$'

    return params


def local_bw(kde):

    """
    Sets the bandwidth used for generating violin plots depending on the
    spread in the data plotted

    Arguments:
    ----------
    kde: array
        kernel density estimates of the data

    Returns:
    --------
    The bandwidth to use

    """

    spread = np.abs(np.amax(kde.dataset) - np.amin(kde.dataset))

    if spread <= 0.1:
        bw = 0.5

    elif (spread > 0.1) and (spread < 0.25):
        bw = 0.4

    elif (spread > 0.25) and (spread < 0.5):
        bw = 0.3

    else:
        bw = 0.2

    return bw


def slice_vplot(vplot, side, ec=None):

    """
    Slices a violin plot in two, either vertically or horizontally

    Arguments:
    ----------
    vplot: matplotlib objects
        violin plots to alter

    side: string
        left, or right, or bottom, or top

    ec: string
        colour to apply to the edge of the violin plots

    Returns:
    --------
    The altered violin plots

    """

    for vp in vplot['bodies']:

        try:
            m1 = np.mean(vp.get_paths()[0].vertices[:, 0])
            m2 = np.mean(vp.get_paths()[0].vertices[:, 1])

            if side == 'left':
                vp.get_paths()[0].vertices[:, 0] = \
                    np.clip(vp.get_paths()[0].vertices[:, 0], -np.inf, m1)

            if side == 'right':
                vp.get_paths()[0].vertices[:, 0] = \
                    np.clip(vp.get_paths()[0].vertices[:, 0], m1, np.inf)

            if side == 'bottom':
                vp.get_paths()[0].vertices[:, 1] = \
                    np.clip(vp.get_paths()[0].vertices[:, 1], -np.inf, m2)

            if side == 'top':
                vp.get_paths()[0].vertices[:, 1] = \
                    np.clip(vp.get_paths()[0].vertices[:, 1], m2, np.inf)

        except IndexError:
            pass

        if ec is not None:
            vp.set_edgecolor(ec)

        else:
            vp.set_edgecolor(plt.rcParams['patch.edgecolor'])

        vp.set_linewidth(plt.rcParams['patch.linewidth'])

    return


def clip_vplot(vplot, mins, maxs, orientation, ec=None):

    """
    Clips violin plots to specific min / max bounds, thus essentially
    cropping them as desired

    Arguments:
    ----------
    vplot: matplotlib objects
        violin plots to alter

    mins: array
        minimum thresholds beyond which the vplot objects cannot extend

    maxs: array
        maximum thresholds beyond which the vplot objects cannot extend

    orientation: string
        either portrait or landscape

    ec: string
        colour to apply to the edge of the violin plot

    Returns:
    --------
    The altered vplots

    """

    for i, vp in enumerate(vplot['bodies']):

        min = mins[i]
        max = maxs[i]

        try:
            if orientation == 'landscape':
                vp.get_paths()[0].vertices[:, 1] = \
                    np.clip(vp.get_paths()[0].vertices[:, 1], min, max)

            if orientation == 'portrait':
                vp.get_paths()[0].vertices[:, 0] = \
                    np.clip(vp.get_paths()[0].vertices[:, 0], min, max)

        except IndexError:
            pass

        if ec is not None:
            vp.set_edgecolor(ec)

        else:
            vp.set_edgecolor(plt.rcParams['patch.edgecolor'])

        vp.set_linewidth(plt.rcParams['patch.linewidth'])

    return


def custom_grid(mlines, plines, ax, orientation):

    """
    Draws gridlines

    Arguments:
    ----------
    mlines: list
        list of positions at which to draw grid lines around each of the
        models' parameters

    plines: list
        list of positions at which to draw grid lines indicative of the
        magnitude spread in parameter values

    ax: matplotlib object
        axis on which to apply the function

    orientation: string
        either portrait or landscape

    Returns:
    --------
    Gridlines at the right positions on the x- and y-axes

    """

    c = plt.rcParams['grid.color']
    lw = plt.rcParams['grid.linewidth']
    bottom, top = ax.get_ylim()
    left, right = ax.get_xlim()

    if orientation == 'landscape':

        for ml in mlines:

            ax.vlines(ml, bottom, top, color=c, lw=lw, zorder=-10)

        for ml in plines:

            ax.hlines(ml, left, right, color=c, lw=lw, zorder=-10)

    if orientation == 'portrait':

        for ml in mlines:

            ax.hlines(ml, left, right, color=c, lw=lw, zorder=-10)

        for ml in plines:

            ax.vlines(ml, bottom, top, color=c, lw=lw, zorder=-10)

    return


def custom_legend(swater, orientation, ec=None):

    """
    Draws a custom legend that is applied within 'calib_info_plot'

    Arguments:
    ----------
    swater: string
        calibration conditions, either wet, or inter, or both

    orientation: string
        either portrait or landscape

    ec: string
        colour to apply to the edge of the legend patches

    Returns:
    --------
    A custom-made legend

    """

    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if swater == 'both':
        leg = [Line2D([], [], linestyle='', marker='*', ms=7., mfc=c[-1],
                      mec='k', label='Best'),
               Patch(facecolor=c[0], edgecolor=ec, alpha=0.7, label='Wet'),
               Patch(facecolor=c[1], edgecolor=ec, alpha=0.7,
                     label='Stressed')]

    else:
        leg = [Line2D([], [], linestyle='', marker='*', mfc=c[-1], mec='k',
               label='Best')]

    return leg


def calib_info_plot(df1, df2, df3, swater='wet', orientation='landscape'):

    """
    Generates plots of the spread in calibrated parameter values,
    obtained for the 11 calibrated gs schemes, under either wet soil
    moisture conditions or an intermediate dry down, or both, and for a
    number of different data samples

    Arguments:
    ----------
    df1: pandas dataframe
        dataframe that contains information on all the parameter
        calibrations

    df2: pandas dataframe
        dataframe that contains information on the calibrations obtained
        from the 3/4 overall best solvers

    df3: pandas dataframe
        dataframe that contains only the best parameter calibrations for
        each model

    swater: string
        calibration conditions, either wet, or inter, or both

    orientation: string
        either portrait, or landscape, or both

    Returns:
    --------
    'model_calibs_swater_orientation.jpg'

    """

    # user-defined plot attributes
    vscale = 0.45  # scaling factor around the median
    pscale = 1.1  # positions for each model's parameters
    pspace = 0.025  # second parameter positions
    wbox = 0.9  # width of the violin plots
    vert = True
    s1 = 'left'
    s2 = 'right'

    if orientation == 'portrait':
        vert = False
        s1 = 'top'
        s2 = 'bottom'

    if swater == 'both':  # colours for the violin plot edge lines
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][-1]
        pspace /= 2.5

    # landscape characteristics
    fs = (6., 3.)

    if orientation == 'portrait':
        fs = (3.25, 6.)

    # declare the figure and the axes
    fig, ax = plt.subplots(nrows=1, figsize=fs)

    # model order?
    models = automate_model_order(df1.copy())

    # modify and order model names across all dfs
    df1 = update_model_names(df1, models)
    df2 = update_model_names(df2, models)
    df3 = update_model_names(df3, models)

    # organise, normalise, and scale the data consistently
    df1, w, i = sorted_data(df1)
    wet1 = scaled_data(w, sc=vscale)
    inter1 = scaled_data(i, sc=vscale)

    # subset of the top 3 solvers
    df2, w, i = sorted_data(df2, norm_wet=w, norm_inter=i)
    wet2 = scaled_data(w, sc=vscale)
    inter2 = scaled_data(i, sc=vscale)

    # best params
    df3, w, i = sorted_data(df3, norm_wet=w, norm_inter=i)
    best_w = scaled_data(w, sc=vscale)
    best_i = scaled_data(i, sc=vscale)

    # param names in model order?
    params = parameter_names(w)

    # where are there 2nd params?
    all = np.array([i for i in range(len(wet1)) if np.nansum(wet1[i]) != 0.])
    isec = np.array([i for i in range(len(all)) if (all[i] % 2) != 0])
    pos = np.arange(float(len(all))) * pscale
    pos[isec] -= 8. * pspace  # second parameter position
    pos2 = pos + pspace

    if orientation == 'portrait':
        pos2 = pos - pspace

    # now that we've reworked the positions, only keep those
    wet1 = [wet1[i] for i in all]
    wet2 = [wet2[i] for i in all]
    inter1 = [inter1[i] for i in all]
    inter2 = [inter2[i] for i in all]
    best_w = [best_w[i] for i in all]
    best_i = [best_i[i] for i in all]

    # all solver data
    Npoints = 100  # smooth violins

    if swater != 'inter':
        vp1 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos, vert=vert, widths=wbox,
                            bw_method=local_bw)

        for vp in vp1['bodies']:

            vp.set_alpha(0.7)

    if swater != 'wet':
        vp2 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2, vert=vert, widths=wbox,
                            bw_method=local_bw)

        for vp in vp2['bodies']:

            vp.set_alpha(0.7)

    if swater == 'both':
        slice_vplot(vp1, s1, ec='gray')
        slice_vplot(vp2, s2, ec='gray')

    # top 3 solver data, if no substantial improvement, no plot
    plt_wet = np.array([(np.amax(wet2[i]) - np.amin(wet2[i])) < 0.75 *
                        (np.amax(wet1[i]) - np.amin(wet1[i]))
                        for i in range(len(wet1))])
    plt_inter = np.array([(np.amax(inter2[i]) - np.amin(inter2[i])) < 0.75 *
                          (np.amax(inter1[i]) - np.amin(inter1[i]))
                          for i in range(len(inter1))])
    wet1 = [wet1[i] for i in range(len(wet1)) if plt_wet[i]]
    inter1 = [inter1[i] for i in range(len(inter1)) if plt_inter[i]]
    wet2 = [wet2[i] for i in range(len(wet2)) if plt_wet[i]]
    inter2 = [inter2[i] for i in range(len(inter2)) if plt_inter[i]]

    if swater != 'inter':
        vp3 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos[plt_wet], vert=vert, widths=wbox,
                            bw_method=local_bw)
        clip_vplot(vp3, [np.amin(e) for e in wet2], [np.amax(e) for e in wet2],
                   orientation, ec=c)

        for vp in vp3['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if swater != 'wet':
        vp4 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2[plt_inter], vert=vert, widths=wbox,
                            bw_method=local_bw)
        clip_vplot(vp4, [np.amin(e) for e in inter2],
                   [np.amax(e) for e in inter2], orientation, ec=c)

        for vp in vp4['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if swater == 'both':
        slice_vplot(vp3, s1, ec=c)
        slice_vplot(vp4, s2, ec=c)

    # best params
    if swater == 'both':
        x = np.append(pos - wbox / 8., pos + wbox / 4. + pspace / 2.)
        y = np.append(best_w, best_i)

    elif swater == 'wet':
        x = pos
        y = best_w

    else:
        x = pos
        y = best_i

    if orientation == 'portrait':
        y = x

        if swater == 'both':
            x = np.append(best_i, best_w)
            y[:len(best_i)] -= wbox / 8.
            y[len(best_i):] -= wbox / 16.

        elif swater == 'wet':
            x = best_w

        else:
            x = best_i

    ax.plot(x, y, lw=0, marker='*', mec='k', zorder=9)

    # add custom legend
    if swater == 'both':  # add custom legend
        handles = custom_legend(swater, orientation, ec='gray')

    else:
        handles = custom_legend(swater, orientation)

    if orientation == 'landscape':
        ax.legend(handles=handles, loc=2, bbox_to_anchor=[-0.025, 1.015])

    else:
        ax.legend(handles=handles, ncol=3, columnspacing=1., handlelength=1.,
                  handletextpad=0.4, frameon=False, loc=2,
                  bbox_to_anchor=[0., 1.03])

    # add grid and format the axes
    lpos = np.asarray([0.05, 0.5, 0.9, 1., 1.1, 2., 20.])
    mpos = np.copy(pos)
    mpos[isec - 1] += (mpos[isec] - mpos[isec - 1]) / 2.
    mpos = np.delete(mpos, isec)
    mlines = np.copy(pos) + pscale / 2.
    mlines[isec] += pspace * 2. / 3.
    mlines = np.delete(mlines, isec - 1)
    custom_grid(mlines, -(vscale ** lpos), ax, orientation)

    if orientation == 'landscape':
        ax.set_yticks(-(vscale ** lpos))

        if swater == 'both':
            ax.set_xlim([np.amin(pos) - 0.5, np.amax(pos) + 0.5])

        else:
            ax.set_xlim([np.amin(pos) - 0.55, np.amax(pos) + 0.6])

        ax.set_xticks(mpos)

    else:
        ax.set_xticks(-(vscale ** lpos))

        if swater == 'both':
            ax.set_ylim([np.amin(pos) - 0.5, np.amax(pos) + 0.5])

        else:
            ax.set_ylim([np.amin(pos) - 0.6, np.amax(pos) + 0.55])

        ax.set_yticks(mpos + 0.15)

    # nicer display of the model names and normalised param values
    pvals = ['0.05', '0.5', '0.9', '', '1.1', '2', '20']
    models[models.index('WUE-LWP')] = r'WUE-$f_{\varPsi_l}$'
    models[models.index('SOX-OPT')] = r'SOX$_\mathrm{\mathsf{opt}}$'

    if orientation == 'landscape':
        ax.set_yticklabels(pvals)
        ax.set_xticklabels(models, va='top', rotation=25.)

        for i in range(len(params)):  # add param names

            t = ax.text(pos[i], -(vscale ** 0.2), params[i], va='top',
                        ha='center')
            t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                            alpha=0.8))

        # move the y labels to the right side
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_ylabel('Normalised parameter values')

    else:
        ax.set_xticklabels(pvals)
        ax.set_yticklabels(models, ha='left', va='top')

        for i in range(len(params)):  # add param names

            yy = pos[i] + 0.125

            if i == 7:
                yy = pos[i] - 0.12

            elif i in [11, 12, 13]:
                yy = pos[i] + 0.37

            elif i == 15:
                yy = pos[i] - 0.05

            t = ax.text(-(vscale ** 5.8), yy, params[i], ha='right', va='top')

        ax.tick_params(axis='y', direction='in', pad=-5.)
        plt.setp(ax.get_yticklabels(), bbox=dict(boxstyle='round', fc='w',
                                                 ec='none'))
        ax.set_xlabel('Normalised parameter values')

    # remove the ticks themselves
    ax.xaxis.set_tick_params(length=0.)
    ax.yaxis.set_tick_params(length=0.)

    base_dir = get_main_dir()
    opath = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    plt.savefig(os.path.join(opath,
                'model_calibs_%s_%s.jpg' % (swater, orientation)))
    plt.close()

    return


def update_solver_names(df):

    """
    Updates the solver names for nicer display

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the solvers to rename

    Returns:
    --------
    df: pandas dataframe
        dataframe where the solvers have been renamed

    """

    # replace the solver names
    df.replace({'solver': {'dual_annealing': 'Dual\nAnnealing',
                           'differential_evolution': 'Differential\nEvolution',
                           'basinhopping': 'Basin-Hopping', 'ampgo': 'AMPGO'}},
               inplace=True)

    return df


def solver_performance(df):

    """
    Orders the solvers from the best performing ones to the worst

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing the data to order

    Returns:
    --------
    df: pandas dataframe
        dataframe in organised order

    """

    # weighted average Rank for each solver
    wr = df.groupby('solver')['Rank'].mean()

    # count where the ranks overlap
    df = df.groupby(['solver', 'Rank']).size().reset_index(name='counts')

    # add the weighted average ranks
    df['waRank'] = df['solver'].map(wr)

    # organise by best to worst solver, as defined by their average rank
    df = df.iloc[df['waRank'].argsort()]
    df.reset_index(drop=True, inplace=True)

    # now assign a unique value to each solver
    df['sv'] = df['solver'].copy()

    sdic = {}
    i = 0

    for solver in pd.unique(df['solver']):

        sdic[solver] = i
        i += 1

    df.replace({'sv': sdic}, inplace=True)
    df = update_solver_names(df)

    return df


def solver_info_plot(df):

    """
    Generates a count plot of the rankings achieved by the various
    solvers in terms of calibration performance

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains information on all the parameter
        calibrations

    Returns:
    --------
    'solver_performance.jpg'

    """

    # declare the figure and the axis
    fig, ax = plt.subplots(figsize=(2.75, 3.))

    if 'training' not in df.columns:
        size = 30.

    else:
        size = 20.

    # solver performance info
    df = solver_performance(df)

    c = plt.rcParams['axes.prop_cycle'].by_key()['color'][-1]

    # counts plot where the points are bigger as more points overlap
    ax.scatter(df['sv'], df['Rank'], marker='o', c='grey', alpha=0.7,
               s=df['counts'] * size)

    # plot the N best data
    ax.scatter(df[df['sv'] < 4]['sv'], df[df['sv'] < 4]['Rank'], marker='o',
               c=c, s=df[df['sv'] < 4]['counts'] * size / 2.)

    # plot the average ranks
    ax.plot(pd.unique(df['sv']), df.groupby('sv')['waRank'].mean(), c='k',
            lw=1.5)

    # format the axes
    ax.set_xticks(np.arange(len(df['solver'].unique())) + 0.5)
    ax.set_xticklabels(df['solver'].unique(), rotation=55., ha='right',
                       va='top')
    ax.xaxis.set_tick_params(length=0.)

    # replace y axis with skill arrow
    ax.get_yaxis().set_visible(False)
    ax.text(-0.125, 0.95, 'Low\nskill', va='center', ha='center',
            transform=ax.transAxes)
    ax.annotate('High\nskill', xy=(-0.125, 0.9), xytext=(-0.125, 0.05),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='<-', lw=0.75))

    for spine in ax.spines.values():  # thinner spines

        spine.set_visible(True)
        spine.set_linewidth(0.25)

    base_dir = get_main_dir()
    opath = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    fig.tight_layout()
    plt.savefig(os.path.join(opath, 'solver_performance.jpg'))
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # user input
    swater = 'both'  # or wet or inter
    orientation = 'portrait'  # or landscape or portrait

    main(swater=swater, orientation=orientation)
