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

# plotting modules
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.patheffects as path_effects
from matplotlib.ticker import ScalarFormatter

# modules required for the custom legend
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib.legend import Legend
import functools
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory

#==============================================================================

def main(fname1, fname2, fname3, calibs='both', orientation='both',
         colours=None):

    base_dir = get_main_dir()
    dirname = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                           'calibrations'), 'idealised')

    # load in the data
    df1 = (pd.read_csv(os.path.join(dirname, fname1), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())
    df2 = (pd.read_csv(os.path.join(dirname, fname2), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())
    df3 = (pd.read_csv(os.path.join(dirname, fname3), header=[0])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    if orientation == 'both':

        for orientation in ['landscape', 'portrait']:

            plt_setup(calibs, orientation, colours=colours)  # rendering
            error_info_plot(df1.copy(), df2.copy(), df3.copy(), calibs=calibs,
                            orientation=orientation)
            calib_info_plot(df1.copy(), df2.copy(), df3.copy(), calibs=calibs,
                            orientation=orientation)

    else:
        plt_setup(calibs, orientation, colours=colours)  # rendering
        error_info_plot(df1.copy(), df2.copy(), df3.copy(), calibs=calibs,
                        orientation=orientation)
        calib_info_plot(df1.copy(), df2.copy(), df3.copy(), calibs=calibs,
                        orientation=orientation)

    solver_info_plot(df1)

    return


class plt_setup(object):

    def __init__(self, calibs, orientation, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use
        plt.rcParams['savefig.orientation'] = orientation

        # colors
        if colours is None:  # use the default colours
            if calibs == 'both':
                colours = ['#53e3d4', '#e2694e', '#53e3d4', '#e2694e',
                           '#ecec3a']

            else:
                colours = ['#001a33', '#fffafa']

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
        plt.rcParams['ytick.labelsize'] = 7.

        if orientation == 'portrait':
            plt.rcParams['xtick.labelsize'] = 7.
            plt.rcParams['ytick.labelsize'] = 6.

        # lines, markers
        plt.rcParams['lines.linewidth'] = 2.
        plt.rcParams['lines.markersize'] = 8.
        plt.rcParams['lines.markeredgewidth'] = 0.5

        # patches (e.g. the shapes in the legend)
        plt.rcParams['patch.edgecolor'] = 'k'
        plt.rcParams['patch.force_edgecolor'] = True  # ensure it's used

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams['legend.framealpha'] = 0.75
        plt.rcParams['legend.edgecolor'] = 'w'
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

    # normalise
    df['norm_v1'] = df['v1'] / df.groupby(by)['v1'].transform('median')
    df['norm_v2'] = df['v2'] / df.groupby(by)['v2'].transform('median')

    return df


def percent_errors(df):

    # normalise
    df['norm_ci1'] = 100. * np.abs(df['ci1'] / df['v1'])
    df['norm_ci2'] = 100. * np.abs(df['ci2'] / df['v2'])

    if len(df) > len(df['Model'].unique()):  # nans
        df['norm_ci1'] = df['norm_ci1'].fillna(500.)
        df['norm_ci2'][~df['v2'].isnull()] = (df['norm_ci2'][~df['v2'].isnull()]
                                                .fillna(500.))

    return df


def automate_model_order(df):

    df1 = percent_errors(df)
    df1['div'] = np.nanmean(np.array([df1['norm_ci1'], df1['norm_ci2']]),
                            axis=0)
    df2 = df1.groupby('Model')['div'].quantile(0.9).sort_values()
    df2.where(df2 < 50., df1.groupby('Model')['div'].mean().sort_values(),
              inplace=True)
    df2.sort_values(inplace=True)
    df2.rename(index={'SOX': 'Eller', 'CGainNet': 'CGain'}, inplace=True)

    return df2.index.to_list()


def update_model_names(df, models):

    # replace the model names
    df.replace({'Model': {'SOX': 'Eller', 'CGainNet': 'CGain'}}, inplace=True)
    df.Model = df.Model.astype('category')
    df.Model.cat.set_categories(models, inplace=True)

    return df


def sorted_data(df, norm_wet=None, norm_inter=None, errors=False):

    if errors:
        w = percent_errors(df[df['training'] == 'wet'].copy())
        i = percent_errors(df[df['training'] == 'inter'].copy())

    else:  # normalise by sub-sample median value
        if norm_wet is None:
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

            w.drop(w.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

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

            i.drop(i.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

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


def scaled_data(data, sc=0.5, errors=False):

    # sorted data in model order
    ordered = [e for __, e in data.groupby('Model')]

    if errors:
        datas = [[e['norm_ci1'].values, e['norm_ci2'].values] for e in ordered]
        datas = [e for ee in datas for e in ee]

    else:
        datas = [[e['norm_v1'].values, e['norm_v2'].values] for e in ordered]
        datas = [-(sc ** e) for ee in datas for e in ee]

    if len(datas[0]) < 2:
        datas = np.asarray([e for ee in datas for e in ee])

    return datas


def parameter_names(df):

    pdf = df.copy().replace({'p1': {'kmax': r'$k_{max}$', 'g1T': r'$g_{1,Tuz}$',
                                    'kmax2': r'$k_{max}$',
                                    'kmaxS1': r'$k_{max}$',
                                    'kmaxS2': r'$k_{max}$',
                                    'kmaxLC': r'$k_{max}$',
                                    'Lambda': r'$\lambda$', 'Alpha': '$a$',
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
    params[params.index('ksc_prev')] = r'$k_{\varPsi_m(t_{o}-1)}$'
    params[params.index('PcritC')] = r'$\varPsi_{\varphi,lim}$'
    params[params.index('PcritM')] = r'$\varPsi_{\varphi,lim}$'

    return params


def slice_vplot(vplot, side, ec=None):

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

    c = plt.rcParams['grid.color']
    lw = plt.rcParams['grid.linewidth']
    bottom, top = ax.get_ylim()
    left, right = ax.get_xlim()

    if orientation == 'landscape':

        for l in mlines:

            ax.vlines(l, bottom, top, color=c, lw=lw, zorder=-10)

        for l in plines:

            ax.hlines(l, left, right, color=c, lw=lw, zorder=-10)

    if orientation == 'portrait':

        for l in mlines:

            ax.hlines(l, left, right, color=c, lw=lw, zorder=-10)

        for l in plines:

            ax.vlines(l, bottom, top, color=c, lw=lw, zorder=-10)

    return


def custom_legend(calibs, orientation, ec=None):

    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if calibs == 'both':
        leg = [Line2D([], [], linestyle='', marker='*', ms=7., mfc=c[-1],
                      mec='k', label='Best'),
               Patch(facecolor=c[0], edgecolor=ec, alpha=0.7, label='Wet'),
               Patch(facecolor=c[1], edgecolor=ec, alpha=0.7, label='Stressed')]

    else:
        leg = [Line2D([], [], linestyle='', marker='*', mfc=c[-1], mec='k',
               label='Best')]

    return leg


def error_info_plot(df1, df2, df3, calibs='wet', orientation='landscape'):

    # user-defined plot attributes
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

    if calibs == 'both':
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
    df1, w, i = sorted_data(df1, errors=True)
    wet1 = scaled_data(w, errors=True)
    inter1 = scaled_data(i, errors=True)

    # subset of the top 3 solvers
    df2, w, i = sorted_data(df2, errors=True)
    wet2 = scaled_data(w, errors=True)
    inter2 = scaled_data(i, errors=True)

    # best params
    df3, w, i = sorted_data(df3, errors=True)
    best_w = scaled_data(w, errors=True)
    best_i = scaled_data(i, errors=True)

    # param names in model order?
    params = parameter_names(w)

    # where are there 2nd params?
    all = np.array([i for i in range(len(wet1)) if np.nansum(wet1[i]) != 0.])
    isec = np.array([i for i in range(len(all)) if (all[i] % 2) != 0])
    pos = np.arange(float(len(all))) * pscale
    pos[isec] -= 8. * pspace # second parameter position
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

    if calibs != 'inter':
        vp1 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos, vert=vert, widths=wbox)

        for vp in vp1['bodies']:

            vp.set_alpha(0.7)

    if calibs != 'wet':
        vp2 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2, vert=vert, widths=wbox)

        for vp in vp2['bodies']:

            vp.set_alpha(0.7)

    if calibs == 'both':
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

    if calibs != 'inter':
        vp3 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos[plt_wet], vert=vert, widths=wbox)
        clip_vplot(vp3, [np.amin(e) for e in wet2], [np.amax(e) for e in wet2],
                   orientation, ec=c)

        for vp in vp3['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if calibs != 'wet':
        vp4 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2[plt_inter], vert=vert, widths=wbox)
        clip_vplot(vp4, [np.amin(e) for e in inter2],
                   [np.amax(e) for e in inter2], orientation, ec=c)

        for vp in vp4['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if calibs == 'both':
        slice_vplot(vp3, s1, ec=c)
        slice_vplot(vp4, s2, ec=c)

    # best params
    if calibs == 'both':
        x = np.append(pos - wbox / 8., pos + wbox / 4. +  pspace / 2.)
        y = np.append(best_w, best_i)

    elif calibs == 'wet':
        x = pos
        y = best_w

    else:
        x = pos
        y = best_i

    if orientation == 'portrait':
        y = x

        if calibs == 'both':
            x = np.append(best_i, best_w)

        elif calibs == 'wet':
            x = best_w

        else:
            x = best_i

    ax.plot(x, y, lw=0, marker='*', mec='k', zorder=9)

    # add custom legend
    if calibs == 'both':  # add custom legend
        handles = custom_legend(calibs, orientation, ec='gray')

    else:
        handles = custom_legend(calibs, orientation)

    if orientation == 'landscape':
        ax.legend(handles=handles, loc=2, bbox_to_anchor=[-0.025, 1.015])

    else:
        ax.legend(handles=handles, ncol=3, columnspacing=1., handlelength=1.,
                  handletextpad=0.4, frameon=False, loc=2,
                  bbox_to_anchor=[0., 1.03])

    # add grid and format the axes
    lpos = np.asarray([0.095, 1, 2, 5, 10, 20, 50, 100, 200, 500])
    mpos = np.copy(pos)
    mpos[isec - 1] += (mpos[isec] - mpos[isec - 1] ) / 2.
    mpos = np.delete(mpos, isec)
    mlines = np.copy(pos) + pscale / 2.
    mlines[isec] += pspace * 2. / 3.
    mlines = np.delete(mlines, isec - 1)
    custom_grid(mlines, lpos, ax, orientation)

    if orientation == 'landscape':
        if calibs == 'both':
            ax.set_xlim([np.amin(pos) - 20., np.amax(pos) + 0.5])

        else:
            ax.set_xlim([np.amin(pos) - 2, np.amax(pos) + 0.6])

        ax.set_xticks(mpos)

    else:
        if calibs == 'both':
            ax.set_ylim([np.amin(pos) - 0.5, np.amax(pos) + 0.5])

        else:
            ax.set_ylim([np.amin(pos) - 0.6, np.amax(pos) + 0.55])

        ax.set_yticks(mpos + 0.15)

    # nicer display of the model names and normalised param values
    models[models.index('WUE-LWP')] = r'WUE-$f_{\varPsi_l}$'
    models[models.index('SOX-OPT')] = r'SOX$_\mathrm{\mathsf{opt}}$'

    if orientation == 'landscape':
        ax.set_xticklabels(models, va='top', rotation=25., size=7.)

        for i in range(len(params)):  # add param names

            t = ax.text(pos[i], 650, params[i], va='top', ha='center')
            t.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                            alpha=0.8))

        ax.set_yscale('log')
        ax.set_ylim([0.091, 550])
        ax.yaxis.set_major_formatter(ScalarFormatter())

        # move the y labels to the right side
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_ylabel(r'Parameter uncertainty ($\%$)')

    else:
        ax.set_yticklabels(models, ha='left', va='top', size=7.)

        for i in range(len(params)):  # add param names

            yy = pos[i] + 0.125

            if i in [9, 14, 15, 16]:
                yy = pos[i] + 0.375

            t = ax.text(650, yy, params[i], ha='left', va='top')

        ax.set_xscale('log')
        ax.set_xlim([0.091, 550])
        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.tick_params(axis='y', direction='in', pad=15.)
        plt.setp(ax.get_yticklabels(), bbox=dict(boxstyle='round', fc='w',
                                                 ec='none'))

        ax.set_xlabel(r'Parameter uncertainty ($\%$)')

    # remove the ticks themselves
    ax.xaxis.set_tick_params(length=0.)
    ax.yaxis.set_tick_params(length=0.)

    base_dir = get_main_dir()
    opath = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    fig.tight_layout()
    plt.savefig(os.path.join(opath,
                'model_calib_errors_%s_%s.png' % (calibs, orientation)))
    plt.savefig(os.path.join(opath,
                'model_calib_errors_%s_%s.jpg' % (calibs, orientation)))

    return


def local_bw(kde):

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


def calib_info_plot(df1, df2, df3, calibs='wet', orientation='landscape'):

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

    if calibs == 'both':  # colours for the violin plot edge lines
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
    pos[isec] -= 8. * pspace # second parameter position
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
    bw = 0.15

    if calibs != 'inter':
        vp1 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos, vert=vert, widths=wbox, bw_method=local_bw)

        for vp in vp1['bodies']:

            vp.set_alpha(0.7)

    if calibs != 'wet':
        vp2 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2, vert=vert, widths=wbox,
                            bw_method=local_bw)

        for vp in vp2['bodies']:

            vp.set_alpha(0.7)

    if calibs == 'both':
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

    if calibs != 'inter':
        vp3 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos[plt_wet], vert=vert, widths=wbox,
                            bw_method=local_bw)
        clip_vplot(vp3, [np.amin(e) for e in wet2], [np.amax(e) for e in wet2],
                   orientation, ec=c)

        for vp in vp3['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if calibs != 'wet':
        vp4 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos2[plt_inter], vert=vert, widths=wbox,
                            bw_method=local_bw)
        clip_vplot(vp4, [np.amin(e) for e in inter2],
                   [np.amax(e) for e in inter2], orientation, ec=c)

        for vp in vp4['bodies']:

            vp.set_alpha(0.7)
            vp.set_hatch('/' * 6)

    if calibs == 'both':
        slice_vplot(vp3, s1, ec=c)
        slice_vplot(vp4, s2, ec=c)

    # best params
    if calibs == 'both':
        x = np.append(pos - wbox / 8., pos + wbox / 4. +  pspace / 2.)
        y = np.append(best_w, best_i)

    elif calibs == 'wet':
        x = pos
        y = best_w

    else:
        x = pos
        y = best_i

    if orientation == 'portrait':
        y = x

        if calibs == 'both':
            x = np.append(best_i, best_w)
            y[:len(best_i)] -= wbox / 8.
            y[len(best_i):] -= wbox / 16.

        elif calibs == 'wet':
            x = best_w

        else:
            x = best_i

    ax.plot(x, y, lw=0, marker='*', mec='k', zorder=9)

    # add custom legend
    if calibs == 'both':  # add custom legend
        handles = custom_legend(calibs, orientation, ec='gray')

    else:
        handles = custom_legend(calibs, orientation)

    if orientation == 'landscape':
        ax.legend(handles=handles, loc=2, bbox_to_anchor=[-0.025, 1.015])

    else:
        ax.legend(handles=handles, ncol=3, columnspacing=1., handlelength=1.,
                  handletextpad=0.4, frameon=False, loc=2,
                  bbox_to_anchor=[0., 1.03])

    # add grid and format the axes
    lpos = np.asarray([0.05, 0.5, 0.9, 1., 1.1, 2., 20.])
    mpos = np.copy(pos)
    mpos[isec - 1] += (mpos[isec] - mpos[isec - 1] ) / 2.
    mpos = np.delete(mpos, isec)
    mlines = np.copy(pos) + pscale / 2.
    mlines[isec] += pspace * 2. / 3.
    mlines = np.delete(mlines, isec - 1)
    custom_grid(mlines, -(vscale ** lpos), ax, orientation)

    if orientation == 'landscape':
        ax.set_yticks(-(vscale ** lpos))

        if calibs == 'both':
            ax.set_xlim([np.amin(pos) - 0.5, np.amax(pos) + 0.5])

        else:
            ax.set_xlim([np.amin(pos) - 0.55, np.amax(pos) + 0.6])

        ax.set_xticks(mpos)

    else:
        ax.set_xticks(-(vscale ** lpos))

        if calibs == 'both':
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
        ax.set_xticklabels(models, va='top', rotation=25., size=7.)

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
        ax.set_yticklabels(models, ha='left', va='top', size=7.)

        for i in range(len(params)):  # add param names

            yy = pos[i] + 0.125

            if i == 6:
                yy = pos[i] - 0.12

            elif i in [13, 16]:
                yy = pos[i] - 0.05

            elif i in [12, 14]:
                yy = pos[i] + 0.37

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

    fig.tight_layout()
    plt.savefig(os.path.join(opath,
                'model_calibs_%s_%s.png' % (calibs, orientation)))
    plt.savefig(os.path.join(opath,
                'model_calibs_%s_%s.jpg' % (calibs, orientation)))

    return


def update_solver_names(df):

    # replace the solver names
    df.replace({'solver': {'dual_annealing': 'Dual\nAnnealing',
                           'differential_evolution': 'Differential\nEvolution',
                           'basinhopping': 'Basin-Hopping', 'ampgo': 'AMPGO'}},
               inplace=True)

    return df


def solver_performance(df):

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

    # declare the figure and the axes
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
    plt.savefig(os.path.join(opath, 'solver_performance.png'))
    plt.savefig(os.path.join(opath, 'solver_performance.jpg'))

    return


#=======================================================================

if __name__ == "__main__":

    # user input
    fname1 = 'overview_of_fits.csv'  # all the solvers' info
    fname2 = 'top_3_fits.csv'  # 3 best solvers
    fname3 = 'best_fit.csv'  # best solvers
    calibs = 'both'  # or wet or inter
    orientation = 'portrait'  # or landscape or portrait

    main(fname1, fname2, fname3, calibs=calibs, orientation=orientation)
