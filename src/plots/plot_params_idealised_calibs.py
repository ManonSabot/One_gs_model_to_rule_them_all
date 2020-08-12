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

def main(fname1, fname2, fname3, models, calibs='both', orientation='both',
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

    # modify the legend options to include subtitles
    for handler in Legend.get_default_handler_map().values():

        handler.legend_artist = subtitles_in_legend(handler.legend_artist)

    if orientation == 'both':

        for orientation in ['landscape', 'portrait']:

            plt_setup(calibs, orientation, colours=colours)  # rendering
            calib_info_plot(df1.copy(), df2.copy(), df3.copy(), models,
                            calibs=calibs, orientation=orientation)

    else:
        plt_setup(calibs, orientation, colours=colours)  # rendering
        calib_info_plot(df1, df2, df3, models, calibs=calibs,
                        orientation=orientation)

    return


def subtitles_in_legend(handler):

    @functools.wraps(handler)

    def wrapper(legend, orig_handle, fontsize, handlebox):

        handle_marker = handler(legend, orig_handle, fontsize, handlebox)

        if handle_marker.get_alpha() == 0:
            handlebox.set_visible(False)

    return wrapper


class plt_setup(object):

    def __init__(self, calibs, orientation, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.05  # padding to use
        plt.rcParams['savefig.jpeg_quality'] = 100
        plt.rcParams['savefig.orientation'] = orientation

        # colors
        if colours is None:  # use the default colours
            if calibs == 'both':
                colours = ['#2e7d9b', '#fc8635', '#2e7d9b', '#fc8635', 'grey',
                           '#ffff99']

            else:
                colours = ['#001a33', '#fffafa']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                               r'\usepackage{mathpazo}',
                                               r'\usepackage{amsmath}']
        plt.rcParams['font.size'] = 6.
        plt.rcParams['xtick.labelsize'] = 6.
        plt.rcParams['ytick.labelsize'] = 6.
        plt.rcParams['axes.labelsize'] = 7.

        # lines
        plt.rcParams['lines.linewidth'] = 2.

        # markers
        plt.rcParams['lines.markersize'] = 8.
        plt.rcParams['lines.markeredgewidth'] = 0.5

        # patches (e.g. the shapes in the legend)
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['patch.edgecolor'] = 'k'
        plt.rcParams['patch.force_edgecolor'] = True  # ensure it's used

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams['legend.framealpha'] = 1
        plt.rcParams['legend.edgecolor'] = 'w'
        plt.rcParams['legend.borderpad'] = 1.25

        if orientation == 'landscape':
            plt.rcParams['legend.fontsize'] = 6.
            plt.rcParams['legend.borderpad'] = 1.5

        # grid
        plt.rcParams['grid.color'] = '#bdbdbd'
        plt.rcParams['grid.linewidth'] = 0.25

        # spines
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['axes.spines.top'] = False


def update_model_names(df, models):

    # replace the model names
    df.replace({'Model': {'SOX': 'Eller'}}, inplace=True)
    df.Model = df.Model.astype('category')
    df.Model.cat.set_categories(models, inplace=True)

    return df


def normalise_params_by_group(df, by):

    # normalise
    df['norm_v1'] = df['v1'] / df.groupby(by)['v1'].transform('median')
    df['norm_v2'] = df['v2'] / df.groupby(by)['v2'].transform('median')

    return df


def model_performance(df):

    # where is there no single best rank within a group?
    minRank = df.groupby(['Model', 'training', 'sub-sample'])['Rank'].min()
    eq_models = minRank[minRank > 1.].index.get_level_values(0)
    eq_trainings = minRank[minRank > 1.].index.get_level_values(1)
    eq_samples = minRank[minRank > 1.].index.get_level_values(2)

    for i in range(len(eq_models)):

        where = np.logical_and(np.logical_and(df['Model'] == eq_models[i],
                               df['training'] == eq_trainings[i]),
                               df['sub-sample'] == eq_samples[i])
        sub = df[where]

        # if min rank duplicated, assign 1 to median params
        if len(sub[sub['Rank'] == sub['Rank'].min()]) > 1:
            idx = sub[sub['v1'] == sub['v1'].median()].index

            if len(idx) > 0:  # if the params are equal, just pick any
                idx = idx[0]

            df.loc[idx, 'Rank'] = 1

    # normalise by the sub-sample's median value to make the data comparable
    w = normalise_params_by_group(df[df['training'] == 'wet'].copy(),
                                  ['Model', 'sub-sample'])
    i = normalise_params_by_group(df[df['training'] == 'inter'].copy(),
                                  ['Model', 'sub-sample'])

    # data in order of model and rank
    df.sort_values(['Model', 'Rank'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    w.sort_values(['Model', 'Rank'], inplace=True)
    w.reset_index(drop=True, inplace=True)

    i.sort_values(['Model', 'Rank'], inplace=True)
    i.reset_index(drop=True, inplace=True)

    return df, w, i


def calib_info(data):

    # sorted data in model order
    ordered = [e for __, e in data.groupby('Model')]
    datas = [[e['norm_v1'].values, e['norm_v2'].values] for e in ordered]
    datas = [-(0.6 ** e) for ee in datas for e in ee]

    # get the best params from full fit out of every model
    ordered2 = [e[e['sub-sample'] == 0] for e in ordered]  # all timeseries

    try:
        dat = [[e.iloc[0, data.columns.get_loc('norm_v1')],
                e.iloc[0, data.columns.get_loc('norm_v2')]] for e in ordered2]

    except IndexError:
        dat = [[e.iloc[0, data.columns.get_loc('norm_v1')],
                e.iloc[0, data.columns.get_loc('norm_v2')]] for e in ordered]

    bests = np.asarray([-(0.6 ** e) for ee in dat for e in ee])

    return datas, bests


def subset_solvers(data, subset, best):

    data = data[data['solver'].isin(subset)]

    # reorder by best solver first
    data['Rank'] = 3.  # reset the ranks

    for i in range(len(best)):

        idx = (data[np.logical_and(data['Model'] == best['Model'].iloc[i] ,
                                   data['solver'] == best['solver'].iloc[i])]
                   .index)
        data.loc[idx, 'Rank'] = 1.

    data.sort_values(['Model', 'Rank'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def slice_vplot(vplot, side, ec=None, alpha=0.7):

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

        vp.set_alpha(alpha)
        vp.set_linewidth(plt.rcParams['patch.linewidth'])


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


def custom_legend(calibs, orientation):

    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if calibs == 'both':
        leg = [Patch(facecolor=c[0], alpha=0.7, label='Well watered'),
               Patch(facecolor=c[1], alpha=0.7, label='Intermediate'),
               Patch(facecolor=c[4], alpha=0.7, label='All solvers'),
               Patch(facecolor=c[5], alpha=0.7, label='Top 3 solvers'),
               Line2D([], [], linestyle='', marker='*', fillstyle='none',
                      mec='k', label='Best param.'),
               Line2D([], [], alpha=0, linestyle='', label='Inset plot:'),
               Line2D([], [], linestyle='', label='------$\;\;$ Mean')]

    else:
        leg = [Patch(facecolor=c[0], alpha=0.7, label='All solvers'),
               Patch(facecolor=c[1], alpha=0.7, label='Top 3 solvers'),
               Line2D([], [], linestyle='', marker='*', fillstyle='none',
                      mec='k', label='Best'),
               Line2D([], [], alpha=0, linestyle='', label='Inset plot:'),
               Line2D([], [], linestyle='', label='------$\;\;$ Mean')]

    return leg


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

    return df


def solver_info_plot(df, ax):

    if 'training' not in df.columns:
        size = 12.

    else:
        size = 8.

    # solver performance info
    df = solver_performance(df)

    c = plt.rcParams['axes.prop_cycle'].by_key()['color'][-2:]

    # counts plot where the points are bigger as more points overlap
    ax.scatter(df['sv'], df['Rank'], marker='o', c=c[0], alpha=0.7,
               s=df['counts'] * size)

    # plot the N best data
    ax.scatter(df[df['sv'] < 3]['sv'], df[df['sv'] < 3]['Rank'], marker='o',
               c=c[1], s=df[df['sv'] < 3]['counts'] * size / 2.)

    # plot the average ranks
    ax.plot(pd.unique(df['sv']), df.groupby('sv')['waRank'].mean(), c='k',
            lw=1.5)

    # format the axes
    ax.set_xticks(np.arange(len(pd.unique(df['solver']))))
    ax.xaxis.set_tick_params(width=0.25, length=2.5, pad=1.5)
    ax.set_xticklabels([i + 1 for i in range(len(pd.unique(df['solver'])))],
                       size=5.5)
    ax.set_xlabel('Solver', labelpad=1.5, fontsize=6.)

    # replace y axis with skill arrow
    ax.get_yaxis().set_visible(False)
    ax.text(-0.125, 0.875, 'Low\nskill', va='center', ha='center',
            transform=ax.transAxes)
    ax.annotate('High\nskill', xy=(-0.125, 0.75), xytext=(-0.125, 0.1),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='<-', lw=0.75))

    for spine in ax.spines.values():  # thinner spines

        spine.set_visible(True)
        spine.set_linewidth(0.25)

    return ax


def calib_info_plot(df1, df2, df3, models, calibs='wet',
                    orientation='landscape'):

    # landscape characteristics
    fs = (6., 4.25)
    iax = [0.2375, 0.685, 0.2, 0.2 * fs[0] / fs[1]]

    if calibs != 'both':
        iax = [0.2325, 0.678, 0.205, 0.205 * fs[0] / fs[1]]

    if orientation == 'portrait':
        fs = (4.25, 6.)
        iax = [0.665, 0.5685, 0.3025, 0.3025 * fs[0] / fs[1]]

        if calibs != 'both':
            iax = [0.655, 0.617, 0.3125, 0.3125 * fs[0] / fs[1]]

    # declare the figure and the axes
    fig, ax = plt.subplots(nrows=1, figsize=fs)
    iax = fig.add_axes(iax)

    if orientation == 'portrait':  # model order for plots
        models.reverse()

    # user-defined plot attributes
    wbox = 0.85
    Npoints = 500  # smooth violins
    bw = 0.3
    vert = True
    s1 = 'left'
    s2 = 'right'

    if orientation == 'portrait':
        vert = False
        s1 = 'bottom'
        s2 = 'top'

    if calibs == 'both':  # colours for the violin plot edge lines
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][-2:]

    # backup the original all solvers df for the inset plot
    df0 = df1.copy()

    # modify and order model names across all dfs
    df1 = update_model_names(df1, models)
    df2 = update_model_names(df2, models)
    df3 = update_model_names(df3, models)

    # model performance info across all solvers
    df1, w, i = model_performance(df1)

    # wet and inter data across all solvers
    wet1, best_w1 = calib_info(w)
    inter1, best_i1 = calib_info(i)

    # subset of the top 3 solvers
    w = subset_solvers(w, df2['solver'].unique(),
                       df3[df3['training'] == 'wet'])
    i = subset_solvers(i, df2['solver'].unique(),
                       df3[df3['training'] == 'inter'])

    # wet and inter data across top 3 solvers
    wet2, best_w2 = calib_info(w)
    inter2, best_i2 = calib_info(i)

    # assign positions to each model's parameters
    pspace = 0.01

    if calibs != 'both':
        pspace = 0.025

    pos = np.arange(float(len(wet1))) * 1.1
    pos[1::2] -= 8. * pspace # second parameter position

    # all solver data
    if calibs != 'inter':
        vp1 = ax.violinplot(wet1, showextrema=False, points=Npoints,
                            positions=pos, vert=vert, widths=wbox, bw_method=bw)

        for vp in vp1['bodies']:
            vp.set_alpha(0.7)

        #ax.plot(np.repeat(pos, len(wet1[0])), [item for sublist in wet1 for item in sublist], 'ro', alpha=0.5)

    if calibs != 'wet':
        vp2 = ax.violinplot(inter1, showextrema=False, points=Npoints,
                            positions=pos + pspace, vert=vert, widths=wbox,
                            bw_method=bw)

        for vp in vp2['bodies']:
            vp.set_alpha(0.7)

        #ax.plot(np.repeat(pos + pspace, len(inter1[0])), [item for sublist in inter1 for item in sublist], 'ro', alpha=0.5)

    if calibs == 'both':
        slice_vplot(vp1, s1, ec=c[0])
        slice_vplot(vp2, s2, ec=c[0])

    # top 3 solver data
    bw *= len(df1['solver'].unique()) / len(df2['solver'].unique())

    if calibs != 'inter':
        vp1 = ax.violinplot(wet2, showextrema=False, points=Npoints,
                            positions=pos, vert=vert, widths=wbox, bw_method=bw)

        for vp in vp1['bodies']:
            vp.set_alpha(0.7)


    if calibs != 'wet':
        vp2 = ax.violinplot(inter2, showextrema=False, points=Npoints,
                            positions=pos + pspace, vert=vert, widths=wbox, bw_method=bw)

        for vp in vp2['bodies']:
            vp.set_alpha(0.7)

    if calibs == 'both':
        slice_vplot(vp1, s1, ec=c[1])
        slice_vplot(vp2, s2, ec=c[1])

    # best all solvers, best top 3 solvers
    best_wet = [best_w1, best_w2]
    best_inter = [best_i1, best_i2]

    for j in range(len(best_wet)):

        wmask = np.ma.masked_invalid(best_wet[j]).mask
        imask = np.ma.masked_invalid(best_inter[j]).mask

        if calibs == 'both':
            x = np.append(pos[~wmask] - wbox / 8.,
                          pos[~imask] + wbox / 4. +  pspace / 2.)
            y = np.append(best_wet[j][~wmask], best_inter[j][~imask])

        elif calibs == 'wet':
            x = pos[~wmask]
            y = best_wet[j][~wmask]

        else:
            x = pos[~imask]
            y = best_inter[j][~imask]

        if orientation == 'portrait':
            y = x

            if calibs == 'both':
                x = np.append(best_wet[j][~wmask], best_inter[j][~imask])

            elif calibs == 'wet':
                x = best_wet[j][~wmask]

            else:
                x = best_inter[j][~imask]

        ax.plot(x, y, lw=0, marker='*', mec='k', zorder=9)

    # add custom legend
    if orientation == 'landscape':
        ax.legend(handles=custom_legend(calibs, orientation), loc=2,
                  bbox_to_anchor=(-0.03, 1.0275)).set_zorder(0)

    else:
        if calibs == 'both':
            ax.legend(handles=custom_legend(calibs, orientation), loc=1,
                      bbox_to_anchor=(1.03, 1.045)).set_zorder(0)

        elif calibs == 'wet':
            ax.legend(handles=custom_legend(calibs, orientation), loc=1,
                      bbox_to_anchor=(1.03, 1.02)).set_zorder(0)

        else:
            ax.legend(handles=custom_legend(calibs, orientation), loc=1,
                      bbox_to_anchor=(1.03, 1.02))

    # add grid and format the axes
    ppos = np.asarray([0.25, 0.5, 0.9, 1., 1.1, 2., 4.])
    mpos = pos[::2] + (pos[1::2] - pos[::2]) / 2.
    mlines = mpos[:-1] + 0.5 * np.diff(mpos)
    custom_grid(mlines, -(0.6 ** ppos), ax, orientation)

    if orientation == 'landscape':
        ax.set_yticks(-(0.6 ** ppos))
        ax.set_ylim(bottom=-(0.6 ** 0.02))  # crops the data but looks nicer
        ax.set_ylim(top=-(0.6 ** 70.))  # crops the data but looks nicer

        if calibs == 'both':
            ax.set_xlim([np.amin(pos) - 0.3, np.amax(pos) + 0.5])

        else:
            ax.set_ylim(top=-(0.6 ** 55.))  # crops the data but looks nicer
            ax.set_xlim([np.amin(pos) - 0.55, np.amax(pos) + 0.6])

        ax.set_xticks(mpos)

    else:
        ax.set_xticks(-(0.6 ** ppos))
        ax.set_xlim(left=-(0.6 ** 0.02))  # crops the data but looks nicer
        ax.set_xlim(right=-(0.6 ** 70))  # crops the data but looks nicer

        if calibs == 'both':
            ax.set_ylim([np.amin(pos) - 0.5, np.amax(pos) + 0.5])

        else:
            ax.set_xlim(right=-(0.6 ** 55))  # crops the data but looks nicer
            ax.set_ylim([np.amin(pos) - 0.6, np.amax(pos) + 0.55])

        ax.set_yticks(mpos)

    # nicer display of the model names and normalised param values
    pvals = ['0.25', '0.5', '0.9', '', '1.1', '2', '4']
    mnames = models[:]  # creates a copy of slice
    idx = [i for i, e in enumerate(mnames) if '-' in e]

    if orientation == 'landscape':
        change_to = [r'WUE-$f_{\varPsi_l}$', 'SOX$_\mathrm{\mathsf{opt}}$']

    else:
        change_to = ['SOX$_\mathrm{\mathsf{opt}}$',
                     'WUE-\n%s' % (r'$f_{\varPsi_l}$')]

    for i, e in enumerate(idx):

        mnames[e] = change_to[i]

    if calibs == 'wet':
        if orientation == 'landscape':
            mnames = mnames[1:]

        else:
            mnames = mnames[:-1]

    if orientation == 'landscape':
        ax.set_yticklabels(pvals)
        ax.set_xticklabels(mnames, rotation=20., size=7.)

        # move the y labels to the right side
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_ylabel('Normalised parameter values')

    else:
        ax.set_xticklabels(pvals)
        ax.set_yticklabels(mnames, size=7.)
        ax.tick_params(axis='y',direction='in', pad=-7)
        ax.set_xlabel('Normalised parameter values')

    # remove the ticks themselves
    ax.xaxis.set_tick_params(length=0., pad=2.5)
    ax.yaxis.set_tick_params(length=0., pad=2.5)

    # finally, add the inset plot of solver performance
    if calibs == 'both':
        solver_info_plot(df0, iax)

    else:
        solver_info_plot(df0[df0['training'] == calibs], iax)

    base_dir = get_main_dir()
    opath = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    fig.tight_layout()
    plt.savefig(os.path.join(opath,
                'model_calibs_%s_%s.png' % (calibs, orientation)))
    plt.savefig(os.path.join(opath,
                'model_calibs_%s_%s.pdf' % (calibs, orientation)))


#=======================================================================

if __name__ == "__main__":

    # user input
    fname1 = 'overview_of_fits.csv'  # all the solvers' info
    fname2 = 'top_3_fits.csv'  # 3 best solvers
    fname3 = 'best_fit.csv'  # best solvers
    models = ['Tuzet', 'Eller', 'ProfitMax', 'CGainNet', 'WUE-LWP', 'CMax',
              'LeastCost', 'SOX-OPT', 'CAP', 'MES']
    calibs = 'both'
    orientation = 'both'

    main(fname1, fname2, fname3, models, calibs=calibs,
         orientation=orientation)
