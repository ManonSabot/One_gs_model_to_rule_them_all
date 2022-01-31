#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script that plots the models' ability to match the observations, as
defined by mechanistic functional shapes or functional biases.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = ""
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (28.05.2021)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import sys  # check for files, versions
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files
import warnings  # ignore warnings

# plotting modules
import matplotlib as mpl  # general matplotlib libraries
import matplotlib.pyplot as plt  # plotting interface
from shapely.geometry import MultiPoint  # contour plots
from scipy.stats import gaussian_kde  # pdfs of the data
from pygam import ExpectileGAM  # fit the data
import statsmodels.api as sm  # smooth
from matplotlib.lines import Line2D  # custom legends
from matplotlib.patches import Patch  # custom legends
import string  # automate subplot lettering

# own modules
from plot_utils import default_plt_setup
from plot_utils import model_order, which_model
from plot_utils import site_spp_order, missing_var
from plot_utils import render_xlabels, render_ylabels

from plot_utils import get_Px  # vulnerability curve

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files

# ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

def main(site_spp=site_spp_order(), subsel1=None, subsel2=None,
         groups=None):

    """
    Main function: plots of the observed functional relations in A to E,
                   gs to Ci, gs to Pleaf, gs to VPD, and Pleaf per
                   biome / ecosystem

    Arguments:
    ----------
    site_spp: list
        site_species combinations to plot

    subsel1: list
        subselection of site_species combinations to plot of the A(E)
        relationship

    subsel2: list
        subselection of site_species combinations to plot for the iWUE
        summary plot

    groups: list
        ecosystems to consider for groupings of Pleaf data

    Returns:
    --------
    too many .jpg files list

    """

    base_dir = get_main_dir()  # working paths
    fig_dir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

    plt_setup()  # figure specs

    # data to plot
    df = pd.read_csv(os.path.join(os.path.join(os.path.join(os.path.join(
                     base_dir, 'output'), 'simulations'), 'obs_driven'),
                     'all_site_spp_simulations.csv'))

    # only keep the relevant site x species data
    df = df[df['site_spp'].isin(site_spp)]

    # A(E) functional shapes
    figname = os.path.join(fig_dir, 'A_E_others.jpg')

    if not os.path.isfile(figname):
        A_E_relationships(df, figname, [e for e in site_spp
                                        if e not in subsel1], VPD_info=False)

    figname = os.path.join(fig_dir, 'A_E_subsel.jpg')

    if not os.path.isfile(figname):
        A_E_relationships(df, figname, subsel1)

    # Ci(gs) contours
    figname = os.path.join(fig_dir, 'Ci_gs.jpg')

    if not os.path.isfile(figname):

        # exclude the site x species where there are no observed Ci
        exclude = missing_var(df, 'Ci', 'site_spp')
        gs_Ci_clusters(df, figname, [e for e in site_spp if e not in exclude])

    # gs(LWP) functional shapes
    figname = os.path.join(fig_dir, 'gs_LWP.jpg')

    if not os.path.isfile(figname):
        gs_LWP_functional(df, figname, site_spp)

    # iWUE summary example plots
    figname = os.path.join(fig_dir, 'iWUE_subsel.jpg')

    if not os.path.isfile(figname):
        iWUE_relationships(df, figname, subsel2)

    # VPD closure plots
    figname = os.path.join(fig_dir, 'VPD_closure.jpg')

    if not os.path.isfile(figname):
        VPD_closure(df.copy(), figname, site_spp)

    # LWP errors
    figname = os.path.join(fig_dir, 'LWP_errors.jpg')

    if not os.path.isfile(figname):  # the dataframe must be ordered
        df.site_spp = df.site_spp.astype('category')
        df.site_spp.cat.set_categories(site_spp, inplace=True)
        df = df.sort_values('site_spp')  # important to get this right

        # plot the data
        LWP_error_plots(df, figname, groups)

    return


class plt_setup(object):

    """
    Matplotlib configuration specific to this figure

    """

    def __init__(self, colours=None):

        # default setup
        default_plt_setup(ticks=True)

        # box plots
        plt.rcParams['boxplot.boxprops.linewidth'] = 0.5
        plt.rcParams['boxplot.whiskerprops.linewidth'] = \
            plt.rcParams['boxplot.boxprops.linewidth']
        plt.rcParams['boxplot.capprops.linewidth'] = \
            plt.rcParams['boxplot.boxprops.linewidth']
        plt.rcParams['boxplot.flierprops.linewidth'] = \
            plt.rcParams['boxplot.boxprops.linewidth'] / 4.
        plt.rcParams['boxplot.medianprops.linewidth'] = \
            plt.rcParams['boxplot.boxprops.linewidth']
        plt.rcParams['boxplot.flierprops.markersize'] = 0.25

        # lines
        plt.rcParams['lines.linewidth'] = 1.75

        # legend
        plt.rcParams['legend.borderpad'] = 0.5


def set_box_colour(bp, colour, alpha=1., lc='k'):

    """
    Alters the colours and rendering of a box plot

    Arguments:
    ----------
    bp: matplotlib objects
        box plot to alter

    colour: string
        face colour to apply to the box plot

    alpha: float
        transparency of the box plot's face colour

    lc: string
        colour to apply to the lines of the box plot, e.g., edge lines,
        whiskers, etc.

    Returns:
    --------
    The altered box plot

    """

    plt.setp(bp['boxes'], color=lc)
    plt.setp(bp['boxes'], facecolor=colour, alpha=alpha)

    plt.setp(bp['whiskers'], color=lc)
    plt.setp(bp['caps'], color=lc)

    plt.setp(bp['medians'], color=lc)
    plt.setp(bp['fliers'], markerfacecolor=lc, markeredgecolor=lc)

    return


def A_E_relationships(df, figname, keep, VPD_info=True):

    """
    Plots scatters of the observed E and A data, modelled data, and each
    site x species fitted functional forms of A(E) depending on VPD

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    keep: list
        sites x species to plot

    VPD_info: bool
        if True, plot the fitted functional A(E)

    Returns:
    --------
    'A_E_other.jpg' (lots of sites) or 'A_E_subsel.jpg' (subselection)

    """

    # decide on how many columns and rows
    Nrows = 0
    Ncols = 0

    while Nrows * Ncols < len(keep):

        Nrows += 1

        if Nrows * Ncols < len(keep):
            Ncols += 1

    # declare figure
    fig = plt.figure(figsize=(Ncols + 2.25, Nrows + 2))
    iter = 0  # keep track of the sites

    for row in range(Nrows - 1, -1, -1):  # adding suplots from the top

        b = row / Nrows
        t = (row + 1) / Nrows
        t = t - (t - b) / (2. * Nrows * Ncols)

        for col in range(Ncols):  # adding from the left

            lf = col / Ncols
            rg = (col + 0.88) / Ncols

            if Nrows < 3:
                wspace = 2.5 * Ncols
                hspace = 0.75 * Nrows

            else:
                wspace = -0.75
                hspace = -0.25

            axes = fig.add_gridspec(nrows=6, ncols=6, left=lf, right=rg, top=t,
                                    bottom=b, wspace=wspace, hspace=hspace)
            ax = fig.add_subplot(axes[:-1, 1:])

            if Nrows < 3:
                ax_y = fig.add_subplot(axes[:-1, 0], xticklabels=[], sharey=ax)
                ax_x = fig.add_subplot(axes[-1, 1:], yticklabels=[], sharex=ax)

                for axis in [ax_y, ax_x]:

                    for s in axis.spines.values():  # remove spines

                        s.set_visible(False)

                    # removing the double tick marks
                    axis.patch.set_visible(False)
                    axis.tick_params(left=False, labelleft=False, bottom=False,
                                     labelbottom=False)

            # E - A relationships
            sub = df.copy()[df['site_spp'] == keep[iter]]

            # plot the obs functional relationship
            x = sub['E']
            y = sub['A']
            xmin = 0.
            xmax = 2.5 * np.nanmax(x)

            if np.nanmin(y) > 0.:
                ymin = np.nanmin(y) / 2.5

            else:
                ymin = np.nanmin(y) * 2.5

            ymax = 2.5 * np.nanmax(y)

            if Nrows < 3:  # distribution of obs
                bp = ax_y.boxplot(y, widths=0.65, showfliers=False,
                                  patch_artist=True)
                set_box_colour(bp, 'w', lc='k')
                ax_y.invert_xaxis()

                bp = ax_x.boxplot(x, vert=False, widths=0.35, showfliers=False,
                                  patch_artist=True)
                set_box_colour(bp, 'w', lc='k')
                ax_x.invert_yaxis()

            # add the obs to the plot
            ax.scatter(x, y, marker='+', s=200. / Nrows, lw=2.5 / Ncols,
                       facecolor='#c0c0c0', edgecolor='none', alpha=0.5,
                       label='Obs.')

            for mod in model_order():

                subsub = sub[['E(%s)' % (mod), 'A(%s)' % (mod)]].dropna()

                xx = subsub['E(%s)' % (mod)]
                yy = subsub['A(%s)' % (mod)]

                # deal with the nans + non-physical values
                valid = np.logical_and(xx >= xmin, xx <= xmax)
                xx = xx[valid]
                yy = yy[valid]

                valid = np.logical_and(yy >= ymin, yy <= ymax)
                xx = xx[valid]
                yy = yy[valid]

                try:  # plot with shading based on density
                    c = mpl.colors.to_rgba(next(ax._get_lines.prop_cycler)
                                           ['color'])
                    dens = gaussian_kde(np.vstack([xx, yy]))(np.vstack([xx,
                                                                        yy]))
                    dens = dens / np.nanmax(dens)  # shade by density
                    c = [(c[0], c[1], c[2], e) for e in dens]
                    ax.scatter(xx, yy, s=2200. / (len(sub) * Ncols), c=c,
                               alpha=0.5, label=which_model(mod))

                except Exception:
                    pass

            # plot average obs behaviour above everything else
            gam = (ExpectileGAM(expectile=0.5, n_splines=5, spline_order=4)
                   .gridsearch(x.values.reshape(-1, 1), y.values))
            px = np.linspace(x.quantile(0.05), x.quantile(0.95), num=500)
            py = gam.predict(px)

            ax.plot(px, py, color='k', ls='--', zorder=20, label='Avg. Obs.')

            if VPD_info:  # plot VPD-specific observed behaviours
                if iter < 2:
                    vpd1 = sub['VPD'] < sub['VPD'].quantile(0.05)
                    vpd2 = sub['VPD'] > sub['VPD'].quantile(0.95)

                else:
                    vpd1 = sub['VPD'] < sub['VPD'].quantile(0.15)
                    vpd2 = sub['VPD'] > sub['VPD'].quantile(0.85)

                for i, vpd in enumerate([vpd1, vpd2]):

                    gam = (ExpectileGAM(expectile=0.5, n_splines=5,
                                        spline_order=4)
                           .gridsearch(x[vpd].values.reshape(-1, 1),
                                       y[vpd].values))
                    px = np.linspace(np.minimum(x[vpd].min(),
                                                x.quantile(0.25)),
                                     np.maximum(x[vpd].max(),
                                                x.quantile(0.75)), num=500)
                    py = gam.predict(px)
                    ax.plot(px, py, color='k', ls='--', zorder=20)

                    if i == 0:
                        txt = ax.annotate('low $D_a$', xy=(px[-1], py[-1]),
                                          xytext=(-20, -2),
                                          textcoords='offset points',
                                          ha='right')

                    else:
                        if iter == 0:
                            p1 = -2
                            p2 = -12

                        if iter == 1:
                            p1 = 8
                            p2 = -6

                        if iter == 2:
                            p1 = -8
                            p2 = -15

                        if iter == 3:
                            p1 = 2
                            p2 = 1

                        txt = ax.annotate('high $D_a$', xy=(px[-1], py[-1]),
                                          xytext=(p1, p2),
                                          textcoords='offset points',
                                          ha='left')

                    txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w',
                                      ec='none', alpha=0.8))

            # tighten the subplots
            __, right = ax.get_xlim()
            ax.set_xlim(0., right)

            # format axes ticks
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.set_xticklabels(ax.get_xticks())  # force LaTex
            ax.set_yticklabels(ax.get_yticks())  # force LaTex
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

            if Nrows < 3:
                pad_x = 12.5
                pad_y = 15.

            else:
                pad_x = 5.
                pad_y = 5.

            if row == 0:
                render_xlabels(ax, r'$E$', r'mmol m$^{-2}$ s$^{-1}$',
                               pad=pad_x)

            if col == 0:
                render_ylabels(ax, r'A$_n$', r'$\mu$mol m$^{-2}$ s$^{-1}$',
                               pad=pad_y)

            species = r'%s %s' % (keep[iter].split('_')[-2],
                                  keep[iter].split('_')[-1])

            if 'Quercus' in species:
                title = r'\textit{%s} (%s)' % (species,
                                               keep[iter].split('_')[0][0])

            else:
                title = r'\textit{%s}' % (species)

            iter += 1  # second or third column

            if Nrows < 3:
                ax.set_title(r'\textbf{(%s)} ' %
                             (string.ascii_lowercase[iter - 1]) + title)

                if (col == Ncols - 1) and (iter != len(keep)):
                    leg = ax.legend(bbox_to_anchor=(1.025, 0.5), loc=2,
                                    handleheight=1., labelspacing=1. / 3.)

            else:
                txt = ax.annotate(title, xy=(0.98, 0.02),
                                  xycoords='axes fraction', ha='right',
                                  va='bottom')

                # subplot labelling
                ax.text(0.025, 0.925,
                        r'\textbf{(%s)}' % (string.ascii_lowercase[iter - 1]),
                        transform=ax.transAxes, weight='bold')

                if (col == Ncols - 1) and (row == 1):
                    if len(keep) == Ncols * Nrows:
                        leg = ax.legend(bbox_to_anchor=(1.01, 0.75), loc=2,
                                        handleheight=0.8, labelspacing=0.05,
                                        fontsize=10.)

                    else:
                        leg = ax.legend(bbox_to_anchor=(-0.25, -0.35), ncol=2,
                                        loc=2, handleheight=1.,
                                        labelspacing=0.05)

            try:  # alter legend handles size and opacity
                if Nrows < 3:
                    leg.legendHandles[1]._sizes = [250. / (Nrows * Ncols)]

                else:
                    leg.legendHandles[1]._sizes = [1000. / (Nrows * Ncols)]

                leg.legendHandles[1].set_alpha(1)

                for e in leg.legendHandles[2:]:

                    if Nrows < 3:
                        e._sizes = [500. / (Nrows * Ncols)]

                    else:
                        e._sizes = [1500. / (Nrows * Ncols)]

                    e.set_alpha(0.9)

            except Exception:
                pass

            if iter == len(keep):
                break

    fig.savefig(figname)
    plt.close()

    return


def chaikins_corner_cutting(coords, refinements=1000):

    """
    Cuts corners to generate curves from a limited number of points

    Arguments:
    ----------
    coords: array
        polygon with corners to cut

    refinements: int
        number of times the cutting procedure should be repeated

    Returns:
    --------
    New polygon with smoothed curved corners

    """

    for _ in range(refinements):

        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]

    return L * 0.75 + R * 0.25


def encircle(ax, x, y, smooth=False, **kw):

    """
    Draws a polygon around x-y data in space

    Arguments:
    ----------
    ax: matplotlib object
        axis on which to plot the polygon

    x: array
        data

    y: array
        data

    smooth: bool
        if True, smoothes the x-y polygon by applying a confindence
        interval on variability in the data and fitting a buffer line
        through the envelope

    ** kw: other polygon properties

    Returns:
    --------
    New polygon with smoothed curved corners

    """

    p = np.c_[x, y]  # concatenate along second axis
    hull = MultiPoint(p).convex_hull

    if smooth:
        hull = hull.buffer(5.).buffer(-5.)

    x, y = hull.exterior.xy
    p = np.array([x, y]).T
    p = chaikins_corner_cutting(p)
    poly = plt.Polygon(p, **kw)
    ax.add_patch(poly)

    return


def gs_Ci_clusters(df, figname, keep):

    """
    Plots contours of modelled Ci(gs) clusters relative to the observed
    Ci and gs

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    keep: list
        sites x species to plot

    Returns:
    --------
    'Ci_gs.jpg'

    """

    # decide on how many columns and rows
    Nrows = 0
    Ncols = 0

    while Nrows * Ncols < len(keep):

        Nrows += 1

        if Nrows * Ncols < len(keep):
            Ncols += 1

    # declare figure
    fig, axes = plt.subplots(Nrows, Ncols, figsize=(Ncols + 2, Nrows + 2.25))
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    axes = axes.flat  # flatten axes

    # Ci - gs relationships
    for i, what in enumerate(keep):

        sub = df.copy()[df['site_spp'] == what]

        left = 1.
        bottom = 1.

        for mod in model_order():

            x = sub['Ci(%s)' % (mod)] / sub['Ci']
            y = sub['gs(%s)' % (mod)] / sub['gs']

            # deal with the nans
            valid = np.logical_and(sub['Ci(%s)' % (mod)] < 9999.,
                                   sub['gs(%s)' % (mod)] < 9999.)
            x = x[valid]
            y = y[valid]

            # interquartile ranges
            valid = np.logical_and(x / y > (x / y).quantile(0.25),
                                   x / y < (x / y).quantile(0.75))

            try:
                c = next(axes[i]._get_lines.prop_cycler)['color']
                encircle(axes[i], x[valid], y[valid], smooth=True, ec=c,
                         fc='None', alpha=0.9, lw=2.)

                # tighten the subplots
                left = np.minimum(left, x[valid].min())
                bottom = np.minimum(bottom, y[valid].min())

            except Exception:
                pass

        axes[i].hlines(1., 0., 1., linestyle=':',
                       transform=axes[i].get_yaxis_transform())
        axes[i].vlines(1., 0., 1., linestyle=':',
                       transform=axes[i].get_xaxis_transform())

        # tighten the subplots
        __, right = axes[i].get_xlim()
        __, top = axes[i].get_ylim()

        # tighten the subplots
        axes[i].set_xlim(np.maximum(-0.1, left - 0.2), right)
        axes[i].set_ylim(np.maximum(-0.1, bottom - 0.2), top)

        # format axes ticks
        axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].set_xticklabels(axes[i].get_xticks())  # force LaTex
        axes[i].set_yticklabels(axes[i].get_yticks())  # force LaTex
        axes[i].xaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))
        axes[i].yaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))

        if i >= (Nrows - 1) * Ncols:
            axes[i].set_xlabel(r'$C_i$ $(sim. : obs.)$')

        if i % Ncols == 0:
            axes[i].set_ylabel(r'$g_s$ $(sim. : obs.)$')

        # subplot titles
        what = what.split('_')
        species = r'\textit{%s %s}' % (what[-2], what[-1])

        if 'Quercus' in what:
            species += ' (%s)' % (what[0][0])

        txt = axes[i].annotate(species, xy=(0.98, 0.03),
                               xycoords='axes fraction', ha='right')
        txt.set_bbox(dict(boxstyle='round,pad=0.05', fc='w', ec='none',
                          alpha=0.8))

        # subplot labelling
        axes[i].text(0.025, 0.9,
                     r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                     transform=axes[i].transAxes, weight='bold')

    # add legend
    handles = ([Line2D([0], [0], c=c, alpha=0.9)
                for c in plt.rcParams['axes.prop_cycle'].by_key()['color']])
    labels = [which_model(mod) for mod in model_order()]

    if len(keep) < len(axes):
        axes[i].legend(handles, labels, bbox_to_anchor=(1.05, 0.98), loc=2,
                       ncol=2, handlelength=0.8)

    else:
        if Nrows > 2:
            i -= Ncols

        axes[i].legend(handles, labels, bbox_to_anchor=(1.025, 0.75), loc=2,
                       handlelength=0.8)

    while len(keep) < len(axes):

        axes[len(keep) - len(axes)].axis('off')
        keep += ['fill']

    fig.savefig(figname)
    plt.close()

    return


def gs_LWP_functional(df, figname, keep):

    """
    Plots fitted modelled functional relationships of gs(LWP), and
    scatters of the observations

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    keep: list
        sites x species to plot

    Returns:
    --------
    'gs_LWP.jpg'

    """

    # decide on how many columns and rows
    Nrows = 0
    Ncols = 0

    while Nrows * Ncols < len(keep):

        Nrows += 1

        if Nrows * Ncols < len(keep):
            Ncols += 1

    # declare figure
    fig, axes = plt.subplots(Nrows, Ncols, figsize=(Ncols + 2.25, Nrows + 2))
    plt.subplots_adjust(hspace=0.275, wspace=0.275)
    axes = axes.flat  # flatten axes

    # gs - LWP relationships
    for i, what in enumerate(keep):

        sub = df.copy()[df['site_spp'] == what]

        ipath = os.path.join(os.path.join(os.path.join(get_main_dir(),
                             'input'), 'simulations'), 'obs_driven')
        ref, __ = read_csv(os.path.join(ipath,
                                        '%s_calibrated.csv' % (what)))
        Pcrit = get_Px(ref.loc[0, 'P50'], ref.loc[0, 'P88'], 50., 88., 95.)

        valid = sub['gs'] < sub['gs'].max()
        yobs = sub['gs'][valid] / sub['gs'].max()

        if what == 'Corrigin_Eucalyptus_capillosa':  # min average obs
            axes[i].scatter(np.linspace(0.45, 0.75, 10), [0.01, ] * 10,
                            marker='+', s=500. / (Nrows * Ncols),
                            lw=2.5 / Ncols, facecolor='#c0c0c0',
                            edgecolor='none', alpha=0.6, label='Obs.')  # gs
            axes[i].scatter([-5.2 / Pcrit, ] * 20, np.linspace(0., 1., 20),
                            marker='+', s=500. / (Nrows * Ncols),
                            lw=2.5 / Ncols, facecolor='#c0c0c0',
                            edgecolor='none', alpha=0.6)  # Pleaf

        else:
            xobs = sub['Pleaf'][valid] / Pcrit
            axes[i].scatter(xobs, yobs, marker='+', s=500. / (Nrows * Ncols),
                            lw=2.5 / Ncols, facecolor='#c0c0c0',
                            edgecolor='none', alpha=0.6, label='Obs.')

        # add P50 info
        axes[i].vlines(-ref.loc[0, 'P50'] / Pcrit, 0., 1., linestyle=':',
                       linewidth=1., transform=axes[i].get_xaxis_transform(),
                       label='P$_{50}$', zorder=20)

        local_max = 0.
        next(axes[i]._get_lines.prop_cycler)

        for mod in model_order()[1:]:

            x = sub['Pleaf(%s)' % (mod)] / Pcrit
            y = (sub['gs(%s)' % (mod)] /
                 sub['gs(%s)' % (mod)][sub['gs(%s)' % (mod)] < 9999.].max())

            # deal with the nans
            valid = np.logical_and(sub['Pleaf(%s)' % (mod)] < 1.1 * sub['Ps'],
                                   np.logical_and(x < 1., sub['gs(%s)' % (mod)]
                                                  < 9999.))

            x = x[valid]
            y = y[valid]

            local_max = np.maximum(local_max, y.max())

            try:  # plot functional relationships
                c = next(axes[i]._get_lines.prop_cycler)['color']
                alpha = 0.5

                if x.min() < 0.95:
                    try:  # plot average obs behaviour
                        gam = (ExpectileGAM(expectile=0.5, n_splines=5,
                                            spline_order=4,
                                            constraints='monotonic_dec')
                               .gridsearch(x.values.reshape(-1, 1),
                                           y.values))
                        px = np.linspace(x.min(),
                                         np.maximum(xobs.max(), x.max()),
                                         num=500)
                        py = gam.predict(px)

                        check = np.gradient(py, px)[~np.isclose(np.gradient(py,
                                                                px), 0.)]

                        if (np.isnan(np.nanmean(check)) or
                           (np.nanmean(check) >= -0.3)):
                            ms = 6.
                            alpha = 0.9

                            raise Exception

                        axes[i].plot(px, py, color=c, alpha=0.9)
                        ms = 1.
                        alpha = 0.3

                    except Exception:
                        pass

                # plot the scatters
                axes[i].scatter(x, y, s=ms, c=c, alpha=alpha,
                                label=which_model(mod))

            except Exception:
                pass

        # format axes ticks
        axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].set_xticklabels(axes[i].get_xticks())  # force LaTex
        axes[i].set_yticklabels(axes[i].get_yticks())  # force LaTex
        axes[i].xaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.2f'))
        axes[i].yaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))

        if i >= (Nrows - 1) * Ncols:
            axes[i].set_xlabel(r'$\Psi$$_{l, norm}$')

        if i % Ncols == 0:
            axes[i].set_ylabel(r'$g_{s, norm}$')

        # subplot titles
        what = what.split('_')
        species = r'\textit{%s %s}' % (what[-2], what[-1])

        if 'Quercus' in what:
            species += ' (%s)' % (what[0][0])

        txt = axes[i].annotate(r'\textbf{(%s)} %s' %
                               (string.ascii_lowercase[i], species),
                               xy=(0.98, 0.98), xycoords='axes fraction',
                               ha='right', va='top')
        txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                          alpha=0.9))

        # tighten the subplots
        left, right = axes[i].get_xlim()
        axes[i].set_xlim(np.maximum(0.05, left), np.minimum(right, 1.05))

        if np.isnan(local_max):
            axes[i].set_ylim(0., 1.05)

        else:
            axes[i].set_ylim(0., np.minimum(local_max + 0.5, 1.05))

    # add legend
    handles, labels = axes[1].get_legend_handles_labels()
    handles[0].set_alpha(1)  # alter handle opacity
    handles = [handles[0], handles[1]] + \
              [Line2D([0], [0], c=c, alpha=0.9) for c in
               plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]]

    if len(keep) < len(axes):
        axes[i].legend(handles, labels, bbox_to_anchor=(1.01, 0.98), loc=2,
                       ncol=2, handlelength=0.8)

    else:
        if Nrows > 2:
            i -= Ncols

        axes[i].legend(handles, labels, bbox_to_anchor=(1.025, 0.75), loc=2,
                       handlelength=0.8)

    while len(keep) < len(axes):

        axes[len(keep) - len(axes)].axis('off')
        keep += ['fill']

    fig.savefig(figname)
    plt.close()

    return


def iWUE_relationships(df, figname, keep):

    """
    Plots both Ci(gs) clusters and gs(LWP) functional shapes

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    keep: list
        sites x species to plot

    Returns:
    --------
    'iWUE_subsel.jpg'

    """

    # decide on how many columns and rows
    Nrows = 2
    Ncols = 0

    while Nrows * Ncols < 2 * len(keep):

        Ncols += 1

    # declare figure
    fig, axes = plt.subplots(Nrows, Ncols, figsize=(Nrows + 2, Ncols + 2))
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    axes = axes.flat  # flatten axes

    # functional relationships
    for i in range(len(axes)):

        if i % 2 == 0:
            what = keep[0]

        else:
            what = keep[1]

        sub = df.copy()[df['site_spp'] == what]

        if i >= Ncols:  # plot the obs on second row
            ipath = os.path.join(os.path.join(os.path.join(get_main_dir(),
                                 'input'), 'simulations'), 'obs_driven')
            ref, __ = read_csv(os.path.join(ipath,
                                            '%s_calibrated.csv' % (what)))
            Pcrit = get_Px(ref.loc[0, 'P50'], ref.loc[0, 'P88'], 50., 88., 95.)

            valid = sub['gs'] < sub['gs'].max()
            xobs = sub['Pleaf'][valid] / Pcrit
            yobs = sub['gs'][valid] / sub['gs'].max()
            axes[i].scatter(xobs, yobs, marker='+', s=125. / Nrows, lw=1.25,
                            facecolor='#c0c0c0', edgecolor='none', alpha=0.6,
                            label='Obs.')

        left = 1.
        bottom = 1.

        for mod in model_order():

            if i < Ncols:
                x = sub['Ci(%s)' % (mod)] / sub['Ci']
                y = sub['gs(%s)' % (mod)] / sub['gs']

                # deal with the actual nans
                valid = np.logical_and(sub['Ci(%s)' % (mod)] < 9999.,
                                       sub['gs(%s)' % (mod)] < 9999.)

            else:
                x = sub['Pleaf(%s)' % (mod)] / Pcrit
                y = (sub['gs(%s)' % (mod)] /
                     sub['gs(%s)' % (mod)][sub['gs(%s)' % (mod)] < 9999.]
                     .max())

                # deal with the actual nans
                valid = np.logical_and(sub['Pleaf(%s)' % (mod)] < sub['Ps'],
                                       np.logical_and(x < 1.,
                                                      sub['gs(%s)' % (mod)]
                                                      < 9999.))

            x = x[valid]
            y = y[valid]

            try:
                c = next(axes[i]._get_lines.prop_cycler)['color']

                if i < Ncols:  # plot interquartile Ci - gs ranges
                    valid = np.logical_and(x / y > (x / y).quantile(0.25),
                                           x / y < (x / y).quantile(0.75))
                    encircle(axes[i], x[valid], y[valid], smooth=True, ec=c,
                             fc='None', alpha=0.9,
                             lw=plt.rcParams['lines.linewidth'])

                    # tighten the subplots
                    left = np.minimum(left, x[valid].min())
                    bottom = np.minimum(bottom, y[valid].min())

                elif mod != 'std':  # plot functional gs - LWP
                    if x.min() < 0.95:
                        try:  # plot average obs behaviour
                            gam = (ExpectileGAM(expectile=0.5, n_splines=5,
                                                spline_order=4,
                                                constraints='monotonic_dec')
                                   .gridsearch(x.values.reshape(-1, 1),
                                               y.values))
                            px = np.linspace(x.min(),
                                             np.maximum(xobs.max(), x.max()),
                                             num=500)
                            py = gam.predict(px)

                            check = np.gradient(py, px)[np.gradient(py, px)
                                                        < 0.]

                            if (np.isnan(np.nanmean(check)) or
                               (np.nanmean(check) > -0.3)):
                                ms = 30.
                                alpha = 0.9

                                raise Exception

                            axes[i].plot(px, py, color=c, alpha=0.9)
                            ms = 5.
                            alpha = 0.5

                        except Exception:
                            pass

                    # plot the scatters for the curves that failed
                    axes[i].scatter(x, y, s=ms, c=c, alpha=alpha)

            except Exception:
                pass

        if i < Ncols:
            axes[i].hlines(1., 0., 1., linestyle=':', linewidth=1.,
                           transform=axes[i].get_yaxis_transform())
            axes[i].vlines(1., 0., 1., linestyle=':', linewidth=1.,
                           transform=axes[i].get_xaxis_transform())

            axes[i].set_xlabel(r'$C_i$ $(sim. : obs.)$')

            if i % 2 == 0:
                axes[i].set_ylabel(r'$g_s$ $(sim. : obs.)$')

            species = r'%s %s' % (what.split('_')[-2], what.split('_')[-1])

            if 'Quercus' in species:
                title = r'\textit{%s} (%s)' % (species, what.split('_')[0][0])

            else:
                title = r'\textit{%s}' % (species)

            axes[i].set_title(title)

        else:  # add P50 info line
            axes[i].vlines(-ref.loc[0, 'P50'] / Pcrit, 0., 1., linestyle=':',
                           linewidth=1.,
                           transform=axes[i].get_xaxis_transform())
            txt = axes[i].annotate('P$_{50}$',
                                   xy=(-ref.loc[0, 'P50'] / Pcrit - 0.02,
                                       0.98), ha='right')
            txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                         alpha=0.8))

            axes[i].set_xlabel(r'$\Psi$$_{l, norm}$')

            if i % 2 == 0:
                axes[i].set_ylabel(r'$g_{s, norm}$')

        # subplot labelling
        axes[i].text(0.9, 0.025,
                     r'\textbf{(%s)}' % (string.ascii_lowercase[i]),
                     transform=axes[i].transAxes, weight='bold')

        # tighten the subplots
        left2, right = axes[i].get_xlim()
        __, top = axes[i].get_ylim()

        if i < Ncols:
            axes[i].set_xlim(np.maximum(0.05, left - 0.1), right)
            axes[i].set_ylim(np.maximum(0.05, bottom - 0.1), top)
            xf = '%.1f'  # format axes ticks

        else:
            axes[i].set_xlim(np.maximum(0.05, left2), np.minimum(right, 1.05))
            axes[i].set_ylim(0., 1.05)
            xf = '%.2f'

        axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axes[i].set_xticklabels(axes[i].get_xticks())  # force LaTex
        axes[i].set_yticklabels(axes[i].get_yticks())  # force LaTex
        axes[i].xaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter(xf))
        axes[i].yaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))

    # add legend
    handles, labels = axes[-1].get_legend_handles_labels()
    handles[0].set_alpha(1)
    handles = [Line2D([0], [0], c=c, alpha=0.9) for c in
               plt.rcParams['axes.prop_cycle'].by_key()['color']] + handles
    labels = [which_model(mod) for mod in model_order()] + labels
    axes[-1].legend(handles, labels, bbox_to_anchor=(1.025, 0.4), loc=3)

    fig.savefig(figname)
    plt.close()

    return


def VPD_closure(df, figname, keep):

    """
    Plots fitted modelled functional relationships of gs closure
    depending with VPD

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    keep: list
        sites x species to plot

    Returns:
    --------
    'VPD_closure.jpg'

    """

    # declare figure
    fig = plt.figure(figsize=(6., 3.))
    gs = fig.add_gridspec(nrows=len(model_order()), ncols=12, wspace=0.15)
    ax = fig.add_subplot(gs[:, :5])

    # generic operation for both subplots
    for what in keep:

        where = df[df['site_spp'] == what].index

        # limit the dry soil effects
        wherewhere = (df.loc[where][df.loc[where, 'Ps'] <
                                    df.loc[where, 'Ps'].quantile(2. / 3.)]
                        .index)
        df.loc[wherewhere] = np.nan

    df = df.dropna(how='all')

    all = df.copy()

    for what in keep:  # behaviour relative to the obs

        where = all[all['site_spp'] == what].index

        for mod in model_order():

            valid = all.loc[where, 'gs(%s)' % (mod)] < 9999.
            nans = all.loc[where, 'gs(%s)' % (mod)][~valid].index
            all.loc[nans, 'gs(%s)' % (mod)] = np.nan
            all.loc[where, 'gs(%s)' % (mod)] /= all.loc[where, 'gs'].max()

        all.loc[where, 'gs'] /= all.loc[where, 'gs'].max()

    # deal with the nans
    all = all.select_dtypes(exclude=['object'])
    all.replace(9999., np.nan, inplace=True)
    all.sort_values(by=['VPD'], inplace=True)
    all.reset_index(drop=True, inplace=True)

    # bin by VPD
    all['distri'] = pd.cut(all['VPD'], bins=16)
    all = all.dropna(how='all')
    all_max = all.groupby(['distri']).max()
    all_min = all.groupby(['distri']).min()
    all = all.groupby(['distri']).mean()

    # find the min/max values and plot
    all_max = all_max[['VPD', 'gs']].dropna(how='all')
    all_min = all_min[['VPD', 'gs']].dropna(how='all')
    all_plot = all[['VPD', 'gs']].dropna(how='all')
    all_plot['VPD'].iloc[0] = all_min['VPD'].min()
    all_plot['VPD'].iloc[-1] = all_max['VPD'].max()

    # draw a box around the area
    ax.hlines(0.3, 2., all_max['VPD'].max(), linestyle='--', linewidth=2.,
              zorder=20)
    ax.vlines(2., 0., 0.3 + 0.0065, linestyle='--', linewidth=2., zorder=20)
    ax.vlines(all_max['VPD'].max(), 0., 0.3 + 0.0065, linestyle='--',
              linewidth=2., zorder=20)

    # draw arrow
    arrow = '->, head_length=0.5, head_width=0.3'
    ax.annotate('High $D_a$\nclosure', ha='center', va='center',
                xy=(3.5, 0.32), xytext=(4.5, 0.5),
                arrowprops=dict(lw=1.5, fc='k', arrowstyle=arrow,
                                connectionstyle='arc3, rad=-0.2'))

    # smooth the lines
    lowess = sm.nonparametric.lowess
    all_plot = all[['VPD', 'gs']].copy()
    all_plot['gs'] = lowess(all_plot['gs'], all_plot['VPD'], frac=0.5,
                            return_sorted=False)
    all_plot['VPD'].iloc[0] = all_min['VPD'].min()
    all_plot['VPD'].iloc[-1] = all_max['VPD'].max()
    ax.fill_between(all_plot['VPD'],
                    np.maximum(0., all_plot['gs'] - all_plot['gs'].std()),
                    np.minimum(all_plot['gs'] + all_plot['gs'].std(), 1.),
                    color='#c0c0c0', alpha=0.5)
    ax.plot(all_plot['VPD'], all_plot['gs'], lw=10., color='#c0c0c0',
            alpha=0.6)

    for mod in model_order():

        c = next(ax._get_lines.prop_cycler)['color']
        all_plot = all[['VPD', 'gs(%s)' % (mod)]].copy()
        all_plot['gs(%s)' % (mod)] = lowess(all_plot['gs(%s)' % (mod)],
                                            all_plot['VPD'], frac=0.5,
                                            return_sorted=False)
        all_plot['VPD'].iloc[0] = all_min['VPD'].min()
        all_plot['VPD'].iloc[-1] = all_max['VPD'].max()
        ax.plot(all_plot['VPD'], all_plot['gs(%s)' % (mod)], color=c,
                label=which_model(mod))

    # tighten
    ax.set_xlim(all_min['VPD'].min(), all_max['VPD'].max() + 0.1)
    ax.set_ylim(0., 0.8)

    # format axes ticks
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.set_xticklabels(ax.get_xticks())  # force LaTex
    ax.set_yticklabels(ax.get_yticks())  # force LaTex
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # subplot labelling
    ax.text(0.025, 0.98, r'\textbf{(%s)}' % (string.ascii_lowercase[0]),
            transform=ax.transAxes, weight='bold')

    # axes labels
    ax.set_ylabel(r'$g_{s,norm}$')
    render_xlabels(ax, r'$D_a$', 'kPa')
    ax.xaxis.labelpad = 5

    # now add relative rates of closure on the second axis
    high = df.copy()

    for what in keep:  # behaviours relative to themselves

        where = high[high['site_spp'] == what].index

        for mod in model_order():

            valid = high.loc[where, 'gs(%s)' % (mod)] < 9999.
            nans = high.loc[where, 'gs(%s)' % (mod)][~valid].index
            high.loc[nans, 'gs(%s)' % (mod)] = np.nan

            # scaling the magnitudes to ask about % diffence closure
            high.loc[where, 'gs(%s)' % (mod)] /= \
                high.loc[where, 'gs(%s)' % (mod)].max()
            high.loc[where, 'gs(%s)' % (mod)] *= high.loc[where, 'gs'].max()

    high = high[high['VPD'] > 2.]  # VPD effect
    high.set_index('VPD', inplace=True)
    high = high[high.filter(like='gs').columns.to_list()]
    high = high[high <= 0.3]

    # relative closure difference
    high.loc[:, high.columns != 'gs'] = (high.loc[:, high.columns != 'gs']
                                             .sub(high['gs'], axis=0)
                                             .divide(high['gs'], axis=0))
    high = high.dropna(how='all')

    models = ['cmax', 'cgn', 'sox2', 'pmax', 'mes', 'wue', 'std', 'lcst',
              'pmax2', 'cap', 'sox1', 'tuz']
    colours = [plt.rcParams['axes.prop_cycle'].by_key()['color']
               [model_order().index(e)] for e in models]

    for i, mod in enumerate(models):

        if i == 0:
            ax = fig.add_subplot(gs[i, 5:])

            # subplot labelling
            ax.text(0.025, 0.92,
                    r'\textbf{(%s)}' % (string.ascii_lowercase[1]),
                    transform=ax.transAxes, weight='bold')

        else:
            ax = fig.add_subplot(gs[i, 5:], sharey=ax)

        # plotting the distribution
        kde = high['gs(%s)' % (mod)].plot.kde(ax=ax, lw=0.5, color='k',
                                              alpha=0.9)

        # grabbing x and y data from the kde plot
        x = kde.get_children()[0]._x
        y = kde.get_children()[0]._y

        # filling the space beneath the distribution
        c = colours[i]
        ax.fill_between(x, y, color=c)

        # adding a horizontal marker with diff to zero
        ax.hlines(y[np.argmax(y)], 0., x[np.argmax(y)], lw=0.75,
                  color='#5a6576', zorder=20)
        ax.scatter(x[np.argmax(y)], y[np.argmax(y)], s=15.,
                   facecolor='#ececec', edgecolor='#5a6576', zorder=20)

        # adding a vertical '0' marker
        if i == 0:
            ax.vlines(0, 0, 0.65, transform=ax.get_xaxis_transform(), lw=0.75,
                      color='#5a6576', zorder=20)

        else:
            ax.vlines(0, 0, 0.8, transform=ax.get_xaxis_transform(), lw=0.75,
                      color='#5a6576', zorder=20)

        # make background transparent
        ax.patch.set_alpha(0)

        # remove borders, axis ticks, and labels
        if i < len(models) - 1:
            plt.tick_params(top=False, bottom=False, left=False, right=False,
                            labelleft=False, labelbottom=False)
            no_spines = ['top', 'right', 'left', 'bottom']

        else:
            plt.tick_params(top=False, left=False, right=False,
                            labelleft=False)
            no_spines = ['top', 'right', 'left']
            ax.spines['bottom'].set_bounds(-2., 6.)

        for s in no_spines:

            ax.spines[s].set_visible(False)

        # add model name
        txt = ax.text(1.025, 0.075, which_model(mod), ha='right',
                      transform=ax.transAxes)
        txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                          alpha=0.8))

        # tighten
        ax.set_xlim(-3., 7.)
        __, top = ax.get_ylim()
        ax.set_ylim(0, top + 0.02)

    # axes ticks and labels
    ax.set_xticklabels(ax.get_xticks())  # force LaTex
    ax.set_xlabel(r'$\Delta$ $g_{s,norm,sc.}$')
    ax.xaxis.labelpad = 5

    gs.update(hspace=-0.725)  # further tighten
    fig.savefig(figname)
    plt.close()

    return


def LWP_error_plots(df, figname, groups):

    """
    Plots boxes of distributions of residuals between modelled and
    observed LWP, where the plots are grouped by ecosystem type

    Arguments:
    ----------
    df: pandas dataframe
        dataframe that contains all the observed plus site x species
        simulations

    figname: string
        name of the figure to produce, including path

    groups: list
        ecosystems to consider for the groupings

    Returns:
    --------
    'LWP_errors.jpg'

    """

    # declare figure
    fig, axes = plt.subplots(3, 2, figsize=(4.75, 6))
    plt.subplots_adjust(hspace=0.1, wspace=0.25)
    axes = axes.flat  # flatten axes

    select = df.filter(like='Pleaf').columns.to_list()

    for i, what in enumerate(groups):

        if what == 'Panama':
            s1 = 'San_Lorenzo'
            s2 = 'Parque_Natural_Metropolitano'
            sub = df[df['site_spp'].str.contains('|'.join([s1, s2]))].copy()

        else:
            sub = df[df['site_spp'].str.contains(what)].copy()

        for c in select:

            sub.loc[sub[c] >= 0., c] = np.nan  # mask nonsense
            sub.loc[sub[c] < -999., c] = np.nan  # mask nonsense

        range = (sub.groupby('site_spp')['Pleaf'].min() -
                 sub.groupby('site_spp')['Pleaf'].max()).abs().max()
        axes[i].fill_between(np.arange(-0.5, len(model_order()) * 2.), -range,
                             range, facecolor='#c0c0c0', edgecolor='none',
                             alpha=0.2)
        axes[i].fill_between(np.arange(-0.5, len(model_order()) * 2.),
                             -0.1 * range, 0.1 * range, facecolor='#c0c0c0',
                             edgecolor='none', alpha=0.3)

        pos = 0.
        next(axes[i]._get_lines.prop_cycler)

        for mod in model_order()[1:]:
            c = next(axes[i]._get_lines.prop_cycler)['color']

            try:

                for site_spp in sub['site_spp'].unique():

                    valid = np.logical_and(sub['site_spp'] == site_spp,
                                           sub['Pleaf(%s)' % (mod)] <
                                           1.01 * sub['Ps'])
                    LWP = (((sub['Pleaf(%s)' % (mod)] - sub['Pleaf'])[valid])
                           .dropna().to_list())
                    bp = axes[i].boxplot(LWP, positions=[pos],
                                         widths=0.9 / len(sub['site_spp']
                                                          .unique()),
                                         showcaps=False, patch_artist=True)

                    if mod == 'std':
                        set_box_colour(bp, c, lc='#c0c0c0')

                    else:
                        set_box_colour(bp, c)

                    pos += 0.95 / len(sub['site_spp'].unique())

            except Exception:
                pass

            pos += 0.5

        if i % 2 == 0:
            render_ylabels(axes[i], r'$\Delta$$\Psi_l$', 'MPa')

        axes[i].set_xlim(-0.6, pos - 0.4)
        axes[i].set_yscale('symlog')

        # axes ticks
        yticks = [-round(range * 2) / 2., -1.5, 0., 1.5, round(range * 2) / 2.]

        if i > 0:
            min = (sub.filter(like='Pleaf').sub(sub['Pleaf'], axis=0).min()
                   .min())
            yticks = [round(min), ] + yticks

        axes[i].set_yticks(yticks)
        axes[i].yaxis.set_major_formatter(mpl.ticker
                                             .FormatStrFormatter('%.1f'))

        # remove xticks
        axes[i].tick_params(axis='x', which='both', bottom=False,
                            labelbottom=False)
        axes[i].tick_params(axis='y', which='minor', left=False)

        # set the subplot's title
        if what == 'Panama':
            what = (r'\textbf{(%s)} Neotropical rainforest'
                    % (string.ascii_lowercase[i]))

        if what == 'ManyPeaksRange':
            what = (r'\textbf{(%s)} Dry tropical rainforest'
                    % (string.ascii_lowercase[i]))

        if what == 'Quercus':
            what = (r'\textbf{(%s)} Mediterranean forest'
                    % (string.ascii_lowercase[i]))

        if what == 'Eucalyptus':
            what = (r'\textbf{(%s)} \textit{Eucalyptus}'
                    % (string.ascii_lowercase[i]))

        if what == 'Sevilleta':
            what = (r'\textbf{(%s)} Piñon-Juniper woodland'
                    % (string.ascii_lowercase[i]))

        txt = axes[i].annotate(what, xy=(0.99, 0.985),
                               xycoords='axes fraction', ha='right', va='top')
        txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                     alpha=0.8))

    # add legend
    axes[-1].axis('off')
    handles = [Patch(facecolor=c, edgecolor='none')
               for c in plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]]
    labels = [which_model(mod) for mod in model_order()[1:]]
    axes[-2].legend(handles, labels, bbox_to_anchor=(1.15, 0.9), loc=2, ncol=2,
                    handleheight=0.7, labelspacing=0.5)

    fig.savefig(figname)
    plt.close()

    return


# ======================================================================

if __name__ == "__main__":

    # user input
    subsel1 = ['San_Lorenzo_Tachigali_versicolor',
               'ManyPeaksRange_Austromyrtus_bidwillii',
               'Vic_la_Gardiole_Quercus_ilex', 'Corrigin_Eucalyptus_capillosa']
    subsel2 = ['San_Lorenzo_Tachigali_versicolor',
               'Vic_la_Gardiole_Quercus_ilex']
    groups = ['Panama', 'ManyPeaksRange', 'Eucalyptus', 'Quercus', 'Sevilleta']

    main(subsel1=subsel1, subsel2=subsel2, groups=groups)
