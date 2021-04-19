# general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory


class plt_setup(object):

    def __init__(self):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # figure spacing
        plt.rcParams['figure.subplot.wspace'] = 0.025

        # colors
        plt.rcParams['axes.facecolor'] = '#737373'

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

        # ticks
        plt.rcParams['xtick.major.size'] = 0.
        plt.rcParams['ytick.major.size'] = 0.
        plt.rcParams['xtick.major.pad'] = 2
        plt.rcParams['ytick.major.pad'] = 6

        # axes limits
        plt.rcParams['axes.autolimit_mode'] = 'data'

        # spines
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.bottom'] = False
        plt.rcParams['axes.spines.top'] = False


def shift_cmap(cmap, start=0., locpoint=0.5, stop=1.0, name='centered'):

    """
    Shift the colours to associate a value standing anywhere in the new cmap
    (relatively to the two extremes start & stop or min & max) with whichever
    value / colour of the input cmap (by default the midpoint).
    If the input cmap is divergent, this will be white by default.
    The locpoint value cannot be the min or max (start or stop).

    """

    # declare a colour + transparency dictionary
    cdict={'red':[], 'green':[], 'blue':[], 'alpha':[]}

    # regular index to compute the colors
    RegInd = np.linspace(start, stop, cmap.N)

    # shifted index to match what the data should be centered on
    ShiftInd = np.hstack([np.linspace(0., locpoint, int(cmap.N / 2),
                                      endpoint=False),
                          np.linspace(locpoint, 1., int(cmap.N / 2))])

    # associate the regular cmap's colours with the newly shifted cmap colour
    for RI, SI in zip(RegInd, ShiftInd):

        # get standard indexation of red, green, blue, alpha
        r, g, b, a = cmap(RI)

        cdict['red'].append((SI, r, r))
        cdict['green'].append((SI, g, g))
        cdict['blue'].append((SI, b, b))
        cdict['alpha'].append((SI, a, a))

    return LinearSegmentedColormap(name, cdict)


def which_model(short):

    if short == 'std2':
        lab = r'Medlyn-$f_{\varPsi_{l,pd}}$'

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

    elif short == 'pmax2':
        lab = 'ProfitMax2'

    elif short == 'cgn':
        lab = 'CGain'

    elif short == 'lcst':
        lab = 'LeastCost'

    elif short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

    elif short == 'cap':
        lab = 'CAP'

    elif short == 'mes':
        lab = 'MES'

    else:
        lab = None

    return lab


def cmap_specs(df, what):

    cmap = cc.cm.CET_D1A

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    mean = np.nanmean(df.drop('variable', axis=1))
    med = np.nanmedian(df.drop('variable', axis=1))

    if (what == 'MASE') or  (what == 'NMSE'):  # on one
        loc_white = np.abs(min - 1.) / (max - min)

    elif what == 'rBIC':  # on 0.5
            loc_white = np.abs(min - 0.5) / (max - min)

    else:  # on zero
        cmap = cc.cm.CET_D1A_r
        loc_white = np.abs(min) / (max + np.abs(min))

    cmap = shift_cmap(cmap, locpoint=loc_white)

    return cmap, min, max


def cbar_specs(what, df):

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    start = 0.  # start for smooth

    if (what == 'NSE') or (what == 'KGE'):
        N1 = 200
        N2 = 600
        cticks = np.array([min, -2, 0, 0.2, 0.4, 0.6, max])

        if what == 'KGE':
            cticks = np.array([min, -0.5, 0, 0.2, 0.4, 0.6, 0.8, max])

    elif (what == 'MASE') or (what == 'NMSE'):
        N1 = 600
        N2 = 200
        start = 1.
        cticks = np.array([min, 0.2, 0.4, 0.6, 0.8, 1., 1.4, 1.8, max])

    elif what == 'rBIC':
        N1 = 600
        N2 = 200
        start = 0.5
        cticks = np.array([min, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, max])

    smooth = (list(np.linspace(cticks[1], start, N1)) +
              list(np.linspace(start, cticks[-2], N2)))
    bounds = [cticks[0]] + smooth + [cticks[-1]]

    return cticks, bounds


def MAP_arrow(ax):

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.patch.set_visible(False)

    # plot arrow
    ax.text(0., 0., 'Xeric', va='center', ha='center', transform=ax.transAxes)
    ax.annotate('Wet', xy=(0., 0.025), xytext=(0., 1.),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='<->', lw=1.))

    return


def heatmap(df, fname, to_plot, what='NSE'):

    variables = np.array(['$g_s$', '$E$', '$A_n$'])[np.array(['gs' in to_plot,
                                                              'E' in to_plot,
                                                              'A' in to_plot])]

    if what == 'All':
        metrics = ['KGE', 'NSE', 'MASE', 'NMSE', 'rBIC']
        hms = []

    else:
        metrics = ['']

    # setup the plots
    if len(to_plot) > 1:
        if what == 'All':
            fig, axes = plt.subplots(len(to_plot), len(metrics),
                                     figsize=(12., 10.), sharey=True)
            plt.subplots_adjust(hspace=0.25, wspace=0.1)
            axes = axes.flat

        else:
            fig, axes = plt.subplots(1, len(to_plot),
                                     figsize=(4.5 * len(to_plot), 5.),
                                     sharey=True)

    else:
        fig, ax = plt.subplots(figsize=(5., 5.))
        axes = [ax]
        df = df[df['variable'] == to_plot[0]]

    # model perf order?
    orders = []

    for i in range(len(to_plot)):

        if what == 'All':

            for metric in metrics:

                mask = np.logical_and(df['variable'] == to_plot[i],
                                      df['metric'] == metric)

                if metric == 'MASE':
                    orders += [df[mask]['median'].sort_values()]

                else:
                    orders += [df[mask]['mean'].sort_values()]

        elif what == 'MASE':
            orders += [df[df['variable'] == to_plot[i]]['median'].sort_values()]

        else:
            orders += [df[df['variable'] == to_plot[i]]['mean'].sort_values()]

    if what == 'All':
        df.drop(['mean', 'median'], axis=1, inplace=True)

    elif what == 'MASE':
        df.drop('median', axis=1, inplace=True)

    else:
        df.drop('mean', axis=1, inplace=True)

    # info needed by shared colormap, limit extremes
    if what == 'All':

        for metric in metrics[:-1]:

            csel =  df.columns.difference(['variable', 'metric'])

            if (metric == 'KGE') or (metric == 'NSE'):
                df.loc[:, csel] = df.loc[:, csel].where(df.loc[:, csel] > -2.,
                                                        -2.)

            else:
                df.loc[:, csel] = df.loc[:, csel].where(df.loc[:, csel] < 2.,
                                                        2.)

    if (what == 'KGE') or (what == 'NSE'):
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] > -2., -2.))

    elif (what == 'MASE') or (what == 'NMSE'):
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] < 2., 2.))

    if what != 'All':
        cmap, vmin, vmax = cmap_specs(df, what)

    for i in range(len(to_plot)):

        # reorder the df based on model perf, only keep relevant var
        if len(to_plot) > 1:
            sub = df.copy()[df['variable'] == to_plot[i]]

        else:
            sub = df.copy()

        sub = sub.drop('variable', axis=1)

        for j, metric in enumerate(metrics):

            if what == 'All':
                cmap, vmin, vmax = cmap_specs(df[df['metric'] == metric]
                                                .drop('metric', axis=1), metric)
                sub2 = sub.copy()[sub['metric'] == metric]
                sub2 = sub2.drop('metric', axis=1)
                sub2 = sub2.reindex(index=orders[i * len(metrics) + j]
                                                .index.to_list())

            else:
                sub2 = sub.reindex(index=orders[i].index.to_list())

            # into the right formats
            models = [which_model(e) for e in sub2.index.to_list()]
            spps = [e.split('_') for e in sub2.columns.to_list()]
            spps = [r'\textit{%s. %s} (%s)' % (e[-2][0], e[-1],
                                               ' '.join(e[:-2])[0])
                    if 'Quercus' in e else r'\textit{%s. %s}' % (e[-2][0],
                                                                 e[-1])
                    for e in spps]
            data = sub2.to_numpy()
            data = data.T

            # plot the data
            if what == 'All':
                hm = axes[i * len(metrics) + j].imshow(data, alpha=0.8,
                                                       cmap=cmap, vmin=vmin,
                                                       vmax=vmax)
                hms += [hm]

                #  format the axes and labels
                axes[i * len(metrics) + j].set_xticks(np.arange(len(models))
                                                      - 0.25)
                axes[i * len(metrics) + j].set_xticklabels(models, rotation=45,
                                                           ha='left')
                axes[i * len(metrics) + j].xaxis.tick_top()
                axes[i * len(metrics) + j].set_yticks(np.arange(len(spps)))
                axes[i * len(metrics) + j].set_yticklabels(spps, ha='right')

                if i == 0:
                    axes[j].set_title(metric)

            else:
                hm = axes[i].imshow(data, alpha=0.8, cmap=cmap, vmin=vmin,
                                    vmax=vmax)

        if what != 'All':  #  format the axes and labels
            axes[i].set_xticks(np.arange(len(models)) - 0.25)
            axes[i].set_xticklabels(models, rotation=45, ha='left')
            axes[i].xaxis.tick_top()
            axes[i].set_yticks(np.arange(len(spps)))
            axes[i].set_yticklabels(spps, ha='right')

            if len(axes) > 1:  # add title below axis
                axes[i].set_title(r'%s for %s' % (what, variables[i]), y=-0.1,
                                  fontsize=10.)

    if what == 'All':  # add axes labels to know what is what

        for j, metric in enumerate(metrics):

            cax = fig.add_axes([j * 0.1575 + 0.136, 0.08, 0.125, 0.02])
            cticks, bounds = cbar_specs(metric, df[df['metric'] == metric]
                                                  .drop('metric', axis=1))
            cbar = plt.colorbar(hms[j], cax=cax, ticks=cticks, format='%.1f',
                                boundaries=bounds, drawedges=False,
                                extend='both', orientation='horizontal')
            cbar.outline.set_edgecolor('w')

    else:
        if len(axes) > 1:  # add colorbar
            cax = fig.add_axes([1. / len(axes), -0.025,
                                int(len(axes) / 2) / len(axes) +
                                plt.rcParams['figure.subplot.wspace'], 0.04])

        else:
            cax = fig.add_axes([0.22, 0.02, 0.585, 0.045])

        cticks, bounds = cbar_specs(what, df)
        cbar = plt.colorbar(hm, cax=cax, ticks=cticks, format='%.1f',
                            boundaries=bounds, drawedges=False, extend='both',
                            orientation='horizontal')
        cbar.outline.set_edgecolor('w')

        # add wet - xeric arrow
        if len(axes) > 1:
            ax = fig.add_axes([0.045, 0.148, 0.05, 0.688])

        else:
            ax = fig.add_axes([0.88, 0.135, 0.05, 0.715])

        MAP_arrow(ax)

    fig.savefig(fname)

    return


# PLOT
what = 'rBIC'
to_plot = ['gs']  #, 'E', 'A']

plt_setup()  # rendering

# Import Data
base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'obs_driven')
ofdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

if what == 'All':

    for metric in ['KGE', 'NSE', 'MASE', 'NMSE', 'rBIC']:

        df = pd.read_csv(os.path.join(ifdir, 'all_%ss.csv' % (metric)))
        df['metric'] = metric

        if metric == 'KGE':
            dfs = df.copy()

        else:
            dfs = pd.concat([dfs, df], ignore_index=True)

    df = dfs

else:
    df = pd.read_csv(os.path.join(ifdir, 'all_%ss.csv' % (what)))

# order by MAP: wet to dry
df.set_index('model', inplace=True)
order = ['San_Lorenzo_Carapa_guianensis', 'San_Lorenzo_Tachigali_versicolor',
         'San_Lorenzo_Tocoyena_pittieri',
         'Parque_Natural_Metropolitano_Calycophyllum_candidissimum',
         'ManyPeaksRange_Alphitonia_excelsa',
         'ManyPeaksRange_Austromyrtus_bidwillii',
         'ManyPeaksRange_Brachychiton_australis',
         'ManyPeaksRange_Cochlospermum_gillivraei',
         'Richmond_Eucalyptus_dunnii', 'Richmond_Eucalyptus_saligna',
         'Richmond_Eucalyptus_cladocalyx', 'Puechabon_Quercus_ilex',
         'Vic_la_Gardiole_Quercus_ilex','Corrigin_Eucalyptus_capillosa',
         'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis']

# make the figure
fname = os.path.join(ofdir, '%s_skill_for_%s.jpg' % (what, '_'.join(to_plot)))

if not os.path.isfile(fname):
    if what == 'All':
        if len(to_plot) < 3:
            to_plot = ['gs', 'E', 'A']

        heatmap(df[['mean', 'median', 'metric', 'variable'] + order], fname,
                to_plot, what=what)

    elif what == 'MASE':
        heatmap(df[['median', 'variable'] + order], fname, to_plot, what=what)

    else:
        heatmap(df[['mean', 'variable'] + order], fname, to_plot, what=what)
