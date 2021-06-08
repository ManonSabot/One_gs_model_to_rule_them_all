# general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as co
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory


class plt_setup(object):

    def __init__(self):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 600.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # figure spacing
        plt.rcParams['figure.subplot.wspace'] = 0.025

        # colors
        #plt.rcParams['axes.facecolor'] = '#737373'

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


def model_order():

    return ['std', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'sox2',
            'pmax2', 'lcst', 'cap', 'mes']


def which_model(short):

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


def cmap_specs(df, what):

    cmap = co.cm.curl

    # extra info
    try:
        min = np.nanmin(df.drop('variable', axis=1))
        max = np.nanmax(df.drop('variable', axis=1))
        mean = np.nanmean(df.drop('variable', axis=1))
        med = np.nanmedian(df.drop('variable', axis=1))

    except Exception:
        min = np.nanmin(df)
        max = np.nanmax(df)
        mean = np.nanmean(df)
        med = np.nanmedian(df)

    if what == 'MASE':  # on one
        loc_white = np.abs(min - 1.) / (max - min)

    elif what == 'rBIC':  # on 0.5
        loc_white = np.abs(min - 0.5) / (max - min)

    else:  # on zero
        cmap = co.cm.curl_r
        loc_white = np.abs(min) / (max + np.abs(min))

    cmap = shift_cmap(cmap, locpoint=loc_white)

    return cmap, min, max


def cbar_specs(what, df):

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    start = 0.  # start for smooth

    if what == 'NSE':
        N1 = 200
        N2 = 600
        cticks = np.array([min, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, max])

    elif what == 'MASE':
        N1 = 600
        N2 = 200
        start = 1.
        cticks = np.array([min, 0.7, 0.9, 1.1, 1.3, 1.5, max])

    elif what == 'R':
        N1 = 200
        N2 = 600
        start = 0.1
        cticks = np.array([min, 0., 0.2, 0.4, 0.6, 0.8, max])

    elif what == 'rBIC':
        N1 = 600
        N2 = 200
        start = 0.5
        cticks = np.array([min, 0.1, 0.3, 0.5, 0.7, max])

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
        metrics = ['NSE', 'R', 'MASE', 'rBIC']
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
                orders += [df[mask]['mean'].sort_values()]

        else:
            orders += [df[df['variable'] == to_plot[i]]['mean'].sort_values()]

    df.drop('mean', axis=1, inplace=True)

    # info needed by shared colormap, limit extremes
    if what == 'All':
        csel =  df.columns.difference(['variable', 'metric'])

    else:
        csel =  df.columns.difference(['variable'])

    df.loc[:, csel] = df.loc[:, csel].where(df.loc[:, csel] < 2., 2.)
    df.loc[:, csel] = df.loc[:, csel].where(df.loc[:, csel] > -2., -2.)

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
                cmap, vmin, vmax = cmap_specs(sub[sub['metric'] == metric]
                                                 .drop('metric', axis=1),
                                              metric)
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
                    if metric == 'R':
                        metric = r'$r$'

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

            cax = fig.add_axes([(j + 1) * 0.15 + 0.04825 * j, 0.08, 0.13, 0.02])
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


def skill_overview(df, fname):

    fig, axes = plt.subplots(2, 2, figsize=(5, 8))
    plt.subplots_adjust(hspace=0.015, wspace=0.15)
    axes = axes.flat

    for i, metric in enumerate(df['metric'].unique()):

        if i == 0:
            axes[i].xaxis.set_ticks_position('top')

        elif i % 2 != 0:
            axes[i].set_yticks([])

            if i < 2:
                axes[i].xaxis.set_ticks_position('top')

            axes[i] = axes[i].twinx()

        # select the relevant data
        all = df[df['metric'] == metric].drop(['metric', 'variable'], axis=1)
        noPsi = (df[np.logical_and(df['metric'] == metric,
                                   df['variable'] != 'Pleaf')]
                   .drop(['metric', 'variable'], axis=1))
        gs = (df[np.logical_and(df['metric'] == metric, df['variable'] == 'gs')]
                .drop(['metric', 'variable'], axis=1))

        # Medlyn cannot plot 'all' because of Pleaf
        all.loc['std'] = np.nan

        # we must first transform NSEs to NNSEs due to inf bound
        if metric == 'NSE':
            all = 1. / (2. - all)
            noPsi = 1. / (2. - noPsi)
            gs = 1. / (2. - gs)

        data = np.zeros(len(model_order()) * 3)
        SEs = np.zeros(len(model_order()) * 3)
        colours = np.asarray(['w',] * len(model_order()) * 3,
                             dtype=np.object)
        col = ['#53e3d4', '#415381', '#e2694e', '#ecec3a']

        for j, mod in enumerate(model_order()):

            # put the bigger out of the three in first place, etc
            values = [np.nanmean(gs.loc[mod].values),
                      np.nanmean(noPsi.loc[mod].values),
                      np.nanmean(all.loc[mod].values)]
            errors = (np.array([np.nanstd(gs.loc[mod].values, ddof=1),
                                np.nanstd(noPsi.loc[mod].values, ddof=1),
                                np.nanstd(all.loc[mod].values, ddof=1)]) /
                      (len(model_order()) ** 0.5))

            if metric == 'NSE':  # transform back
                values = [2. - 1. / e for e in values]
                errors = (np.array([np.nanstd(2. - 1. / gs.loc[mod].values,
                                              ddof=1),
                                    np.nanstd(2. - 1. / noPsi.loc[mod].values,
                                              ddof=1),
                                    np.nanstd(2. - 1. / all.loc[mod].values,
                                              ddof=1)]) /
                          (len(model_order()) ** 0.5))

            for k in range(len(values)):

                data[j + k * len(model_order())] = values[k]
                SEs[j + k * len(model_order())] = errors[k]
                colours[j + k * len(model_order())] = col[k]

            if mod == 'std':
                data[j + 2 * len(model_order())] = np.nan
                SEs[j + 2 * len(model_order())] = np.nan

        # rearrange the order things are plotted in
        if (metric == 'NSE') or (metric == 'rBIC'):
            ascend = False

        else:
            ascend = True

        if metric == 'MASE':
            print('meeting MASE:', gs.mean(axis=1)[gs.mean(axis=1) < 1.01])

        # first, figure out order relative to Medlyn
        order1 = ((gs.mean(axis=1) +
                   noPsi.reset_index().groupby('model').mean().mean(axis=1))
                  .sort_values(ascending=ascend)).index.to_list()

        # then figure out order on each side of Medlyn
        order2 = ((gs.mean(axis=1) +
                   all.reset_index().groupby('model').mean().mean(axis=1))
                  .sort_values(ascending=ascend)).index.to_list()
        left = [e for e in order2 if e in order1[:order1.index('std')]]
        right = [e for e in order2 if e in order1[order1.index('std') + 1:]]

        # final order
        order = left + ['std'] + right
        idx = [model_order().index(e) for e in order]

        # apply new order
        for j in range(3):

            data[j * len(model_order()): (j + 1) * len(model_order())] = \
                 data[j * len(model_order()): (j + 1) * len(model_order())][idx]
            SEs[j * len(model_order()): (j + 1) * len(model_order())] = \
                  SEs[j * len(model_order()): (j + 1) * len(model_order())][idx]
            colours[j * len(model_order()): (j + 1) * len(model_order())] = \
              colours[j * len(model_order()): (j + 1) * len(model_order())][idx]

        pos = np.tile(np.arange(float(len(model_order()))), 3)
        pos[len(model_order()):] -= 0.2
        pos[2 * len(model_order()):] -= 0.2

        for j, e in enumerate(data):
            axes[i].errorbar(data[j], pos[j], xerr=SEs[j], fmt='o',
                             color=colours[j], alpha=0.8)

        if i < 2:
            if i % 2 == 0:
                x = 0.

            else:
                x = 1.

            axes[i].axvline(x=x, ymin=0.025, ymax=0.975, c=col[-1], ls='--',
                            lw=2., zorder=0)

        # bound the subplots for readability
        if i == 0:
            axes[i].set_xlim(-0.6, 0.6)

        if i == 1:
            axes[i].set_xlim(0.8, 3.5)

        if i == 2:
            __, max = axes[i].get_xlim()
            axes[i].set_xlim(0.25, max)

        ticks = pos[:len(model_order())] - 0.15
        ticks[-1] -= 0.2 / 3.
        axes[i].set_yticks(ticks)

        if i % 2 != 0:
            axes[i].set_yticklabels([which_model(e) for e in order], ha='left')
            axes[i].invert_xaxis()

        else:
            axes[i].set_yticklabels([which_model(e) for e in order], ha='right')

        if i == 3:
            axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

        else:
            axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))

        axes[i].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # add perf arrow
    axes[1].annotate('Less skill', xy=(-0.1, 0.05), xytext=(-0.1, 0.95),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.))
    axes[-1].annotate('Less skill', xy=(-0.1, 0.95), xytext=(-0.1, 0.05),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.))
    axes[-1].text(-0.1, 1., 'More skill', va='center', ha='center',
                  transform=axes[-1].transAxes)
    axes[-1].annotate('', xy=(-0.3, 1.), xytext=(-1.15, 1.),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.))
    axes[-1].annotate('', xy=(0.95, 1.), xytext=(0.1, 1.),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.))

    # add subplot numbering in box descriptor
    axes[1].annotate('', xy=(1.8, 0.9), xycoords='axes fraction',
                     xytext=(1.65, 0.8),
                     arrowprops=dict(arrowstyle='-',
                     connectionstyle='angle,angleA=90,angleB=180,rad=0'))
    axes[1].annotate('', xy=(1.65, 1.), xycoords='axes fraction',
                     xytext=(1.5, 0.9),
                     arrowprops=dict(arrowstyle='-',
                     connectionstyle='angle,angleA=180,angleB=90,rad=0'))
    axes[1].text(1.605, 0.95, r'\textbf{(a)}', transform=axes[1].transAxes,
                 ha='right', va='bottom', weight='bold')
    axes[1].text(1.7, 0.95, r'\textbf{(b)}', transform=axes[1].transAxes,
                 ha='left', va='bottom', weight='bold')
    axes[1].text(1.605, 0.85, r'\textbf{(c)}', transform=axes[1].transAxes,
                 ha='right', va='top', weight='bold')
    axes[1].text(1.7, 0.85, r'\textbf{(d)}', transform=axes[1].transAxes,
                 ha='left', va='top', weight='bold')

    leg = [Line2D([0], [0], marker='o', color=col[0], alpha=0.8, label=r'$g_s$'),
           Line2D([0], [0], marker='o', color=col[1], alpha=0.8,
                  label=r'No $\Psi$$_l$'),
           Line2D([0], [0], marker='o', color=col[2], alpha=0.8, label='All')]
    axes[1].legend(handles=leg, loc=2, frameon=False,
                    bbox_to_anchor=[1.475, 0.775])

    fig.savefig(fname)


# PLOT
what = 'NSE'
to_plot = ['gs']  #, 'E', 'A']

plt_setup()  # rendering

# Import Data
base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'obs_driven')
ofdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

if what == 'All':

    for i, metric in enumerate(['NSE', 'MASE', 'R', 'rBIC']):

        df = pd.read_csv(os.path.join(ifdir, 'all_%ss.csv' % (metric)))
        df['metric'] = metric

        if i == 0:
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


fname = os.path.join(ofdir, 'summary_of_skill.jpg')

if not os.path.isfile(fname):
    skill_overview(df[['metric', 'variable'] + order], fname)

# make the figure
fname = os.path.join(ofdir, '%s_skill_for_%s.jpg' % (what, '_'.join(to_plot)))

if not os.path.isfile(fname):
    if what == 'All':
        if len(to_plot) < 3:
            to_plot = ['gs', 'E', 'A']

        heatmap(df[['mean', 'metric', 'variable'] + order], fname, to_plot,
                what=what)

    else:
        heatmap(df[['mean', 'variable'] + order], fname, to_plot, what=what)
