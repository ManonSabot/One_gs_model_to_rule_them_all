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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LogNorm

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
        plt.rcParams['savefig.pad_inches'] = 0.05  # padding to use
        plt.rcParams['savefig.jpeg_quality'] = 100

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

    elif short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

    elif short == 'wue':
        lab = r'WUE-$f_{\varPsi_l}$'

    elif short == 'cgn':
        lab = 'CGain'

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


def model_perf(df, variable, what):

    if what == 'NSE':
        order = (df[df['variable'] == variable].drop('variable', axis=1)
                   .median(axis=1).sort_values())

    else:
        order = (df[df['variable'] == variable].drop('variable', axis=1)
                   .mean(axis=1).sort_values())

    return order


def cmap_specs(df):

    cmap = cc.cm.dimgray_r
    norm = None

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    mean = np.nanmean(df.drop('variable', axis=1))
    med = np.nanmedian(df.drop('variable', axis=1))

    if min < 0.:
        cmap = cc.cm.CET_D1A_r
        loc_zero = np.abs(min) / (max + np.abs(min))
        cmap = shift_cmap(cmap, locpoint=loc_zero)

    else:
        thresh = mean / med

        if (thresh < 0.8) or (thresh > 1.2):
            if (thresh < 0.1) or (thresh > 10.):
                norm = LogNorm()

            else:
                norm = PowerNorm(gamma=0.4)

    return cmap, norm, min, max


def cbar_specs(what, df):

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    start = 0.  # start for smooth

    if what == 'NSE':
        N1 = 50
        N2 = 150
        cticks = np.array([min, -2.5, -1., 0., 0.1, 0.2, 0.4, 0.6, max])

    elif what == 'log':
        N1 = 100
        N2 = 100
        cticks = np.sort(np.append([min, max],
                         np.log(np.array([0.1, 0.2, 0.5, 1., 2., 5.]))))

    elif what == 'MASE':
        N2 = 20
        start = 0.5
        cticks = np.array([min, start, 0.75, 1., 1.4, 2., max])

    elif what == 'MAPE':
        N2 = 200
        start = 0.25
        cticks = np.array([min, start, 0.5, 0.75, 1., 2., 4., max])

    elif what == 'SMAPE':
        N2 = 200
        start = 0.1
        cticks = np.array([min, start, 0.25, 0.5, 0.75, max])

    if what == 'RMSE':
        N2 = 200
        start = 0.1
        cticks = np.array([min, start, 0.25, 0.5, 1., 2., 4., 6., max])

    smooth = list(np.linspace(start, cticks[-2], N2))

    if min < 0.:
        smooth = list(np.linspace(cticks[1], 0., N1)) + smooth

    bounds = [cticks[0]] + smooth + [cticks[-1]]

    return cticks[1:], bounds


def MAP_arrow(ax):

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.patch.set_visible(False)

    # plot arrow
    ax.text(0., 0., 'Xeric', va='center', ha='center', transform=ax.transAxes)
    ax.annotate('Mesic', xy=(0., 0.025), xytext=(0., 1.),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='<->', lw=1.))

    return


def heatmap(df, fname, to_plot, what='NSE'):

    variables = np.array(['$g_s$', '$E$', '$A_n$'])[np.array(['gs' in to_plot,
                                                              'E' in to_plot,
                                                              'A' in to_plot])]

    # setup the plots
    if len(to_plot) > 1:
        fig, axes = plt.subplots(1, len(to_plot),
                                 figsize=(4.5 * len(to_plot), 5.), sharey=True)

    else:
        fig, ax = plt.subplots(figsize=(5., 5.))
        axes = [ax]
        df = df[df['variable'] == to_plot[0]]

    # model perf order?
    orders = []

    for i in range(len(to_plot)):

        if len(to_plot) > 1:
            sub = df.copy()[df['variable'] == to_plot[i]]

        else:
            sub = df.copy()

        orders += [model_perf(sub, to_plot[i], what)]

    # info needed by shared colormap, limit extremes
    if what == 'NSE':
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] > -5., -5.))

    elif what == 'MAPE':
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] < 500., 500.))

    elif what == 'MASE':
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] < 5., 5.))

    elif what == 'RMSE':
        df.loc[:, df.columns != 'variable'] = \
            (df.loc[:, df.columns != 'variable']
               .where(df.loc[:, df.columns != 'variable'] < 2., 2.))

    cmap, norm, vmin, vmax = cmap_specs(df)

    for i in range(len(to_plot)):

        # reorder the df based on model perf, only keep relevant var
        if len(to_plot) > 1:
            sub = df.copy()[df['variable'] == to_plot[i]]

        else:
            sub = df.copy()

        sub = sub.drop('variable', axis=1)
        sub = sub.reindex(index=orders[i].index.to_list())

        # into the right formats
        models = [which_model(e) for e in sub.index.to_list()]
        spps = [e.split('_') for e in sub.columns.to_list()]
        spps = ['$%s$. $%s$ (%s)' % (e[-2][0], e[-1], ' '.join(e[:-2])[0])
                if 'Quercus' in e else '$%s$. $%s$' % (e[-2][0], e[-1])
                for e in spps]
        data = sub.to_numpy()
        data = data.T

        # plot the data
        hm = axes[i].imshow(data, alpha=0.8, cmap=cmap, norm=norm, vmin=vmin,
                            vmax=vmax)

        #  format the axes and labels
        axes[i].set_xticks(np.arange(len(models)) - 0.25)
        axes[i].set_xticklabels(models, rotation=45, ha='left')
        axes[i].xaxis.tick_top()
        axes[i].set_yticks(np.arange(len(spps)))
        axes[i].set_yticklabels(spps, ha='right')

        if len(axes) > 1:  # add title below axis
            axes[i].set_title(r'%s for %s' % (what, variables[i]), y=-0.1,
                              fontsize=10.)

    # add colorbar and arrow
    cticks, bounds = cbar_specs(what, df)

    if len(axes) > 1:
        cax = fig.add_axes([1. / len(axes), -0.025,
                            int(len(axes) / 2) / len(axes) +
                            plt.rcParams['figure.subplot.wspace'], 0.04])

    else:
        cax = fig.add_axes([0.25, 0.05, 0.52, 0.03])

    if vmin < 0.:
        cbar = plt.colorbar(hm, cax=cax, ticks=cticks, boundaries=bounds,
                            extend='both', orientation='horizontal')

    else:
        cbar = plt.colorbar(hm, cax=cax, ticks=cticks, boundaries=bounds,
                            spacing='proportional', extend='both',
                            orientation='horizontal')

    if what == 'SMAPE':
        cbar.ax.set_xticklabels([int(e) if str(e).split('.')[1] == '0'
                                 else round(e, 2) for e in cticks])

    else:
        cbar.ax.set_xticklabels([int(e) if str(e).split('.')[1] == '0'
                                 else round(e, 1) for e in cticks])

    if len(axes) == 1:
        cbar.set_label(r'%s for %s' % (what, variables[0]))

    # add mesic - xeric arrow
    if len(axes) > 1:
        ax = fig.add_axes([0.045, 0.148, 0.05, 0.688])

    else:
        ax = fig.add_axes([-0.025, 0.135, 0.05, 0.715])

    MAP_arrow(ax)

    fig.savefig(fname, dpi=300)

    return


# PLOT
what = 'MASE'
to_plot = ['gs', 'E', 'A']

plt_setup()  # rendering

# Import Data
base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'obs_driven')
ofdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'plots'), 'obs_performance')

df = pd.read_csv(os.path.join(ifdir, 'all_%ss.csv' % (what)))

# order by MAP: wet to dry
df.set_index('model', inplace=True)
order = ['San_Lorenzo_Carapa_guianensis',
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
if what == 'NSE':
    ofdir = os.path.dirname(ofdir)

fname = os.path.join(ofdir, '%s_skill_for_%s.png' % (what, '_'.join(to_plot)))
heatmap(df[['variable'] + order], fname, to_plot, what=what)
