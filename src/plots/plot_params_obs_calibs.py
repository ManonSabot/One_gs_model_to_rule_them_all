# general modules
import os
import sys
import numpy as np
import pandas as pd  # read/write dataframes, csv files

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit  # fit the functional shapes
from scipy.integrate import quad  # integrate on a range
import string   # automate subplot lettering

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import conv, cst
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import f, Weibull_params


class plt_setup(object):

    def __init__(self, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # colors
        if colours is None:  # use the default colours
            colours = ['#1a1a1a', '#6f32c7', '#a182bf', '#197aff', '#9be2fd',
                       '#009231', '#a6d96a', '#6b3b07', '#ff8e12', '#ffe020',
                       '#f10c80', '#ffc2cd']

        plt.rcParams['axes.prop_cycle'] = cycler(color=colours)

        # labels, text, annotations
        plt.rcParams['text.usetex'] = True  # use LaTeX
        preamble = [r'\usepackage[sfdefault,light]{merriweather}',
                    r'\usepackage{mathpazo}', r'\usepackage{amsmath}']
        plt.rcParams['text.latex.preamble'] = '\n'.join(preamble)
        plt.rcParams['font.size'] = 6.
        plt.rcParams['axes.labelsize'] = 7.
        plt.rcParams['xtick.labelsize'] = 6.
        plt.rcParams['ytick.labelsize'] = 6.

        # lines and markers
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.color'] = '#d3d3d3'
        plt.rcParams['scatter.marker'] = '.'
        plt.rcParams['scatter.edgecolors'] = 'k'

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams["legend.columnspacing"] = 0.8
        plt.rcParams['legend.edgecolor'] = 'w'

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


def render_xlabels(ax, name, unit, fs=7.):

    ax.set_xlabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit))

    return


def render_ylabels(ax, name, unit, fs=7.):

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit))

    return


def plot_obs(ax, x, y, which='gs'):

    if which == 'gs':
        ax.scatter(x, y, marker='+', s=25., edgecolor='none', alpha=0.5)

    else:
        vp = ax.violinplot(y, showextrema=False, positions=[x], widths=0.8)
        plt.setp(vp['bodies'], facecolor='w', edgecolor='#d3d3d3', alpha=1)
        ax.vlines(x, np.amin(y), np.amax(y), zorder=1)
        ax.vlines(x, np.percentile(y, 25), np.percentile(y, 75), lw=6, zorder=2)
        ax.scatter(x, np.percentile(y, 50), marker='_', color='w', zorder=3)


def fsig_tuzet(psi_leaf, sf, psi_f):
    """
    An empirical logistic function to describe the sensitivity of stomata
    to leaf water potential.

    Sigmoid function assumes that stomata are insensitive to psi_leaf at
    values close to zero and that stomata rapidly close with decreasing
    psi_leaf.

    Parameters:
    -----------
    psi_leaf : float
        leaf water potential (MPa)

    Returns:
    -------
    fw : float
        sensitivity of stomata to leaf water potential [0-1]

    Reference:
    ----------
    * Tuzet et al. (2003) A coupled model of stomatal conductance,
      photosynthesis and transpiration. Plant, Cell and Environment 26,
      10971116

    """
    num = 1.0 + np.exp(sf * psi_f)
    den = 1.0 + np.exp(sf * (psi_f - psi_leaf))
    fw = num / den

    return fw


def envelope(x, y, reject=0):

    """
    Upper envelope peaks of y to x

    """

    # declare the first values
    u_x = [x[0],]
    u_y = [y[0],]
    lastPeak = 0

    # detect peaks and mark their location
    for i in range(1, len(y) - 1):

        if ((y[i] - y[i - 1]) > 0. and ((y[i] - y[i + 1]) > 0) and
            ((i - lastPeak) > reject)):
            u_x.append(x[i])
            u_y.append(y[i])
            lastPeak = i

    # append the last values
    u_x.append(x[-1])
    u_y.append(y[-1])

    return u_x, u_y


def fit_Tuzet(df):

    # smooth out noise
    smoothed = gaussian_filter(df['gs'], df['gs'].std())

    # point where the signal goes above the background noise
    base = 0.3  # background noise is +/- 15% of max gs
    supp = (df['gs'][df['Pleaf'] < -df['Pleaf'].std()] - base).std()
    m = smoothed < (base - df['gs'].std() * supp)
    x0 = np.maximum(df['Pleaf'][m].max(), df['Pleaf'][np.isclose(df['gs'], 1.)])
    x1 = df['Pleaf'][m].min()

    # now sort df by LWP
    df.sort_values(by=['Pleaf'], ascending=False, inplace=True)
    LWP, gs = envelope(df['Pleaf'].to_numpy(), df['gs'].to_numpy())

    try:
        obs_popt, __ = curve_fit(fsig_tuzet, LWP, gs, p0=[2., (x0+x1) / 2.],
                                 bounds=([0.01, df['Pleaf'].min()],
                                         [10, df['Pleaf'].max()]))

        return x0, x1, obs_popt

    except Exception:
            return (0.,) * 3


def get_calib_kmax(df):

    params = []
    models = []

    for what in df['training'].unique().dropna():

        sub = df.copy()[df['training'] == what]
        sub['v3'] = sub['v1']
        sub[sub['Model'] == 'Tuzet']['v3'] = sub['v2']
        keep = np.logical_or(sub['p2'].str.contains('kmaxT').fillna(False),
                             np.logical_or(sub['p1'].str.contains('kmax'),
                                           sub['p1'].str.contains('krl')))

        # set model order
        sub['order'] = sub['Model'].replace({'Tuzet': 0, 'Eller': 1,
                                             'ProfitMax': 2, 'SOX-OPT': 3,
                                             'ProfitMax2': 4, 'LeastCost': 5,
                                             'CAP': 6, 'MES': 7})

        sub = sub[keep].sort_values(by=['solver', 'order'])
        sub.reset_index(inplace=True)
        params += [np.log(sub['v3'].values)]
        models += [sub['Model'].values]

    return params, models


def obs_calibs(df1, df2, figname):

    fig = plt.figure(figsize=(6.5, 8.))
    gs = fig.add_gridspec(nrows=96, ncols=16, hspace=0.3, wspace=0.2)
    ax2 = fig.add_subplot(gs[52:, 6:])  # conductance data

    ipath = os.path.join(os.path.join(os.path.join(get_main_dir(),
                         'input'), 'simulations'), 'obs_driven')

    labels = []

    for i, what in enumerate(df1['site_spp'].unique().dropna()):

        if i < 13:
            nrow = int(i / 4) * 16
            ncol = (i % 4) * 4
            ax1 = fig.add_subplot(gs[nrow:nrow + 16, ncol:ncol + 4])

        else:
            nrow += 16
            ax1 = fig.add_subplot(gs[nrow:nrow + 16, :4])

        sub = df1.copy()[df1['site_spp'] == what]
        sub = sub.select_dtypes(exclude=['object', 'category'])
        sub = sub[sub['Pleaf'] > -9999.]
        sub['gs'] /= sub['gs'].max()

        for day in sub['doy'].unique():

            mask = sub['doy'] == day
            plot_obs(ax1, sub['Pleaf'][mask], sub['gs'][mask])

        x0, x1, obs_popt = fit_Tuzet(sub)
        x = np.linspace(sub['Pleaf'].max(), sub['Pleaf'].min(), 500)
        ax1.plot(x, fsig_tuzet(x, obs_popt[0], obs_popt[1]), 'k', zorder=30)
        ax1.vlines(x0, 0., 1., linestyle=':')
        ax1.vlines(x1, 0., 1., linestyle=':')

        # get the integrated VC given by the obs and site params
        ref, __ = read_csv(os.path.join(ipath, '%s_calibrated.csv' % (what)))
        b, c = Weibull_params(ref.iloc[0])
        int_VC = np.zeros(len(sub))

        for j in range(len(sub)):

            int_VC[j], __ = quad(f, sub['Pleaf'].iloc[j], sub['Ps'].iloc[j],
                                 args=(b, c))

        plot_obs(ax2, i, np.log(sub['E'] / int_VC), which='kmax')

        # subplot titles (including labelling)
        what = what.split('_')
        species = r'\textit{%s %s}' % (what[-2], what[-1])
        labels += [r'\textit{%s. %s}' % (what[-2][0], what[-1])]

        if 'Quercus' in what:
            species += ' (%s)' % (what[0][0])
            labels[-1] += ' (%s)' % (what[0][0])

        txt = ax1.annotate(r'\textbf{(%s)} %s' % (string.ascii_lowercase[i],
                                                  species),
                           xy=(0.025, 0.98), xycoords='axes fraction',
                           ha='left', va='top')
        txt.set_bbox(dict(boxstyle='round,pad=0.1', fc='w', ec='none',
                     alpha=0.8))

        # format axes ticks
        ax1.xaxis.set_major_locator(mpl.ticker.NullLocator())

        if (i == 13) or ((ncol > 0) and (nrow == 32)):
            render_xlabels(ax1, r'$\Psi_{l}$', 'MPa')

        if ncol == 0:
            ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax1.set_ylabel(r'$g_{s, norm}$')

        else:
            ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax1.set_yticklabels([])

    ax2.annotate(r'\textbf{(%s)}' % (string.ascii_lowercase[i + 1]),
                 xy=(0.05, 0.98), xycoords='axes fraction', ha='right',
                 va='top')

    # add max conductance parameter values
    params, models = get_calib_kmax(df2)
    params = np.asarray(params)
    locs = np.arange(len(df1['site_spp'].unique()))

    # update colour list
    colours = (['#6023b7', '#af97c5', '#009231', '#6b3b07', '#ff8e12',
                '#ffe020', '#f10c80', '#ffc2cd']) * len(params)

    for i in range(params.shape[1]):

        if i < 8:
            ax2.scatter(locs, params[:, i], s=50, linewidths=0.25, c=colours[i],
                        alpha=0.9, label=models[0][i], zorder=4)

        else:
            ax2.scatter(locs, params[:, i], s=50, linewidths=0.25, c=colours[i],
                        alpha=0.9, zorder=4)

    # tighten the subplot
    ax2.set_xlim(locs[0] - 0.8, locs[-1] + 0.8)
    ax2.set_ylim(np.log(0.025) - 0.1, np.log(80.))

    # ticks
    ax2.set_xticks(locs + 0.5)
    ax2.set_xticklabels(labels, ha='right', rotation=40)
    ax2.xaxis.set_tick_params(length=0.)

    yticks = [0.025, 0.25, 1, 5, 25, 75]
    ax2.set_yticks([np.log(e) for e in yticks])
    ax2.set_yticklabels(yticks)
    render_ylabels(ax2, r'k$_{max}$', 'mmol m$^{-2}$ s$^{-1}$ MPa$^{-1}$')

    handles, labels = ax2.get_legend_handles_labels()
    labels[3] = 'SOX$_\mathrm{\mathsf{opt}}$'
    ax2.legend(handles, labels, ncol=3, labelspacing=1. / 3., columnspacing=0.5,
               loc=3)

    # save
    fig.savefig(figname)


# Draw Plot
plt_setup()

# Import Data
base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'obs_driven')
ofdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')
df1 = pd.read_csv(os.path.join(ifdir, 'all_site_spp_simulations.csv'))

ifdir = ifdir.replace('simulations', 'calibrations')
df2 = pd.read_csv(os.path.join(ifdir, 'overview_of_fits.csv'))

site_spp = ['San_Lorenzo_Carapa_guianensis', 'San_Lorenzo_Tachigali_versicolor',
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

# temporarily address the issue of many extra sites
df1 = df1[df1['site_spp'].isin(site_spp)]

# organise the dfs in order
df1.site_spp = df1.site_spp.astype('category')
df1.site_spp.cat.set_categories(site_spp, inplace=True)
df1 = df1.sort_values('site_spp')
df2.training = df2.training.astype('category')
df2.training.cat.set_categories(site_spp, inplace=True)
df2 = df2.sort_values('training')

subsel = []

for s in site_spp:  # find out where there are no LWP obs

    if len(df1['Pleaf'][df1['site_spp'] == s].dropna()) < 1:
        subsel += [s]

# exclude sites without LWP from site_spp
df1 = df1[~df1['site_spp'].isin(subsel)]
df2 = df2[~df2['training'].isin(subsel)]

figname = os.path.join(ofdir, 'obs_data_calibs.png')

if not os.path.isfile(figname):
    obs_calibs(df1, df2, figname)
