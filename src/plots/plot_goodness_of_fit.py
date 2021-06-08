# general modules
import os  # check for files, paths
import sys  # check for files, paths
import numpy as np
import pandas as pd  # read/write dataframes, csv files

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files

#==============================================================================


class plt_setup(object):

    def __init__(self, colours=None):

        # saving the figure
        plt.rcParams['savefig.dpi'] = 1200.  # resolution
        plt.rcParams['savefig.bbox'] = 'tight'  # no excess side padding
        plt.rcParams['savefig.pad_inches'] = 0.01  # padding to use

        # colors
        if colours is None:  # use the default colours
            colours = ['#1a1a1a', '#6023b7', '#af97c5', '#197aff', '#9be2fd',
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
        plt.rcParams['lines.linewidth'] = 1.75
        plt.rcParams['scatter.marker'] = '.'

        # boxes
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

        # patches (e.g. the shapes in the legend)
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['patch.edgecolor'] = 'k'
        plt.rcParams['patch.force_edgecolor'] = True  # ensure it's used

        # legend
        plt.rcParams['legend.fontsize'] = 7.
        plt.rcParams["legend.columnspacing"] = 0.8
        plt.rcParams['legend.edgecolor'] = 'w'
        plt.rcParams['legend.borderpad'] = 0.5

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

    elif short == 'cgn':
        lab = 'CGain'

    elif short == 'sox2':
        lab = 'SOX$_\mathrm{\mathsf{opt}}$'

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


def render_xlabels(ax, name, unit, fs=7., pad=0.):

    ax.set_xlabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit),
                  labelpad=pad)

    return


def render_ylabels(ax, name, unit, fs=7., pad=0.):

    ax.set_ylabel(r'{\fontsize{%dpt}{3em}\selectfont{}%s }' % (fs, name) +
                  r'{\fontsize{%dpt}{3em}\selectfont{}(%s)}' % (0.9 * fs, unit),
                  labelpad=pad)

    return


def studentize(df, vars=['gs', 'Ci', 'E', 'A', 'Pleaf']):

    # calc mean and standard deviation
    mean = df.mean(axis=0)
    std = df.std(axis=0)

    # variables to consider
    columns = [c for c in df.columns if any(var in c for var in vars)]

    if 'gs' in vars:  # give each model the avg obs attributes
        mean.loc[[e for e in columns if 'gs(' in e]] = mean.gs
        std.loc[[e for e in columns if 'gs(' in e]] = std.gs

    if 'Ci' in vars:
        mean.loc[[e for e in columns if 'Ci(' in e]] = mean.Ci
        std.loc[[e for e in columns if 'Ci(' in e]] = std.Ci

    if 'E' in vars:
        mean.loc[[e for e in columns if 'E(' in e]] = mean.E
        std.loc[[e for e in columns if 'E(' in e]] = std.E

    if 'A' in vars:
        mean.loc[[e for e in columns if 'A(' in e]] = mean.A
        std.loc[[e for e in columns if 'A(' in e]] = std.A

    if 'Pleaf' in vars:
        mean.loc[[e for e in columns if 'Pleaf(' in e]] = mean.Pleaf
        std.loc[[e for e in columns if 'Pleaf(' in e]] = std.Pleaf

    # now studentize each model's variables and the obs
    icol = [df.columns.get_loc(e) for e in columns]
    df.iloc[:, icol] = (df.iloc[:, icol].subtract(mean, axis=1)
                          .div(std, axis=1))

    return df


def one_2_one(df, figname, what):

    models = model_order()

    # decide on how many columns and rows
    Nrows = 0
    Ncols = 0

    while Nrows * Ncols < len(models):

        Nrows += 1

        if Nrows * Ncols < len(models):
            Ncols += 1

    fig, axes = plt.subplots(Nrows, Ncols, figsize=(Ncols + 2, Nrows + 2.25),
                             sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.1, wspace=0.05)
    axes = axes.flat

    df = df[df['site_spp'] == what]
    df.reset_index(drop=True, inplace=True)

    # limit to roughly midday LWP
    not_midday = df[np.logical_or(df['hod'] < 12., df['hod'] > 14.)].index
    df.loc[not_midday, 'Pleaf'] = np.nan

    # studentize the data
    df = studentize(df)  # rescale across all
    columns = [c for c in df.columns if any(var in c for var in
               ['gs', 'Ci', 'E', 'A', 'Pleaf'])]
    df[columns] = df[columns].where(df[columns] > -1000.)
    df[columns] = df[columns].where(df[columns] < 1000.)

    for i, mod in enumerate(models):

        x = df[['gs', 'E', 'A', 'Ci']].values.flatten()
        y = df[['gs(%s)' % (mod), 'E(%s)' % (mod), 'A(%s)' % (mod),
                'Ci(%s)' % (mod)]].values.flatten()

        axes[i].scatter(df['gs'], df['gs(%s)' % (mod)], alpha=0.5,
                        label=r'$g_{s}$')
        axes[i].scatter(df['Ci'], df['Ci(%s)' % (mod)], alpha=0.5,
                        label=r'$C_{i}$')
        axes[i].scatter(df['A'], df['A(%s)' % (mod)], alpha=0.5,
                        label=r'$A_{n}$')
        axes[i].scatter(df['E'], df['E(%s)' % (mod)], alpha=0.5, label=r'$E$')

        if mod != 'std':
            axes[i].scatter(df['Pleaf'], df['Pleaf(%s)' % (mod)],
                            label=r'$\Psi$$_{l, midday}$')

        mask = np.logical_and(np.isfinite(x), np.isfinite(y))
        slope, intercept, r, p, sem = stats.linregress(y[mask], x[mask])

        axes[i].plot([-2, 2], [-2, 2], c='orange', ls='--', lw=2.)
        axes[i].plot([-2.75, 2], [intercept - 2. * slope,
                                  intercept + 2. * slope], c='orange', lw=2.)

        axes[i].text(0.05, 0.8, '%s:\nslope=%s, $r^{2}$=%s' %
                                (which_model(mod), round(slope, 2),
                                 round(r ** 2., 2)), ha='left',
                     transform=axes[i].transAxes, weight='bold')

        if i % Ncols == 0:
            axes[i].set_ylabel('Simulated variables')

        if i >= (Nrows - 1) * Ncols:
            axes[i].set_xlabel('Observed variables')

    # format axes ticks
    axes[-1].set_xlim(-2.9, 2.9)
    axes[-1].set_ylim(-2.9, 2.9)
    axes[-1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    axes[-1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    # add legend
    axes[-1].legend(bbox_to_anchor=(1., 0.75))

    fig.savefig(figname)
    plt.close()

    return


# Draw Plot
plt_setup()

# Import Data
base_dir = get_main_dir()  # dir paths
ifdir = os.path.join(os.path.join(os.path.join(base_dir, 'output'),
                     'simulations'), 'obs_driven')
ofdir = os.path.join(os.path.join(base_dir, 'output'), 'plots')

df = pd.read_csv(os.path.join(ifdir, 'all_site_spp_simulations.csv'))

figname = os.path.join(ofdir, 'goodness_of_fit.jpg')

if not os.path.isfile(figname):
    one_2_one(df, figname, 'Richmond_Eucalyptus_cladocalyx')
