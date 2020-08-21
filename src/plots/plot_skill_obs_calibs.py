# general modules
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    if short == 'std1':
        lab = r'Medlyn-$\beta$'

    elif short == 'std2':
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

    cmap = plt.cm.OrRd
    cmap2 = cc.cm.CET_D1A
    cmap3 = cc.cm.CET_D1A_r
    norm = None

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    mean = np.nanmean(df.drop('variable', axis=1))
    med = np.nanmedian(df.drop('variable', axis=1))

    # what is the data distribution
    loc_zero = np.abs(min) / (max + np.abs(min))

    if min < 0.:

        if loc_zero > 0.5:
            cmap = cmap3

            if loc_zero > 0.75:
                norm = PowerNorm(gamma=np.log(0.5) / np.log(loc_zero))

            else:
                cmap = shift_cmap(cmap, locpoint=loc_zero)

        else:
            cmap = cmap2

            if loc_zero < 0.25:
                norm = PowerNorm(gamma=np.log(0.5) / np.log(loc_zero))

            else:
                cmap = shift_cmap(cmap, locpoint=loc_zero)

    else:
        thresh = mean / med

        if (thresh < 0.8) or (thresh > 1.2):
            norm = PowerNorm(gamma=0.4)

    return cmap, norm, min, max


def format_grid(ax, xmax, ymax):

    # grid
    #ax.grid(False, 'major')
    #ax.grid(True, 'minor', c='w', lw=0.1)

    # remove the ticks
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.tick_params(axis='both', which='both', length=0)

    # make sure left and bottom aren't cropped
    ax.set_xlim([-0.5, xmax + 0.5])
    ax.set_ylim([-0.5, ymax + 0.5])

    # remove spines
    for s in ax.spines.values():

        s.set_visible(False)

    return


def cbar_specs(what, df):

    # extra info
    min = np.nanmin(df.drop('variable', axis=1))
    max = np.nanmax(df.drop('variable', axis=1))
    start = 0.  # start for smooth

    if what == 'NSE':
        N1 = 50
        N2 = 150
        cticks = np.array([min, -2.5, -1., 0., 0.1, 0.25, 0.5, 0.75, max])

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
        cticks = np.array([min, start, 0.25, 0.5, 1., 1.5, max])

    if what == 'RMSE':
        N2 = 200
        start = 0.1
        cticks = np.array([min, start, 0.25, 0.5, 1., 2., 4., 6., max])

    smooth = list(np.linspace(start, cticks[-2], N2))

    if min < 0.:
        smooth = list(np.linspace(cticks[1], 0., N1)) + smooth

    bounds = [cticks[0]] + smooth + [cticks[-1]]

    return cticks[1:], bounds


def MAP_arrow(ax, direction='horizontal'):

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.patch.set_visible(False)

    for s in ax.spines.values():  # remove spines

        s.set_visible(False)

    if direction == 'horizontal':
        x1 = 0.
        x2 = x1 + 0.075
        x3 = 1.
        y1 = 0.
        y2 = y1
        y3 = y1
        head = '->'

    if direction == 'vertical':
        x1 = 0.
        x2 = x1
        x3 = x1
        y1 = 0.
        y2 = y1 + 0.05
        y3 = 1.
        head = '<-'

    # plot arrow
    ax.text(x1, y1, 'Mesic', va='center', ha='center',
            transform=ax.transAxes)
    ax.annotate('Xeric', xy=(x2, y2), xytext=(x3, y3),
                xycoords='axes fraction', va='center', ha='center',
                arrowprops=dict(arrowstyle='<-', lw=1.))

    return


def heatmap(df, gs=True, E=True, A=True, what='NSE'):

    to_plot = np.array(['gs', 'E', 'A'])[np.array([gs, E, A])]
    variables = np.array(['$g_s$', '$E$', '$A_n$'])[np.array([gs, E, A])]

    # setup the plotss
    fig, axes = plt.subplots(1, len(to_plot), figsize=(4.5 * len(to_plot), 5.),
                             sharey=True)

    if len(to_plot) < 2:
        axes = [axes]
        df = df[df['variable'] == to_plot[0]]

    else:
        plt.subplots_adjust(wspace=0.025)

    # info needed for shared colorbar
    cmap, norm, vmin, vmax = cmap_specs(df)

    for i in range(len(to_plot)):

        axes[i].set_facecolor('#737373')  # set background colour

        # reorder the df based on model perf, only keep relevant var
        if len(to_plot) > 1:
            sub = df.copy()[df['variable'] == to_plot[i]]

        else:
            sub = df.copy()

        order = model_perf(sub, to_plot[i], what)
        sub = sub.drop('variable', axis=1)
        sub = sub.reindex(index=order.index.to_list())

        # into the right formats
        models = [which_model(e) for e in sub.index.to_list()]
        spps = [e.split('_') for e in sub.columns.to_list()]
        spps = ['$%s$. $%s$ (%s)' % (e[-2][0], e[-1], ' '.join(e[:-2])[0])
                if 'Quercus' in e else '$%s$. $%s$' % (e[-2][0], e[-1])
                for e in spps]
        data = sub.to_numpy()

        if len(to_plot) > 1:
            data = data.T

        # plot the data
        hm = axes[i].imshow(data, alpha=0.8, cmap=cmap, norm=norm, vmin=vmin,
                            vmax=vmax)

        # x and y axes labels
        xlabs = spps
        ylabs = models

        if len(to_plot) > 1:
            xlabs = models
            ylabs = spps

        if len(to_plot) > 1:
            axes[i].set_xticks(np.arange(len(xlabs)))
            axes[i].set_xticklabels(xlabs, rotation=90, ha='center')
            axes[i].xaxis.tick_top()

        else:
            axes[i].set_xticks(np.arange(len(xlabs)) + 0.4)
            axes[i].set_xticklabels(xlabs, rotation=45, ha='right')

        axes[i].set_yticks(np.arange(len(ylabs)))
        axes[i].set_yticklabels(ylabs, ha='right')

        # format the axes
        format_grid(axes[i], len(xlabs) - 1, len(ylabs) - 1)

        if len(axes) > 1:  # add title below axis
            axes[i].set_title(r'%s for %s' % (what, variables[i]), y=-0.1,
                              fontsize=10.)

    # add colorbar and arrow
    cticks, bounds = cbar_specs(what, df)

    if len(axes) > 1:
        cax = fig.add_axes([1. / len(axes), -0.025,
                            int(len(axes) / 2) / len(axes) + 0.025, 0.04])
        #cbar = plt.colorbar(hm, cax=cax, orientation='horizontal')

        if vmin < 0.:
            cbar = plt.colorbar(hm, cax=cax, ticks=cticks, boundaries=bounds,
                                extend='both', orientation='horizontal')

        else:
            cbar = plt.colorbar(hm, cax=cax, ticks=cticks, boundaries=bounds,
                                spacing='proportional', extend='both',
                                orientation='horizontal')

        if (what == 'NSE') or (what == 'MAPE') or (what == 'MASE'):
            cbar.ax.set_xticklabels([round(e, 2) for e in cticks])

        else:
            cbar.ax.set_xticklabels([round(e, 1) for e in cticks])

        # add mesic - xeric arrow
        ax = fig.add_axes([0.03, 0.148, 0.05, 0.688])
        MAP_arrow(ax, direction='vertical')

    else:
        cax = make_axes_locatable(axes[0]).append_axes('right', size='5%',
                                                        pad=0.2)
        cbar = plt.colorbar(hm, cax=cax, ticks=cticks, boundaries=bounds,
                            extend='both')
        cbar.ax.set_yticklabels([round(e, 2) for e in cticks])
        cbar.set_label(r'%s for %s' % (what, variables[0]))

        # add mesic - xeric arrow
        ax = fig.add_axes([0.175, 0.825, 0.6, 0.05])
        MAP_arrow(ax)

    fig.savefig(r'%s_skill_for_%s.png' % (what, '_'.join(to_plot)), dpi=300,
                bbox_inches='tight')

    return


# Import Data
what = 'SMAPE'
df = pd.read_csv('/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/output/simulations/obs_driven/all_%ss.csv' % (what))

plt.rcParams['text.usetex'] = True  # use LaTeX
plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                       r'\usepackage{mathpazo}',
                                       r'\usepackage{amsmath}']

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
heatmap(df[['variable'] + order], what=what) #, E=False, A=False)
