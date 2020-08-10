# general modules
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


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
        lab = 'CGainNet'

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

    elif what == 'log':
        order = (df[df['variable'] == variable].drop('variable', axis=1)
                   .mean(axis=1).sort_values())

    else:
        order = (df[df['variable'] == variable].drop('variable', axis=1)
                   .mean(axis=1).sort_values())

    return order


def cmap_specs(data):

    cmap = plt.cm.OrRd
    cmap2 = cc.cm.CET_D1A
    cmap3 = cc.cm.CET_D1A_r
    norm = None

    # what is the data distribution
    loc_zero = np.abs(np.nanmin(data)) / (np.nanmax(data) +
                                          np.abs(np.nanmin(data)))

    if any(data.flatten() < 0.):

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
        thresh = np.nanmean(data) / np.nanmedian(data)

        if (thresh < 0.8) or (thresh > 1.2):
            norm = PowerNorm(gamma=0.5)

    return cmap, norm


def format_grid(ax, xmax, ymax):

    # grid
    #ax.grid(False, 'major')
    #ax.grid(True, 'minor', c='w')

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


def heatmap(df, gs=True, E=True, A=True, what='NSE'):

    # default plot characteristics
    size_scale = 500.
    to_plot = np.array(['gs', 'E', 'A'])[np.array([gs, E, A])]
    labels = np.array(['g$_s$', 'E', 'A$_n$'])[np.array([gs, E, A])]

    # setup the plotss
    fig, axes = plt.subplots(1, len(to_plot), figsize=(4.5 * len(to_plot), 5.),
                             sharey=True)
    plt.subplots_adjust(wspace=0.05)

    for i in range(len(to_plot)):

        axes[i].set_facecolor('#737373')  # set background colour

        # reorder the df based on model perf, only keep relevant var
        sub = df.copy()[df['variable'] == to_plot[i]]
        order = model_perf(sub, to_plot[i], what)
        sub = sub.drop('variable', axis=1)
        sub = sub.reindex(index=order.index.to_list())

        # into the right format
        x_labels = sub.index.to_list()
        y_labels = sub.columns.to_list()
        data = sub.to_numpy()

        # from column names to integer coordinates
        x_num = (np.repeat(np.arange(len(x_labels)), len(y_labels))
                          .reshape(len(x_labels), -1))
        y_num = np.tile(np.arange(len(y_labels)), (len(x_labels), 1))

        # plot the data
        cmap, norm = cmap_specs(data)
        #hm = ax.scatter(x_num, y_num, s=np.abs(data) * size_scale, c=data,
        #                marker='s', cmap=cmap, alpha=0.8)
        hm = axes[i].scatter(x_num, y_num, s=size_scale, c=data, ec='none',
                             marker='s', cmap=cmap, norm=norm, alpha=0.8)

        # models labels
        axes[i].set_xticks(np.arange(len(x_labels)))
        axes[i].set_xticklabels([which_model(e) for e in x_labels], rotation=90,
                                ha='center')

        # species labels
        y_labels = [e.split('_') for e in y_labels]
        spp = ['$%s$. $%s$ (%s)' % (e[-2][0], e[-1], ' '.join(e[:-2])[0])
               if 'Quercus' in e else '$%s$. $%s$' % (e[-2][0], e[-1])
               for e in y_labels]
        axes[i].set_yticks(np.arange(len(y_labels)))
        axes[i].set_yticklabels(spp, ha='right')

        # setup the axes
        format_grid(axes[i], np.amax(x_num), np.amax(y_num))

        # add the colorbar
        cax = make_axes_locatable(axes[i]).append_axes('top', size='4%',
                                  pad='3%')

        if (norm is not None) and ((np.nanmin(data) < 0.) or
                                   (np.nanmax(data) > 1.)):
            cticks =[round(np.nanpercentile(data, 5), 1),
                     round(np.nanpercentile(data, 25), 1),
                     round(np.nanpercentile(data, 50), 1),
                     round(np.nanpercentile(data, 75), 1),
                     round(np.nanpercentile(data, 95), 1)]

            if any(data.flatten() < 0.):
                cticks.insert(2, 0)

            else:
                cticks.insert(0, 0)

            cbar = plt.colorbar(hm, cax=cax, ticks=cticks,
                                orientation='horizontal')

        else:
            cbar = plt.colorbar(hm, cax=cax, extend='both',
                                orientation='horizontal')

        cbar.set_label(r'%s of %s' % (what, labels[i]))
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

    fig.savefig(r'%s_skill_for_%s.png' % (what, '_'.join(to_plot)), dpi=300,
                bbox_inches='tight')

    return


# Import Data
what = 'log'
df = pd.read_csv('/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/output/simulations/obs_driven/all_%ss.csv' % (what))

plt.rcParams['text.usetex'] = True  # use LaTeX
plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                       r'\usepackage{mathpazo}',
                                       r'\usepackage{amsmath}']

# order by MAP
df.set_index('model', inplace=True)
order = ['Richmond_Eucalyptus_dunnii', 'ManyPeaksRange_Alphitonia_excelsa',
         'ManyPeaksRange_Austromyrtus_bidwillii',
         'ManyPeaksRange_Brachychiton_australis',
         'ManyPeaksRange_Cochlospermum_gillivraei',
         'Richmond_Eucalyptus_saligna', 'Puechabon_Quercus_ilex',
         'Vic_la_Gardiole_Quercus_ilex', 'Richmond_Eucalyptus_cladocalyx',
         'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis',
         'Corrigin_Eucalyptus_capillosa']

# make the figure
heatmap(df[['variable'] + order], what=what)
