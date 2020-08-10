# general modules
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy import stats

# plotting
import matplotlib.pyplot as plt
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


def heatmap(ax, x_labels, y_labels, data, what='RMSE'):

    # Mapping from column names to integer coordinates
    x_num = (np.repeat(np.arange(len(x_labels)), len(y_labels))
                      .reshape(len(x_labels), -1))
    y_num = np.tile(np.arange(len(y_labels)), (len(x_labels), 1))

    size_scale = 500.
    cmap = plt.cm.OrRd
    norm = None

    loc_zero = np.abs(np.nanmin(data)) / (np.nanmax(data) +
                                          np.abs(np.nanmin(data)))

    if any(data.flatten() < 0.):

        if loc_zero > 0.5:
            cmap = cc.cm.CET_D1A_r

            if loc_zero > 0.75:
                norm = PowerNorm(gamma=np.log(0.5) / np.log(loc_zero))

            else:
                cmap = shift_cmap(cmap, locpoint=loc_zero)

        else:
            cmap = cc.cm.CET_D1A

            if loc_zero < 0.25:
                norm = PowerNorm(gamma=np.log(0.5) / np.log(loc_zero))

            else:
                cmap = shift_cmap(cmap, locpoint=loc_zero)

    else:
        thresh = np.nanmean(data) / np.nanmedian(data)
        print(thresh)

        if (thresh < 0.8) or (thresh > 1.2):
            norm = PowerNorm(gamma=0.5)

    #hm = ax.scatter(x_num, y_num, s=np.abs(data) * size_scale, c=data,
    #                marker='s', cmap=cmap, alpha=0.8)

    hm = ax.scatter(x_num, y_num, s=size_scale, c=data, ec='none', marker='s',
                    cmap=cmap, norm=norm, alpha=0.8)

    # Show column labels on the axes
    x_labels = [e.split('_') for e in x_labels]
    spp = ['$%s$. $%s$ (%s)' % (e[-2][0], e[-1], ' '.join(e[:-2])[0])
           if 'Quercus' in e else '$%s$. $%s$' % (e[-2][0], e[-1])
           for e in x_labels]
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(spp, rotation=90, ha='center')
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([which_model(e) for e in y_labels])

    # grid
    #ax.grid(False, 'major')
    #ax.grid(True, 'minor', c='w')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.tick_params(axis='both', which='both', length=0)

    # make sure left and bottom aren't cropped
    ax.set_xlim([-0.5, np.amax(x_num) + 0.5])
    ax.set_ylim([-0.5, np.amax(y_num) + 0.5])

    # remove spines
    for s in ax.spines.values():

        s.set_visible(False)

    # add the colorbar
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

        cbar = plt.colorbar(hm, ticks=cticks)

    else:
        cbar = plt.colorbar(hm, extend='both')

    cbar.set_label(what)

    return


# Import Data
what = 'RMSE'
df = pd.read_csv('/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/output/simulations/obs_driven/all_%ss.csv' % (what))

# order the df by 'skill'
df.set_index('model', inplace=True)

if what == 'NSE':
    order = (((df[df['variable'] == 'gs'].drop('variable', axis=1)
                 .median(axis=1) +
               df[df['variable'] == 'A'].drop('variable', axis=1)
                 .median(axis=1) +
               df[df['variable'] == 'E'].drop('variable', axis=1)
                 .median(axis=1)))
             .sort_values())
    #order = (df[df['variable'] == 'gs'].drop('variable', axis=1).median(axis=1)
    #          .sort_values())

elif what == 'log':  # that's not quite right, must be closest to zero is better
    order = (df[df['variable'] == 'gs'].drop('variable', axis=1).mean(axis=1)
               .sort_values())

elif what == 'RMSE':
    order = ((df[df['variable'] == 'gs'].drop('variable', axis=1) *
              df[df['variable'] == 'A'].drop('variable', axis=1) *
              df[df['variable'] == 'E'].drop('variable', axis=1)).mean(axis=1)
             .sort_values())
    order = (df[df['variable'] == 'gs'].drop('variable', axis=1).mean(axis=1)
               .sort_values())

else:
    order = ((df[df['variable'] == 'gs'].drop('variable', axis=1) +
              df[df['variable'] == 'A'].drop('variable', axis=1) +
              df[df['variable'] == 'E'].drop('variable', axis=1)).mean(axis=1)
             .sort_values())
    order = (df[df['variable'] == 'gs'].drop('variable', axis=1).mean(axis=1)
             .sort_values())

# only keep the gs' RMSEs
df = df[df['variable'] == 'gs']
df = df.drop('variable', axis=1)

# reorder the df
df = df.reindex(index=order.index.to_list())  # models

# by MAP
order = ['Richmond_Eucalyptus_dunnii', 'ManyPeaksRange_Alphitonia_excelsa',
         'ManyPeaksRange_Austromyrtus_bidwillii',
         'ManyPeaksRange_Brachychiton_australis',
         'ManyPeaksRange_Cochlospermum_gillivraei',
         'Richmond_Eucalyptus_saligna', 'Puechabon_Quercus_ilex',
         'Vic_la_Gardiole_Quercus_ilex', 'Richmond_Eucalyptus_cladocalyx',
         'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis',
         'Corrigin_Eucalyptus_capillosa']

df = df[order]

plt.rcParams['text.usetex'] = True  # use LaTeX
plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                       r'\usepackage{mathpazo}',
                                       r'\usepackage{amsmath}']

fig, ax = plt.subplots(figsize=(6., 4.4))
ax.set_facecolor('#737373')
heatmap(ax, df.columns.to_list(), df.index.to_list(), df.to_numpy().transpose(),
        what=what)

# Decoration
fig.savefig(r'gs_model_skill_%s_by_gs.png' % (what), dpi=1200, bbox_inches='tight')
