#!/usr/bin/env python
# -*- coding: utf-8 -*-

# general modules
import os
import numpy as np  # array manipulations, math operators
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from pygam import LinearGAM  # fit the functional shapes


#######################################################################
def read_csv(fname):

    """
    Reads csv file with two headers, one for the variables, one for
    their units.

    Arguments:
    ----------
    fname: string
        input filename (with path)

    drop_units: boolean
        if True, drops the second row of the columns, i.e. the units

    Returns:
    --------
    df: pandas dataframe
        dataframe containing all the csv data, units header dropped

    columns: array
        original columns names and units present in csv file

    """

    df = (pd.read_csv(fname, header=[0]).dropna(axis=0, how='all')
          .dropna(axis=1, how='all').squeeze())
    columns = df.columns

    return df


def process_datasets():

    dfs = []
    spps = []
    info = open(os.path.join(os.getcwd(), 'info_data.txt'), 'w+')

    # open the datasets
    for file in os.listdir(os.getcwd()):

        if file.endswith('.csv'):
            df = read_csv(file)

            # time to hod
            df['hod'] = df['Time'].copy()
            hod = [e.split(':') for e in df['hod'].values]
            hod = np.asarray([float(e[0]) + round(float(e[1]) / 30.) / 2.
                              for e in hod])
            df['hod'] = hod

            # add measure of WUE
            df['WUE'] = df['A'].copy().values / (1.57 * df['gs'].copy().values)

            if 'MartinStPaul' in file:
                df['E'] *= 1.e3  # mol to mmol

            # quality checks on the data, and normalise the variables
            df = df[df['hod'] > 6.]
            df = df[df['PARin'] > 50.]
            df = df[df['VPD'] >= 0.5]
            df = df[df['gs'] > 0.]
            df = df[df['E'] > 0.]

            try:
                df = df[np.logical_and(df['Ci'] > 100., df['Ci'] < 375.)]
                df['Ci'] /= np.nanmax(df['Ci'].values)

            except KeyError:
                pass

            try:
                df = df[df['LWP'] < 0.]
                df['LWP'] *= -1.

            except KeyError:
                pass

            try:
                df = df[df['LWPpd'] < 0.]
                df['LWPpd'] *= -1.

            except KeyError:
                pass

            df['gs'] /= np.nanmax(df['gs'].values)
            df['E'] /= np.nanmax(df['E'].values)
            df['A'] /= np.nanmax(df['A'].values)
            df['WUE'] /= np.nanmax(df['WUE'].values)

            dfs += [df]

            try:
                species = np.unique(df['species'])

            except KeyError:
                species = np.unique(df['Species'])


            info.write('%s: %d data points\n' % (file, len(df)))
            info.write('%d species (%s)\n'
                       % (len(species), ', '.join(species)))
            info.write('LWPmin: -%s MPa, LWPmax: -%s MPa\n'
                       % (str(df['LWPpd'].max()), str(df['LWPpd'].min())))

            if 'LAI' in df.columns:
                info.write('LAI available\n')

            if 'SWC' in df.columns:
                info.write('SWC available\n')

            if 'Totalheight' in df.columns:
                info.write('height available\n')
                
            info.write('\n\n')

            spps += [species]

    spps = np.concatenate(spps).ravel()
    info.close()

    return dfs, spps


def ini_fig(N):

    fig = plt.figure(N, figsize=(16, 3))
    plt.subplots_adjust(wspace=0.025)
    axes = fig.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

    return


#######################################################################

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# plots' info
x = ['VPD', 'gs', 'LWPpd', 'hod', 'VPD', 'hod', 'hod', 'hod', 'VPD',
     'E', 'hod', 'hod']
xlab = ['VPD (kPa)', 'g$_{s, norm}$', '$\Psi_{leaf,pd}$ (-MPa)', 'hod (h)',
        'VPD (kPa)', 'hod (h)', 'hod (h)', 'hod (h)', 'VPD (kPa)',
        'E$_{norm}$ (-)', 'hod (h)', 'hod (h)']
y = ['gs', 'Ci', 'gs', 'gs', 'LWP', 'LWP', 'Ci', 'WUE', 'WUE', 'A', 'E',
     'A']
ylab = ['g$_{s, norm}$', 'Ci$_{norm}$ (-)', 'g$_{s, norm}$', 'g$_{s, norm}$',
        '$\Psi_{leaf, pd}$ (-MPa)', '$\Psi_{leaf}$ (-MPa)', 'Ci$_{norm}$ (-)',
        'WUE$_{norm}$ (-)', 'WUE$_{norm}$ (-)', 'A$_{norm}$ (-)',
        'E$_{norm}$ (-)', 'A$_{norm}$ (-)']
plot = ['%s_%s' % (y[i], x[i]) for i in range(len(x))]

# get all the data
dfs, spps = process_datasets()

# plot all, species in different colors
cols = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
        '#35978f']
colours = dict(zip(spps, cols))

# attempt the same as below, but restricted to plotting pdfs
fig1 = plt.figure(1, figsize=(16, 3))
plt.subplots_adjust(wspace=0.025)
axes = fig1.subplots(nrows=1, ncols=3, sharey=True)
ax1, ax2, ax3 = fig1.axes
#fig2 = plt.figure(2)
#axx1, axx2, axx3 = fig2.axes

# split by species, and then by dry / wet / ambient (or location)
for i in range(len(dfs)):

    try:
        species = np.unique(dfs[i]['species'])
        subdfs = [dfs[i][dfs[i]['species'] == e] for e in species]

    except KeyError:
        species = np.unique(dfs[i]['Species'])
        subdfs = [dfs[i][dfs[i]['Species'] == e] for e in species]

    for j in range(len(subdfs)):

        try:
            which = np.unique(subdfs[j]['Season'])

            if (len(which) <= 1) or (not any('dry' in e for e in which) and
               not any('wet' in e for e in which)):
                raise KeyError

            which = np.sort(which)[::-1]
            sub = [subdfs[j][subdfs[j]['Season'] == e] for e in which]        

            for df in sub:

                if df['Season'].iloc[0] == 'wet':
                    ax = ax1
                    #axx = axx1

                if df['Season'].iloc[0] == 'intermediate':
                    ax = ax2
                    #axx = axx2           

                if df['Season'].iloc[0] == 'dry':
                    ax = ax3
                    #axx = axx3

                try:
                    df = df[['LWP', 'LWPpd', 'hod']].dropna(axis=0, how='any')
                    x = df['LWP'].values - df['LWPpd'].values
                    xx = (df[df['hod'] == 12.]['LWP']).values

                    pdf1 = stats.gaussian_kde(x, bw_method='silverman')
                    spread1 = np.linspace(np.amin(x), np.amax(x), 200)

                    #pdf2 = stats.gaussian_kde(xx)
                    #spread2 = np.linspace(np.amin(xx), np.amax(xx), 200)

                    try:
                        ax.plot(spread1, pdf1(spread1), c=colours.get(species[j]),
                                alpha=0.6)
                        #axx.plot(spread2, pdf2(spread2), c=colours.get(species[j]),
                        #         alpha=0.6)

                    except ValueError:
                        pass

                except KeyError:
                    pass

        except KeyError:
            try:
                which = np.unique(subdfs[j]['Treatment'])

                if len(which) <= 1:
                    raise KeyError

                which = np.sort(which)[::-1]
                sub = [subdfs[j][subdfs[j]['Treatment'] == e] for e in which]

                for df in sub:

                    if df['Treatment'].iloc[0] == 'wet':
                        ax = ax1
                        #axx = axx1

                    if df['Treatment'].iloc[0] == 'intermediate':
                        ax = ax2
                        #axx = axx2           

                    if df['Treatment'].iloc[0] == 'dry':
                        ax = ax3
                        #axx = axx3

                    try:
                        df = df[['LWP', 'LWPpd', 'hod']].dropna(axis=0, how='any')
                        x = df['LWP'].values - df['LWPpd'].values
                        xx = (df[df['hod'] == 12.]['LWP']).values

                        pdf1 = stats.gaussian_kde(x, bw_method='silverman')
                        spread1 = np.linspace(np.amin(x), np.amax(x), 200)

                        #pdf2 = stats.gaussian_kde(xx)
                        #spread2 = np.linspace(np.amin(xx), np.amax(xx), 200)

                        try:
                            ax.plot(spread1, pdf1(spread1), c=colours.get(species[j]),
                                    alpha=0.6)
                            #axx.plot(spread2, pdf2(spread2), c=colours.get(species[j]),
                            #         alpha=0.6)

                        except ValueError:
                            pass

                    except KeyError:
                        pass

            except KeyError:
                which = np.sort(np.unique(subdfs[j]['opt']))[::-1]
                sub = [subdfs[j][subdfs[j]['opt'] == e] for e in which]

                for df in sub:

                    if df['opt'].iloc[0] == 'wet':
                        ax = ax1
                        #axx = axx1

                    if df['opt'].iloc[0] == 'intermediate':
                        ax = ax2
                        #axx = axx2           

                    if df['opt'].iloc[0] == 'dry':
                        ax = ax3
                        #axx = axx3

                    try:
                        df = df[['LWP', 'LWPpd', 'hod']].dropna(axis=0, how='any')
                        x = df['LWP'].values - df['LWPpd'].values
                        xx = (df[df['hod'] == 12.]['LWP']).values

                        pdf1 = stats.gaussian_kde(x, bw_method='silverman')
                        spread1 = np.linspace(np.amin(x), np.amax(x), 200)

                        #pdf2 = stats.gaussian_kde(xx)
                        #spread2 = np.linspace(np.amin(xx), np.amax(xx), 200)

                        try:
                            ax.plot(spread1, pdf1(spread1), c=colours.get(species[j]),
                                    alpha=0.6)
                            #axx.plot(spread2, pdf2(spread2), c=colours.get(species[j]),
                            #         alpha=0.6)

                        except ValueError:
                            pass

                    except KeyError:
                        pass

    dfs[i] = sub

# make the plots nice
fig1 = plt.figure(1)
ax1, ax2, ax3 = fig1.axes
#fig2 = plt.figure(2)
#axx1, axx2, axx3 = fig2.axes

ax1.set_title('Wet')
ax2.set_title('Intermediate')
ax3.set_title('Dry')

#axx1.set_title('Wet')
#axx2.set_title('Intermediate')
#axx3.set_title('Dry')

ax1.set_xlabel('Psi - Psi_pd', fontsize=14.)
ax2.set_xlabel('Psi - Psi_pd', fontsize=14.)
ax3.set_xlabel('Psi - Psi_pd', fontsize=14.)

#axx1.set_xlabel('Psi_midday', fontsize=14.)
#axx2.set_xlabel('Psi_midday', fontsize=14.)
#axx3.set_xlabel('Psi_midday', fontsize=14.)

fig1.savefig('pdf_Psi.png', dpi=1200, bbox_inches='tight')
#fig2.savefig('pdf_Psi_mid.png', dpi=1200, bbox_inches='tight')

exit(1)

# initialise the figures
for i in range(len(plot)):

    ini_fig(i + 1)

# split by species, and then by dry / wet / ambient (or location)
for i in range(len(dfs)):

    try:
        species = np.unique(dfs[i]['species'])
        subdfs = [dfs[i][dfs[i]['species'] == e] for e in species]

    except KeyError:
        species = np.unique(dfs[i]['Species'])
        subdfs = [dfs[i][dfs[i]['Species'] == e] for e in species]

    for j in range(len(subdfs)):

        try:
            which = np.unique(subdfs[j]['Season'])

            if (len(which) <= 1) or (not any('dry' in e for e in which) and
               not any('wet' in e for e in which)):
                raise KeyError

            which = np.sort(which)[::-1]
            sub = [subdfs[j][subdfs[j]['Season'] == e] for e in which]

            for k in range(len(plot)):

                fig = plt.figure(k + 1)
                ax1, ax2, ax3 = fig.axes           

                for df in sub:

                    if df['Season'].iloc[0] == 'wet':
                        ax = ax1

                    if df['Season'].iloc[0] == 'intermediate':
                        ax = ax2             

                    if df['Season'].iloc[0] == 'dry':
                        ax = ax3

                    try:
                        df = df[[x[k], y[k]]].dropna(axis=0, how='any')
                        xx = df[x[k]].values
                        yy = df[y[k]].values

                        # pre-process
                        if x[k] == 'hod':  # average the data per hod
                            bins = df.groupby([x[k]]).mean()
                            weights = df.groupby([x[k]])[x[k]].value_counts()
                            px = bins.index.values
                            py = bins[y[k]].values

                        else:  # bin the distribution
                            df.sort_values(by=[x[k]], inplace=True)
                            bins = pd.cut(df[x[k]], bins=7)
                            weights = bins.value_counts()
                            bins = df.groupby(bins).mean()
                            px = bins[x[k]].values
                            py = bins[y[k]].values

                        # deal with NaNs 
                        ok = ~np.isnan(px)
                        xp = ok.ravel().nonzero()[0]
                        fp = px[ok]
                        xxx = np.isnan(px).ravel().nonzero()[0]
                        px[np.isnan(px)] = np.interp(xxx, xp, fp)

                        ok = ~np.isnan(py)
                        yp = ok.ravel().nonzero()[0]
                        fp = py[ok]
                        yyy = np.isnan(py).ravel().nonzero()[0]
                        py[np.isnan(py)] = np.interp(yyy, yp, fp)
                            
                        # 4th order gam fit
                        gam = (LinearGAM(n_splines=5, spline_order=4)
                                        .gridsearch(px.reshape(-1, 1), py,
                                                    weights=weights.values))
                        px = np.linspace(np.percentile(xx, 10),
                                         np.percentile(xx, 90), num=500)
                        py = gam.predict(px)
                        px = px[py >= 0.]
                        py = py[py >= 0.]

                        try:
                            ax.scatter(xx, yy, c=colours.get(species[j]),
                                       alpha=0.2)
                            ax.plot(px, py, color=colours.get(species[j]),
                                    linewidth=3.)

                        except ValueError:
                            pass

                    except KeyError:
                        pass

        except KeyError:
            try:
                which = np.unique(subdfs[j]['Treatment'])

                if len(which) <= 1:
                    raise KeyError

                which = np.sort(which)[::-1]
                sub = [subdfs[j][subdfs[j]['Treatment'] == e] for e in which]

                for k in range(len(plot)):

                    fig = plt.figure(k + 1)
                    ax1, ax2, ax3 = fig.axes           

                    for df in sub:

                        if df['Treatment'].iloc[0] == 'wet':
                            ax = ax1

                        if df['Treatment'].iloc[0] == 'intermediate':
                            ax = ax2             

                        if df['Treatment'].iloc[0] == 'dry':
                            ax = ax3

                        try:
                            df = df[[x[k], y[k]]].dropna(axis=0, how='any')
                            xx = df[x[k]].values
                            yy = df[y[k]].values

                            # pre-process
                            if x[k] == 'hod':  # average the data per hod
                                bins = df.groupby([x[k]]).mean()
                                weights = (df.groupby([x[k]])[x[k]]
                                             .value_counts())
                                px = bins.index.values
                                py = bins[y[k]].values

                            else:  # bin the distribution
                                df.sort_values(by=[x[k]], inplace=True)
                                bins = pd.cut(df[x[k]], bins=7)
                                weights = bins.value_counts()
                                bins = df.groupby(bins).mean()
                                px = bins[x[k]].values
                                py = bins[y[k]].values

                            # deal with NaNs 
                            ok = ~np.isnan(px)
                            xp = ok.ravel().nonzero()[0]
                            fp = px[ok]
                            xxx  = np.isnan(px).ravel().nonzero()[0]
                            px[np.isnan(px)] = np.interp(xxx, xp, fp)

                            ok = ~np.isnan(py)
                            yp = ok.ravel().nonzero()[0]
                            fp = py[ok]
                            yyy  = np.isnan(py).ravel().nonzero()[0]
                            py[np.isnan(py)] = np.interp(yyy, yp, fp)
                                
                            # 4th order gam fit
                            gam = (LinearGAM(n_splines=5, spline_order=4)
                                            .gridsearch(px.reshape(-1, 1), py,
                                                       weights=weights.values))
                            px = np.linspace(np.percentile(xx, 10),
                                             np.percentile(xx, 90), num=500)
                            py = gam.predict(px)
                            px = px[py >= 0.]
                            py = py[py >= 0.]

                            try:
                                ax.scatter(xx, yy, c=colours.get(species[j]),
                                           alpha=0.2)
                                ax.plot(px, py, color=colours.get(species[j]),
                                        linewidth=3.)

                            except ValueError:
                                pass

                        except KeyError:
                            pass

            except KeyError:
                which = np.sort(np.unique(subdfs[j]['opt']))[::-1]
                sub = [subdfs[j][subdfs[j]['opt'] == e] for e in which]

                for k in range(len(plot)):

                    fig = plt.figure(k + 1)
                    ax1, ax2, ax3 = fig.axes           

                    for df in sub:

                        if df['opt'].iloc[0] == 'wet':
                            ax = ax1

                        if df['opt'].iloc[0] == 'intermediate':
                            ax = ax2             

                        if df['opt'].iloc[0] == 'dry':
                            ax = ax3

                        try:
                            df = df[[x[k], y[k]]].dropna(axis=0, how='any')
                            xx = df[x[k]].values
                            yy = df[y[k]].values

                            # pre-process
                            if x[k] == 'hod':  # average the data per hod
                                bins = df.groupby([x[k]]).mean()
                                weights = (df.groupby([x[k]])[x[k]]
                                             .value_counts())
                                px = bins.index.values
                                py = bins[y[k]].values

                            else:  # bin the distribution
                                df.sort_values(by=[x[k]], inplace=True)
                                bins = pd.cut(df[x[k]], bins=7)
                                weights = bins.value_counts()
                                bins = df.groupby(bins).mean()
                                px = bins[x[k]].values
                                py = bins[y[k]].values

                            # deal with NaNs 
                            ok = ~np.isnan(px)
                            xp = ok.ravel().nonzero()[0]
                            fp = px[ok]
                            xxx  = np.isnan(px).ravel().nonzero()[0]
                            px[np.isnan(px)] = np.interp(xxx, xp, fp)

                            ok = ~np.isnan(py)
                            yp = ok.ravel().nonzero()[0]
                            fp = py[ok]
                            yyy  = np.isnan(py).ravel().nonzero()[0]
                            py[np.isnan(py)] = np.interp(yyy, yp, fp)
                                
                            # 4th order gam fit
                            gam = (LinearGAM(n_splines=5, spline_order=4)
                                            .gridsearch(px.reshape(-1, 1), py,
                                                       weights=weights.values))
                            px = np.linspace(np.percentile(xx, 10),
                                             np.percentile(xx, 90), num=500)
                            py = gam.predict(px)
                            px = px[py >= 0.]
                            py = py[py >= 0.]

                            try:
                                ax.scatter(xx, yy, c=colours.get(species[j]),
                                           alpha=0.2)
                                ax.plot(px, py, color=colours.get(species[j]),
                                        linewidth=3.)

                            except ValueError:
                                pass

                        except KeyError:
                            pass

    dfs[i] = sub

# make the plots nice
for i in range(len(plot)):

    fig = plt.figure(i + 1)
    ax1, ax2, ax3 = fig.axes

    ax1.set_title('Wet')
    ax2.set_title('Intermediate')
    ax3.set_title('Dry')

    ax1.set_ylabel(ylab[i], fontsize=14.)
    ax1.set_xlabel(xlab[i], fontsize=14.)
    ax2.set_xlabel(xlab[i], fontsize=14.)
    ax3.set_xlabel(xlab[i], fontsize=14.)

    fig.savefig('%s.png' % plot[i], dpi=1200, bbox_inches='tight')

fig = plt.figure(i + 2, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
handles = [Line2D([0], [0], color=colours.get(j), marker='o') for j in spps]
plt.legend(handles, spps, frameon=False)
plt.tight_layout()
fig.savefig('legend.png', dpi=1200, bbox_inches='tight')

