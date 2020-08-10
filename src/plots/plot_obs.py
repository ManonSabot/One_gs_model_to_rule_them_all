# general modules
import numpy as np
import pandas as pd  # read/write dataframes, csv files

# plotting
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.spatial import ConvexHull
from pygam import ExpectileGAM  # fit the functional shapes


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


def chaikins_corner_cutting(coords, refinements=5000):

    for _ in range(refinements):

        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]

    return L * 0.75 + R * 0.25


def encircle(ax, x, y, smooth=False, **kw):

    p = np.c_[x, y]  # concatenate along second axis
    hull = ConvexHull(p)
    p = p[hull.vertices, :]

    if smooth:
        p = chaikins_corner_cutting(p)

    poly = plt.Polygon(p, **kw)
    ax.add_patch(poly)

    return


def E_A_relationships_backup(df, colours):

    keep = ['Richmond_Eucalyptus_dunnii', 'ManyPeaksRange_Alphitonia_excelsa',
            'ManyPeaksRange_Austromyrtus_bidwillii',
            'ManyPeaksRange_Brachychiton_australis',
            'ManyPeaksRange_Cochlospermum_gillivraei',
            'Richmond_Eucalyptus_saligna', 'Puechabon_Quercus_ilex',
            'Vic_la_Gardiole_Quercus_ilex', 'Richmond_Eucalyptus_cladocalyx',
            'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis',
            'Corrigin_Eucalyptus_capillosa']

    fig, axes = plt.subplots(round(len(keep) / 3), 3, figsize=(15, 20))

    i = 0  # loop over the axes
    iter = 0  # first column

    # E - A relationships
    for what in keep:

        sub = df.copy()[df['site_spp'] == what]

        # plot the obs functional relationship
        x = sub['E']
        y = sub['A']
        xmin = np.nanmin(x) / 3.
        xmax = 3. * np.nanmax(x)
        ymin = np.nanmin(y) / 3.
        ymax = 3. * np.nanmax(y)

        encircle(axes[i][iter], x, y, ec='none', fc='gray', alpha=0.15)
        gam = (ExpectileGAM(expectile=0.5, n_splines=5, spline_order=4)
                           .gridsearch(x.values.reshape(-1, 1), y.values))
        px = np.linspace(x.min(), x.max(), num=500)
        py = gam.predict(px)
        axes[i][iter].plot(px, py, color='k', ls='--', linewidth=4.)

        j = 0

        for mod in ['std2', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
                    'sox2', 'cap', 'mes']:

            subsub = sub[['E(%s)' % (mod), 'A(%s)' % (mod)]].dropna()

            xx = subsub['E(%s)' % (mod)]
            yy = subsub['A(%s)' % (mod)]

            # deal with the nans + non-physical values
            valid = np.logical_and(xx >= xmin, xx < xmax)
            xx = xx[valid]
            yy = yy[valid]

            valid = np.logical_and(yy >= ymin, yy < ymax)
            xx = xx[valid]
            yy = yy[valid]

            try:
                axes[i][iter].scatter(xx, yy, marker='+', c=colours[j], s=40.,
                                      label=which_model(mod))

            except Exception:
                pass

            j += 1

        # tighten the subplots
        __, right = axes[i][iter].get_xlim()
        axes[i][iter].set_xlim(np.maximum(-0.25, xmin),
                               np.minimum(3. * xmax, right))

        __, high = axes[i][iter].get_ylim()
        axes[i][iter].set_ylim(np.maximum(-0.25, ymin),
                               np.minimum(3. * ymax, high))

        site = what.split('_')[0]

        if site == 'Vic':
            site = '%s %s %s' % (site, what.split('_')[1], what.split('_')[2])
            species = '$%s$ $%s$' % (what.split('_')[3], what.split('_')[4])

        else:
            species = '$%s$ $%s$' % (what.split('_')[1], what.split('_')[2])

        if 'Quercus' in species:
            title = '%s (%s)' % (species, site)

        else:
            title = species

        axes[i][iter].set_title(title)

        iter += 1  # second or third column

        if iter == len(axes[0]):
            iter = 0
            i += 1

    fig.savefig('E_A_functional.png', dpi=300, bbox_inches='tight')


def set_box_color(bp, colour):

    plt.setp(bp['boxes'], color='grey', linewidth=1.)
    plt.setp(bp['boxes'], facecolor=colour)

    if colour == 'w':
        plt.setp(bp['whiskers'], color='grey')
        plt.setp(bp['caps'], color='grey')

    else:
        plt.setp(bp['whiskers'], color=colour)
        plt.setp(bp['caps'], color=colour)

    plt.setp(bp['medians'], color='grey', linewidth=1.)
    plt.setp(bp['fliers'], color='grey', markersize=1.5)


def E_A_relationships(df, colours):

    keep = ['Richmond_Eucalyptus_dunnii', 'ManyPeaksRange_Alphitonia_excelsa',
            'ManyPeaksRange_Austromyrtus_bidwillii',
            'ManyPeaksRange_Brachychiton_australis',
            'ManyPeaksRange_Cochlospermum_gillivraei',
            'Richmond_Eucalyptus_saligna', 'Puechabon_Quercus_ilex',
            'Vic_la_Gardiole_Quercus_ilex', 'Richmond_Eucalyptus_cladocalyx',
            'Sevilleta_Juniperus_monosperma', 'Sevilleta_Pinus_edulis',
            'Corrigin_Eucalyptus_capillosa']

    fig = plt.figure(figsize=(16, 20))

    iter = 0  # keep track of the sites

    for row in range(3, -1, -1):  # adding suplots from the top

        b = row / 4.
        t = (row + 1) / 4.
        t = t - (t - b) / 15.

        for col in range(3):  # adding from the left

            l = col / 3.
            r = (col + 1) / 3.
            r = r - (r - l) / 15.

            gs = fig.add_gridspec(nrows=6, ncols=6, left=l, right=r, top=t,
                                  bottom=b, wspace=0., hspace=0.)
            ax = fig.add_subplot(gs[:-1, 1:])
            ax_y = fig.add_subplot(gs[:-1, 0], xticklabels=[], sharey=ax)
            ax_x = fig.add_subplot(gs[-1, 1:], yticklabels=[], sharex=ax)

            for axis in [ax_y, ax_x]:

                # removing the spines
                axis.spines['right'].set_visible(False)
                axis.spines['top'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.patch.set_visible(False)

                # removing the double tick marks
                axis.tick_params(left=False, labelleft=False, bottom=False,
                                 labelbottom=False)

            # E - A relationships
            sub = df.copy()[df['site_spp'] == keep[iter]]

            # plot the obs functional relationship
            x = sub['E']
            y = sub['A']
            xmin = np.nanmin(x) / 2.5
            xmax = 2.5 * np.nanmax(x)
            ymin = np.nanmin(y) / 2.5
            ymax = 2.5 * np.nanmax(y)

            encircle(ax, x, y, ec='none', fc='gray', alpha=0.15)
            gam = (ExpectileGAM(expectile=0.5, n_splines=5, spline_order=4)
                               .gridsearch(x.values.reshape(-1, 1), y.values))
            px = np.linspace(x.min(), x.max(), num=500)
            py = gam.predict(px)
            ax.plot(px, py, color='k', ls='--', linewidth=4.)

            bp = ax_y.boxplot(y, widths=0.35, patch_artist=True)
            set_box_color(bp, 'w')
            ax_y.invert_xaxis()

            bp = ax_x.boxplot(x, vert=False, widths=0.35, patch_artist=True)
            set_box_color(bp, 'w')
            ax_x.invert_yaxis()

            j = 0

            for mod in ['std2', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn',
                        'lcst', 'sox2', 'cap', 'mes']:

                subsub = sub[['E(%s)' % (mod), 'A(%s)' % (mod)]].dropna()

                xx = subsub['E(%s)' % (mod)]
                yy = subsub['A(%s)' % (mod)]

                # deal with the nans + non-physical values
                valid = np.logical_and(xx >= xmin, xx < xmax)
                xx = xx[valid]
                yy = yy[valid]

                valid = np.logical_and(yy >= ymin, yy < ymax)
                xx = xx[valid]
                yy = yy[valid]

                try:
                    ax.scatter(xx, yy, marker='+', c=colours[j], s=40.,
                               label=which_model(mod))

                except Exception:
                    pass

                j += 1

            # tighten the subplots
            __, right = ax.get_xlim()
            ax.set_xlim(np.maximum(-0.25, xmin), np.minimum(3. * xmax, right))

            __, high = ax.get_ylim()
            ax.set_ylim(np.maximum(-0.25, ymin), np.minimum(3. * ymax, high))

            pad = 40.

            if row == 0:
                ax.set_xlabel('E (mmol m$^{-2}$ s$^{-1}$)',
                              labelpad=4. / 5. * pad)

            if col == 0:
                ax.set_ylabel('A$_n$ ($\mu$mol m$^{-2}$ s$^{-1}$)',
                              labelpad=pad)

            site = keep[iter].split('_')[0]

            if site == 'Vic':
                site = '%s %s %s' % (site, keep[iter].split('_')[1],
                                     keep[iter].split('_')[2])

            species = '$%s$ $%s$' % (keep[iter].split('_')[-2],
                                     keep[iter].split('_')[-1])

            if 'Quercus' in species:
                title = '%s (%s)' % (species, site)

            else:
                title = species

            ax.set_title(title)

            iter += 1  # second or third column

    fig.savefig('E_A_functional.png', dpi=300, bbox_inches='tight')


def gs_Ci_clusters(df, colours):

    keep = ['Richmond_Eucalyptus_dunnii', 'Richmond_Eucalyptus_saligna',
            'Puechabon_Quercus_ilex', 'Vic_la_Gardiole_Quercus_ilex',
            'Richmond_Eucalyptus_cladocalyx', 'Sevilleta_Juniperus_monosperma',
            'Sevilleta_Pinus_edulis', 'Corrigin_Eucalyptus_capillosa']

    fig, axes = plt.subplots(round(len(keep) / 3), 3, figsize=(15, 15))

    i = 0  # loop over the axes
    iter = 0  # first column

    # Ci - gs relationships
    for what in keep:

        sub = df.copy()[df['site_spp'] == what]

        j = 0

        for mod in ['std2', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
                    'sox2', 'cap', 'mes']:

            xy = sub[['Ci', 'gs', 'Ci(%s)' % (mod), 'gs(%s)' % (mod)]].dropna()

            x = xy['gs(%s)' % (mod)] / xy['gs']
            y = xy['Ci(%s)' % (mod)] / xy['Ci']

            # deal with the nans
            valid = np.logical_and(xy['Ci(%s)' % (mod)] < 9999.,
                                   xy['gs(%s)' % (mod)] < 9999.)
            x = x[valid]
            y = y[valid]

            # interquartile ranges
            valid = np.logical_and(x / y > (x / y).quantile(0.25),
                                   x / y < (x / y).quantile(0.75))

            try:
                encircle(axes[i][iter], x[valid], y[valid], smooth=True,
                         ec=colours[j], fc='None', linewidth=2.,
                         label=which_model(mod))

            except Exception:
                print(what, mod)
                pass

            j += 1

        axes[i][iter].hlines(1., 0., 1., linestyle=':',
                             transform=axes[i][iter].get_yaxis_transform())
        axes[i][iter].vlines(1., 0., 1., linestyle=':',
                             transform=axes[i][iter].get_xaxis_transform())

        # tighten the subplots
        __, right = axes[i][iter].get_xlim()
        axes[i][iter].set_xlim(0., right)

        site = what.split('_')[0]

        if site == 'Vic':
            site = '%s %s %s' % (site, what.split('_')[1], what.split('_')[2])
            species = '%s %s' % (what.split('_')[3], what.split('_')[4])

        else:
            species = '%s %s' % (what.split('_')[1], what.split('_')[2])

        if 'Quercus' in species:
            title = '%s (%s)' % (species, site)

        else:
            title = species

        axes[i][iter].set_title(title)

        iter += 1  # second or third column

        if iter == len(axes[0]):
            iter = 0
            i += 1

    axes[-1][-1].axis('off')
    axes[0][-1].legend(bbox_to_anchor=(1., -2.4), loc=4, frameon=False)

    fig.savefig('Ci_gs_clustered_test2.png', dpi=300, bbox_inches='tight')


def LWP_box_plots(df, colours):

    # histograms
    groups = ['ManyPeaksRange', 'Richmond', 'Sevilleta', 'Quercus']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    select = df.filter(like='Pleaf').columns.to_list()

    i = 0  # loop over the axes
    iter = 0  # first column

    for what in groups:

        sub = df.copy()[df['site_spp'].str.contains(what)]

        for c in select:

            sub[c] -= sub['Ps_pd']  # we're looking at deltaPleaf
            sub[c][sub[c] >= 0.] =  np.nan  # mask nonsense
            sub[c][sub[c] < -100.] = np.nan  # mask nonsense

        pos = 0.
        bp = axes[i][iter].boxplot([sub['Pleaf'].dropna()], positions=[pos],
                                   widths=0.8, patch_artist=True)
        set_box_color(bp, 'w')
        interquartiles = [sub['Pleaf'].dropna().quantile(0.25),
                          sub['Pleaf'].dropna().quantile(0.75)]
        axes[i][iter].fill_between(np.arange(-0.5, 12.), interquartiles[0],
                                   interquartiles[1], color='#f0f0f0',
                                   zorder=-1)

        j = 0

        for mod in ['std2', 'tuz', 'sox1', 'wue', 'cmax', 'pmax', 'cgn', 'lcst',
                    'sox2', 'cap', 'mes']:

            pos += 1.

            try:
                if sub['Pleaf(%s)' % (mod)].dropna().quantile(0.75) > -16.:
                    bp = axes[i][iter].boxplot([sub['Pleaf(%s)' % (mod)]
                                                   .dropna()], positions=[pos],
                                               widths=0.8, patch_artist=True)
                    set_box_color(bp, colours[j])

            except Exception:
                pass

            j += 1

        if axes[i][iter].get_ylim()[0] < -16.:
            axes[i][iter].set_ylim(-16., 0.1)

        else:
            axes[i][iter].set_ylim(axes[i][iter].get_ylim()[0], 0.1)

        axes[i][iter].set_xlim(-0.5, 11.5)
        axes[i][iter].set_title(what)

        i += 1

        if i == len(axes):
            i = 0
            iter += 1  # second or third column

    fig.savefig('LWP_boxes.png', dpi=1200, bbox_inches='tight')

# Import Data
df = pd.read_csv('/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/output/simulations/obs_driven/all_site_spp_simulations.csv')

# Draw Plot
colours = ['#1a1a1a', '#984ea3', '#decbe4', '#0571b0', '#92c5de', '#1a9641',
           '#a6d96a', '#ca0020', '#f4a582', '#a6611a', '#dfc27d']
#plt.rcParams['axes.prop_cycle'] = cycler(color=colours)
plt.rcParams['text.usetex'] = True  # use LaTeX
plt.rcParams['text.latex.preamble'] = [r'\usepackage{avant}',
                                       r'\usepackage{mathpazo}',
                                       r'\usepackage{amsmath}']

#E_A_relationships(df, colours)
Ci_gs_clusters(df, colours)
#LWP_box_plots(df, colours)
