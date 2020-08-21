# general modules
import os
import sys
import numpy as np
import pandas as pd  # read/write dataframes, csv files
from scipy.integrate import quad  # integrate on a range

# plotting
import matplotlib.pyplot as plt

# change the system path to load modules from TractLSM
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))

# own modules
from TractLSM import conv, cst
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files

fdir = '/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/input/calibrations/obs_driven/'

df = pd.read_csv('/mnt/c/Users/le_le/Work/One_gs_model_to_rule_them_all/output/calibrations/obs_driven/overview_of_fits.csv')

colours = ['#a6611a', '#ca0020', '#dfc27d', '#1a9641', '#f4a582', '#decbe4'] * 3


fig, ax = plt.subplots()
pos = 0.
x = []

for file in os.listdir(fdir):

    if '_x.csv' in file:
        df1, __ = read_csv(os.path.join(fdir, file))
        df2, __ = read_csv(os.path.join(fdir, file.replace('_x.csv', '_y.csv')))

        if 'Pleaf' in df2.columns:

            site_spp = file.split('_x.csv')[0].split('_')

            if 'Quercus' in site_spp:
                x += ['$%s.$ $%s$ (%s)' % (site_spp[-2][0], site_spp[-1],
                                          site_spp[0][0])]

            else:
                x += ['$%s.$ $%s$' % (site_spp[-2][0], site_spp[-1])]

            thresh1 = df2['E'].quantile(0.25)  # only keep 'wetter' soil
            thresh2 = df1['Ps'].quantile(0.25)
            df2 = df2[np.logical_and(df2['E'] >= thresh1, df1['Ps'] >= thresh2)]
            df2.reset_index(inplace=True)

            kmax = []

            for i in range(len(df2)):

                kmax += [df2.loc[i, 'E'] / np.abs(df2.loc[i, 'Pleaf'] -
                                                  df1.loc[i, 'Ps'])]

            kmax = np.array(kmax)
            bp = ax.boxplot(kmax[~np.isnan(kmax)], positions=[pos],
                            widths=0.8, patch_artist=True, zorder=0)

            # add param values
            where = df['training'].unique()
            iwhere = [(e in file) for e in where]
            where = where[iwhere]

            try:
                sub = df.copy()[df['training'] == where[0]]

                keep = np.logical_or(sub['p1'].str.contains('kmax'),
                                      sub['p1'].str.contains('krl'))
                sub = sub[keep].sort_values(by=['solver', 'Model'])
                sub.reset_index(inplace=True)

                for j in range(len(sub)):

                    ax.scatter([pos], [sub.loc[j, 'v1']], c=colours[j],
                               label=sub.loc[j, 'Model'], zorder=20)

            except IndexError:
                pass

            pos += 1.

        else:
            pass

ax.set_ylim(0, 16.)
ax.set_xticklabels(x, rotation=90)

h, l = ax.get_legend_handles_labels()
ax.legend(h[:6], l[:6], loc=1)
fig.savefig('param_spread.png', dpi=300, bbox_inches='tight')
