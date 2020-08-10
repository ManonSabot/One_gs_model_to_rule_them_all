"""
# test SOX
fname3 = os.path.join(os.path.join(base_dir, 'output'),
                      '%s_%s.csv' % ('testSOX', which1))

if not os.path.isfile(fname3):  # create file if it doesn't exist

    fig, ax = plt.subplots()
    D = df1['VPD'].copy()
    PAR = df1['PPFD'].copy()
    Ca = df1['CO2'].copy()

    df1['Vmax25'] = 100.
    df1['P50'] = 2.
    df1['P88'] = 2.5
    df1['kmax'] = 1.
    df1['VPD'] = 0.5
    df1['Ps'] = -0.3
    df1['coszen'] = 0.85
    df1['PPFD'][df1['PPFD'] > 50.] = 600.
    PAR2 = df1['PPFD'].copy()

    df3 = hrun(fname3, df1, len(df1.index), 'Farquhar', models=['SOX12'])

    mask = np.logical_and(df3['gs(xi1)'] > 0., df3['gs(xi2)'] > 0.)
    ax.scatter(df3['gs(xi2)'][mask], df3['gs(xi1)'][mask], label='temperature')

    df1['Tair'] = 25.
    df1['VPD'] = D

    df3 = hrun(fname3, df1, len(df1.index), 'Farquhar', models=['SOX12'])

    mask = np.logical_and(df3['gs(xi1)'] > 0., df3['gs(xi2)'] > 0.)
    ax.scatter(df3['gs(xi2)'][mask], df3['gs(xi1)'][mask], label='VPD')

    df1['VPD'] = 0.5
    df1['PPFD'] = PAR

    df3 = hrun(fname3, df1, len(df1.index), 'Farquhar', models=['SOX12'])

    mask = np.logical_and(df3['gs(xi1)'] > 0., df3['gs(xi2)'] > 0.)
    ax.scatter(df3['gs(xi2)'][mask], df3['gs(xi1)'][mask], label='PAR')

    df1['PPFD'] = PAR2
    df1['CO2'] = np.random.uniform(0.5 * df1['CO2'], 4. * df1['CO2'], len(df1))

    df3 = hrun(fname3, df1, len(df1.index), 'Farquhar', models=['SOX12'])

    mask = np.logical_and(df3['gs(xi1)'] > 0., df3['gs(xi2)'] > 0.)
    ax.scatter(df3['gs(xi2)'][mask], df3['gs(xi1)'][mask], label='Ca')

    df1['CO2'] = Ca    
    df1['Ps'] = np.random.uniform(-0.05, -2.5, len(df1))

    df3 = hrun(fname3, df1, len(df1.index), 'Farquhar', models=['SOX12'])

    mask = np.logical_and(df3['gs(xi1)'] > 0., df3['gs(xi2)'] > 0.)
    ax.scatter(df3['gs(xi2)'][mask], df3['gs(xi1)'][mask], label='Ppd')

    ax.plot([0., 0.3], [0., 0.3], '-k')

    ax.set(xlabel='numerical SOX', ylabel='analytical SOX',
           title='Whatay joke')
    ax.legend()

    plt.show()
    exit(1)
"""
