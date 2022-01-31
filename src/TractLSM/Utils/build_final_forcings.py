# -*- coding: utf-8 -*-

"""
Generate csv driver input, either random (from weather generator) or
from site-level forcing, including parameters that the user can manually
change.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Forcing data csv generator"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (02.03.2018)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import warnings  # catch warnings
import numpy as np  # array manipulations, math operators
import pandas as pd  # read/write dataframes, csv files

# own modules
try:
    from general_utils import read_csv  # read in files
    from default_params import default_params  # forcing parameters
    import weather_generator as weather  # alternative forcings
    from calculate_solar_geometry import cos_zenith  # geom

except (ImportError, ModuleNotFoundError):
    from TractLSM.Utils.general_utils import read_csv
    from TractLSM.Utils.default_params import default_params
    import TractLSM.Utils.weather_generator as weather
    from TractLSM.Utils.calculate_solar_geometry import cos_zenith


# ======================================================================

def main(fname, d, Ndays=1, year=None):

    """
    Main function: either uses the weather generator to generate N days
                   of met data or writes a csv file containing
                   site-level forcings. Then, concatenates the remaining
                   necessary parameters which can be defined by the user
                   or be the default values.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    Ndays: int
        number of met data days to generate

    year: int
        selected year, e..g, subsampling from observational data

    Returns:
    --------
    The comprehensive csv forcing file.

    """

    if Ndays is not None:
        ppfd, tdays, precip, vpd, u = get_met_data_wg(d, Ndays=Ndays)
        write_csv_wg(fname, d, ppfd, tdays, precip, vpd, u)

    warnings.simplefilter('ignore',
                          category=pd.io.pytables.PerformanceWarning)
    add_vars_to_csv(fname, d)

    return


# ======================================================================

# ~~~ Other functions are defined here ~~~

def get_met_data_wg(d, Ndays):

    """
    Generates weather data for Ndays(s) data using the weather
    generator.

    Arguments:
    ----------
    d: class or pandas series
        contains the non-default parameters

    Ndays: int
        number of met data days to generate

    Returns:
    --------
    ppfd: array
        Nday diurnal course of the par [umol m-2 s-1]

    tdays: array
        Nday diurnal time course of the temperature [degC]

    precip: array
        Nday diurnal time course of rainfall [mm]

    vpd: array
        Nday diurnal course of the vapor pressure deficit [kPa]

    u: array
        Nday diurnal course of the wind speed [m s-1]

    """

    # declare empty arrays for output
    ppfd = np.zeros((Ndays, 48))
    tdays = np.zeros((Ndays, 48))
    precip = np.zeros((Ndays, 48))
    vpd = np.zeros((Ndays, 48))
    u = np.zeros((Ndays, 48))

    for i in range(Ndays):

        doy = d.doy + float(i)

        if i == 0:
            sw_rad_day = d.sw_rad_day
            tmin = d.tmin
            tmax = d.tmax
            rain_day = d.rain_day
            rh = d.RH
            u0 = d.u

        # randomly change the inputs everyday
        tmin += np.random.uniform(-tmin / 5., tmin / 5.)
        tmax += np.random.uniform(-tmax / 5., tmax / 5.)

        # bound the inputs for them to still be physically acceptable
        if tmax >= d.tmax * 5. / 3.:
            tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)
            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        if tmin <= - d.tmin * 15. / 2.:
            tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)
            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        if tmax <= tmin:
            if tmin >= 18.:
                tmin = d.tmin + np.random.uniform(-tmin / 5., tmin / 5.)

            tmax = d.tmax + np.random.uniform(-tmax / 5., tmax / 5.)

        # randomly change the inputs everyday
        rain_day += np.random.uniform(-rain_day / 5., rain_day / 5.)

        # bound the inputs for them to still be physically acceptable
        if (rain_day < 0.) or (rain_day > d.rain_day * 20.):
            rain_day = d.rain_day

        if i > 0:  # randomly change the inputs after 1st day
            u0 += np.random.uniform(-u0 / 2., u0 / 2.)

            # bound the inputs for them to still be acceptable
            if (u0 < 0.) or (u0 > 30.):
                u0 = d.u

        ppfd[i], tdays[i], precip[i], vpd[i], u[i] = weather.main(d.lat, d.lon,
                                                                  doy,
                                                                  sw_rad_day,
                                                                  tmin, tmax,
                                                                  rain_day, rh,
                                                                  u0)

    return ppfd, tdays, precip, vpd, u


def write_csv_wg(fname, d, ppfd, tdays, precip, vpd, u):

    """
    Writes the forcing data generated by the weather generator into a
    csv.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    ppfd: array
        Nday diurnal course of the par [umol m-2 s-1]

    tdays: array
        Nday diurnal time course of the temperature [degC]

    precip: array
        Nday diurnal time course of rainfall [mm]

    vpd: array
        Nday diurnal course of the vapor pressure deficit [kPa]

    u: array
        Nday diurnal course of the wind speed [m s-1]

    Returns:
    --------
    Creates the csv forcing file.

    """

    if d is None:
        d = default_params()

    elif type(d) == str:
        d = read_csv(d, drop_units=False)

    # generate the time series (WG.hours is N 0.5 h / day)
    WG = weather.WeatherGenerator(d.lat, d.lon)
    time = np.zeros(len(WG.hours) * ppfd.shape[0])
    step = WG.hours[1] - WG.hours[0]
    time[0] = d.doy * 24.

    for i in range(1, len(time)):

        time[i] = time[i-1] + step

    # column names
    varnames = ('doy', 'hod', 'PPFD', 'Tair', 'precip', 'VPD', 'u')
    units = ('[-]', '[h]', '[umol m-2 s-1]', '[deg C]', '[mm d-1]', '[kPa]',
             '[m s-1]')
    ppfd = np.concatenate(ppfd, axis=0)  # same size as the time series
    tdays = np.concatenate(tdays, axis=0)
    precip = np.concatenate(precip, axis=0)
    vpd = np.concatenate(vpd, axis=0)
    u = np.concatenate(u, axis=0)

    # create the doy series that matches the time series
    doy = np.asarray(time / 24.).astype(int)
    hod = np.tile(np.arange(0.5, 24.5, 0.5), len(np.unique(doy)))

    # is the sun up?
    cos_zen = [cos_zenith(doy[i], hod[i], d.lat, d.lon)
               for i in range(len(hod))]
    ppfd[np.where(90. - np.degrees(np.arccos(cos_zen)) <= 0.)] = 0.
    ppfd[np.where(ppfd <= 50.)] = 0.

    # write the csv
    these_headers = pd.MultiIndex.from_tuples(list(zip(varnames, units)))
    df = pd.DataFrame([doy, hod, ppfd, tdays, precip, vpd, u]).T
    df.columns = these_headers
    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return


def update_site_params(df, p):

    """
    Updates a pandas series object used to store the model's parameters
    with the site-specific parameters.

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all the data

    p: pandas series
        site parameters

    Returns:
    --------
    p: pandas series
        updated model's parameters (with site info)

    """

    ds = df.iloc[0]

    try:
        if str(ds['Vmax25']) != str(pd.np.NaN):
            p.Vmax25 = ds['Vmax25']

    except KeyError:
        pass

    try:
        if str(ds['albedo_l']) != str(pd.np.NaN):
            p.albedo_l = ds['albedo_l']

    except KeyError:
        pass

    try:
        if str(ds['max_leaf_width']) != str(pd.np.NaN):
            p.max_leaf_width = ds['max_leaf_width']

    except KeyError:
        pass

    try:
        if str(ds['P50']) != str(pd.np.NaN):
            p.P50 = ds['P50']

    except KeyError:
        pass

    try:
        if str(ds['P88']) != str(pd.np.NaN):
            p.P88 = ds['P88']

    except KeyError:
        pass

    try:
        if str(ds['ratiocrit']) != str(pd.np.NaN):
            p.ratiocrit = ds['ratiocrit']

    except KeyError:
        pass

    try:
        if str(ds['Psie']) != str(pd.np.NaN):
            p.Psie = ds['Psie']
            p.Ps = ds['Psie']

    except KeyError:
        pass

    return p


def add_vars_to_csv(fname, d):

    """
    Appends the parameters that the user can manually change to the
    forcing csv file.

    Arguments:
    ----------
    fname: string
        csv output filename (with path), preferably stored in the input/
        folder. The corresponding met (and flux and LAI) data must be
        stored in input/fluxsites/.

    d: class or pandas series
        contains the non-default parameters

    Returns:
    --------
    The comprehensive csv forcing file.

    """

    mismatch = False
    df1 = read_csv(fname, drop_units=False)

    columns = ['Patm', 'u', 'CO2', 'O2', 'Vmax25', 'gamstar25', 'Tref', 'JV',
               'Rlref', 'TRlref', 'Kc25', 'Ko25', 'alpha', 'c1', 'c2', 'c3',
               'c4', 'eps_l', 'albedo_l', 'tau_l', 'chi_l', 'kn', 'Ev', 'Ej',
               'Egamstar', 'Ec', 'Eo', 'deltaSv', 'deltaSj', 'Hdv', 'Hdj',
               'height', 'LAI', 'max_leaf_width', 'g1', 'g1T', 'Kappa',
               'Lambda', 'Eta', 'P50', 'P88', 'kmax', 'kmax2', 'kmaxT',
               'kmaxWUE', 'kmaxCN', 'kmaxCM', 'kmaxLC', 'kmaxS1', 'kmaxS2',
               'krlC', 'krlM', 'ratiocrit', 'sref', 'srefT', 'PrefT', 'PcritC',
               'PcritM', 'Alpha', 'Beta', 'ground_area', 'Ztop', 'Zbottom',
               'Ps', 'Psie', 'hyds', 'theta_sat', 'fc', 'pwp', 'bch']
    units = ['[kPa]', '[m s-1]', '[Pa]', '[kPa]', '[umol m-2 s-1]', '[Pa]',
             '[deg C]', '[-]', '[umol m-2 s-1]', '[deg C]', '[Pa]', '[Pa]',
             '[mol(photon) mol(e-)-1]', '[-]', '[-]', '[-]', '[-]', '[-]',
             '[-]', '[-]', '[-]', '[-]', '[J mol-1]', '[J mol-1]', '[J mol-1]',
             '[J mol-1]', '[J mol-1]', '[J mol-1 K-1]', '[J mol-1 K-1]',
             '[J mol-1]', '[J mol-1]', '[m]', '[m2 m-2]', '[m]', '[kPa0.5]',
             '[-]', '[umol m-2 s-1]', '[mol mol-1]', '[mol mol-1]', '[-MPa]',
             '[-MPa]', '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[-]', '[MPa-1]', '[MPa-1]', '[MPa]',
             '[MPa]', '[MPa]', '[mol m-2 s-1 MPa-2]', '[mol m-2 s-1 MPa-1]',
             '[m2]', '[m]', '[m]', '[MPa]', '[MPa]', '[m s-1]', '[m3 m-3]',
             '[m3 m-3]', '[m3 m-3]', '[-]']

    try:
        df2 = pd.DataFrame([(d.Patm, d.u, d.CO2, d.O2, d.Vmax25, d.gamstar25,
                             d.Tref, d.JV, d.Rlref, d.TRlref, d.Kc25, d.Ko25,
                             d.alpha, d.c1, d.c2, d.c3, d.c4, d.eps_l,
                             d.albedo_l, d.tau_l, d.chi_l, d.kn, d.Ev, d.Ej,
                             d.Egamstar, d.Ec, d.Eo, d.deltaSv, d.deltaSj,
                             d.Hdv, d.Hdj, d.height, d.LAI, d.max_leaf_width,
                             d.g1, d.g1T, d.Kappa, d.Lambda, d.Eta, d.P50,
                             d.P88, d.kmax, d.kmax2, d.kmaxT, d.kmaxWUE,
                             d.kmaxCN, d.kmaxCM, d.kmaxLC, d.kmaxS1, d.kmaxS2,
                             d.krlC, d.krlM, d.ratiocrit, d.sref, d.srefT,
                             d.PrefT, d.PcritC, d.PcritM, d.Alpha, d.Beta,
                             d.ground_area, d.Ztop, d.Zbottom, d.Ps, d.Psie,
                             d.hyds, d.theta_sat, d.fc, d.pwp, d.bch)],
                           columns=columns)

    except AttributeError:  # always yield error if d is csv
        mismatch = True  # diffs if d is obj? all params must be present
        d2 = default_params()
        df2 = pd.DataFrame([(d2.Patm, d2.u, d2.CO2, d2.O2, d2.Vmax25,
                             d2.gamstar25, d2.Tref, d2.JV, d2.Rlref, d2.TRlref,
                             d2.Kc25, d2.Ko25, d2.alpha, d2.c1, d2.c2, d2.c3,
                             d2.c4, d2.eps_l, d2.albedo_l, d2.tau_l, d2.chi_l,
                             d2.kn, d2.Ev, d2.Ej, d2.Egamstar, d2.Ec, d2.Eo,
                             d2.deltaSv, d2.deltaSj, d2.Hdv, d2.Hdj, d2.height,
                             d2.LAI, d2.max_leaf_width, d2.g1, d2.g1T,
                             d2.Kappa, d2.Lambda, d2.Eta, d2.P50, d2.P88,
                             d2.kmax, d2.kmax2, d2.kmaxT, d2.kmaxWUE,
                             d2.kmaxCN, d2.kmaxCM, d2.kmaxLC, d2.kmaxS1,
                             d2.kmaxS2, d2.krlC, d2.krlM, d2.ratiocrit,
                             d2.sref, d2.srefT, d2.PrefT, d2.PcritC, d2.PcritM,
                             d2.Alpha, d2.Beta, d2.ground_area, d2.Ztop,
                             d2.Zbottom, d2.Ps, d2.Psie, d2.hyds, d2.theta_sat,
                             d2.fc, d2.pwp, d2.bch)], columns=columns)

    these_headers = list(zip(df2.columns, units))
    df2.columns = pd.MultiIndex.from_tuples(these_headers)

    # if data is already in df1, avoid over-writing it
    for ind in df2.columns.levels[0]:

        if ind in df1.columns.levels[0]:
            df2 = df2.drop([ind], axis=1)

    df = pd.concat([df1, df2], axis=1)
    original_columns = df.columns
    df.columns = df.columns.droplevel(level=1)

    # if params specified in site file or class object, overwrite df
    if mismatch:
        if '.csv' in d:
            df3 = read_csv(d, drop_units=False)

            try:  # site params in csv
                df3.columns = df3.columns.droplevel(level=1)
                df3 = df3.set_index('Site')
                df3 = df3[~df3.index.duplicated(keep='first')]
                sites = df3.index.tolist()
                sites = [str(e) for e in sites if str(e) != str(pd.np.NaN)]

                for site in sites:

                    if site in fname:
                        ds = df3.loc[site, :]

            except KeyError:  # specific params in csv
                ds = df3.iloc[0]

            for ind in ds.index:

                if ((ind in df.columns) and not pd.isna(ds.loc[ind])):
                    df.iloc[0, df.columns.get_loc(ind)] = ds.loc[ind]

            # ini water potential cannot be above saturation: update
            if ((df.iloc[0, df.columns.get_loc('Ps')] >
                 df.iloc[0, df.columns.get_loc('Psie')]) or
               (df.iloc[0, df.columns.get_loc('Ps')] == df2.Ps)):
                df.iloc[0, df.columns.get_loc('Ps')] = \
                    df.iloc[0, df.columns.get_loc('Psie')]

        else:  # class object

            for key in vars(d).keys():

                df.loc[:, key] = vars(d)[key]

    df.columns = pd.MultiIndex.from_tuples(original_columns)
    df.to_csv(fname, index=False, na_rep='', encoding='utf-8')

    return
