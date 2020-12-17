#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

__title__ = ""
__author__ = "[Manon Sabot]"
__version__ = "1.0 (16.01.2019)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

# general modules
import os
import itertools
import difflib
import re
import numpy as np  # array manipulations, math operators
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit  # fit the functional shapes

from default_params import default_params


#==============================================================================

def ref_varnames():

    info = open(os.path.join(os.getcwd(), 'ReadMe.md'), 'r')

    # skip the headers (i.e. 5 first lines)
    for i in range(5):

        next(info)

    varnames = []

    for line in info:

        varnames += [line.strip().split(' | ')[0]]

    return varnames


def compare_lists(ref, to_compare):

    combis = list(itertools.product(ref, to_compare))

    match = [e for e in to_compare if any(re in e for re in ref)]

    to_replace = {}
    mismatch = [e[0] for e in combis if ((e[0].lower() == e[1].lower()) and
                e[0] not in match)]  # cap mismatch

    for e in combis:  # 1-letter mismatches
        letter_mismatches = [ee for ee in difflib.ndiff(e[0], e[1])]

        if letter_mismatches[0] != ' ':
            if len(letter_mismatches) == 1:  # ignore mismatches > 1 letter

                try:  # if number, ignore mismatch
                    float(letter_mismatches[0])

                except ValueError:  # not a number, add mismatch to list
                    mismatch += [e[0]]

    if len(mismatch) > 0:
        to_replace = { e[1] : e[0] for e in combis if e[0] in mismatch }

    missing = [e for e in ref if ((e not in match) and (e not in mismatch))]

    extra = [e for e in to_compare if ((e not in match) and
             (e not in mismatch))]

    return match, to_replace, missing, extra


def add_atm_forcing(df):

    location = df['Location'].iloc[0]  # which site is this

    # is there a met file?
    rmet = os.path.join(os.getcwd(), 'Met_data')

    for file in os.listdir(rmet):

        if file.endswith('.csv') and ((location.replace(' ', '') in file) or
            any(loc in file for loc in location.split())):
            met = (pd.read_csv(os.path.join(rmet, file), header=[0])
                     .dropna(axis=0, how='all').dropna(axis=1, how='all')
                     .squeeze())

            try:  # replace some of the variable names
                met.rename(columns={'DD': 'day', 'MM': 'month',
                                    'Year Month Day Hour Minutes in YYYY':
                                    'year', 'Air Temperature in degrees C':
                                    'Tair', 'Station level pressure in hPa':
                                    'Patm', 'Wind speed in km/h': 'u'},
                           inplace=True)

                # value formats
                met['Tair'] = pd.to_numeric(met['Tair'], downcast='float')
                met['Patm'] = pd.to_numeric(met['Patm'], downcast='float')
                met['u'] = pd.to_numeric(met['u'], downcast='float')

                # unit consistency
                met['Patm'] /= 10.  # hPa to kPa
                met['u'] *= 5. / 18. # km h-1 to m s-1

                # format the dates and hods
                met['Date'] = pd.to_datetime(met[['year', 'month', 'day']])
                met['decimaltime'] = (met['HH24'] +
                                      met['MI format in Local time'] / 60.)

                # assign these variables to df
                df = df.merge(met[['Date', 'decimaltime', 'Tair', 'Patm', 'u']],
                              on=['Date', 'decimaltime'])

            except KeyError:
                met.rename(columns={'Hour': 'decimaltime', 'Temperature':
                                    'Tair', 'Atm_pressure': 'Patm',
                                    'Wind_speed': 'u'},
                           inplace=True)

                # unit consistency
                met['Patm'] /= 10.  # hPa to kPa

                # format the dates and hods
                met['Date'] = (pd.to_datetime(met['Year'], format='%Y') +
                               pd.to_timedelta(met['Julian_day'] - 1, unit='d'))
                met['decimaltime'] += met['Min'] / 60.

                # assign these variables to df
                df = df.merge(met[['Date', 'decimaltime', 'Tair', 'Patm', 'u']],
                              on=['Date', 'decimaltime'])

    return df


def calc_atm_forcing(Tair=None, VPD=None, RH=None, ele=None):

    if Tair is not None:  # calculate the saturation vapour pressure
        es = 0.61078 * np.exp(17.27 * Tair / (Tair + 237.3))  # Tetens eq., kPa

    if ele is None:
        if (VPD is not None) and (RH is not None) and (Tair is None):
            aterm = np.log(VPD / (0.61078 * (1. - RH / 100.)))  # ancillary
            Tair = 237.3 * aterm / (17.27 - aterm)  # reworked Tetens, degC

            return Tair

        if (Tair is not None) and (RH is not None) and (VPD is None):
            VPD = es * (1. - RH / 100)  # kPa

            return VPD

    elif Tair is not None:

        TairK = Tair + 273.15  # degK
        Patm = (101.325 * (TairK / (TairK + 0.0065 * ele)) ** (9.80665 /
                (287.04 * 0.0065)))  # kPa

        return Patm


def unify_headers_units(df):

    # conventional variable names
    ref = ref_varnames()

    # Datetime formats
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    # which are the same, which have to change, and which are missing?
    match, to_replace, missing, __ = compare_lists(ref, df.columns.to_list())

    # deal with small mismatches by replacing the problem headers
    df.rename(columns=to_replace, inplace=True)

    # from ppm to Pa
    if 'Ca_Pa' in df.columns:
        df['CO2S'] = df['Ca_Pa']
        df.drop('Ca_Pa', axis=1, inplace=True)

    elif 'CO2S' in df.columns:
        df['CO2S'] *= 1.e-3 * 101.325

    if 'Ci_Pa' in df.columns:
        df['Ci'] = df['Ci_Pa']
        df.drop('Ci_Pa', axis=1, inplace=True)

    elif 'Ci' in df.columns:
        df['Ci'] *= 1.e-3 * 101.325

    if 'Quercus' in df['Species'].unique()[0]:
        df['Trmmol'] *= 1.e3  # the South of France Tr data is in mols!

    try:  # make sure the site names are in the same format
        location = np.unique(df['Location'])

        for i in range(len(location)):

            loc = re.sub( r'([A-Z])', r' \1', location[i]).split()

            if len(loc) > 1:  # several uppercase letters in loc name

                try:
                    idx = [loc.index(e) for e in loc if len(e) == 1][0]
                    location[i] = ('').join(loc[:idx])

                except IndexError:
                    location[i] = ('').join(loc)

    except KeyError:
        df['Location'] = 'Site'  # artificially add a site name

    if 'decimaltime' in missing:
        time = pd.to_datetime(df['Time'], format='%H:%M:%S')
        df['decimaltime'] = (time.dt.hour + (time.dt.minute + time.dt.second)
                             / 60.)

    # if the data is missing Patm, Tair or VPD, first look for a met file
    if ('Tair' in missing) or ('Patm' in missing) or ('VPD' in missing):
        df = add_atm_forcing(df)
        match, __, missing, __ = compare_lists(ref, df.columns.to_list())

        # correct high biases in both directions
        if (df['Tair'] - df['Tleaf']).abs().mean() > 3.:
            over = df['Tair'] - df['Tleaf'] > 0.
            under = df['Tair'] - df['Tleaf'] < 0.
            o_bias = (df['Tair'] - df['Tleaf'])[over].mean()
            u_bias = (df['Tair'] - df['Tleaf'])[under].mean()

            # making sure we're preserving the "direction" of differences
            over = np.logical_and(over, df['Tair'] - o_bias >= df['Tleaf'])
            under = np.logical_and(under, df['Tair'] - u_bias <= df['Tleaf'])
            df['Tair'][over] = df['Tair'][over] - o_bias
            df['Tair'][under] = df['Tair'][under] - u_bias

    # add using Tetens
    if ('Tair' in missing) or ('Patm' in missing) or ('VPD' in missing):
        if ('VPD' in match) and ('RH' in match) and ('Tair' in missing):
            df['Tair'] = calc_atm_forcing(VPD=df['VPD'], RH=df['RH'])

            # which data are still missing?
            match, __, missing, __ = compare_lists(ref, df.columns.to_list())

        if 'Tair' in match:
            if ('RH' in match) and ('VPD' in missing):
                df['VPD'] = calc_atm_forcing(Tair=df['Tair'], RH=df['RH'])

            if 'Patm' in missing:
                df['Patm'] = calc_atm_forcing(Tair=df['Tair'],
                                              ele=df['altitude'])

    # which data are still missing and which extra variables are there?
    match, __, missing, extra = compare_lists(ref, df.columns.to_list())

    # can't use dataset if specific forcing combinations are missing
    if ('Patm' in missing) and ((('RH' in missing) and (('Tair' in missing) or
        ('VPD' in missing))) or (('Tair' in missing) and ('VPD' in missing))):

        if 'GrowthTair' in match:

            try:
                df['Tair'] = df['GrowthTair'].astype(float)

                if ('Patm' in missing) and (('RH' in missing) or
                    (('VPD' in missing) and ('VPD_A' in missing) and
                    ('VPD_L' in missing))):

                    return

            except ValueError:

                return

            try:
                df['VPD'] = df['VPD_A']

            except KeyError:

                try:
                    df['VPD'] = df['VPD_L']

                except KeyError:

                    return

        else:

            return

    # can't use dataset if it's missing all possible soil moisture stresses
    if ('SWP' in missing) and ('SWC' in missing) and ('LWPpredawn' in missing):

        return

    else:  # common mismatch: 'Funtype' vs 'PFT'; drop IbDB
        if ('PFT' in match) and not ('PFT2v' in match):
            if 'Funtype' in extra:
                if not df['PFT'].equals(df['Funtype']):
                    df['PFT2v'] = df['Funtype']

                    try:
                        df.drop(['Funtype', 'Funtype2'], axis=1, inplace=True)

                    except KeyError:
                        df.drop(['Funtype'], axis=1, inplace=True)

                elif 'Funtype2' in extra:
                    df['PFT2v'] = df['Funtype2']
                    df.drop(['Funtype', 'Funtype2'], axis=1, inplace=True)

            elif 'Funtype2' in extra:
                if not df['PFT'].equals(df['Funtype2']):
                    df['PFT2v'] = df['Funtype2']
                    df.drop(['Funtype2'], axis=1, inplace=True)

        if ('PFT2v' in match) and not ('PFT' in match):
            if 'Funtype' in extra:
                if not df['PFT2v'].equals(df['Funtype']):
                    df['PFT'] = df['Funtype']

                    try:
                        df.drop(['Funtype', 'Funtype2'], axis=1, inplace=True)

                    except KeyError:
                        df.drop(['Funtype'], axis=1, inplace=True)

                elif 'Funtype2' in extra:
                    df['PFT'] = df['PFT2v']
                    df['PFT2v'] = df['Funtype2']
                    df.drop(['Funtype', 'Funtype2'], axis=1, inplace=True)

            elif 'Funtype2' in extra:
                if not df['PFT2v'].equals(df['Funtype2']):
                    df['PFT'] = df['PFT2v']
                    df['PFT2v'] = df['Funtype2']
                    df.drop(['Funtype2'], axis=1, inplace=True)

            else:
                if 'PFT' not in df.columns.to_list():
                    df['PFT'] = df['PFT2v']
                    df.drop(['PFT2v'], axis=1, inplace=True)

        # remove unused extras but keep some specific extras
        __, __, __, extra = compare_lists(ref, df.columns.to_list())

        try:
            extra.remove('u')  # to keep

        except ValueError:
            pass

        try:
            extra.remove('Country')

        except ValueError:
            pass

        try:
            extra.remove('Dataset')

        except ValueError:
            pass

        if len(extra) > 0:
            df.drop(extra, axis=1, inplace=True)

        return df, missing


def VC_Manzoni(P50, a):

    """
    One can solve for any second WP using Px = P50 * (x / 1 - x) ** (1 / a)
    from Manzoni et al. 2013 (Advances in Water Resources)

    """

    P88 = P50 * ((0.88 / 0.12) ** (1. / a))

    return P88


def VC_Sack(Px1, Px2, x1, x2):


    """
    Finds the leaf water potential associated with a specific x% decrease in
    hydraulic conductance, using the plant vulnerability curve.

    Arguments:
    ----------
    Px: float
        leaf water potential [MPa] at which x% decrease in hydraulic conductance
        is observed

    x: float
        percentage loss in hydraulic conductance

    Returns:
    --------
    P88: float
        leaf water potential [MPa] at which 88% decrease in hydraulic
        conductance is observed
    """

    Px1 = np.abs(Px1)
    Px2 = np.abs(Px2)
    x1 /= 100. # normalise between 0-1
    x2 /= 100.

    # c is derived from both expressions of b
    try:
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))

    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))

    b = Px1 / ((- np.log(1 - x1)) ** (1. / c))
    P88 = -b * ((- np.log(0.12)) ** (1. / c)) # MPa

    return P88


def cleaned_up_Manzoni_data():

    rname = os.path.join(os.getcwd(), 'Hydraulic_traits')

    df = (pd.read_csv(os.path.join(os.path.join(rname, 'original_files'),
                                   'Manzoni_from_TRY.txt'), header=[0],
                      sep='\t', encoding='latin-1', low_memory=False)
            .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    # create a new df for the VC curves only
    new_names = {'Xylem water potential at which 50% of conductivity is lost (P50)':
                 'P50',
                 'Xylem embolism: Shape parameter for the vulnerability curve, a':
                 'a', 'Reference / source': 'Ref', 'Reference 2': 'Ref2',
                 'Reference 3': 'Ref3'}
    df['DataName'].replace(new_names, inplace=True)  # rename the variables

    # first we need to organise the df by unique obs ID
    df.set_index('ObservationID', inplace=True)
    idx = df[df['DataName'] == 'P50'].index.to_list()
    df = df.loc[idx]  # only keep the obs ID where there's P50 data

    # now reorganise into a "cleaner" df
    df.reset_index(inplace=True)  # reset idx to group by unique combis
    df2 = (df[['ObservationID', 'AccSpeciesName']]
             .groupby(['ObservationID', 'AccSpeciesName']).first())
    df2['Species'] = df2.index.droplevel(level=0)  # so far, df2 is a multi-index
    df2.index = df2.index.droplevel(level=1)  # ObservationID is the index
    df.set_index('ObservationID', inplace=True)  # reset as the index

    for var in df['DataName'].unique():

        df2[var] = np.nan
        sub = df[df['DataName'] == var]

        for i in df2.index.to_list():

            try:
                df2.loc[i, var] = sub.loc[i, 'OrigValueStr']

            except KeyError:
                pass

    df2[['P50', 'a']] = df2[['P50', 'a']].astype(float)
    df2['P88'] = VC_Manzoni(df2['P50'], df2['a'])
    df2 = df2[np.logical_or(df2['P88'] > -20, df2['P88'].isnull())]  # rm crazys

    # now reorganise the columns
    df2 = df2[['Species', 'P50', 'P88', 'Latitude', 'Longitude', 'Altitude',
               'Ref', 'Ref2', 'Ref3']]
    df2.to_csv(os.path.join(rname, 'Manzoni_2013_dataset.csv'), index=False,
               encoding='utf-8')

    return df2


def cleaned_up_Sack_data():

    rname = os.path.join(os.getcwd(), 'Hydraulic_traits')

    df = (pd.read_csv(os.path.join(os.path.join(rname, 'original_files'),
                                   'Sack_from_TRY.txt'), header=[0], sep='\t',
                      encoding='latin-1', low_memory=False)
            .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())

    # create a new df for the VC curves only
    new_names = {'Stem P20': 'P20', 'Stem P50': 'P50', 'Stem P80': 'P80',
                 'Reference': 'Ref'}
    df['DataName'].replace(new_names, inplace=True)  # rename the variables

    # first we need to organise the df by unique obs ID
    df.set_index('ObservationID', inplace=True)
    idx = df[df['DataName'] == 'P50'].index.to_list()
    df = df.loc[idx]  # only keep the obs ID where there's P50 data

    # now reorganise into a "cleaner" df
    df.reset_index(inplace=True)  # reset idx to group by unique combis
    df2 = (df[['ObservationID', 'AccSpeciesName']]
             .groupby(['ObservationID', 'AccSpeciesName']).first())
    df2['Species'] = df2.index.droplevel(level=0)  # so far, df2 is a multi-index
    df2.index = df2.index.droplevel(level=1)  # ObservationID is the index
    df.set_index('ObservationID', inplace=True)  # reset as the index

    for var in df['DataName'].unique():

        df2[var] = np.nan
        sub = df[df['DataName'] == var]

        for i in df2.index.to_list():

            try:
                df2.loc[i, var] = sub.loc[i, 'OrigValueStr']

            except KeyError:
                pass

    df2[['P20', 'P50', 'P80']] = df2[['P20', 'P50', 'P80']].astype(float)
    df2['P88'] = VC_Sack(df2['P20'], df2['P80'], 20, 80)
    df2 = df2[np.logical_or(df2['P88'] > -20, df2['P88'].isnull())]  # rm crazys

    # now reorganise the columns
    try:
        df2 = df2[['Species', 'P50', 'P88', 'Latitude', 'Longitude', 'Altitude',
                   'Ref']]

    except KeyError:
        df2 = df2[['Species', 'P50', 'P88', 'Latitude', 'Longitude',
                   'Altitude']]

    df2.to_csv(os.path.join(rname, 'Sack_unpublished_dataset.csv'), index=False,
               encoding='utf-8')

    return df2


def check_VC_data(species):

    rname = os.path.join(os.getcwd(), 'Hydraulic_traits')

    df1 = (pd.read_csv(os.path.join(rname, 'SurEau_dataset.csv'),
                       header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    if 'Quercus' in species:
        df2 = (pd.read_csv(os.path.join(rname, 'Choat_Quercus_data.csv'),
                           header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    if 'Eucalyptus' in species:
        df2 = (pd.read_csv(os.path.join(rname, 'Bourne_Eucalyptus_data.csv'),
                           header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    df3 = (pd.read_csv(os.path.join(rname, 'Choat_2012_dataset.csv'),
                       header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    if os.path.isfile(os.path.join(rname, 'Manzoni_2013_dataset.csv')):
        df4 = (pd.read_csv(os.path.join(rname, 'Manzoni_2013_dataset.csv'),
                           header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    else:
        df4 = cleaned_up_Manzoni_data()

    if os.path.isfile(os.path.join(rname, 'Sack_unpublished_dataset.csv')):
        df5 = (pd.read_csv(os.path.join(rname, 'Sack_unpublished_dataset.csv'),
                           header=[0]).dropna(axis=0, how='all')
                 .dropna(axis=1, how='all').squeeze())

    else:
        df5 = cleaned_up_Sack_data()

    df6 = (pd.read_csv(os.path.join(rname, 'Wu_Wolfe_2020_dataset.csv'),
                       header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    df7 = (pd.read_csv(os.path.join(rname,
                                    'Meinzer_2008_Mendez_2012_datasets.csv'),
                       header=[0]).dropna(axis=0, how='all')
             .dropna(axis=1, how='all').squeeze())

    traits = []  # empty string

    try:
        df1.set_index('Species', drop=False, inplace=True)
        traits += ['(P12: %s, P50: %s)' % (str(-df1.loc[species, 'P12']),
                                           str(-df1.loc[species, 'P50']))]

    except KeyError:
        pass

    if 'Quercus' in species:
        try:
            df2.set_index('Species', drop=False, inplace=True)
            traits += ['(P50: %s, P88: %s)' % (str(df2.loc[species, 'P50']),
                                               str(df2.loc[species, 'P88']))]

        except KeyError:
            pass

    if 'Eucalyptus' in species:
        try:
            df2.set_index('Species', drop=False, inplace=True)
            traits += ['(P50: %s, P88: %s)' % (str(df2.loc[species, 'P50']),
                                               str(df2.loc[species, 'P88']))]

        except KeyError:
            pass

    try:
        df3.set_index('Species', drop=False, inplace=True)
        traits += ['(P50: %s, P88: %s)' % (str(df3.loc[species, 'P50']),
                                           str(df3.loc[species, 'P88']))]

    except KeyError:
        pass

    try:  # in this case there can be several values, we take the median
        df4.set_index('Species', drop=False, inplace=True)
        P50 = df4.loc[species, 'P50']
        P88 = df4.loc[species, 'P88']

        try:
            P50 = P50.median()
            P88 = P88.median()

        except AttributeError:
            pass

        traits += ['(P50: %s, P88: %s)' % (str(-P50), str(-P88))]

    except KeyError:
        pass

    try:  # in this case there can be several values, we take the median
        df5.set_index('Species', drop=False, inplace=True)
        P50 = df5.loc[species, 'P50']
        P88 = df5.loc[species, 'P88']

        try:
            P50 = P50.median()
            P88 = P88.median()

        except AttributeError:
            pass

        traits += ['(P50: %s, P88: %s)' % (str(-P50), str(-P88))]

    except KeyError:
        pass

    try:
        df6.set_index('Species', drop=False, inplace=True)
        traits += ['(P50: %s, P88: %s)' % (str(df6.loc[species, 'P50']),
                                           str(df6.loc[species, 'P88']))]

    except KeyError:
        pass

    try:
        df7.set_index('Species', drop=False, inplace=True)
        traits += ['(P50: %s, P88: %s)' % (str(df7.loc[species, 'P50']),
                                           str(df7.loc[species, 'P88']))]

    except KeyError:
        pass

    traits = ', '.join(traits)

    return traits


def add_vars_to_csv(df, site, species):

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

    columns = ['O2', 'Vmax25', 'gamstar25', 'Tref', 'JV', 'Rlref', 'TRlref',
               'Kc25', 'Ko25', 'alpha', 'c1', 'c2', 'c3', 'c4', 'eps_l',
               'albedo_l', 'tau_l', 'chi_l', 'kn', 'Ev', 'Ej', 'Egamstar',
               'Ec', 'Eo', 'deltaSv', 'deltaSj', 'Hdv', 'Hdj', 'height', 'LAI',
               'max_leaf_width', 'g1', 'g1T', 'Kappa', 'Lambda', 'Eta', 'P50',
               'P88', 'kmax', 'kmax2', 'kmaxT', 'kmaxWUE', 'kmaxCN', 'kmaxCM',
               'kmaxLC', 'kmaxS1', 'kmaxS2', 'ksc_prev', 'krlC', 'krlM',
               'ratiocrit', 'ksrmaxC', 'ksrmaxM', 'sref', 'srefT', 'PrefT',
               'PcritC', 'PcritM', 'Alpha', 'Beta', 'ground_area', 'Ztop',
               'Zbottom', 'Psie', 'hyds', 'theta_sat', 'fc', 'pwp', 'bch']
    units = ['[kPa]', '[umol m-2 s-1]', '[Pa]', '[deg C]', '[-]',
             '[umol m-2 s-1]', '[deg C]', '[Pa]', '[Pa]',
             '[mol(photon) mol(e-)-1]', '[-]', '[-]', '[-]', '[-]', '[-]',
             '[-]', '[-]', '[-]', '[-]', '[J mol-1]', '[J mol-1]', '[J mol-1]',
             '[J mol-1]', '[J mol-1]', '[J mol-1 K-1]', '[J mol-1 K-1]',
             '[J mol-1]', '[J mol-1]', '[m]', '[m2 m-2]', '[m]', '[kPa0.5]',
             '[-]', '[umol m-2 s-1]', '[mol mol-1]', '[-]', '[-MPa]', '[-MPa]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]', '[-]',
             '[mmol m-2 s-1 MPa-1]', '[mmol m-2 s-1 MPa-1]', '[MPa-1]',
             '[MPa-1]', '[MPa]', '[MPa]', '[MPa]', '[mol m-2 s-1 MPa-2]',
             '[mol m-2 s-1 MPa-1]', '[m2]', '[m]', '[m]', '[MPa]', '[m s-1]',
             '[m3 m-3]', '[m3 m-3]', '[m3 m-3]', '[-]']

    d = default_params()
    df2 = pd.DataFrame([(d.O2, d.Vmax25, d.gamstar25, d.Tref, d.JV, d.Rlref,
                         d.TRlref, d.Kc25, d.Ko25, d.alpha, d.c1, d.c2, d.c3,
                         d.c4, d.eps_l, d.albedo_l, d.tau_l, d.chi_l, d.kn,
                         d.Ev, d.Ej, d.Egamstar, d.Ec, d.Eo, d.deltaSv,
                         d.deltaSj, d.Hdv, d.Hdj, d.height, d.LAI,
                         d.max_leaf_width, d.g1, d.g1T, d.Kappa, d.Lambda,
                         d.Eta, d.P50, d.P88, d.kmax, d.kmax2, d.kmaxT,
                         d.kmaxWUE, d.kmaxCN, d.kmaxCM, d.kmaxLC, d.kmaxS1,
                         d.kmaxS2, d.ksc_prev, d.krlC, d.krlM, d.ratiocrit,
                         d.ksrmaxC, d.ksrmaxM, d.sref, d.srefT, d.PrefT,
                         d.PcritC, d.PcritM, d.Alpha, d.Beta, d.ground_area,
                         d.Ztop, d.Zbottom, d.Psie, d.hyds, d.theta_sat, d.fc,
                         d.pwp, d.bch)], columns=columns)

    df2.columns = pd.MultiIndex.from_tuples(list(zip(columns, units)))
    df = pd.concat([df, df2], axis=1)
    original_columns = df.columns
    df.columns = df.columns.droplevel(level=1)

    # if params specified in site file or class object, overwrite df
    df3 = (pd.read_csv('site_level_traits.csv', header=[0, 1])
             .dropna(axis=0, how='all').dropna(axis=1, how='all').squeeze())
    df3.columns = df3.columns.droplevel(level=1)

    # which site?
    check_site = [e in site for e in df3['Site']]

    if not any(check_site):
        check_site = [site in e for e in df3['Site']]

    idx = [i for i in range(len(check_site)) if check_site[i]][0]

    if any(check_site):
        subset = df3[df3['Site'] == df3.loc[idx, 'Site']]

    # if several species at that site, also subset by species
    if len(subset) > 1:
        subset = df3[df3['Species'] == species]

    params = [e for e in df3.columns.to_list() if e not in
              ['Site', 'Species', 'Reference']]

    for p in params:

        if not pd.isnull(subset[p].values):
            df.loc[0, p] = subset[p].values

    # deal with empirical P parameters
    df.loc[0, 'PrefT'] = -df.loc[0, 'P50']  # Tuzet model
    df.loc[0, 'PcritC'] = -df.loc[0, 'P50']  # CAP model
    df.loc[0, 'PcritM'] = -df.loc[0, 'P50']  # MES model

    df.columns = original_columns

    return df


def fLWP(Pleaf, srefT, PrefT):

    return (1. + np.exp(srefT * PrefT)) / (1. + np.exp(srefT * (PrefT - Pleaf)))


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


def Tuzet_params(df):

    # smooth out noise
    df = df[df['Pleaf'] > -9999.]
    df['gs'] /= df['gs'].max()

    # smooth out noise within interquartile
    smoothed = gaussian_filter(df['gs'], df['gs'].std())

    # point where the signal goes above the background noise
    base = 0.3  # background noise is +/- 15% of max gs
    supp = (df['gs'][df['Pleaf'] < -df['Pleaf'].std()] - base).std()
    m = smoothed < (base - df['gs'].std() * supp)

    # searchable space
    x0 = np.maximum(df['Pleaf'][m].max(), df['Pleaf'][np.isclose(df['gs'], 1.)])
    x1 = df['Pleaf'][m].min()

    # sort by LWP
    print('gets here')
    df1 = df.sort_values(by=['Pleaf'], ascending=False)
    LWP, gs = envelope(df1['Pleaf'].to_numpy(), df1['gs'].to_numpy())

    # fitted params
    obs_popt, __ = curve_fit(fsig_tuzet, LWP, gs, p0=[2., (x0 + x1) / 2.],
                             bounds=([0.01, df['Pleaf'].min()],
                                     [10, df['Pleaf'].max()]))
    print(obs_popt)
    #obs_popt, __ = curve_fit(fLWP, df['Pleaf'], df['gs'],
    #                         p0=[2., (x0 + x1) / 2.],
    #                         bounds=([0.01, x1], [10, x0]))

    return obs_popt[0], obs_popt[1]


def format_x_y_files(df, loc):

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df['year'] = df['Date'].dt.year
    df['doy'] = df['Date'].dt.dayofyear

    df.rename(columns={'decimaltime': 'hod', 'LWPpredawn': 'Ps',
                       'PARin': 'PPFD', 'CO2S': 'CO2', 'Photo': 'A',
                       'Trmmol': 'E', 'Cond': 'gs', 'BLCond': 'gb',
                       'LWP': 'Pleaf'}, inplace=True)

    # forcing data
    columns1 = ['year', 'doy', 'hod', 'Ps', 'PPFD', 'Tair', 'Tleaf', 'VPD',
                'Patm', 'u', 'CO2', 'gb']
    units1 = ['[-]', '[-]', '[h]', '[MPa]', '[umol m-2 s-1]', '[deg C]',
              '[deg C]', '[kPa]', '[kPa]', '[m s-1]', '[Pa]', '[mol m-2 s-1]']

    # leaf level output variables
    columns2 = ['year', 'doy', 'hod', 'A', 'E', 'Ci', 'gs', 'gb', 'Pleaf',
                'Tleaf']
    units2 = ['[-]', '[-]', '[h]', '[umol m-2 s-1]', '[mmol m-2 s-1]', '[Pa]',
              '[mol m-2 s-1]', '[mol m-2 s-1]', '[MPa]', '[deg C]']

    # make sure the absolutely necessary forcings and output are valid
    try:
        df1 = df[columns1 + ['gs']].copy()

    except KeyError:
        try:
            df['u'] = 0.  # wind speed is not used anyway
            df1 = df[columns1 + ['gs']].copy()

        except KeyError:
            if 'Sevilleta' in df['Location'].unique()[0]:
                df['gb'] = 5.  # default LICOR for conifers

            else:
                df['gb'] = 2.84  # common broadleaf setting

            df1 = df[columns1 + ['gs']].copy()

    df1.dropna(axis=0, how='any', inplace=True)
    df = df.loc[df1.index]

    # make sure the df is in chronological order
    df.sort_values(['year', 'doy', 'hod'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) <= 30:  # then the spp at this location can't be used
        raise Exception

    # put the site and species info appart
    site = df['Location'].unique()[0]
    spp = df['Species'].unique()[0]

    # first, write the forcing data file
    df1 = df[columns1].copy()
    df1.columns = pd.MultiIndex.from_tuples(list(zip(columns1, units1)))

    # add the parameters
    df1 = add_vars_to_csv(df1, site, spp)

    # add the Tuzet Pref a priori based on the data
    try:
        srefT, PrefT = Tuzet_params(df.copy())
        df1.loc[0, 'srefT'] = srefT
        df1.loc[0, 'PrefT'] = PrefT

    except Exception:
        pass

    df1.to_csv('%s_%s_x.csv' % (loc, spp.replace(' ', '_')), index=False,
                encoding='utf-8')

    # second, write the leaf level output data file
    idxs = [i for i, __ in reversed(list(enumerate(columns2)))
            if columns2[i] not in df.columns]  # check that all are present

    if idxs != []:
        columns2 = list(np.delete(columns2, idxs))
        units2 = list(np.delete(units2, idxs))

    df2 = df[columns2]
    df2.columns = pd.MultiIndex.from_tuples(list(zip(columns2, units2)))
    df2.to_csv('%s_%s_y.csv' % (loc, spp.replace(' ', '_')), index=False,
                encoding='utf-8')

    return


def clean_up_data():

    rdata = os.path.join(os.getcwd(), 'Leaf_gas_exchange')

    # summary file where the datasets' info are stored
    ifname = os.path.join(os.getcwd(), 'info_data.txt')

    if os.path.isfile(ifname):
        os.remove(ifname)  # clean write

    info = open(ifname, 'w+')

    iter = 0  # iterator

    for file in os.listdir(rdata):

        if file.endswith('.csv'):  # loop over the datasets

            info.write(file)  # summary info to text file
            info.write('\n======================')

            df = (pd.read_csv(os.path.join(rdata, file), header=[0])
                    .dropna(axis=0, how='all').dropna(axis=1, how='all')
                    .squeeze())
            df = df.loc[:, ~df.eq(-9999).all()]  # get rid of -9999 everywhere

            if 'ManyPeaksRange' in file:  # Ca fixed to something plausible
                df['CO2S'] = 375.  # actually right for 2003, not 400!

            # make sure the headers follow the convention and adjust units
            df, missing = unify_headers_units(df)

            if missing is None:
                info.write('\nNot included: missing critical forcing\n')
                info.write('\n\n')

            elif (('Cond' in missing) or ('Trmmol' in missing) or
                  ('Photo' in missing)):
                info.write('\nNot included: missing critical output\n')
                info.write('\n\n')

            else:  # apply quality and sanity checks on the data
                try:
                    if (df['LWP'] >= 0.).all():  # if in -MPa
                        df['LWP'] *= -1.  # MPa

                except KeyError:
                    pass

                try:
                    if (df['LWPpredawn'] >= 0.).all():  # if in -MPa
                        df['LWPpredawn'] *= -1.  # MPa

                except KeyError:
                    pass

                # other criteria
                df = df[df['PARin'] > 50.]  # low PAR creates pbms for photo
                df = df[df['VPD'] >= 0.05]  # Medlyn model won't work < 0.05
                df = df[np.logical_and(df['Cond'] >= 0., df['Trmmol'] >= 0.)]

                try:  # check that the LWP and LWPpd values make sense
                    df = df[df['LWP'] < df['LWPpredawn']]

                except KeyError:
                    pass

                try:
                    df = df[np.logical_and(df['Ci'] > 0.,
                                           df['Ci'] < df['CO2S'])]

                except KeyError:
                    pass

                # how many datapoints per species per location are left?
                df2 = (df.copy().groupby(['Species', 'Location']).size()
                         .reset_index().rename(columns={0:'Count'}))

                # only keep subsets of more than 30 data points
                df2 = df2[df2['Count'] > 30]

                if df2.empty:
                    info.write('\nNot included: not enough spp data / site\n')
                    info.write('\n\n')

                else:
                    species = df2['Species'].to_list()
                    Nlocs = df2['Location'].unique()
                    Nspecies = df2['Count'].to_list()
                    Nspecies = ['%s (%d)' % (species[i], Nspecies[i])
                                for i in range(len(Nspecies))
                                if Nspecies[i] > 30]

                    info.write('\n%d species (size=N) at %d site.s:\n%s\n'
                               % (len(np.unique(species)), len(Nlocs),
                                  ', '.join(Nspecies)))

                    count = 0

                    for spp in np.unique(species):

                        hydraulics = check_VC_data(spp)

                        if hydraulics != '':
                            info.write('\nHydraulic traits available for %s\n'
                                       % (spp))

                        # rm species for which we don't have trait data
                        if hydraulics == '':
                            df2 = df2[df2['Species'] != spp]
                            count += 1

                    if (count > 0) and (count < len(np.unique(species))):
                        info.write('\nMissing hydraulic traits for %d species\n'
                                   % (count))

                    if df2.empty:
                        info.write('\nNot included: no hydraulic traits\n')
                        info.write('\n\n')

                    else:  # restrict the df to those species per location
                        for i in range(len(df2['Species'])):

                            crit1 = df['Species'] == df2.loc[df2.index[i],
                                                             'Species']
                            crit2 = df['Location'] == df2.loc[df2.index[i],
                                                              'Location']
                            tmp = np.logical_and(crit1, crit2)

                            if i == 0:
                                keep = tmp

                            else:
                                keep = np.logical_or(keep, tmp)

                        df = df[keep]

                        # split by unique species + location and save
                        for spp in np.unique(species):

                            for loc in Nlocs:

                                if len(Nlocs) == 1:
                                    df3 = df[df['Species'] == spp]

                                    if len(df3) > 30:
                                        loc = file.split('.csv')[0]

                                        try:
                                            format_x_y_files(df3, loc)

                                        except Exception:
                                            pass

                                else:
                                    df3 = df[np.logical_and(df['Species'] ==
                                             spp, df['Location'] == loc)]

                                    if len(df3) > 30:
                                        loc = loc.replace(' ', '_')

                                        try:
                                            format_x_y_files(df3, loc)

                                        except Exception:
                                            pass

                        try:
                            info.write('\nLWPpdmax: %s MPa and LWPpdmin: %s MPa'
                                        % (str(df['LWPpredawn'].max()),
                                           str(df['LWPpredawn'].min())))
                            info.write(' across all remaining data\n')

                        except KeyError:
                            pass

                        info.write('\nThe dataset is missing:\n%s\n' %
                                   (', '.join(missing)))

                        if 'SWC' not in missing:
                            info.write('N.B: SWC is available\n')

                        if 'SWP' not in missing:
                            info.write('N.B: SWP is available\n')

                        if 'LAI' not in missing:
                            info.write('N.B: LAI is available\n')

                        if 'Totalheight' not in missing:
                            info.write('N.B: height is available\n')

                        info.write('\n\n')

    info.close()

    return


clean_up_data()
