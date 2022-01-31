# -*- coding: utf-8 -*-

"""
Support functions needed to calibrate the models and sort / analyse the
calibration outputs.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

"""

__title__ = "Useful ancillary calibration functions"
__author__ = "Manon E. B. Sabot"
__version__ = "2.0 (15.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

# general modules
import os  # check for paths
import numpy as np  # array manipulations, math operators
import lmfit  # non-linear model minimizers
import random  # pick a random day for the forcings to be generated
from itertools import groupby  # organise text info

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from TractLSM.Utils import read_csv  # read in files
from TractLSM.SPAC import water_potential  # soil module
from TractLSM.SPAC import net_radiation  # preset it for model training
from TractLSM import InForcings  # generate met data & default params
from TractLSM import hrun  # run the reference gs model

from calibrations import fres  # tailor-made minimizing function

import warnings  # ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ======================================================================

class NLMFIT(object):

    """
    Wrapper around the models_2_fit.py that sets up the lmfit
    calibration method depending on the minimizer.s chosen, defines the
    parameters to calibrate and initializes them, and finally saves the
    output.

    """

    def __init__(self, method='powell', store=None, inf_gb=True):

        # fitting method
        self.method = method  # which solver is used

        # MCMC specifications
        self.steps = 15000
        self.nchains = 4
        self.burn = 1000
        self.thin = 2

        if store is None:  # default storing path for the outputs
            self.base_dir = get_main_dir()  # working paths
            self.opath = os.path.join(os.path.join(self.base_dir, 'output'),
                                      'calibrations')

        else:  # user defined storing path for the outputs
            self.opath = store

        self.inf_gb = inf_gb  # whether to calculate gb or not

    def get_P95(self, Px1, Px2, x1, x2):

        """
        Finds the leaf water potential associated with a 95% decrease in
        hydraulic conductance, using the plant vulnerability curve.

        Arguments:
        ----------
        Px: float
            leaf water potential [MPa] at which x% decrease in hydraulic
            conductance is observed

        x: float
            percentage loss in hydraulic conductance

        Returns:
        --------
        P95: float
            leaf water potential [MPa] at which there is 95% loss of
            hydraulic conductance

        """

        x1 /= 100.  # normalise between 0-1
        x2 /= 100.

        try:  # c is derived from both expressions of b
            c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                             np.log(Px2))

        except ValueError:
            c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                             np.log(Px1))

        b = Px1 / ((- np.log(1 - x1)) ** (1. / c))
        P95 = -b * ((- np.log(0.05)) ** (1. / c))  # MPa

        return P95

    def param_space(self, pname, P50=None, P88=None):

        """
        Sets bounds on the parameter values.

        Arguments:
        ----------
        pname: string
            parameter name

        P50: float
            leaf water potential [MPa] at which there is 50% loss of
            hydraulic conductance

        P88: float
            leaf water potential [MPa] at which there is 88% loss of
            hydraulic conductance

        Returns:
        --------
        Minimum and maximum bounds on any specific parameter.

        """

        if 'sref' in pname:

            return 0.01, 10.

        elif 'g1' in pname:

            return 0.01, 12.5

        elif 'kmax' in pname:

            return 0.005, 20.

        elif 'krl' in pname:

            return 0.005, 20.

        elif (P50 is not None) and (P88 is not None) and (('Pref' in pname) or
                                                          ('Pcrit' in pname)):

            return self.get_P95(P50, P88, 50, 88), -0.15

        elif (P88 is not None) and (('Pref' in pname) or ('Pcrit' in pname)):

            return -P88, -0.15

        elif (P88 is None) and (('Pref' in pname) or ('Pcrit' in pname)):

            return None, -0.15

        elif pname == 'Alpha':

            return 0.5, 80.

        elif pname == 'Beta':

            return 0.1, 8.

        elif pname == 'Lambda':

            return 0.01, 10.

        else:

            return 0.01, 50.

    def run(self, X, Y, model):

        """
        Finds the leaf water potential associated with a 95% decrease in
        hydraulic conductance, using the plant vulnerability curve.

        Arguments:
        ----------
        X: recarray object or pandas series
            met data & params used to force the chosen model

        target: recarray object or pandas series
            target observations to calibrate against

        model: string
            model to calibrate. The choices are: 'CAP', 'CGain', 'Cmax',
            'Eller' (i.e., SOX empirical), 'LeastCost', 'MES', 'Medlyn',
            'ProfitMax', 'ProfitMax2', 'Tuzet', or 'SOX-OPT' (i.e., the
            optimization version of SOX), 'WUE-LWP'

        Returns:
        --------
        Saves the calibrated parameters and additional information on
        'goodness-of-fit' for the best calibration run in text files
        stored in the specified 'self.opath'.

        """

        p0 = X.iloc[0]  # read in the input info
        params = lmfit.Parameters()  # empty parameter class
        success = True  # check for success

        if model == 'Medlyn':
            min, max = self.param_space('g1')
            params.add('g1', p0.g1, min=min, max=max)
            min, max = self.param_space('sref')
            params.add('sref', p0.sref, min=min, max=max)

        if model == 'Eller':
            min, max = self.param_space('kmax')
            params.add('kmaxS1', p0.kmaxS1, min=min, max=max)

        if (model == 'ProfitMax') or (model == 'ProfitMax2'):
            min, max = self.param_space('kmax')
            params.add('kmax', p0.kmax, min=min, max=max)

        # the following models all require the Sperry kmax as an input!
        if model == 'Tuzet':
            min, max = self.param_space('g1')
            params.add('g1T', p0.g1T, min=min, max=max)

            if 'Tleaf' in X.columns:  # vary g1 and kmax
                min, max = self.param_space('kmax')
                params.add('kmaxT', p0.kmax, min=min, max=max)

            else:  # vary g1 and Pref, sref fixed
                min, max = self.param_space('PrefT', P50=p0.P50, P88=p0.P88)

                if any(X['Ps_pd'] > p0.PrefT):
                    params.add('PrefT', p0.PrefT, min=min, max=max)

                else:
                    params.add('PrefT', -p0.P88, min=min, max=max)

        if model == 'WUE-LWP':
            min, max = self.param_space('Lambda')
            params.add('Lambda', p0.Lambda, min=min, max=max)

        if model == 'CGain':
            min, max = self.param_space('Kappa')
            params.add('Kappa', p0.Kappa, min=min, max=max)

        if model == 'CMax':
            min, max = self.param_space('Alpha')
            params.add('Alpha', p0.Alpha, min=min, max=max)
            min, max = self.param_space('Beta')
            params.add('Beta', p0.Beta, min=min, max=max)

        if model == 'SOX-OPT':
            min, max = self.param_space('kmax')
            params.add('kmaxS2', p0.kmaxS2, min=min, max=max)

        if model == 'LeastCost':
            min, max = self.param_space('kmax')
            params.add('kmaxLC', p0.kmaxLC, min=min, max=max)
            min, max = self.param_space('Eta')
            params.add('Eta', p0.Eta, min=min, max=max)

        if model == 'CAP':
            min, max = self.param_space('krl')
            params.add('krlC', p0.krlC, min=min, max=max)
            min, max = self.param_space('Pcrit', P50=p0.P50, P88=p0.P88)

            if any(X['Ps_pd'] > p0.PcritC):
                params.add('PcritC', p0.PcritC, min=min, max=max)

            else:
                params.add('PcritC', -p0.P88, min=min, max=max)

        if model == 'MES':
            min, max = self.param_space('krl')
            params.add('krlM', p0.krlM, min=min, max=max)
            min, max = self.param_space('Pcrit', P50=p0.P50, P88=p0.P88)

            if any(X['Ps_pd'] > p0.PcritM):
                params.add('PcritM', p0.PcritM, min=min, max=max)

            else:
                params.add('PcritM', -p0.P88, min=min, max=max)

        if not os.path.isdir(self.opath):  # create output dir
            os.makedirs(self.opath)

        # run the minimizer
        if self.method == 'emcee':
            out = lmfit.minimize(fres, params, args=(model, X, Y,
                                                     self.inf_gb,),
                                 method=self.method, steps=self.steps,
                                 nwalkers=self.nchains, burn=self.burn,
                                 thin=self.thin, is_weighted=False,
                                 progress=False, nan_policy='omit')

        else:
            out = lmfit.minimize(fres, params, args=(model, X, Y,
                                                     self.inf_gb,),
                                 method=self.method, nan_policy='omit')

            for param in out.params.values():

                if np.isclose(param.value, param.init_value):
                    params[param.name] = lmfit.Parameter(name=param.name,
                                                         value=1.5 *
                                                         param.init_value)
                    out = lmfit.minimize(fres, params,
                                         args=(model, X, Y, self.inf_gb,),
                                         method=self.method,
                                         nan_policy='omit')

        if not os.path.isfile(os.path.join(self.opath, '%s.txt' % (model))):
            txt = open(os.path.join(self.opath, '%s.txt' % (model)), 'w+')

        else:  # append to existing file
            txt = open(os.path.join(self.opath, '%s.txt' % (model)), 'a+')

        txt.write('\n')
        txt.write(lmfit.fit_report(out))

        if not success:
            txt.write('\n##  Warning: had to fix first parameter value')

        txt.write('\n')
        txt.close()  # close text file

        return out.params.valuesdict()


def soil_water(df, profile):

    """
    Generates idealised soil moisture profiles, where the volumetric
    water content is made to decline exponentially at different rates,
    depending on whether the soil remains 'wet' by the end of the
    decline (i.e., above the field capacity), or becomes 'dry' (i.e.,
    below the field capacity but above the wilting point).

    Arguments:
    ----------
    df: pandas dataframe
        dataframe containing all input data & params

    profile: string
        either 'wet' or 'inter'

    Returns:
    --------
    sw: array
        volumetric soil water content [m3 m-3]

    Ps: array
        soil water potential [MPa]

    """

    # initialize the soil moisture array
    sw = np.full(len(df), df['theta_sat'][0])

    if profile == 'wet':
        # starting value from which the decay is applied
        start = sw[0]

        # rate of decline
        rate = -1.5 / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))

        # minimum sw at the end of the decline period
        sw_min = (sw[0] + df['fc'][0]) / 2.25

    if profile == 'inter':
        start = 0.9 * sw[0]
        rate = -6. / len(df) * (np.log(sw[0]) - np.log(df['fc'][0]))
        sw_min = df['pwp'][0]

    track = 1  # track the iterations

    for i in range(len(df)):

        if i == 0:
            sw[i] = start

        else:
            sw[i] = sw[i-1]

        if df['PPFD'].iloc[i] > 50.:  # daytime hours
            sw[i] = np.maximum(start / (1. - rate * track), sw_min)
            track += 1

    # get the soil water potentials matching the soil moisture profile
    Ps = np.asarray([water_potential(df.iloc[0], sw[i])
                     for i in range(len(sw))])

    return sw, Ps


def check_idealised_files(ifile, ofile):

    """
    Ensures the idealised forcing files (i.e., met data) and idealised
    gs timeseries files exist.

    Arguments:
    ----------
    ifile: string
        input file name, including path

    ofile: string
        output file name, including path

    Returns:
    --------
    The input and output files if they don't already exist.

    """

    # check that the 4 week forcing file exists
    if not os.path.isfile(ifile):  # create file if it doesn't exist
        params = InForcings().defparams
        params.doy = random.randrange(92, 275)  # random day within GS
        InForcings().run(ifile, params, Ndays=7*4)

    # check that the output file from the reference model exists
    if not os.path.isfile(ofile):
        df1, __ = read_csv(ifile)

        # add the soil moisture profile to the input data
        df1['sw'], df1['Ps'] = soil_water(df1, os.path.basename(ofile)
                                                 .split('_')[1])
        df1['Ps_pd'] = df1['Ps'].copy()  # pre-dawn soil water potential
        df1['Ps_pd'].where(df1['PPFD'] <= 50., np.nan, inplace=True)

        # fixed value for the wind speed
        df1['u'] = df1['u'].iloc[0]

        # non time-sensitive: last valid propagated until next valid
        df1.fillna(method='ffill', inplace=True)

        __ = hrun(ofile, df1, len(df1.index), 'Farquhar', models=['Medlyn'],
                  inf_gb=True)

    return


def prep_training_N_target(ifile, ofile, profile=None):

    """
    Prepares the input and output data files by adding in missing
    variables, converting units, and ensuring only daytime hours are
    included.

    Arguments:
    ----------
    ifile: string
        input file name, including path

    ofile: string
        output file name, including path

    profile: string
        soil moisture profile, 'wet' or 'inter'

    Returns:
    --------
    df1: pandas dataframe
        dataframe containing all input data & params

    Y: pandas series
        stomatal conductance [mmol m-2 s-1]

    """

    # read in the input and output data
    df1, __ = read_csv(ifile)
    df2, __ = read_csv(ofile)

    if profile is not None:  # generate soil moisture profile
        df1['sw'], df1['Ps'] = soil_water(df1, profile)

    # add the predawn soil moisture profile to the input data
    df1['Ps_pd'] = df1['Ps'].copy()  # pre-dawn soil water potentials

    if profile is not None:  # idealised data
        df1['Ps_pd'].where(df1['PPFD'] <= 50., np.nan, inplace=True)
        df1['u'] = df1['u'].iloc[0]  # fix the wind speed

    # non time-sensitive: last valid value propagated until next valid
    df1.fillna(method='ffill', inplace=True)

    # add Rnet to the input (because of no ET, soil albedo feedbacks)
    df1['Rnet'] = net_radiation(df1)
    df1['scale2can'] = 1.  # muted scaling leaf to canopy

    if profile is not None:  # idealised data, drop below min threshold
        Y = df2['gs(std)'][df1['PPFD'] > 50.] * 1000.  # mmol m-2 s-1
        df1 = df1[df1['PPFD'] > 50.]
        df1.reset_index(inplace=True, drop=True)

    else:  # obs
        Y = df2['gs'] * 1000.  # in mmol m-2 s-1

    return df1, Y


def extract_calib_info(fname):

    """
    Processes any text output file that contains calibration information
    as outputted by the LMFIT package.

    Arguments:
    ----------
    fname: string
        input file name, including path

    Returns:
    --------
    info: list
        a nested list that contains each of the minimizers' info in
        different sublists

    """

    # read in the text file
    f = open(fname, 'r')
    lines = f.readlines()

    # reading criteria
    k1 = 'fitting method'
    k2 = 'function evals'
    k3 = 'data points'
    k4 = 'Bayesian info crit'
    k5 = ' ('  # calibrated parameters
    k6 = '(init'  # calibrated parameters
    k7 = '+/-'  # calibrated parameters
    k8 = ':'  # calibrated parameters
    k9 = '(fixed'  # calibrated parameters
    k10 = '=='  # calibrated parameters

    # info to keep
    info = [e.split('=') if (k1 in e) else [e.split('=')[1]] if ((k2 in e) or
            (k3 in e) or (k4 in e)) else
            [(e.split(k6)[0].split(k5)[0].split(k7)[0].split(k8)[0]),
             (e.split(k6)[0].split(k5)[0].split(k7)[0].split(k8)[1]),
             e.split(k6)[0].split(k5)[0].split(k7)[1]] if (k7 in e) else
            [e.split(k6)[0].split(':')[0], e.split(k6)[0].split(':')[1], 'nan']
            if (k6 in e) else [e.split(k9)[0].split(':')[0],
                               e.split(k9)[0].split(':')[1], 'nan']
            if (k9 in e) else [e.split(k10)[0].split(':')[0],
                               e.split(k10)[0].split(':')[1], 'nan']
            if (k10 in e) else [''] for e in lines]

    # remove end lines and formatting issues
    info = [e.strip('\n') for sub in info for e in sub if e != '']
    info = [e.replace(' ', '') if (':' in e) else e.strip() for e in info]

    # split into sublists containing each solver's info
    info = [list(sub) for e, sub in groupby(info, lambda x: k1 not in x) if e]

    return info
