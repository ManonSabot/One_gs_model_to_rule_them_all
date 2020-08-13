# -*- coding: utf-8 -*-

"""
Apologies, this is completely hardwired right now... Will get it fixed soonish!

"""

__title__ = ""
__author__ = "[Manon Sabot]"
__version__ = "1.0 (16.01.2019)"
__email__ = "m.e.b.sabot@gmail.com"


#==============================================================================

import warnings # ignore these warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# general modules
import os
import sys
import importlib
import numpy as np # array manipulations, math operators
import lmfit  # non-linear model optimizer

# own modules
from TractLSM.Utils import get_main_dir  # get the project's directory
from models_2_fit import fres

#==============================================================================

class NLMFIT(object):

    def __init__(self, method='powell', store=None, inf_gb=True):

        # fitting method
        self.method = method  # which solver is used

        # MCMC-specific
        self.steps = 11000
        self.nchains = 3
        self.burn = 1000
        self.thin = 2

        if store is None:  # default storing path for the outputs
            self.base_dir = get_main_dir()  # working paths
            self.opath = os.path.join(os.path.join(self.base_dir, 'output'),
                                      'calibrations')

        else:  # user defined storing path for the outputs
            self.opath = store

        self.inf_gb = inf_gb  # whether to calculate gb or not

    def run(self, X, Y, model, g1=False):

        p0 = X.iloc[0]  # read in the input info
        params = lmfit.Parameters()  # empty parameter class

        if model == 'Medlyn-LWP':
            params.add('sref', p0.sref, min=0.01, max=10.)

            if g1:
                params.add('g1', p0.g1, min=0.05, max=20.)

        if model == 'SOX':
            params.add('kmaxS1', p0.kmaxS1, min=0.01, max=30.)

        if model == 'ProfitMax':
            params.add('kmax', p0.kmax, min=0.05, max=30.)

        # the following models all require the Sperry kmax as an input!
        if model == 'Tuzet':  # only vary g1 and Pref at first, sref fixed
            params.add('g1T', p0.g1T, min=0.05, max=20.)

            if p0.PrefT < p0.Ps_pd:
                params.add('PrefT', p0.PrefT, min=-3. * p0.P88, max=-0.15)

            else:
                params.add('PrefT', -p0.P88, min=-3. * p0.P88, max=-0.15)

        if model == 'WUE-LWP':
            params.add('Lambda', p0.Lambda, min=0.1, max=100.)

        if model == 'CGainNet':
            params.add('beta', p0.beta, min=0.1, max=100.)

        if model == 'CMax':
            params.add('Alpha', p0.Alpha, min=0.5, max=80.)
            params.add('Beta', p0.Beta, min=0.1, max=8.)

        if model == 'SOX-OPT':
            params.add('kmaxS2', p0.kmaxS2, min=0.01, max=30.)
            params.add('factor', 0.7, min=0.05, max=0.95)  # ancillary term
            params.add('ksc_prev', 0.7 * p0.kmaxS2, min=0.0005, max=28.5,
                       expr='factor * kmaxS2')

        if model == 'LeastCost':
            params.add('kmaxLC', p0.kmaxLC, min=0.05, max=30.)
            params.add('BoA', p0.BoA, min=0.1, max=250.)

        if model == 'CAP':
            params.add('krlC', p0.krlC, min=0.05, max=100.)

            if p0.PcritC < p0.Ps_pd:
                params.add('PcritC', p0.PcritC, min=-3. * p0.P88, max=-0.15)

            else:
                params.add('PcritC', -p0.P88, min=-3. * p0.P88, max=-0.15)

        if model == 'MES':
            params.add('krlM', p0.krlM, min=0.05, max=100.)

            if p0.PcritC < p0.Ps_pd:
                params.add('PcritM', p0.PcritM, min=-3. * p0.P88, max=-0.15)

            else:
                params.add('PcritM', -p0.P88, min=-3. * p0.P88, max=-0.15)

        # run the minimizer
        if self.method == 'emcee':
            out = lmfit.minimize(fres, params, args=(model, X, Y, self.inf_gb),
                                 method=self.method, steps=self.steps,
                                 nwalkers=self.nchains, burn=self.burn,
                                 thin=self.thin, is_weighted=False,
                                 progress=False)

        else:
            out = lmfit.minimize(fres, params, args=(model, X, Y, self.inf_gb),
                                 method=self.method)

        if not os.path.isdir(self.opath):  # create output dir
            os.makedirs(self.opath)

        if not os.path.isfile(os.path.join(self.opath, '%s.txt' % (model))):
            txt = open(os.path.join(self.opath, '%s.txt' % (model)), 'w+')

        else:  # append to existing file
            txt = open(os.path.join(self.opath, '%s.txt' % (model)), 'a+')

        txt.write('\n')
        txt.write(lmfit.fit_report(out))
        txt.write('\n')
        txt.close()  # close text file

        if model == 'Tuzet':  # test two-step fitting

            # retrieve the first (now solved for) parameter name and value
            p1name = str(out.params.valuesdict().popitem(last=False)[0])
            p1val = out.params.valuesdict().popitem(last=False)[1]

            # reset the input parameter dic accordingly
            params[p1name] = lmfit.Parameter(name=p1name, value=p1val,
                                             vary=False)

            # now vary sref alongside Pref
            params.add('srefT', p0.srefT, min=0.01, max=10.)

            # re-run the minimizer
            if self.method == 'emcee':
                out = lmfit.minimize(fres, params, args=(model, X, Y,
                                                         self.inf_gb),
                                     method=self.method, steps=self.steps,
                                     nwalkers=self.nchains, burn=self.burn,
                                     thin=self.thin, is_weighted=False,
                                     progress=False)

            else:
                out = lmfit.minimize(fres, params, args=(model, X, Y,
                                                         self.inf_gb),
                                     method=self.method)

            if not os.path.isfile(os.path.join(self.opath,
                                               '%s2.txt' % (model))):
                txt = open(os.path.join(self.opath, '%s2.txt' % (model)), 'w+')

            else:  # append to existing file
                txt = open(os.path.join(self.opath, '%s2.txt' % (model)), 'a+')

            txt.write('\n')
            txt.write(lmfit.fit_report(out))
            txt.write('\n')
            txt.close()  # close text file

        return out.params.valuesdict()
