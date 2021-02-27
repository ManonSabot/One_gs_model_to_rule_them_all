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
import numpy as np
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
        P88: float
            leaf water potential [MPa] at which 88% decrease in hydraulic
            conductance is observed
        """

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
        P95 = -b * ((- np.log(0.05)) ** (1. / c)) # MPa

        return P95

    def param_space(self, pname, P50=None, P88=None):

        if 'sref' in pname:

            return 0.01, 10.

        elif 'g1' in pname:

            return 0.01, 12.5

        elif 'kmax' in pname:

            return 0.005, 20.

        elif pname == 'ksc_prev':

            return 0.05 * 0.005, 0.95 * 20.

        elif 'krl' in pname:

            #return 0.01, 100.
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

    def run(self, X, Y, model, g1=False):

        p0 = X.iloc[0]  # read in the input info
        params = lmfit.Parameters()  # empty parameter class
        success = True  # check for success

        if model == 'Medlyn-LWP':
            min, max = self.param_space('sref')
            params.add('sref', p0.sref, min=min, max=max)

            if g1:
                min, max = self.param_space('g1')
                params.add('g1', p0.g1, min=min, max=max)

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
            params.add('factor', 0.7, min=0.05, max=0.95)  # ancillary term
            min, max = self.param_space('ksc_prev')
            params.add('ksc_prev', 0.7 * p0.kmaxS2, min=min, max=max,
                       expr='factor * kmaxS2')

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
            out = lmfit.minimize(fres, params, args=(model, X, Y, self.inf_gb,),
                                 method=self.method, steps=self.steps,
                                 nwalkers=self.nchains, burn=self.burn,
                                 thin=self.thin, is_weighted=False,
                                 progress=False, nan_policy='omit')

        else:
            out = lmfit.minimize(fres, params, args=(model, X, Y, self.inf_gb,),
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

        """
        # two-step test: recalibration of both Pref and sref
        if success and (model == 'Tuzet') and not ('Tleaf' in X.columns):

            # retrieve the first (now solved for) parameter name and value
            p1name = str(out.params.valuesdict().popitem(last=False)[0])
            p1val = out.params.valuesdict().popitem(last=False)[1]

            # reset the input parameter dic accordingly
            params[p1name].set(value=p1val, vary=False)

            # add sref to vary alongside Pref
            min, max = self.param_space('srefT')
            params.add('srefT', p0.srefT, min=min, max=max)

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
        """

        return out.params.valuesdict()
