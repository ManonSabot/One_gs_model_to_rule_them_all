# -*- coding: utf-8 -*-

"""
Default parameter class, necessary to run the model.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* ten Berge, Hein FM. Heat and water transfer in bare topsoil and lower
  atmosphere. No. 33. Pudoc, 1990.
* Campbell, G. S., & Norman, J. M. “An Introduction to Environmental
  Biophysics” 2nd Edition, Springer-Verlag, New York, 1998.
* Choat, B., Jansen, S., Brodribb, T. J., Cochard, H., Delzon, S.,
  Bhaskar, R., ... & Jacobsen, A. L. (2012). Global convergence in the
  vulnerability of forests to drought. Nature, 491(7426), 752.
* Kattge, J., & Knorr, W. (2007). Temperature acclimation in a
  biochemical model of photosynthesis: a reanalysis of data from 36
  species. Plant, cell & environment, 30(9), 1176-1190.
* Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley,
  P. C., Kirschbaum, M. U. F., ... & Wang, K. (2002). Temperature
  response of parameters of a biochemically based model of
  photosynthesis. II. A review of experimental data. Plant, Cell &
  Environment, 25(9), 1167-1179.
* Peltoniemi, M. S., Duursma, R. A., & Medlyn, B. E. (2012). Co-optimal
  distribution of leaf nitrogen and hydraulic conductance in plant
  canopies. Tree Physiology, 32(5), 510-519.

"""

__title__ = "Default parameter class necessary to run the model"
__author__ = "Manon E. B. Sabot"
__version__ = "8.0 (09.10.2020)"
__email__ = "m.e.b.sabot@gmail.com"


# ======================================================================

class default_params(object):  # default inputs needed to run model

    def __init__(self):

        # information used by the weather generator
        self.doy = 150.  # day of the year
        self.tmin = 5.  # degC
        self.tmax = 25.  # degC
        self.rain_day = 0.  # mm d-1
        self.RH = 30.  # relative humidity, %
        self.sw_rad_day = 1080. * 10.  # W m-2, 10 daylight hours
        self.Patm = 101.325  # kPa
        self.u = 2.  # m s-1

        # location of the former Duke FACE
        self.lat = 36.  # alternative 38.569120
        self.lon = -79.  # alternative -80.018519

        # gas concentrations
        self.CO2 = 41.55  # Pa, ~410 ppm as of early 2020
        self.O2 = 20.8  # kPa

        # photosynthesis related
        self.Vmax25 = 35.  # max carboxyl rate @ 25 degC (umol m-2 s-1)
        self.gamstar25 = 4.22  # CO2 compensation point @ 25 degC (Pa)
        self.Tref = 25.  # ref T for Vmax25, gamstar, deltaS, Hd
        self.JV = 1.67  # Jmax25 to Vmax25 ratio (Medlyn et al., 2002)
        self.Rlref = self.Vmax25 * 0.015  # resp @ TRlref (umol m-2 s-1)
        self.TRlref = 25.  # T for the ref keaf respiration (degC)
        self.Kc25 = 39.96  # Michaelis-Menten cst for carboxylation (Pa)
        self.Ko25 = 27.48  # Michaelis-Menten cst for oxygenation (kPa)
        self.alpha = 0.3  # quantum yield photo (mol(photon) mol(e-)-1)

        # Farquhar model
        self.c1 = 0.7  # curvature of light response
        self.c2 = 0.99  # transition Je vs Jc (Peltoniemi et al., 2012)

        # Collatz model
        self.c3 = 0.83  # curvature of light response
        self.c4 = 0.93  # transition Je vs Jc

        # energies of activation
        self.Ev = 60000.  # Vcmax, J mol-1
        self.Ej = 30000.  # Jmax, J mol-1
        self.Egamstar = 37830.  # gamstar, J mol-1
        self.Ec = 79430.  # carboxylation, J mol-1
        self.Eo = 36380.  # oxygenation, J mol-1

        # inhibition at higher temperatures (Kattge & Knorr, 2007)
        self.deltaSv = 650.  # Vmax entropy factor (J mol-1 K-1)
        self.deltaSj = 650.  # Jmax entropy factor (J mol-1 K-1)
        self.Hdv = 200000.  # Vmax decrease rate above opt T (J mol-1)
        self.Hdj = 200000.  # Jmax decrease rate above opt T (J mol-1)

        # relating to light / rad (C & N is Campbell & Norman)
        self.eps_l = 0.97  # leaf emiss. LW (Table 11.3 C & N 1998)
        self.albedo_l = 0.062  # leaf refl. SW vis (CABLE)
        self.tau_l = 0.05  # leaf transmis. SW vis (CABLE)
        self.chi_l = 9.99999978E-03  # leaf angle dist (spherical = 0)
        self.kn = 0.001  # extinction coef. of nitrogren (CABLE)

        # canopy / leaves
        self.height = 20.  # canopy height (m)
        self.LAI = 1.  # leaf area index (m2 m-2)
        self.max_leaf_width = 0.001  # m

        # stomata
        self.g1 = 2.4942  # sensitivity of gs to VPD and An (kPa0.5)
        self.g1T = 3.5  # sensitivity of gs to LWP and An (unitless)
        self.Kappa = 5.  # max water use in C units (umol m-2 s-1)
        self.Lambda = 5.  # WUE term (mol C mol-1 H2O)
        self.Eta = 5.  # cost of Rd:Vcmax to sapwood:E (mol C mol-1 H2O)

        # hydraulics
        self.P50 = 3.13  # xylem pressure at P50 (-MPa) - P. tadea
        self.P88 = 4.9  # same at P88 (-MPa) (Choat et al., 2012)
        self.kmax = 1.  # max plant hydr cond / LAI (mmol m-2 s-1 MPa-1)
        self.kmax2 = 1.  # max plant hydr cond / LAI in ProfitMax2
        self.kmaxWUE = self.kmax  # max plant hydr cond / LAI, WUE-LWP
        self.kmaxCN = self.kmax  # max plant hydr cond / LAI, CNetGain
        self.kmaxCM = self.kmax  # max plant hydr cond / LAI, CMax
        self.kmaxT = self.kmax  # max plant hydr cond / LAI, Tuzet
        self.kmaxS1 = self.kmax  # kmax from SOX analytical
        self.kmaxS2 = self.kmax  # kmax from SOX-Opt
        self.kmaxLC = self.kmax  # max plant hydr cond / LAI, LeastCost
        self.krlC = self.kmax  # fixed root-leaf conductance, CAP
        self.krlM = self.kmax   # fixed root-leaf conductance, MES
        self.ratiocrit = 0.05  # degree stom control? kcrit = N%(kmax)

        # empirical moisture stress functions
        self.sref = 2.  # sensitivity of the stomates to LWP (MPa-1)
        self.srefT = 2.  # sensitivity of the stomates to LWP (MPa-1)
        self.PrefT = -self.P50  # reference LWP for sensitivity (MPa)
        self.PcritC = -self.P50  # crit. LWP of stomatal closure (MPa)
        self.PcritM = -self.P50  # crit. LWP of stomatal closure (MPa)
        self.Alpha = 5.  # coef parabolic cost func. (mol m-2 s-1 MPa-2)
        self.Beta = 1.  # coef parabolic cost func. (mol m-2 s-1 MPa-1)

        # soil
        self.ground_area = 1.  # m2
        self.Ztop = 0.02  # top soil layer depth (m)
        self.Zbottom = 1.  # bottom soil layer depth (m)
        self.Psie = -0.0008  # air entry point water potential (MPa)
        self.Ps = self.Psie  # initial soil water potential (MPa)
        self.hyds = 9.305051e-6  # saturation hydr conductivity (m s-1)
        self.fc = 0.213388  # field capacity (m3 m-3)
        self.theta_sat = 0.4079013  # saturation soil moisture (m3 m-3)
        self.pwp = 0.1027524  # plant wilting point (m3 m-3)
        self.bch = 5.222721  # Clapp and Hornberger index

        return
