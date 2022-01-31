# -*- coding: utf-8 -*-

"""
Get the solar zenith angle: this is used to realistically limit the
daytime hours when writing the final csv files.

This file is part of the TractLSM model.

Copyright (c) 2022 Manon E. B. Sabot

Please refer to the terms of the MIT License, which you should have
received along with the TractLSM.

References:
-----------
* De Pury & Farquhar (1997) PCE, 20, 537-557.
* Hughes, David W.; Yallop, B. D.; Hohenkerk, C. Y. (1989), "The
  Equation of Time", Monthly Notices of the Royal Astronomical Society
  238: 1529–1535
* Spencer, J. W. (1971). Fourier series representation of the position
  of the sun.

"""

__title__ = "Solar geometry at a location"
__author__ = ["Manon E. B. Sabot", "Martin De Kauwe"]
__version__ = "3.0 (12.12.2019)"
__email__ = ["m.e.b.sabot@gmail.com", "mdekauwe@gmail.com"]


# ======================================================================

# general modules
import numpy as np  # array manipulations, math operators


# ======================================================================

def day_angle(doy):

    """
    Calculation of day angle. De Pury & Farquhar eq. A18

    Arguments:
    ----------
    doy: int or float
        day of year

    Returns:
    ---------
    fractional year / day angle [radians]

    """
    return 2. * np.pi * doy / 365.25


def solar_declination(doy):

    """
    Solar Declination Angle is a function of day of year and is
    indepenent of location, varying between 23deg45' and -23deg45'. De
    Pury & Farquhar eq. A14

    Arguments:
    ----------
    doy: int or float
        day of year

    gamma: float
        fractional year / day angle [radians]

    Returns:
    --------
    Solar Declination Angle [radians]

    """

    return np.radians(-23.45) * np.cos(2. * np.pi * (doy + 10.) / 365.25)


def eq_of_time(gamma):

    """
    Equation of time - correction for the difference btw solar time and
    the clock time, Hughes et al. (1989)

    Arguments:
    ----------
    gamma: float
        fractional year / day angle [radians]

    Returns:
    --------
    Equation of time [minutes]

    """

    # Spencer '71. This best matches the de Pury worked example (pg 554)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) -
                    0.032077 * np.sin(gamma) - 0.014615 * np.cos(2. * gamma) -
                    0.04089 * np.sin(2. * gamma))

    return eqt


def solar_midday(eqt, longitude):

    """
    Calculation of solar midday. De Pury & Farquhar eq. A16

    Arguments:
    ----------
    eqt: float
        equation of time [minutes]

    longitude: float
        longitude [deg]

    Returns:
    --------
    Solar midday [hours]

    """

    # international standard meridians are multiples of 15 deg E/W
    lonmed = round(longitude / 15.) * 15.

    return 12. + (4. * (lonmed - longitude) - eqt) / 60.


def hour_angle(hod, gamma, longitude):

    """
    Calculation of hour angle, relative to solar midday. De Pury &
    Farquhar eq. A15

    Arguments:
    ----------
    hod: float
        hour of the day (0.5 to 24)

    gamma: float
        fractional year / day angle [radians]

    longitude: float
        longitude [deg]

    Returns:
    ---------
    Hour angle [radians]

    """

    eqt = eq_of_time(gamma)
    t0 = solar_midday(eqt, longitude)

    return np.pi * (hod - t0) / 12.


def cos_zenith(doy, hod, latitude, longitude):

    """
    The solar zenith angle is the angle between the zenith and the
    centre of the sun's disc. The solar elevation angle is the altitude
    of the sun, the angle between the horizon and the centre of the
    sun's disc. Since these two angles are complementary, the cosine of
    either one of them equals the sine of the other, i.e.
    cos theta = sin beta.

    Arguments:
    ----------
    doy: int or float
        day of year

    hod: float
        hour of the day (0.5 to 24)

    latitude: float
        latitude [deg]

    longitude: float
        longitude [deg]

    Returns:
    --------
    cosine of the zenith angle of the sun (0-1)

    """

    gamma = day_angle(doy)
    rdec = solar_declination(doy)
    h = hour_angle(hod, gamma, longitude)

    # A13 - De Pury & Farquhar
    cos_zenith = (np.sin(np.radians(latitude)) * np.sin(rdec) +
                  np.cos(np.radians(latitude)) * np.cos(rdec) * np.cos(h))

    return np.maximum(0., np.minimum(1., cos_zenith))
