from nose import tools
import numpy as np
from random_data.spectra import CrossSpectralDensity
from random_data.array import Array, coefficient_of_determination


def test_getSpectralDensities():
    # Sampling properties
    Fs = 200e3
    t0 = 0
    t = np.arange(0, 1e-3, 1. / Fs)
    locations = np.arange(3)

    # Signal properties
    f0 = 50e3
    y1 = np.cos(2 * np.pi * f0 * t)
    y2 = np.cos((2 * np.pi * f0 * t) + (np.pi / 4))
    y3 = np.cos((2 * np.pi * f0 * t) + (np.pi / 2))
    signals = np.array([y1, y2, y3])

    # Spectral estimation properties
    Tens = 1e-4
    Nreal_per_ens = 5

    # Individually compute cross-spectral densities
    csd12 = CrossSpectralDensity(
        y1, y2, Fs=Fs, t0=t0, Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd13 = CrossSpectralDensity(
        y1, y3, Fs=Fs, t0=t0, Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd23 = CrossSpectralDensity(
        y2, y3, Fs=Fs, t0=t0, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Create an array object and compute corresponding
    # cross-spectral densities
    A = Array(signals, locations, Fs=Fs, t0=t0,
              Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Ensure cross-spectral densities in array object `A`
    # are indeed the cross-spectral densities they are supposed to be
    np.testing.assert_equal(csd12.Gxy, A.csd[0].Gxy)
    np.testing.assert_equal(csd13.Gxy, A.csd[1].Gxy)
    np.testing.assert_equal(csd23.Gxy, A.csd[2].Gxy)

    return


def test_getSlice():
    # Sampling properties
    Fs = 200e3
    t0 = 0
    t = np.arange(0, 1e-3, 1. / Fs)
    locations = np.arange(3)

    # Signal properties
    f0 = 50e3
    y1 = np.cos(2 * np.pi * f0 * t)
    y2 = np.cos((2 * np.pi * f0 * t) + (np.pi / 4))
    y3 = np.cos((2 * np.pi * f0 * t) + (np.pi / 2))
    signals = np.array([y1, y2, y3])

    # Spectral estimation properties
    Tens = 1e-4
    Nreal_per_ens = 5

    # Create an array object and compute corresponding
    # cross-spectral densities
    A = Array(signals, locations, print_locations=False,
              Fs=Fs, t0=t0,
              Tens=Tens, Nreal_per_ens=Nreal_per_ens,
              print_params=False, print_status=False)

    # Test slice routines with indexing
    for tind in [0, -1]:
        for find in [0, -1]:
            # Slice by index
            theta_xy = A.getSlice('theta_xy', tind=tind, find=find)
            gamma2xy = A.getSlice('gamma2xy', tind=tind, find=find)
            Gxy = A.getSlice('Gxy', tind=tind, find=find)

            # Ensure slicing routine does as expected
            # by comparing against "manual" slices
            for cind in np.arange(len(A.yloc)):
                np.testing.assert_equal(
                    theta_xy[cind],
                    A.csd[cind].theta_xy[find, tind])
                np.testing.assert_equal(
                    gamma2xy[cind],
                    A.csd[cind].gamma2xy[find, tind])
                np.testing.assert_equal(
                    Gxy[cind],
                    A.csd[cind].Gxy[find, tind])

    # Test slice routines with physical time and frequency values
    # by using smallest and largest bins in frequency and time
    for t in [t0, t[-1]]:
        for f in [0, 0.5 * Fs]:
            # Slice by physical time and frequency
            theta_xy = A.getSlice('theta_xy', t=t, f=f)
            gamma2xy = A.getSlice('gamma2xy', t=t, f=f)
            Gxy = A.getSlice('Gxy', t=t, f=f)

            # Determine corresponding indices
            if t == t0:
                tind = 0
            else:
                tind = -1

            if f == 0:
                find = 0
            else:
                find = -1

            # Ensure slicing routine does as expected
            # when slicing by physical time and frequency values
            for cind in np.arange(len(A.yloc)):
                np.testing.assert_equal(
                    theta_xy[cind],
                    A.csd[cind].theta_xy[find, tind])
                np.testing.assert_equal(
                    gamma2xy[cind],
                    A.csd[cind].gamma2xy[find, tind])
                np.testing.assert_equal(
                    Gxy[cind],
                    A.csd[cind].Gxy[find, tind])

    return


def test_coefficient_of_determination():
    tools.assert_equal(1, coefficient_of_determination(0., 1.))
    tools.assert_equal(0, coefficient_of_determination(1., 1.))
