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


def test_getTimeSlice():
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
    A = Array(signals, locations, Fs=Fs, t0=t0,
              Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    theta_xy_t0 = A.getTimeSlice('theta_xy', 0)
    gamma2xy_t0 = A.getTimeSlice('gamma2xy', 0)
    Gxy_tend = A.getTimeSlice('Gxy', -1)

    for cind in np.arange(len(A.yloc)):
        np.testing.assert_equal(
            theta_xy_t0[cind, :],
            A.csd[cind].theta_xy[:, 0])
        np.testing.assert_equal(
            gamma2xy_t0[cind, :],
            A.csd[cind].gamma2xy[:, 0])
        np.testing.assert_equal(
            Gxy_tend[cind, :],
            A.csd[cind].Gxy[:, -1])

    return


def test_coefficient_of_determination():
    tools.assert_equal(1, coefficient_of_determination(0., 1.))
    tools.assert_equal(0, coefficient_of_determination(1., 1.))
