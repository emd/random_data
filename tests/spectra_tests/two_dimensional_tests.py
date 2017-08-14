from nose import tools
import numpy as np
import random_data as rd
from random_data.spectra.two_dimensional import (
    TwoDimensionalAutoSpectralDensity)


# Construct signal from which to initialize `SpatialCrossCorrelation`:
# ====================================================================
#
# Temporal grid:
# --------------
Fs = 1.
t0 = 0
T = 10000
t = np.arange(t0, T, 1. / Fs)

# Spatial grid:
# -------------
Fs_spatial = 1.
z0 = 0
Ztot = 10
z = np.arange(z0, Ztot, 1. / Fs_spatial)

# 2d computational grids:
# -----------------------
tt = np.outer(np.ones(len(z)), t)
zz = np.outer(z, np.ones(len(t)))

# Coherent signal in presence of white noise:
# -------------------------------------------
A = 1.
f0 = 0.1
xi0 = 0.25
x = A * np.cos(2 * np.pi * ((xi0 * zz) + (f0 * tt)))
x += np.random.rand(x.shape[0], x.shape[1])
x -= np.mean(x)  # avoid low-f, low-xi leakage

# Compute `SpatialCrossCorrelation` object:
# =========================================
corr = rd.array.SpatialCrossCorrelation(
    x, z, Fs=Fs, t0=t0,
    Tens=(0.5 * T), Nreal_per_ens=100)


def test_TwoDimensionalAutoSpectralDensity_ValueError():
    # Must pass `SpatialCrossCorrelation` object, not array
    tools.assert_raises(
        ValueError,
        TwoDimensionalAutoSpectralDensity,
        corr.Gxy)

    # Must use a method that has been implemented
    tools.assert_raises(
        ValueError,
        TwoDimensionalAutoSpectralDensity,
        corr, **{'spatial_method': 'ARMA'})

    return


def test_TwoDimensionalAutoSpectralDensity_Fourier():
    # Estimate 2d autospectral density via Fourier method:
    # ----------------------------------------------------
    asd2d = TwoDimensionalAutoSpectralDensity(
        corr, spatial_method='fourier',
        fourier_params={'window': np.hanning})

    # Test that peak appears in right place:
    # --------------------------------------
    peak = np.max(np.abs(asd2d.Sxx))
    ind = np.where(np.abs(asd2d.Sxx) == peak)

    # Passing implies accuracy within 5%
    xi_ratio = asd2d.xi[ind[0]] / xi0
    np.testing.assert_almost_equal(xi_ratio, 1, decimal=1)

    # Passing implies accuracy within 5%
    f_ratio = asd2d.f[ind[1]] / f0
    np.testing.assert_almost_equal(f_ratio, 1, decimal=1)

    # Test power conservation:
    # ------------------------
    # ???

    return


def test_TwoDimensionalAutoSpectralDensity_Burg():
    # Estimate 2d autospectral density via Burg AR method:
    # ----------------------------------------------------
    asd2d = TwoDimensionalAutoSpectralDensity(
        corr, spatial_method='burg',
        burg_params={'p': 5, 'Nxi': 101})

    # Test that peak appears in right place:
    # --------------------------------------
    peak = np.max(np.abs(asd2d.Sxx))
    ind = np.where(np.abs(asd2d.Sxx) == peak)

    # Passing implies accuracy within 5%
    xi_ratio = asd2d.xi[ind[0]] / xi0
    np.testing.assert_almost_equal(xi_ratio, 1, decimal=1)

    # Passing implies accuracy within 5%
    f_ratio = asd2d.f[ind[1]] / f0
    np.testing.assert_almost_equal(f_ratio, 1, decimal=1)

    # Test power conservation:
    # ------------------------
    # ???

    return
