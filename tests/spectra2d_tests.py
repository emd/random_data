from nose import tools
import numpy as np
import random_data as rd


def coherent_2d_signal(z, t, A=0.03, f0=0.1, xi0=0.25):
    'Get 2d signal of a coherent fluctuation.'
    # 2d computational grids
    tt = np.outer(np.ones(len(z)), t)
    zz = np.outer(z, np.ones(len(t)))

    x = A * np.cos(2 * np.pi * ((xi0 * zz) + (f0 * tt)))
    x += np.random.rand(x.shape[0], x.shape[1])
    x -= np.mean(x)  # avoid low-f, low-xi leakage

    return x


def average_autospectral_density(x, Fs, t0, Tens, Nreal_per_ens):
    'Get average 1d autospectral density from 2d signal `x`.'
    # Initialize w/ signal from first position
    asd = rd.spectra.AutoSpectralDensity(
        x[0, :], Fs=Fs, t0=t0,
        Tens=T, Nreal_per_ens=Nreal_per_ens)

    Gxx = asd.Gxx

    # Loop through remainder of positions, accumulating
    # autospectral density at each
    for posind in (np.arange(x.shape[0] - 1) + 1):
        asd = rd.spectra.AutoSpectralDensity(
            x[posind, :], Fs=Fs, t0=t0,
            Tens=T, Nreal_per_ens=Nreal_per_ens)

        Gxx += asd.Gxx

    # Divide by number of positions to obtain average
    Gxx /= x.shape[0]

    return np.squeeze(Gxx)


# Signal and spectral-estimation parameters:
# ------------------------------------------
# Temporal-grid parameters
Fs = 1.
t0 = 0
T = 10000

# Spatial-grid parameters
Fs_spatial = 1.
z0 = 0
Z = 50

# Coherent spectral parameters
A = 0.03
f0 = 0.1 * Fs
xi0 = 0.25 * Fs_spatial

# Broadband spectral parameters
fc = 0.1 * Fs
pole = 2
vph = 1.0
Lz = 5

# Spectral-estimation parameters
Nreal_per_ens = 100

# Create signals:
# ---------------
# Create broadband signal
sig_broadband = rd.signals.RandomSignal2d(
    Fs=Fs, t0=t0, T=T, fc=fc, pole=pole,
    Fs_spatial=Fs_spatial, z0=z0, Z=Z, vph=vph, Lz=Lz)

# Subtract mean to minimize low-f, low-xi leakage
x_broadband = sig_broadband.x - np.mean(sig_broadband.x)

# Extract spatial and temporal grid of broadband signal and
# use to construct a coherent signal
z = sig_broadband.z()
t = sig_broadband.t()
x_coherent = coherent_2d_signal(z, t, A=A, f0=f0, xi0=xi0)

# Mixed broadband and coherent signal
x_mixed = x_broadband + x_coherent
x_mixed -= np.mean(x_mixed)

# Compute spatial correlations for each type of signal:
# -----------------------------------------------------
corr_coherent = rd.array.SpatialCrossCorrelation(
    x_coherent, z, Fs=Fs, t0=t0,
    Nreal_per_ens=Nreal_per_ens)

corr_broadband = rd.array.SpatialCrossCorrelation(
    x_broadband, z, Fs=Fs, t0=t0,
    Nreal_per_ens=Nreal_per_ens)

corr_mixed = rd.array.SpatialCrossCorrelation(
    x_mixed, z, Fs=Fs, t0=t0,
    Nreal_per_ens=Nreal_per_ens)

# Compute average 1d autospectral densities:
# ------------------------------------------
asd1d_Gxx_av_coherent = average_autospectral_density(
    x_coherent, Fs, t0, t[-1] - t[0], Nreal_per_ens)
asd1d_Gxx_av_broadband = average_autospectral_density(
    x_broadband, Fs, t0, t[-1] - t[0], Nreal_per_ens)
asd1d_Gxx_av_mixed = average_autospectral_density(
    x_mixed, Fs, t0, t[-1] - t[0], Nreal_per_ens)


def test_TwoDimensionalAutoSpectralDensity_ValueError():
    # Must pass `SpatialCrossCorrelation` object, not array
    tools.assert_raises(
        ValueError,
        rd.spectra2d.TwoDimensionalAutoSpectralDensity,
        corr_coherent.Gxy)

    # Must use a method that has been implemented
    tools.assert_raises(
        ValueError,
        rd.spectra2d.TwoDimensionalAutoSpectralDensity,
        corr_coherent, **{'spatial_method': 'ARMA'})

    return


def test_TwoDimensionalAutoSpectralDensity_Fourier_with_coherent_signal():
    # Estimate 2d autospectral density via Fourier method:
    # ----------------------------------------------------
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_coherent, spatial_method='fourier',
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

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_coherent)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_coherent,
        np.ones(len(Sf_asd2d)),
        decimal=1)

    return


def test_TwoDimensionalAutoSpectralDensity_Fourier_with_broadband_signal():
    # Estimate 2d autospectral density via Fourier method:
    # ----------------------------------------------------
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_broadband, spatial_method='fourier',
        fourier_params={'window': np.hanning})

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_broadband)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_broadband,
        np.ones(len(Sf_asd2d)),
        decimal=1)

    return


def test_TwoDimensionalAutoSpectralDensity_Fourier_with_mixed_signal():
    # Estimate 2d autospectral density via Fourier method:
    # ----------------------------------------------------
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_mixed, spatial_method='fourier',
        fourier_params={'window': np.hanning})

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_mixed)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_mixed,
        np.ones(len(Sf_asd2d)),
        decimal=1)

    return


def test_TwoDimensionalAutoSpectralDensity_Burg_with_coherent_signal():
    # Estimate 2d autospectral density via Burg AR method:
    # ----------------------------------------------------
    # Note: Choice of poles `p` explained in below section
    # regarding "local" power conservation.
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_coherent, spatial_method='burg',
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

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_coherent)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    # Note: For a coherent signal, we need more than one pole
    # if we want to get agreement with the 1d Fourier routine
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_coherent,
        np.ones(len(Sf_asd2d)),
        decimal=1)

    return


def test_TwoDimensionalAutoSpectralDensity_Burg_with_broadband_signal():
    # Estimate 2d autospectral density via Burg AR method:
    # ----------------------------------------------------
    # Note: Choice of poles `p` explained in below section
    # regarding "local" power conservation.
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_broadband, spatial_method='burg',
        burg_params={'p': 1, 'Nxi': 101})

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_broadband)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    # Note: For a purely broadband signal with a single phase velocity,
    # we only need one pole if we want to get agreement with the
    # 1d Fourier routine. The Burg and Fourier 2d spectral estimates
    # are obtained via entirely different means, each with its own
    # strengths and weaknesses, so we will take local power conservation
    # within 50% to be "agreement". (Both 2d estimates are very sensitive
    # to the number of measurement positions `len(z)`, and the Burg
    # method is also sensitive to the number of poles... so a general
    # "apples-to-apples" comparison is hard...)
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_broadband,
        np.ones(len(Sf_asd2d)),
        decimal=0.5)

    return


def test_TwoDimensionalAutoSpectralDensity_Burg_with_mixed_signal():
    # Estimate 2d autospectral density via Burg AR method:
    # ----------------------------------------------------
    # Note: Choice of poles `p` explained in below section
    # regarding "local" power conservation.
    asd2d = rd.spectra2d.TwoDimensionalAutoSpectralDensity(
        corr_mixed, spatial_method='burg',
        burg_params={'p': 10, 'Nxi': 101})

    # Test global power conservation:
    # -------------------------------
    P_asd2d = np.sum(np.abs(asd2d.Sxx)) * asd2d.df * asd2d.dxi
    P_x = np.var(x_mixed)
    np.testing.assert_almost_equal(
        P_asd2d / P_x,
        1,
        decimal=1)

    # Test local (i.e. at each frequency) power conservation:
    # -------------------------------------------------------
    # Note: For a mixed broadband & coherent signal, we need several
    # poles to identify both the broadband and coherent components.
    # Because the Burg and Fourier 2d spectral estimates are obtained
    # via entirely different means, each with its own strengths and
    # weaknesses, we will take local power conservation within 50%
    # to be "agreement". (Both 2d estimates are very sensitive
    # to the number of measurement positions `len(z)`, and the Burg
    # method is also sensitive to the number of poles... so a general
    # "apples-to-apples" comparison is hard...)
    Sf_asd2d = np.sum(np.abs(asd2d.Sxx), axis=0) * asd2d.dxi
    np.testing.assert_almost_equal(
        Sf_asd2d / asd1d_Gxx_av_mixed,
        np.ones(len(Sf_asd2d)),
        decimal=0.5)

    return
