from nose import tools
import numpy as np
from random_data.spectra import SpectralDensity


def test_SpectralDensity_signal_input():
    # Different length signals should fail
    x = np.random.randn(50e3)
    y = np.random.randn(50e3 + 1)
    tools.assert_raises(ValueError, SpectralDensity, x, {'y': y})

    # If `x` and `y` are different, we are computing cross-spectral density
    tools.assert_equal(SpectralDensity(x, y=y[:-1]).kind, 'cross-spectral')

    # If `x` and `y` are equal, we are computing autospectral density
    y = x
    tools.assert_equal(SpectralDensity(x, y=y).kind, 'autospectral')

    # If `y` is not specified, we are computing autospectral density
    tools.assert_equal(SpectralDensity(x).kind, 'autospectral')


def test_SpectralDensity__getNumPtsPerReal():
    x = np.random.randn(50e3)

    # Create `SpectralDensity` object
    Fs = 1.0
    Tens=40960
    Nreal_per_ens = 1
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # With `Fs` = 1, the number of points in an ensemble is *equal* to `Tens`.
    # Further, with only a single realization in the ensemble, the number of
    # points in the ensemble should be equal to `Tens`.
    Nexp = np.int(Tens)
    tools.assert_equal(sd._getNumPtsPerReal(Fs, Tens, Nreal_per_ens), Nexp)

    # Increasing number of realizations by factor of 10 should decrease
    # number of points in realization by a factor of 10
    c = 10
    tools.assert_equal(sd._getNumPtsPerReal(Fs, Tens, c * Nreal_per_ens),
                       Nexp / c)

    # ... and increasing sampling rate by factor of 10 should increase
    # the number of points in a realization by factor of 10
    c = 10
    tools.assert_equal(sd._getNumPtsPerReal(c * Fs, Tens, Nreal_per_ens),
                       c * Nexp)


def test_SpectralDensity_getFrequencies():
    x = np.random.randn(100)

    # For `Fs` = 1 and one realization per ensemble, the number
    # of points per realization is simply `Tens`. The number of points
    # per realization directly determines the frequency resolution
    # of the spectral density estimate.
    Fs = 1.0
    Nreal_per_ens = 1
    Tens = 16.
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    fexp = np.arange((Tens // 2) + 1) / Tens
    np.testing.assert_equal(sd.f, fexp)

    # If we have `c` realizations per ensemble, the frequency resolution
    # should decrease by a factor of `c`
    c = 2
    Nreal_per_ens *= c
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=Nreal_per_ens)
    np.testing.assert_equal(sd.f, fexp[::c])

    # ... and if we increase `Fs` by `c`, the frequency resolution should
    # stay the same, but the frequency range will increase by `c`
    Fs *= c
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=Nreal_per_ens)
    np.testing.assert_equal(sd.f, c * fexp)


def test_SpectralDensity_getTimes():
    x = np.random.randn(256)

    # For `Fs` = 1 and `Tens` = len(x), there is only one ensemble,
    # and the returned time base should correspond to the midpoint.
    Fs = 1.0
    Nreal_per_ens = 1
    Tens = len(x)
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    Texp = Tens / 2.
    np.testing.assert_equal(sd.t, Texp)

    # For `Fs` = 1 and `Tens` = len(x) / 4, there are four ensembles,
    # and the returned time base should correspond to the midpoints.
    c = 4
    Tens = len(x) / c
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    Texp = Tens * np.arange(0.5, c)
    np.testing.assert_equal(sd.t, Texp)


def test_SpectralDensity_white_noise():
    # White noise of a given power
    noise_power = np.sqrt(2)
    x = np.sqrt(noise_power) * np.random.randn(1e6)

    # Compute autospectral density of `x`
    asd = SpectralDensity(x)

    # Average over time, then integrate over frequency
    noise_power_estimate = np.sum(np.mean(asd.Gxy, axis=-1)) * asd.df

    tools.assert_almost_equal(noise_power, noise_power_estimate, places=1)
