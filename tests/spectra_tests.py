from nose import tools
import numpy as np
from random_data.spectra import AutoSpectralDensity, CrossSpectralDensity


def test_AutoSpectralDensity_signal_input():
    # Complex signal should fail
    x = np.random.randn(50e3)
    xc = x.astype('complex128')
    tools.assert_raises(ValueError, AutoSpectralDensity, xc)


def test_CrossSpectralDensity_signal_input():
    # Different length signals should fail
    x = np.random.randn(50e3)
    y = np.random.randn(50e3 + 1)
    tools.assert_raises(ValueError, CrossSpectralDensity, x, y)

    # Complex signals should fail
    xc = x.astype('complex128')
    yc = y.astype('complex128')
    tools.assert_raises(ValueError, CrossSpectralDensity, xc, y)
    tools.assert_raises(ValueError, CrossSpectralDensity, x, yc)
    tools.assert_raises(ValueError, CrossSpectralDensity, xc, yc)


def test_AutoSpectralDensity_white_noise():
    # White noise of a given power
    noise_power = np.sqrt(2)
    x = np.sqrt(noise_power) * np.random.randn(1e6)

    # Compute autospectral density of `x`
    asd = AutoSpectralDensity(x)

    # Average over time, then integrate over frequency
    noise_power_estimate = np.sum(np.mean(asd.Gxx, axis=-1)) * asd.df

    tools.assert_almost_equal(noise_power, noise_power_estimate, places=1)


def test_CrossSpectralDensity_vs_AutoSpectralDensity():
    # "Detrending" can really fuck things up if applied incorrectly, so
    # its generally best to just avoid it
    detrend = None

    # White noise of a given power
    noise_power = np.sqrt(2)
    x = np.sqrt(noise_power) * np.random.randn(1e6)

    # Compute autospectral density of `x`
    # asd = AutoSpectralDensity(x)
    asd = AutoSpectralDensity(x, detrend=detrend)

    # Ensure that autospectral density is real
    tools.assert_is(asd.Gxx.dtype.type, np.float64)

    # Now, the autospectral density of `x` is simply
    # the cross-spectral density of `x` against itself.
    # Thus, we should be able to compute the autospectral
    # density using `random_data.spectra.CrossSpectralDensity`, and
    # this should be in agreement with the autospectral density
    # computed from `random_data.spectra.AutoSpectralDensity`
    # (other than the fact that the cross-spectral density
    # data type will be, by definition, complex rather than real)
    csd = CrossSpectralDensity(x, x.copy(), detrend=detrend)

    # Ensure that cross-spectral density is complex (by definition) but
    # has null imaginary component
    tools.assert_is(csd.Gxy.dtype.type, np.complex128)
    np.testing.assert_equal(csd.Gxy.imag, 0)

    np.testing.assert_equal(asd.Gxx, csd.Gxy)


def test_CrossSpectralDensity_getCoherence():
    # Sampling parameters
    Fs = 32.
    t0 = 0.
    tf = 1000.
    t = np.arange(t0, tf, 1. / Fs)

    # Sinusoidal signal @ `f0`
    f0 = 1
    x = np.cos(2 * np.pi * f0 * t)

    # `x`, phase shifted by `ph0`
    ph0 = np.pi / 4
    y = np.cos((2 * np.pi * f0 * t) + ph0)

    # Let's also add some noise...
    noise_power = 1e0
    x += np.sqrt(noise_power) * np.random.randn(len(t))
    y += np.sqrt(noise_power) * np.random.randn(len(t))

    # Compute cross spectral density and magnitude-squared coherence
    Tens = 100.
    csd = CrossSpectralDensity(x, y, Fs=Fs, Tens=Tens, Nreal_per_ens=10)

    # Coherence should be real-valued!
    tools.assert_true(np.isrealobj(csd.gamma2xy))

    # Should have 0 <= gamma2xy <= 1 for all f and all t
    tools.assert_true(np.alltrue(np.greater_equal(csd.gamma2xy, 0)))
    tools.assert_true(np.alltrue(np.less_equal(csd.gamma2xy, 1)))


def test_CrossSpectralDensity_getPhaseAngle():
    # Sampling parameters
    Fs = 32.
    t0 = 0.
    tf = 1000.
    t = np.arange(t0, tf, 1. / Fs)

    # Sinusoidal signal @ `f0`
    f0 = 1
    x = np.cos(2 * np.pi * f0 * t)

    # `x`, phase shifted by `ph0`
    ph0 = np.pi / 4
    y = np.cos((2 * np.pi * f0 * t) + ph0)

    Tens = 100.
    csd = CrossSpectralDensity(x, y, Fs=Fs, Tens=Tens, Nreal_per_ens=10)
    csd.getPhaseAngle()

    f_ind = np.where(np.abs(csd.f - f0) == np.min(np.abs(csd.f - f0)))[0]
    ph0_est = np.mean(csd.theta_xy[f_ind, :], axis=-1)

    tools.assert_almost_equal(ph0, ph0_est, places=3)
