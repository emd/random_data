from nose import tools
import numpy as np
from random_data.spectra.nonparametric import (
    AutoSpectralDensity, CrossSpectralDensity,
    wrap, phase_angle_bins, _next_largest_divisor_for_integer_quotient)


def test_AutoSpectralDensity_signal_input():
    # Complex signal should fail
    x = np.random.randn(np.int(50e3))
    xc = x.astype('complex128')
    tools.assert_raises(ValueError, AutoSpectralDensity, xc)

    return


def test_CrossSpectralDensity_signal_input():
    # Different length signals should fail
    x = np.random.randn(np.int(50e3))
    y = np.random.randn(np.int(50e3) + 1)
    tools.assert_raises(ValueError, CrossSpectralDensity, x, y)

    # Complex signals should fail
    xc = x.astype('complex128')
    yc = y.astype('complex128')
    tools.assert_raises(ValueError, CrossSpectralDensity, xc, y)
    tools.assert_raises(ValueError, CrossSpectralDensity, x, yc)
    tools.assert_raises(ValueError, CrossSpectralDensity, xc, yc)

    return


def test_AutoSpectralDensity_white_noise():
    # White noise of a given power
    noise_power = np.sqrt(2)
    x = np.sqrt(noise_power) * np.random.randn(np.int(1e6))

    # Compute autospectral density of `x`
    asd = AutoSpectralDensity(x)

    # Average over time, then integrate over frequency
    noise_power_estimate = np.sum(np.mean(asd.Gxx, axis=-1)) * asd.df

    tools.assert_almost_equal(noise_power, noise_power_estimate, places=1)

    return


def test_CrossSpectralDensity_vs_AutoSpectralDensity():
    # "Detrending" can really fuck things up if applied incorrectly, so
    # its generally best to just avoid it
    detrend = None

    # White noise of a given power
    noise_power = np.sqrt(2)
    x = np.sqrt(noise_power) * np.random.randn(np.int(1e6))

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

    return


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

    return


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

    return


def test_wrap():
    # Test (1):
    # ---------
    # Use standard limits such that "wrapped" angles lie between [-pi, pi)
    theta_min = -np.pi
    theta_max = np.pi

    # Angles between [-2 * pi , 2 * pi] in 0.5 * pi increments
    theta = np.pi * np.array([
        -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0])

    # ... which should be wrapped onto these values
    theta_expected = np.pi * np.array([
        0.0, 0.5, -1.0, -0.5, 0.0,
        0.5, -1.0, -0.5, 0.0])

    np.testing.assert_allclose(
        theta_expected,
        wrap(theta, theta_min, theta_max),
        atol=(10 * np.finfo('float64').eps))

    # Test (2):
    # ---------
    # Rotate wrapped region by pi / 4
    theta_min += np.pi / 4
    theta_max += np.pi / 4

    # New expected values after wrapping
    theta_expected = np.pi * np.array([
        0.0, 0.5, 1.0, -0.5, 0.0,
        0.5, 1.0, -0.5, 0.0])

    np.testing.assert_allclose(
        theta_expected,
        wrap(theta, theta_min, theta_max),
        atol=(10 * np.finfo('float64').eps))

    return


def test__next_largest_divisor_for_integer_quotient():
    # Specified `divisor` yields an integer when dividing into `dividend`
    dividend = 10
    divisor = 2
    tools.assert_equal(
        divisor,
        _next_largest_divisor_for_integer_quotient(dividend, divisor))

    # Specified `divisor` does *not* yield  an integer when dividing
    # into `dividend`
    dividend = 10
    divisor = 4
    tools.assert_equal(
        5,
        _next_largest_divisor_for_integer_quotient(dividend, divisor))

    # 45 degree toroidal spacing of DIII-D's interferometers
    dividend = 2 * np.pi
    divisor = np.pi / 4
    tools.assert_equal(
        divisor,
        _next_largest_divisor_for_integer_quotient(dividend, divisor))

    # 33 degree toroidal spacing of DIII-D's "typical" magnetic probes
    dividend = 2 * np.pi
    divisor = 33 * (np.pi / 180)
    tools.assert_equal(
        36 * (np.pi / 180),
        _next_largest_divisor_for_integer_quotient(dividend, divisor))

    return


def test_phase_angle_bins():
    # Use typical `theta_min` of -pi and 0; also, test with "wild" offsets
    theta_min_test_vals = np.array([
        -np.pi,
        0,
        -10 * np.pi,
        np.pi / 2
    ])

    # (1) Test with *even* number of bins when
    #     `dtheta0` divides (2 * pi) into integer number of bins:
    # ===========================================================
    dtheta0 = np.pi / 2

    for theta_min in theta_min_test_vals:
        theta_max = theta_min + (2 * np.pi)
        bins_expected = np.arange(theta_min, theta_max, dtheta0)

        bins, dtheta = phase_angle_bins(dtheta0, theta_min)

        # Use "almost equal" to avoid errors w/ round-off errors
        np.testing.assert_array_almost_equal(bins, bins_expected)
        tools.assert_equal(dtheta, dtheta0)

    # (2) Test with *even* number of bins when
    #     `dtheta0` does *not* divide (2 * pi) into integer number of bins:
    # =====================================================================
    dtheta_expected = np.pi / 2
    dtheta0 = 0.99 * dtheta_expected

    for theta_min in theta_min_test_vals:
        theta_max = theta_min + (2 * np.pi)
        bins_expected = np.arange(theta_min, theta_max, dtheta_expected)

        bins, dtheta = phase_angle_bins(dtheta0, theta_min)

        # Use "almost equal" to avoid errors w/ round-off errors
        np.testing.assert_array_almost_equal(bins, bins_expected)
        tools.assert_equal(dtheta, dtheta_expected)

    # (3) Test with *odd* number of bins when
    #     `dtheta0` divides (2 * pi) into integer number of bins
    #     (note that the odd case requires a bit more care...):
    # ==========================================================
    dtheta0 = (2 * np.pi) / 5

    # (3a) [-pi, pi):
    # ---------------
    theta_min = -np.pi
    theta_max = theta_min + (2 * np.pi)
    bins_expected = dtheta0 * np.arange(-2, 3)

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta0)

    # (3b) [0, 2 * pi):
    # -----------------
    theta_min = 0
    theta_max = theta_min + (2 * np.pi)
    bins_expected = dtheta0 * np.arange(0, 5)

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta0)

    # (3c) [-10 * pi, -8 * pi):
    # -------------------------
    theta_min = -10 * np.pi  # equivalent to 0 radians
    theta_max = theta_min + (2 * np.pi)
    bins_expected = theta_min + (dtheta0 * np.arange(0, 5))

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta0)

    # (3d) [0.4 * pi, 2.4 * pi):
    # --------------------------
    theta_min = 0.4 * np.pi
    theta_max = theta_min + (2 * np.pi)
    bins_expected = theta_min + (dtheta0 * np.arange(0, 5))

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta0)

    # (4) Test with *odd* number of bins when
    #     `dtheta0` does *not* divide (2 * pi) into integer number of bins
    #     (note that the odd case requires a bit more care...):
    # =====================================================================
    dtheta_expected = (2 * np.pi) / 5
    dtheta0 = 0.99 * dtheta_expected

    # (4a) [-pi, pi):
    # ---------------
    theta_min = -np.pi
    theta_max = theta_min + (2 * np.pi)
    bins_expected = dtheta_expected * np.arange(-2, 3)

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta_expected)

    # (4b) [0, 2 * pi):
    # -----------------
    theta_min = 0
    theta_max = theta_min + (2 * np.pi)
    bins_expected = dtheta_expected * np.arange(0, 5)

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta_expected)

    # (4c) [-10 * pi, -8 * pi):
    # -------------------------
    theta_min = -10 * np.pi  # equivalent to 0 radians
    theta_max = theta_min + (2 * np.pi)
    bins_expected = theta_min + (dtheta_expected * np.arange(0, 5))

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta_expected)

    # (4d) [0.4 * pi, 2.4 * pi):
    # --------------------------
    theta_min = 0.4 * np.pi
    theta_max = theta_min + (2 * np.pi)
    bins_expected = theta_min + (dtheta_expected * np.arange(0, 5))

    bins, dtheta = phase_angle_bins(dtheta0, theta_min)

    # Use "almost equal" to avoid errors w/ round-off errors
    np.testing.assert_array_almost_equal(bins, bins_expected)
    tools.assert_equal(dtheta, dtheta_expected)

    return
