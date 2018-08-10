from nose import tools
import numpy as np
import matplotlib.mlab as mlab
from random_data.signals import RandomSignal
from random_data.spectra.parametric import (
    forward_prediction_error, backward_prediction_error, total_error_energy,
    next_order_a_pp, levinson_recursion, burg_coefficients,
    BurgAutoSpectralDensity)


def test_forward_prediction_error():
    # Should raise value error if `len(a) > len(x)`
    x = np.zeros(10)
    a = np.zeros(len(x) + 1)
    a[0] = 1
    tools.assert_raises(ValueError, forward_prediction_error, *[x, a])

    # Zero-order AR model should result in forward error equal
    # to the input signal
    x = np.arange(10.)
    a = np.array([1.0])
    np.testing.assert_equal(
        forward_prediction_error(x, a),
        x)

    # Pseudo 1st-order AR model with `a[0] = 1` and `a[1] = 0` should
    # give same forward error as zero-order AR model, but with the
    # length of error reduced by one
    x = np.arange(10.)
    a = np.array([1.0, 0.])
    np.testing.assert_equal(
        forward_prediction_error(x, a),
        x[1:])

    # True 1st-order AR model w/ real coefficients
    x = np.arange(10.)
    a = np.array([1.0, 0.5])
    error_expected = (a[0] * x[1:]) + (a[1] * x[:-1])
    np.testing.assert_equal(
        forward_prediction_error(x, a),
        error_expected)

    # True 1st-order AR model w/ complex coefficients
    x = np.arange(10.)
    a = np.array([1.0, 0.5 * (1 + 1j)])
    error_expected = (a[0] * x[1:]) + (a[1] * x[:-1])
    np.testing.assert_equal(
        forward_prediction_error(x, a),
        error_expected)

    return


def test_backward_prediction_error():
    # As `backward_prediction_error` simply calls `forward_prediction_error`,
    # most of the tests are already done in `test_forward_prediction_error`.
    # Thus, here, we only need to check behavior of 1st order AR models w/
    # real and complex coefficients

    # True 1st-order AR model w/ real coefficients
    x = np.arange(10.)
    a = np.array([1.0, 0.5])
    error_expected = (a[1] * x[1:]) + (a[0] * x[:-1])
    np.testing.assert_equal(
        backward_prediction_error(x, a),
        error_expected)

    # True 1st-order AR model w/ complex coefficients
    x = np.arange(10.)
    a = np.array([1.0, 0.5 * (1 + 1j)])
    error_expected = ((np.conj(a[1]) * x[1:]) + (np.conj(a[0]) * x[:-1]))
    np.testing.assert_equal(
        backward_prediction_error(x, a),
        error_expected)

    return


def test_total_error_energy():
    # `total_error_energy` is more or less a wrapper around
    # `forward_prediction_error` and `backward_prediction_error`, which
    # have been extensively tested above. Here, we just want to ensure
    # that the "wrapping" does indeed give the correct total error energy
    # under a simple test case.
    #
    # Zero-order AR model results in forward and backward errors that are
    # equal to the input signal, allowing an easy check for
    # `total_error_energy`.
    x = (1 - 1j) * np.arange(10.)
    a = np.array([1.0])
    np.testing.assert_equal(
        total_error_energy(x, a),
        2 * np.sum(np.abs(x) ** 2))

    return


def test_next_order_a_pp():
    # A ValueError should be raised if `len(a) >= len(x)`
    x = np.zeros(5)
    a = np.ones(len(x))
    tools.assert_raises(
        ValueError,
        next_order_a_pp,
        *[x, a])

    # Zero-order AR model produces forward and backwards errors
    # that are equal to the input signal. If the input signal's
    # entries are all equal, then we expect a_pp = -1.
    x = (1 + 1j) * np.ones(10)
    a = np.array([1.0])
    np.testing.assert_equal(
        next_order_a_pp(x, a),
        -1)

    # Analytic considerations constrain |a_{pp}| <= 1
    x = np.random.rand(100)
    a = np.random.rand(10)
    a[0] = 1
    tools.assert_less_equal(
        np.abs(next_order_a_pp(x, a)),
        1)

    return


def test_levinson_recursion():
    # For `anext_pp = -1` and uniform, real `a`, the Levinson recursion
    # should give all zeros
    a = np.ones(5)
    anext_pp = -1
    np.testing.assert_equal(
        levinson_recursion(a, anext_pp),
        np.zeros(len(a) - 1))

    # Test with complex, ramping `a`
    a = (1 + 1j) * np.arange(3)
    anext_pp = -1
    np.testing.assert_equal(
        levinson_recursion(a, anext_pp),
        np.array([-1 + 3j, 1 + 3j]))

    return


def test_burg_coefficients():
    # Ensure we are generating correct order
    x = np.random.rand(10) - 0.5
    p = 5
    a = burg_coefficients(x, p)
    tools.assert_equal(
        len(a),
        p + 1)

    return


def test_BurgAutoSpectralDensity():
    # Construct timebase and *real* signal (symmetric spectrum):
    # ==========================================================
    Fs = 1.
    t0 = 0
    T = 1e3
    t = np.arange(t0, T, 1. / Fs)

    f0 = 0.25
    A = 1.
    x = A * np.cos(2 * np.pi * f0 * t)

    # Check that ValueError is raised for orders equal to or
    # exceeding the length of the signal `x`:
    # --------------------------------------------------------
    tools.assert_raises(
        ValueError,
        BurgAutoSpectralDensity,
        *[len(x), x])

    # Compute two-sided autospectral density estimates:
    # -------------------------------------------------
    NFFT = 128
    noverlap = NFFT // 2
    asd_welch, f = mlab.psd(
        x, NFFT=NFFT, Fs=Fs,
        window=mlab.window_none,  # No windowing for sharpest resolution
        noverlap=noverlap,
        sides='twosided')

    # Order-2 AR autospectral-density estimate sufficient
    # for a single sinusoidal signal
    Nf = len(f)
    asd_burg = BurgAutoSpectralDensity(
        2, x, Fs=Fs, Nf=Nf, normalize=True)

    # Examine positive-frequency spectral components:
    # -----------------------------------------------
    sl = slice(Nf // 2, Nf)

    # Are peaks at same place?
    maxind_welch = np.where(asd_welch[sl] == np.max(asd_welch[sl]))[0]
    maxind_burg = np.where(asd_burg.Sxx[sl] == np.max(asd_burg.Sxx[sl]))[0]
    np.testing.assert_equal(maxind_burg, maxind_welch)

    # Do the peaks have approximately the same power?
    np.testing.assert_almost_equal(
        asd_burg.Sxx[maxind_burg],
        asd_welch[maxind_welch])

    # Examine negative-frequency spectral components:
    # -----------------------------------------------
    sl = slice(0, Nf // 2)

    # Are peaks at same place?
    maxind_welch = np.where(asd_welch[sl] == np.max(asd_welch[sl]))[0]
    maxind_burg = np.where(asd_burg.Sxx[sl] == np.max(asd_burg.Sxx[sl]))[0]
    np.testing.assert_equal(maxind_burg, maxind_welch)

    # Do the peaks have approximately the same power?
    np.testing.assert_almost_equal(
        asd_burg.Sxx[maxind_burg],
        asd_welch[maxind_welch])

    # Construct *complex* signal (asymmetric spectrum):
    # =================================================
    x = A * np.exp(1j * 2 * np.pi * f0 * t)

    # Compute two-sided autospectral density estimates:
    # -------------------------------------------------
    asd_welch, f = mlab.psd(
        x, NFFT=NFFT, Fs=Fs,
        window=mlab.window_none,  # No windowing for sharpest resolution
        noverlap=noverlap,
        sides='twosided')

    # Order-1 AR autospectral-density estimate sufficient
    # for a single complex exponential signal
    Nf = len(f)
    asd_burg = BurgAutoSpectralDensity(
        1, x, Fs=Fs, Nf=Nf, normalize=True)

    # Examine position and magnitude of peak:
    # ---------------------------------------
   # Are peaks at same place?
    maxind_welch = np.where(asd_welch == np.max(asd_welch))[0]
    maxind_burg = np.where(asd_burg.Sxx == np.max(asd_burg.Sxx))[0]
    np.testing.assert_equal(maxind_burg, maxind_welch)

    # Do the peaks have approximately the same power?
    np.testing.assert_almost_equal(
        asd_burg.Sxx[maxind_burg],
        asd_welch[maxind_welch])

    return


def test_BurgAutoSpectralDensity_windowing():
    # Check that raw vs. windowed signals give rise to roughly same PSD,
    # say within +/- 25%, due to reasons explained in Marple
    #
    # Could also have a similar test fail for Fourier spectra

    # Create random signal w/ lots of power at low frequencies such that
    # FFT-based spectral estimates will suffer from leakage if data blocks
    # are not smoothly tapered/windowed to zero at their edges:
    # --------------------------------------------------------------------
    # Sampling parameters
    Fs = 10.  # 1.
    t0 = 0
    T = 1e3

    # Random signal parameters
    f0_broad = 0.
    tau_broad = 4. / Fs
    G0 = 1.
    noise_floor = 1e-6
    seed = None

    # Coherent signal parameters
    f0 = 2.5  # 0.25
    A = 0.3

    # Generate random signal
    sig = RandomSignal(
        Fs=Fs, t0=t0, T=T,
        f0=f0_broad, tau=tau_broad, G0=G0,
        noise_floor=noise_floor, seed=seed)
    t = sig.t()
    x = sig.x.copy()

    # Add coherent signal
    x += (A * np.sin(2 * np.pi * f0 * t))
    x -= np.mean(x)

    # Windowing should "minimally" affect Burg autospectral-density estimate
    # (here, "minimally" means within a few tens of percent; when talking
    # about windowing FFT-based calculations, we're usually worried about
    # e.g. strong, low-frequency components of the signal contaminating
    # the higher-frequency spectral estimates, resulting in order-of-magnitude
    # errors in the spectral estimates):
    # -----------------------------------------------------------------------
    Nf = 128
    asd_burg = BurgAutoSpectralDensity(
        2, x, Fs=Fs, Nf=Nf, normalize=True)
    asd_burg_hanning = BurgAutoSpectralDensity(
        2, np.sqrt(8. / 3) * mlab.window_hanning(x),
        Fs=Fs, Nf=Nf, normalize=True)

    # A unity ratio implies that windowing has no effect on the spectral
    # estimate, with larger divergences from unity indicating that windowing
    # has a larger effect on the estimate. Variations about unity are
    # typically +/-15%. The below ensures that the raw and windowed estimates
    # differ by < 150% in all locations.
    burg_ratio = asd_burg_hanning.Sxx / asd_burg.Sxx
    np.testing.assert_almost_equal(
        burg_ratio,
        np.ones(len(burg_ratio)),
        decimal=0)

    # In contrast, windowing should significantly affect FFT-based
    # autospectral-density estimates:
    # ------------------------------------------------------------
    NFFT = Nf
    noverlap = NFFT // 2
    asd_welch, f = mlab.psd(
        x, NFFT=NFFT, Fs=Fs,
        window=mlab.window_none,
        noverlap=noverlap,
        sides='twosided')
    asd_welch_hanning, f = mlab.psd(
        x, NFFT=NFFT, Fs=Fs,
        window=mlab.window_hanning,
        noverlap=noverlap,
        sides='twosided')

    welch_ratio = asd_welch_hanning / asd_welch

    # Numpy does *not* have the logical converse of
    # `np.testing.assert_almost_equal`, so we have to
    # hack something together ourselves...
    #
    # If the below test passes, it shows that the raw and windowed
    # FFT spectral estimates differ significantly (by > 50% in one
    # or more locations, where the less-stringent constraint of 50%
    # results from application of triangle inequality to the logic in
    # `np.testing.assert_almost_equal`).
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_almost_equal,
        *[welch_ratio, np.ones(len(welch_ratio))],
        **{'decimal': 0})

    return
