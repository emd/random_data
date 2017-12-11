from nose import tools
import numpy as np
from matplotlib import mlab
from random_data.ensemble import (
    Ensemble, _largest_power_of_2_leq, closest_index)
from random_data.signals import RandomSignal


def test__largest_power_of_2_leq():
    x = 16

    # exact
    tools.assert_equal(x, _largest_power_of_2_leq(x))

    # should round down to `x`
    tools.assert_equal(x, _largest_power_of_2_leq(x + 1))

    # should round down to `x` / 2
    tools.assert_equal(x / 2, _largest_power_of_2_leq(x - 1))


def test_Ensemble_ValueError():
    x = np.random.randn(np.int(50e3))

    # `Nreal_per_ens` should be a positive integer
    tools.assert_raises(
        ValueError, Ensemble, x, **{'Nreal_per_ens': -1})
    tools.assert_raises(
        ValueError, Ensemble, x, **{'Nreal_per_ens': 1.})

    # `Npts_per_real` should be a positive integer
    tools.assert_raises(ValueError, Ensemble, x, **{'Npts_per_real': -1})
    tools.assert_raises(ValueError, Ensemble, x, **{'Npts_per_real': 1.})

    # ... but a valid specification of `Npts_per_real` should work
    Npts_per_real = 10
    tools.assert_equal(
        Ensemble(x, Npts_per_real=Npts_per_real).Npts_per_real,
        Npts_per_real)

    # `Npts_overlap` should be a positive integer, less than `Npts_per_real`
    tools.assert_raises(ValueError, Ensemble, x, **{'Npts_overlap': -1})
    tools.assert_raises(ValueError, Ensemble, x, **{'Npts_overlap': 1.})
    tools.assert_raises(ValueError, Ensemble,
                        x, **{'Npts_per_real': 100, 'Npts_overlap': 101})

    # ... but a valid choice of `Npts_overlap` should work
    Npts_per_real = 100
    Npts_overlap = Npts_per_real // 2
    tools.assert_equal(
        Ensemble(
            x, Npts_per_real=Npts_per_real,
            Npts_overlap=Npts_overlap).Npts_overlap,
        Npts_overlap)

    # Similarly, 0 <= `fraction_overlap` < 1
    tools.assert_raises(ValueError, Ensemble,
                        x, **{'fraction_overlap': -1})
    tools.assert_raises(ValueError, Ensemble,
                        x, **{'fraction_overlap': 1})

    # ... but a valid choice of `fraction_overlap` should work
    Npts_per_real = 100
    fraction_overlap = 0.5
    Npts_overlap = np.int(fraction_overlap * Npts_per_real)
    tools.assert_equal(
        Ensemble(
            x, Npts_per_real=Npts_per_real,
            fraction_overlap=fraction_overlap).Npts_overlap,
        Npts_overlap)


def test_Ensemble_getNumPtsPerReal():
    x = np.random.randn(np.int(50e3))

    # Create `Ensemble` object
    Fs = 1.0
    Tens = 2 ** 15  # 32768
    Nreal_per_ens = 1
    fraction_overlap = 0
    ens = Ensemble(
        x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        fraction_overlap=fraction_overlap)

    # With `Fs` = 1, the number of points in an ensemble is *equal* to `Tens`.
    # Further, with only a single realization in the ensemble, the number of
    # points in the realization should be equal to the number of points
    # in the whole ensemble, i.e. `Tens`.
    Nexp = np.int(Tens)
    tools.assert_equal(
        ens.getNumPtsPerReal(Fs, Tens, Nreal_per_ens, fraction_overlap), Nexp)

    # Increasing number of realizations by factor of 2 should decrease
    # number of points in realization by a factor of 2
    c = 2
    tools.assert_equal(
        ens.getNumPtsPerReal(Fs, Tens, c * Nreal_per_ens, fraction_overlap),
        Nexp / c)

    # ... and increasing sampling rate by factor of 2 should increase
    # the number of points in a realization by factor of 2
    c = 2
    tools.assert_equal(
        ens.getNumPtsPerReal(c * Fs, Tens, Nreal_per_ens, fraction_overlap),
        c * Nexp)

    # ... and if we have 3 realizations per ensemble with 50% overlap
    # between adjacent realizations, we expect the number of points
    # per realization to decrease by 2
    tools.assert_equal(
        ens.getNumPtsPerReal(Fs, Tens, 3, 0.5),
        Nexp // 2)


def test_Ensemble_getNumPtsPerEns():
    x = np.random.randn(np.int(50e3))

    # Create `Ensemble` object
    Fs = 1.0
    Tens = 2 ** 15  # 32768
    Npts_per_ens = np.int(Fs * Tens)

    # Test (1): With *single* realization per ensemble and *no* overlap
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=1, fraction_overlap=0)
    tools.assert_equal(ens.getNumPtsPerEns(), Npts_per_ens)

    # Test (2): With *multiple* realizations per ensemble and *no* overlap
    #
    # Note: Choose `Nreal_per_ens` = 2 below because return value of
    # `getNumPtsPerEns(...)` is influenced by `self.Npts_per_real`,
    # which is itself determined by `getNumPtsPerReal(...)`.
    # The routine for determining the number of points per realization
    # performs division by the factor
    #
    #    denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
    #
    # To avoid round-off from integer division effects,
    # `denominator` should be a power of 2. For `fraction_overlap` = 0,
    # this is readily accomplished with `Nreal_per_ens` = 2
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=2, fraction_overlap=0)
    tools.assert_equal(ens.getNumPtsPerEns(), Npts_per_ens)

    # Test (3): With *one* realization per ensemble and *overlap*
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=1, fraction_overlap=0.5)
    tools.assert_equal(ens.getNumPtsPerEns(), Npts_per_ens)

    # Test (4): With *multiple* realizations per ensemble and *overlap*
    #
    # Note: Choose `Nreal_per_ens` = 3 below because return value of
    # `getNumPtsPerEns(...)` is influenced by `self.Npts_per_real`,
    # which is itself determined by `getNumPtsPerReal(...)`.
    # The routine for determining the number of points per realization
    # performs division by the factor
    #
    #    denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
    #
    # To avoid round-off from integer division effects,
    # `denominator` should be a power of 2. For `fraction_overlap` = 0.5,
    # this is readily accomplished with `Nreal_per_ens` = 3
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=3, fraction_overlap=0.5)
    tools.assert_equal(ens.getNumPtsPerEns(), Npts_per_ens)


def test_Ensemble_getFrequencies():
    x = np.random.randn(100)

    # Test (1)
    # --------
    # For `Fs` = 1 and one realization per ensemble, the number
    # of points per realization is simply `Tens`. The number of points
    # per realization directly determines the frequency resolution
    # of the spectral density estimate.
    Fs = 1.0
    Nreal_per_ens = 1
    Tens = 16.
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    fexp = np.arange((Tens // 2) + 1) / Tens
    np.testing.assert_equal(ens.f, fexp)

    # Test (2)
    # --------
    # If we have `c` realizations per ensemble, the frequency resolution
    # should decrease by a factor of `c`
    c = 2
    ens = Ensemble(x, Fs=Fs, Tens=Tens, Nreal_per_ens=(c * Nreal_per_ens))
    np.testing.assert_equal(ens.f, fexp[::c])

    # Test (3)
    # --------
    # ... and if we increase `Fs` by `c` and decrease `Tens` by `c`
    # (we decrease `Tens` by `c` to ensure we have the same number
    # of points per realization and ensemble as in (1)),
    # the expected frequencies will increase by a factor `c`
    # relative to those in (1)
    ens = Ensemble(x, Fs=(c * Fs), Tens=(Tens / float(c)),
                   Nreal_per_ens=Nreal_per_ens)
    np.testing.assert_equal(ens.f, c * fexp)


def test_Ensemble_getFFTs():
    # Construct a random signal
    Fs = 4e6            # [Fs] = samples / s
    t0 = 0              # [t0] = s
    T = 100e-3          # [T] = s
    fc = 200e3          # [fc] = Hz
    pole = 2
    sig = RandomSignal(Fs, t0, T, fc=fc, pole=pole)

    # Create ensemble object
    Tens = 5e-3         # [Tens] = s
    Nreal_per_ens = 100
    ens = Ensemble(
        sig.x, Fs=sig.Fs, t0=sig.t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # FFT procedure's should raise ValueError if signal is complex
    detrend = mlab.detrend_none
    window = mlab.window_hanning
    tools.assert_raises(
        ValueError,
        ens.getFFTs,
        sig.x.astype('complex'),
        **{'detrend': detrend, 'window': window})

    # Compute one-sided autospectral density from FFTs
    Xk = ens.getFFTs(sig.x)
    XkstarXk = np.real(np.conj(Xk) * Xk)
    EXkstarXk = np.mean(np.mean(XkstarXk, axis=-1), axis=-1)
    C0 = 2 / (ens.Npts_per_real * ens.Fs)
    Gxx = C0 * EXkstarXk

    # Compute one-sided autospectral density from mlab functions
    # using same spectral-estimation parameters as above
    Gxx_mlab, f_mlab = mlab.psd(
        sig.x, NFFT=ens.Npts_per_real, Fs=ens.Fs,
        detrend=detrend, window=window)

    # Test that the two spectral estimates are equal
    # (within numerical precision; also, neglect off-by-two
    # discrepancy at DC and Nyquist frequencies)
    sl = slice(1, -1)
    rtol = 1. / np.sqrt(Nreal_per_ens)
    np.testing.assert_allclose(Gxx_mlab[sl], Gxx[sl], rtol=rtol)
    np.testing.assert_allclose(f_mlab, ens.f)

    return


def test_Ensemble_getTimes():
    t0 = 0
    Fs = 1.

    # `len(x)` and `Tens` both a power of 2 to avoid round-off errors
    # with methods defined in `Ensemble` class
    x = np.random.randn(2 ** 8)
    Tens = 2 ** 6

    # No overlap and only one realization per ensemble so that the
    # midpoint of each ensemble can be easily computed
    ens = Ensemble(x, Fs=Fs, t0=t0, Tens=Tens,
                   Nreal_per_ens=1, fraction_overlap=0)

    # Construct expected time base
    Nens_exp = len(x) / Tens
    Texp = t0 + (Tens * np.arange(0.5, Nens_exp, 1))

    np.testing.assert_equal(ens.t, Texp)


def test_closest_index():
    N = 10
    half_N = N // 2
    v = np.arange(N)

    # Check that works as expected with rounding
    tools.assert_equal(closest_index(v, half_N), half_N)
    tools.assert_equal(closest_index(v, half_N + 0.1), half_N)
    tools.assert_equal(closest_index(v, half_N + 0.6), half_N + 1)

    # Check that works as expected against boundary cases
    tools.assert_equal(closest_index(v, -1), 0)
    tools.assert_equal(closest_index(v, N), N - 1)
