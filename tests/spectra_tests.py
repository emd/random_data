from nose import tools
import numpy as np
from random_data.spectra import SpectralDensity, _closest_power_of_2


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

    # Complex signals should fail
    xc = x.astype('complex128')
    yc = y.astype('complex128')
    tools.assert_raises(ValueError, SpectralDensity, xc)
    tools.assert_raises(ValueError, SpectralDensity, xc, {'y': y})
    tools.assert_raises(ValueError, SpectralDensity, x, {'y': yc})
    tools.assert_raises(ValueError, SpectralDensity, xc, {'y': yc})


def test_SpectralDensity_Npts_input():
    x = np.random.randn(50e3)

    # `Nreal_per_ens` should be a positive integer
    tools.assert_raises(ValueError, SpectralDensity, x, {'Nreal_per_ens': -1})
    tools.assert_raises(ValueError, SpectralDensity, x, {'Nreal_per_ens': 1.})

    # `Npts_per_real` should be a positive integer
    tools.assert_raises(ValueError, SpectralDensity, x, {'Npts_per_real': -1})
    tools.assert_raises(ValueError, SpectralDensity, x, {'Npts_per_real': 1.})

    # ... but a valid specification of `Npts_per_real` should work
    Npts_per_real = 10
    tools.assert_equal(
        SpectralDensity(x, Npts_per_real=Npts_per_real).Npts_per_real,
        Npts_per_real)

    # `Npts_overlap` should be a positive integer, less than `Npts_per_real`
    tools.assert_raises(ValueError, SpectralDensity, x, {'Npts_overlap': -1})
    tools.assert_raises(ValueError, SpectralDensity, x, {'Npts_overlap': 1.})
    tools.assert_raises(ValueError, SpectralDensity,
                        x, {'Npts_per_real': 100, 'Npts_overlap': 101})

    # ... but a valid choice of `Npts_overlap` should work
    Npts_per_real = 100
    Npts_overlap = Npts_per_real // 2
    tools.assert_equal(
        SpectralDensity(
            x, Npts_per_real=Npts_per_real,
            Npts_overlap=Npts_overlap).Npts_overlap,
        Npts_overlap)

    # Similarly, 0 <= `fraction_overlap` < 1
    tools.assert_raises(ValueError, SpectralDensity,
                        x, {'fraction_overlap': -1})
    tools.assert_raises(ValueError, SpectralDensity,
                        x, {'fraction_overlap': 1})

    # ... but a valid choice of `fraction_overlap` should work
    Npts_per_real = 100
    fraction_overlap = 0.5
    Npts_overlap = np.int(fraction_overlap * Npts_per_real)
    tools.assert_equal(
        SpectralDensity(
            x, Npts_per_real=Npts_per_real,
            fraction_overlap=fraction_overlap).Npts_overlap,
        Npts_overlap)


def test_SpectralDensity__getNumPtsPerReal():
    x = np.random.randn(50e3)

    # Create `SpectralDensity` object
    Fs = 1.0
    Tens = 2 ** 15  # 32768
    Nreal_per_ens = 1
    fraction_overlap = 0
    sd = SpectralDensity(
        x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens,
        fraction_overlap=fraction_overlap)

    # With `Fs` = 1, the number of points in an ensemble is *equal* to `Tens`.
    # Further, with only a single realization in the ensemble, the number of
    # points in the realization should be equal to the number of points
    # in the whole ensemble, i.e. `Tens`.
    Nexp = np.int(Tens)
    tools.assert_equal(
        sd._getNumPtsPerReal(Fs, Tens, Nreal_per_ens, fraction_overlap), Nexp)

    # Increasing number of realizations by factor of 2 should decrease
    # number of points in realization by a factor of 2
    c = 2
    tools.assert_equal(
        sd._getNumPtsPerReal(Fs, Tens, c * Nreal_per_ens, fraction_overlap),
        Nexp / c)

    # ... and increasing sampling rate by factor of 2 should increase
    # the number of points in a realization by factor of 2
    c = 2
    tools.assert_equal(
        sd._getNumPtsPerReal(c * Fs, Tens, Nreal_per_ens, fraction_overlap),
        c * Nexp)

    # ... and if we have 3 realizations per ensemble with 50% overlap
    # between adjacent realizations, we expect the number of points
    # per realization to decrease by 2
    tools.assert_equal(
        sd._getNumPtsPerReal(Fs, Tens, 3, 0.5),
        Nexp // 2)


def test_SpectralDensity__getNumPtsPerEns():
    x = np.random.randn(50e3)

    # Create `SpectralDensity` object
    Fs = 1.0
    Tens = 2 ** 15  # 32768
    Npts_per_ens = np.int(Fs * Tens)

    # Test (1): With *single* realization per ensemble and *no* overlap
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=1, fraction_overlap=0)
    tools.assert_equal(sd._getNumPtsPerEns(), Npts_per_ens)

    # Test (2): With *multiple* realizations per ensemble and *no* overlap
    #
    # Note: Choose `Nreal_per_ens` = 2 below because return value of
    # `_getNumPtsPerEns(...)` is influenced by `self.Npts_per_real`,
    # which is itself determined by `_getNumPtsPerReal(...)`.
    # The routine for determining the number of points per realization
    # performs division by the factor
    #
    #    denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
    #
    # To avoid round-off from integer division effects,
    # `denominator` should be a power of 2. For `fraction_overlap` = 0,
    # this is readily accomplished with `Nreal_per_ens` = 2
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=2, fraction_overlap=0)
    tools.assert_equal(sd._getNumPtsPerEns(), Npts_per_ens)

    # Test (3): With *one* realization per ensemble and *overlap*
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=1, fraction_overlap=0.5)
    tools.assert_equal(sd._getNumPtsPerEns(), Npts_per_ens)

    # Test (4): With *multiple* realizations per ensemble and *overlap*
    #
    # Note: Choose `Nreal_per_ens` = 3 below because return value of
    # `_getNumPtsPerEns(...)` is influenced by `self.Npts_per_real`,
    # which is itself determined by `_getNumPtsPerReal(...)`.
    # The routine for determining the number of points per realization
    # performs division by the factor
    #
    #    denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
    #
    # To avoid round-off from integer division effects,
    # `denominator` should be a power of 2. For `fraction_overlap` = 0.5,
    # this is readily accomplished with `Nreal_per_ens` = 3
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=3, fraction_overlap=0.5)
    tools.assert_equal(sd._getNumPtsPerEns(), Npts_per_ens)


def test_SpectralDensity_getFrequencies():
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
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    fexp = np.arange((Tens // 2) + 1) / Tens
    np.testing.assert_equal(sd.f, fexp)

    # Test (2)
    # --------
    # If we have `c` realizations per ensemble, the frequency resolution
    # should decrease by a factor of `c`
    c = 2
    sd = SpectralDensity(x, Fs=Fs, Tens=Tens,
                         Nreal_per_ens=(c * Nreal_per_ens))
    np.testing.assert_equal(sd.f, fexp[::c])

    # Test (3)
    # --------
    # ... and if we increase `Fs` by `c` an decrease `Tens` by `c`
    # (we decrease `Tens` by `c` to ensure we have the same number
    # of points per realization and ensemble as in (1)),
    # the expected frequencies will increase by a factor `c`
    # relative to those in (1)
    sd = SpectralDensity(x, Fs=(c * Fs), Tens=(Tens / float(c)),
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


def test_SpectralDensity_getPhaseAngle():
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
    csd = SpectralDensity(x, y, Fs=Fs, Tens=Tens, Nreal_per_ens=10)
    csd.getPhaseAngle()

    f_ind = np.where(np.abs(csd.f - f0) == np.min(np.abs(csd.f - f0)))[0]
    ph0_est = np.mean(csd.theta_xy[f_ind, :], axis=-1)

    tools.assert_almost_equal(ph0, ph0_est, places=3)


def test__closest_power_of_2():
    x = 16

    # exact
    tools.assert_equal(x, _closest_power_of_2(x))

    # should round up to `x`
    tools.assert_equal(x, _closest_power_of_2(x - 1))

    # should round down to `x`
    tools.assert_equal(x, _closest_power_of_2(x + 1))
