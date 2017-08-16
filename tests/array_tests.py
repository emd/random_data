from nose import tools
import numpy as np
from random_data.spectra import CrossSpectralDensity
from random_data.array import (
    ArrayStencil, CrossSpectralDensityArray,
    FittedCrossPhaseArray, coefficient_of_determination,
    _find_first_nan, _get_timebase_indices)
from random_data.ensemble import closest_index


def test_ArrayStencil_getUniqueCorrelationPairs():
    # No unique correlation pairs
    stencil = ArrayStencil([1], include_autocorrelations=False)
    tools.assert_equal(0, len(stencil.separation))
    tools.assert_equal(0, len(stencil.xind))
    tools.assert_equal(0, len(stencil.yind))

    # Single unique (auto)correlation pair
    stencil = ArrayStencil([1], include_autocorrelations=True)
    tools.assert_equal(0, stencil.separation[0])
    tools.assert_equal(0, stencil.xind[0])
    tools.assert_equal(0, stencil.yind[0])

    # Uniform grid *without* autocorrelation
    stencil = ArrayStencil([1, 2, 3], include_autocorrelations=False)
    np.testing.assert_equal(np.array([1, 1, 2]), stencil.separation)
    np.testing.assert_equal(np.array([0, 1, 0]), stencil.xind)
    np.testing.assert_equal(np.array([1, 2, 2]), stencil.yind)

    # Uniform grid *with* autocorrelation
    stencil = ArrayStencil([1, 2, 3], include_autocorrelations=True)
    np.testing.assert_equal(np.array([0, 0, 0, 1, 1, 2]), stencil.separation)
    np.testing.assert_equal(np.array([0, 1, 2, 0, 1, 0]), stencil.xind)
    np.testing.assert_equal(np.array([0, 1, 2, 1, 2, 2]), stencil.yind)

    # Uniform, non-monotonic grid *without* autocorrelation
    # (a bit less intuitive than a monotonic grid, which is one
    # reason to avoid non-monotonicity; but should still work)
    stencil = ArrayStencil([2, 3, 1], include_autocorrelations=False)
    np.testing.assert_equal(np.array([-2, -1, 1]), stencil.separation)
    np.testing.assert_equal(np.array([1, 0, 0]), stencil.xind)
    np.testing.assert_equal(np.array([2, 2, 1]), stencil.yind)

    # Uniform, non-monotonic grid *with* autocorrelation
    # (a bit less intuitive than a monotonic grid, which is one
    # reason to avoid non-monotonicity; but should still work)
    stencil = ArrayStencil([2, 3, 1], include_autocorrelations=True)
    np.testing.assert_equal(np.array([-2, -1, 0, 0, 0, 1]), stencil.separation)
    np.testing.assert_equal(np.array([1, 0, 0, 1, 2, 0]), stencil.xind)
    np.testing.assert_equal(np.array([2, 2, 0, 1, 2, 1]), stencil.yind)

    # Negative, non-integer, and/or non-uniform grids should *not*
    # provide any novel testing relative to that performed above, so
    # we can stop here :)

    return


def test_ArrayStencil_getSeparationGCD():
    # No unique correlation pairs
    stencil = ArrayStencil([1], include_autocorrelations=False)
    tools.assert_equal(0, stencil.separation_gcd)

    # Single unique (auto)correlation pair
    stencil = ArrayStencil([1], include_autocorrelations=True)
    tools.assert_equal(0, stencil.separation_gcd)

    # Uniform grid
    stencil = ArrayStencil([0, 2, 4])
    tools.assert_equal(2, stencil.separation_gcd)

    # Non-uniform grid
    stencil = ArrayStencil([0, 4, 12])
    tools.assert_equal(4, stencil.separation_gcd)

    # Non-monotonic grid with "negative" separation values
    stencil = ArrayStencil([2, 3, 1], include_autocorrelations=False)
    tools.assert_equal(1, stencil.separation_gcd)

    return


def test_ArrayStencil_getMask():
    # Uniform grid
    stencil = ArrayStencil([1, 2, 3])
    np.testing.assert_equal(stencil.getMask(), [1, 1, 1])

    # Non-uniform grid
    stencil = ArrayStencil([0.5, 2, 3])
    np.testing.assert_equal(stencil.getMask(), [1, 0, 0, 1, 0, 1])

    # Non-monotonic, non-uniform grid
    stencil = ArrayStencil([2, 3, 0.5])
    np.testing.assert_equal(stencil.getMask(), [1, 0, 0, 1, 0, 1])

    return


def test_ArrayStencil_getMaskGapSizes():
    # Uniform grid
    stencil = ArrayStencil([0, 1, 2])
    np.testing.assert_equal(stencil.getMaskGapSizes(), [0, 0, 0])

    # Non-uniform grid with a gap size of unity
    stencil = ArrayStencil([0, 2, 3])
    np.testing.assert_equal(stencil.getMaskGapSizes(), [0, 1, 0, 0])

    # Non-monotonic, non-uniform grid with gap size of unity
    stencil = ArrayStencil([2, 3, 0])
    np.testing.assert_equal(stencil.getMaskGapSizes(), [0, 1, 0, 0])

    # Non-uniform grid with various gap sizes
    stencil = ArrayStencil([0, 2, 3, 4, 7, 8, 12, 14])

    #        locations = [0, x, 2, 3, 4, x, x, 7, 8, x, x, x, 12, x, 14]
    #             mask = [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0,  1, 0,  1]
    expected_gap_sizes = [0, 1, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3,  0, 1,  0]

    np.testing.assert_equal(
        stencil.getMaskGapSizes(),
        expected_gap_sizes)

    return


def test_ArrayStencil_getUniqueSeparation():
    # Uniform grid
    stencil = ArrayStencil([1, 2, 3])
    np.testing.assert_equal(
        stencil.getUniqueSeparation(),
        np.array([0, 1, 2]))

    # Non-uniform grid
    stencil = ArrayStencil([0.5, 2, 3])
    np.testing.assert_equal(
        stencil.getUniqueSeparation(),
        [0.0, 1.0, 1.5, 2.5])

    # Non-monotonic, non-uniform grid
    stencil = ArrayStencil([2, 3, 0.5])
    np.testing.assert_equal(
        stencil.getUniqueSeparation(),
        [-2.5, -1.5, 0.0, 1.0])

    return


def test_ArrayStencil_getCrossCorrelation_SineWave():
    # Sine wave properties
    f0 = 0.125
    A = 2.

    def check_error(y, stencil, A, f0):
        xcorr = stencil.getCrossCorrelation(y)

        # The expected form of the cross correlation is from
        # Bendat & Piersol's "Random Data", 4th ed., pg. 124,
        # Table 5.1 -- Special Autocorrelation functions.
        xcorr_exp = np.cos(2 * np.pi * f0 * stencil.unique_separation)
        xcorr_exp *= 0.5 * (A ** 2)

        # Compute absolute error
        delta = np.abs(xcorr - xcorr_exp)

        # According to Bendat & Piersol's "Random Data", 4th ed., pg. 285,
        # Table 8.1 -- Record lengths and Averages for Basic Estimates,
        # the random error in the autocorrelation scales ~1 / sqrt(N),
        # where N is the number of realizations.
        N = np.zeros(len(stencil.unique_separation))

        for sind, separation in enumerate(stencil.unique_separation):
            N[sind] = len(np.where(stencil.separation == separation)[0])

        # 3 is a "fudge factor" -- this is more of a semi-quantitative
        # test rather than a rigorous quantitative test
        max_error = 5. / np.sqrt(N)

        # The errors at the very end of the computational domain tend
        # to be larger than `max_error`, so restrict ourselves to, say,
        # 90% of computational domain
        sl = slice(0, np.int(0.9 * len(xcorr)))

        np.testing.assert_array_less(
            delta[sl],
            max_error[sl])

        return

    # Uniform timebase:
    # -----------------
    # Of course, with a large uniform grid, it is much faster
    # to perform this calculation via the FFT, but we can
    # check this bounding case to ensure proper performance.
    #
    Fs = 1.
    t0 = 0.
    T = (20. / f0) - t0
    t = np.arange(t0, T, 1. / Fs)
    stencil = ArrayStencil(t)
    y = A * np.sin(2 * np.pi * f0 * t)
    check_error(y, stencil, A, f0)

    # Non-uniform timebase:
    # ---------------------
    # Randomly sample from `t`
    tr = t[np.random.rand(len(t)) > 0.5]
    stencilr = ArrayStencil(tr)
    yr = A * np.sin(2 * np.pi * f0 * tr)
    check_error(yr, stencilr, A, f0)

    return


def test_ArrayStencil_getAverageForEachSeparation():
    # Non-uniform stencil with non-uniform spacing of unique separations
    stencil = ArrayStencil([0, 1, 5], include_autocorrelations=True)
    Nsep = len(stencil.separation)

    # Real-valued (integer) array:
    # ----------------------------
    A = np.arange(Nsep)

    # Compute average of `A` for each separation
    uniform_separation, A_avg = stencil.getAverageForEachSeparation(A)

    # Expected values
    uniform_separation_exp = np.array([0, 1, 2, 3, 4, 5])
    A_avg_exp = np.array([
        1.,
        3.,
        np.nan,
        np.nan,
        4.,
        5.])

    np.testing.assert_equal(
        uniform_separation,
        uniform_separation_exp)

    np.testing.assert_equal(
        A_avg,
        A_avg_exp)

    # Complex-valued array:
    # ---------------------
    A = (1 + 1j) * np.arange(Nsep)

    # Compute average of `A` for each separation
    uniform_separation, A_avg = stencil.getAverageForEachSeparation(A)

    # Expected values
    uniform_separation_exp = np.array([0, 1, 2, 3, 4, 5])
    A_avg_exp = np.array([
        1. + 1.j,
        3. + 3.j,
        np.nan + (1j * np.nan),
        np.nan + (1j * np.nan),
        4. + 4.j,
        5. + 5.j])

    np.testing.assert_equal(
        uniform_separation,
        uniform_separation_exp)

    np.testing.assert_equal(
        A_avg,
        A_avg_exp)

    return


def test_CrossSpectralDensityArray_getSpectralDensities():
    # Sampling properties
    Fs = 200e3
    t0 = 0
    t = np.arange(0, 1e-3, 1. / Fs)
    locations = np.array([0, 1, 3])

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
        y1, y2, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd13 = CrossSpectralDensity(
        y1, y3, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd23 = CrossSpectralDensity(
        y2, y3, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Create an array object and compute corresponding
    # cross-spectral densities
    A = CrossSpectralDensityArray(
        signals, locations, include_autocorrelations=False,
        Fs=Fs, t0=t0, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Ensure cross-spectral densities in array object `A`
    # are indeed the cross-spectral densities they are supposed to be.
    # Note that the cross-spectral density objects are sorted in `A`
    # by increasing spatial separation between measurement locations.
    np.testing.assert_equal(csd12.Gxy, A.Gxy[0])  # smallest separation
    np.testing.assert_equal(csd23.Gxy, A.Gxy[1])  # middle separation
    np.testing.assert_equal(csd13.Gxy, A.Gxy[2])  # largest separation

    # Individually compute autospectral densities, but force
    # computation of complex-valued spectral density
    csd11 = CrossSpectralDensity(
        y1, y1.copy(), Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd22 = CrossSpectralDensity(
        y2, y2.copy(), Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)
    csd33 = CrossSpectralDensity(
        y3, y3, Fs=Fs, t0=t0,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Create an array object and compute corresponding
    # cross-spectral densities, including autocorrelations
    A = CrossSpectralDensityArray(
        signals, locations, include_autocorrelations=True,
        Fs=Fs, t0=t0, Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Ensure cross-spectral densities in array object `A`
    # are indeed the cross-spectral densities they are supposed to be.
    # Note that the cross-spectral density objects are sorted in `A`
    # by increasing spatial separation between measurement locations.
    np.testing.assert_equal(csd11.Gxy, A.Gxy[0])  # smallest separation
    np.testing.assert_equal(csd22.Gxy, A.Gxy[1])
    np.testing.assert_equal(csd33.Gxy, A.Gxy[2])
    np.testing.assert_equal(csd12.Gxy, A.Gxy[3])
    np.testing.assert_equal(csd23.Gxy, A.Gxy[4])
    np.testing.assert_equal(csd13.Gxy, A.Gxy[5])  # largest separation

    return


def test_coefficient_of_determination():
    tools.assert_equal(1, coefficient_of_determination(0., 1.))
    tools.assert_equal(0, coefficient_of_determination(1., 1.))

    return


def test_FittedCrossPhaseArray_fitPhaseAngles():
    # Measurement locations
    locations = np.arange(0, 2 * np.pi)
    Nsig = len(locations)

    # Signal parameters
    Fs = 200e3
    t0 = 0.
    tf = 0.1
    t = np.arange(t0, tf, 1. / Fs)
    Npts = len(t)
    f0 = 50e3

    # Spectral estimation parameters
    Tens = 5e-3
    Nreal_per_ens = 10

    # Initialize
    signals = np.zeros((Nsig, Npts))

    # True mode number
    # (For uniform spacing of `dzeta` radian between measurement locations,
    # the Nyquist mode number is floor(pi / dzeta). For dzeta = 1 radian,
    # then, the Nyquist mode number is 3).
    n_array = np.array([0, -1, 2, -3])

    for n in n_array:
        # Signal generation: use purely coherent modes because
        # if mode-number extraction does not work in this ideal case,
        # it certainly won't work in the presence of noise
        for i in np.arange(Nsig):
            signals[i, :] = np.cos(
                (2 * np.pi * f0 * t) + (n * (locations[i] - locations[0])))

        # Perform fit
        A = FittedCrossPhaseArray(
            signals, locations, Fs=Fs,
            Tens=Tens, Nreal_per_ens=Nreal_per_ens)

        # Determine index corresponding to coherent frequency, `f0`
        find = closest_index(A.f, f0)

        # Compare fitted mode number to true mode number
        np.testing.assert_allclose(n, A.mode_number[find, :], atol=0.0)

    return


def test__find_first_nan():
    # Must be array of floats
    a = np.arange(10.)

    # Test case with no `np.nan` in `a`
    tools.assert_equal(
        _find_first_nan(a),
        None)

    # Insert `np.nan` into `a`
    ind = [3, 5, 7]
    a[ind] = np.nan

    tools.assert_equal(
        _find_first_nan(a),
        ind[0])

    return


def test__get_timebase_indices():
    # (a) timebase: 0 <= t <= 9 w/ dt = 1:
    # ------------------------------------
    Fs = 1.
    t0 = 0.
    Npts = 10

    # `tlim` specifies a subset of timebase
    tlim = [3, 7]
    np.testing.assert_equal(
        _get_timebase_indices(tlim, Fs, t0, Npts),
        [3, 4, 5, 6, 7])

    # `tlim` specifies full timebase
    tlim = [0, 9]
    np.testing.assert_equal(
        _get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # `tlim` beyond full timebase
    tlim = [-1, 11]
    np.testing.assert_equal(
        _get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # `tlim` is `None`
    tlim = None
    np.testing.assert_equal(
        _get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # (b) timebase: 1 <= t <= 5 w/ dt = 0.5:
    # --------------------------------------
    Fs = 2.
    t0 = 1.
    Npts = 9

    # The tests in (a) checked that we have appropriate behavior
    # at the boundary cases. Here, we just want to check that
    # things work when specifying non-zero `t0` and non-unity `Fs`,
    # so we'll just perform a single check with `tlim` specifying
    # a subset of the timebase.
    tlim = [3, 4]
    np.testing.assert_equal(
        _get_timebase_indices(tlim, Fs, t0, Npts),
        [4, 5, 6])

    return
