from nose import tools
import numpy as np
from random_data.signals.spikes import (
    _subset_boundary_values, _index_times, SpikeHandler)


def test__subset_boundary_values():
    # Monotonicity test:
    # ==================
    x = np.arange(10)
    x[5] = 0                # `x` is now nonmonotonic
    tools.assert_raises(
        ValueError,
        _subset_boundary_values, *[x])

    # Trivial subsets:
    # ================

    # Trivial subset test 1: only one subset in `x`
    # ---------------------------------------------
    min_subset_spacing = 2
    x = np.arange(10)

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        np.array([x[0]]))

    np.testing.assert_equal(
        subset_stop_vals,
        np.array([x[-1]]))

    # Trivial subset test 2: uniform spacing by `min_subset_spacing`
    # such that each point in `x` is a distinct subset
    # --------------------------------------------------------------
    min_subset_spacing = 1
    x = np.array([0, 1, 2, 3])

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        x)

    np.testing.assert_equal(
        subset_stop_vals,
        x)

    # Trivial subset test 3: non-uniform spacing by >= `min_subset_spacing`
    # such that each point in `x` is a distinct subset
    # ---------------------------------------------------------------------
    min_subset_spacing = 1
    x = np.array([0, 1, 10, 33])

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        x)

    np.testing.assert_equal(
        subset_stop_vals,
        x)

    # Non-trivial subsets:
    # ====================

    # Non-trivial subset test 1: all subsets have length > 1
    # -------------------------------------------------------
    min_subset_spacing = 2

    x = np.array([
        0, 1, 2, 3,         # subset 1
        5, 6, 7, 8,         # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 13])

    # Non-trivial subset test 2: middle subset has unity length
    # ---------------------------------------------------------
    min_subset_spacing = 2

    x = np.array([
        0, 1, 2, 3,         # subset 1
        5,                  # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 5, 13])

    # Non-trivial subset test 3: first subset has unity length
    # --------------------------------------------------------
    min_subset_spacing = 2

    x = np.array([
        3,                  # subset 1
        5, 6, 7, 8,         # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [3, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 13])

    # Non-trivial subset test 4: last subset has unity length
    # -------------------------------------------------------
    min_subset_spacing = 2

    x = np.array([
        0, 1, 2, 3,     # subset 1
        5, 6, 7, 8,     # subset 2
        10])            # subset 3

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 10])

    # Non-trivial subset test 5: bouncing < `min_subset_spacing`
    # ----------------------------------------------------------
    min_subset_spacing = 3

    x = np.array([
        0, 2, 3,            # subset 1
        6, 8,               # subset 2
        11, 13, 15, 16,     # subset 3
        20])                # subset 4

    vals = _subset_boundary_values(
        x, min_subset_spacing=min_subset_spacing)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 6, 11, 20])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 16, 20])

    return


def test__index_times():
    # Non-uniformly spaced indices, beginning at 0
    ind = np.array([0, 10, 30])

    # For this "trivial" choice of `Fs` and `t0`,
    # the returned times are simply equal to `ind`
    Fs = 1
    t0 = 0
    np.testing.assert_equal(
        _index_times(ind, Fs, t0),
        ind)

    # Increasing `t0` by unity augments returned times
    # by unity as well
    t0 = 1
    np.testing.assert_equal(
        _index_times(ind, Fs, t0),
        ind + 1)

    # Doubling sampling rate decreases returned times
    # by a factor of two
    Fs = 2
    t0 = 0
    np.testing.assert_equal(
        _index_times(ind, Fs, t0),
        ind / 2)

    return


def test_SpikeHandler__init__():
    # Sinusoidal signal parameters
    A = np.sqrt(2)      # RMS amplitude of unity
    Fs = 1.             # sampling rate
    f0 = Fs / 20        # frequency of sinusoid
    t0 = 0              # initial time of waveform
    Nperiods = 10       # number of periods in waveform

    # Generate stationary signal
    t = np.arange(t0, Nperiods / f0, 1. / Fs)
    x = A * np.cos(2 * np.pi * f0 * t)

    # Single, one-point spike:
    # ========================
    debounce_dt = None  # debouncing not needed for single-point spike
    sigma_mult = 5
    spike_amplitude = 10

    # In middle of signal:
    # --------------------
    spike_ind = 50          # w/ t0 = 0 & Fs=1, this is also spike *time*
    spike = np.zeros(len(x))
    spike[spike_ind] = spike_amplitude

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=None)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([0, spike_ind + 1]))

    # At start of signal:
    # -------------------
    spike_ind = 0           # w/ t0 = 0 & Fs=1, this is also spike *time*
    spike = np.zeros(len(x))
    spike[spike_ind] = spike_amplitude

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=None)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([spike_ind + 1]))

    # At end of signal:
    # -----------------
    spike_ind = len(x) - 1  # w/ t0 = 0 & Fs=1, this is also spike *time*
    spike = np.zeros(len(x))
    spike[spike_ind] = spike_amplitude

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=None)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([0, spike_ind + 1]))

    # Single, multi-point w/ "bouncing" spike:
    # ========================================
    debounce_dt = 4

    # In middle of signal:
    # --------------------
    spike_start_ind = 50    # w/ t0 = 0 & Fs=1, this is also spike *time*
    spike = np.zeros(len(x))

    # Spike jumps above & below signal baseline with
    # two empty spaces between spike peak & valley
    spike[spike_start_ind:(spike_start_ind + 5)] = np.array([
        spike_amplitude, 0, 0, -spike_amplitude, -spike_amplitude])

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=debounce_dt)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_start_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([0, spike_start_ind + 5]))

    # At beginning of signal:
    # -----------------------
    spike_start_ind = 0     # w/ t0 = 0 & Fs=1, this is also spike *time*
    spike = np.zeros(len(x))

    # Spike jumps above & below signal baseline with
    # two empty spaces between spike peak & valley
    spike[spike_start_ind:(spike_start_ind + 5)] = np.array([
        spike_amplitude, 0, 0, -spike_amplitude, -spike_amplitude])

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=debounce_dt)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_start_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([spike_start_ind + 5]))

    # At end of signal:
    # -----------------
    spike_start_ind = len(x) - 5
    spike = np.zeros(len(x))

    # Spike jumps above & below signal baseline with
    # two empty spaces between spike peak & valley
    spike[spike_start_ind:] = np.array([
        spike_amplitude, 0, 0, -spike_amplitude, -spike_amplitude])

    # Spike detection
    sh = SpikeHandler(x + spike, Fs=Fs, t0=t0,
                      sigma_mult=sigma_mult, debounce_dt=debounce_dt)

    np.testing.assert_equal(
        sh.spike_start_times,
        np.array([spike_start_ind]))

    np.testing.assert_equal(
        sh.spike_free_start_times,
        np.array([0, spike_start_ind + 5]))

    return


def test_SpikeHandler__getSpikeFreeTimeWindows():
    # ValueError tests:
    # =================
    x = np.zeros(100)
    x[50] = 10
    sh = SpikeHandler(x)

    # `window_fraction` should have length 2
    window_fraction = [0.2, 0.3, 0.4]
    tools.assert_raises(
        ValueError,
        sh._getSpikeFreeTimeWindows,
        **{'window_fraction': window_fraction})

    # Values in `window_fraction` should be between 0 & 1, inclusive
    window_fraction = [-0.1, 1.]
    tools.assert_raises(
        ValueError,
        sh._getSpikeFreeTimeWindows,
        **{'window_fraction': window_fraction})

    window_fraction = [0, 1.1]
    tools.assert_raises(
        ValueError,
        sh._getSpikeFreeTimeWindows,
        **{'window_fraction': window_fraction})

    # Functionality tests:
    # ====================

    # Two, single-point peaks:
    # ------------------------
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    # 20% - 80% window
    tstart, tstop = sh._getSpikeFreeTimeWindows(
        window_fraction=[0.2, 0.8])

    # Shift of 2 in second entry of each array expected because
    # first peak happens at t = 101 and start of second spike-free
    # region begins at t = 102 (contrast this with first region,
    # which begins at t = 0). Similar reasoning applies for third
    # entry of array in `tstart` test.
    np.testing.assert_equal(tstart, np.array([20, 122, 224]))
    np.testing.assert_equal(tstop, np.array([80, 182]))

    # 0% - 100% window
    tstart, tstop = sh._getSpikeFreeTimeWindows(
        window_fraction=[0., 1.0])

    # Shift of 2 in second entry of each array expected because
    # first peak happens at t = 101 and start of second spike-free
    # region begins at t = 102 (contrast this with first region,
    # which begins at t = 0). Similar reasoning applies for third
    # entry of array in `tstart` test.
    np.testing.assert_equal(tstart, np.array([0, 102, 204]))
    np.testing.assert_equal(tstop, np.array([100, 202]))

    # 35% - 53% window
    tstart, tstop = sh._getSpikeFreeTimeWindows(
        window_fraction=[0.35, 0.53])

    # Shift of 2 in second entry of each array expected because
    # first peak happens at t = 101 and start of second spike-free
    # region begins at t = 102 (contrast this with first region,
    # which begins at t = 0). Similar reasoning applies for third
    # entry of array in `tstart` test.
    np.testing.assert_equal(tstart, np.array([35, 137, 239]))
    np.testing.assert_equal(tstop, np.array([53, 155]))

    return


def test_SpikeHandler_addSpike():
    # SpikeHandler instance for two, single-point peaks:
    # ==================================================
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    # `tlim` should have length 2:
    # ============================
    tlim = [10, 20, 30]
    tools.assert_raises(
        ValueError,
        sh.addSpike,
        *[tlim])

    # Raise a ValueError if `tlim` straddles identified spike(s):
    # ===========================================================

    # Two, single-point peaks; initial point is spike-free:
    # -----------------------------------------------------
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    tlim = [100, 101]
    tools.assert_raises(
        ValueError,
        sh.addSpike,
        *[tlim])

    # Three, single-point peaks; initial point is a spike:
    # ----------------------------------------------------
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[0] = 10.
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    tlim = [100, 101]
    tools.assert_raises(
        ValueError,
        sh.addSpike,
        *[tlim])

    # Test a valid insertion:
    # =======================

    # Two, single-point peaks; initial point is spike-free:
    # -----------------------------------------------------
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    tlim = [150, 155]
    sh.addSpike(tlim)

    np.testing.assert_equal(
        sh.spike_start_times,
        [101, 150, 203])

    np.testing.assert_equal(
        sh.spike_free_start_times,
        [0, 102, 156, 204])

    # Three, single-point peaks; initial point is a spike:
    # ----------------------------------------------------
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[0] = 10.
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    tlim = [150, 155]
    sh.addSpike(tlim)

    np.testing.assert_equal(
        sh.spike_start_times,
        [0, 101, 150, 203])

    np.testing.assert_equal(
        sh.spike_free_start_times,
        [1, 102, 156, 204])

    return


def test_SpikeHandler_getSpikeFreeTimeIndices():
    # Generate signal w/ two, single-point peaks for tests:
    # =====================================================
    Fs = 1.
    t0 = 0.
    x = np.zeros(301)
    x[101] = 10.
    x[203] = 10.

    sh = SpikeHandler(x, Fs=Fs, t0=t0)

    # Tests with same timebase as that of input signal:
    # =================================================
    timebase = t0 + (np.arange(len(x)) / Fs)

    # 20% - 80% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0.2, 0.8])

    # Shift of 2 in second `arange(...)` expected because
    # first peak happens at t = 101 and start of second spike-free
    # region begins at t = 102 (contrast this with first region,
    # which begins at t = 0). Similar reasoning applies for third
    # `arange(...)`.
    ind_expected = np.concatenate((
        np.arange(20, 81),
        np.arange(122, 183),
        np.arange(224, 301)))

    np.testing.assert_equal(ind, ind_expected)

    # 0% - 100% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0., 1.0])

    # Shift of 2 in second `arange(...)` expected because
    # first peak happens at t = 101 and start of second spike-free
    # region begins at t = 102 (contrast this with first region,
    # which begins at t = 0). Similar reasoning applies for third
    # `arange(...)`.
    ind_expected = np.concatenate((
        np.arange(0, 101),
        np.arange(102, 203),
        np.arange(204, 301)))

    np.testing.assert_equal(ind, ind_expected)

    # Tests with same timebase offset by one from that of input signal:
    # =================================================================
    timebase = t0 + (np.arange(len(x)) / Fs) - 1

    # 20% - 80% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0.2, 0.8])

    # Subtracting one from `timebase` of original signal means
    # that we need to *add* one to expected indices from
    # corresponding tests with original timebase. (*Except* for
    # last argument in final `arange(...)`, as this value is
    # fixed by the size of `timebase`, which still has length 300).
    ind_expected = np.concatenate((
        np.arange(21, 82),
        np.arange(123, 184),
        np.arange(225, 301)))

    np.testing.assert_equal(ind, ind_expected)

    # 0% - 100% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0., 1.0])

    # Subtracting one from `timebase` of original signal means
    # that we need to *add* one to expected indices from
    # corresponding tests with original timebase. (*Except* for
    # last argument in final `arange(...)`, as this value is
    # fixed by the size of `timebase`, which still has length 300).
    ind_expected = np.concatenate((
        np.arange(1, 102),
        np.arange(103, 204),
        np.arange(205, 301)))

    np.testing.assert_equal(ind, ind_expected)

    # Tests with same timebase 2x slower than that of input signal:
    # =============================================================
    timebase = np.arange(t0, len(x) / Fs, 1 / (0.5 * Fs))

    # 20% - 80% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0.2, 0.8])

    # For `timebase` that is 2x slower than that of original input signal,
    # indices reduced by a factor of two as well relative to those
    # determined for the original timebase. (Also, need to account for
    # "off-by-one" type factors associated w/ using `arange(...)` etc.).
    ind_expected = np.concatenate((
        np.arange(10, 41),
        np.arange(61, 92),
        np.arange(112, 151)))

    np.testing.assert_equal(ind, ind_expected)

    # 0% - 100% window:
    # -----------------
    ind = sh.getSpikeFreeTimeIndices(
        timebase, window_fraction=[0., 1.0])

    # For `timebase` that is 2x slower than that of original input signal,
    # indices reduced by a factor of two as well relative to those
    # determined for the original timebase. (Also, need to account for
    # "off-by-one" type factors associated w/ using `arange(...)` etc.).
    #
    # In this particular case, note that the timebase is too slow
    # and the fraction of the window is too large to exclude any
    # points from the spike-free region.
    ind_expected = np.concatenate((
        np.arange(0, 51),
        np.arange(51, 102),
        np.arange(102, 151)))

    np.testing.assert_equal(ind, ind_expected)

    return
