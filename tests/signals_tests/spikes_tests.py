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


def test_SpikeHandler():
    # Sinusoidal signal parameters
    A = np.sqrt(2)      # RMS amplitude of unity
    Fs = 1.             # sampling rate
    f0 = Fs / 20        # frequency of sinusoid
    t0 = 0              # initial time of waveform
    Nperiods = 10       # number of periods in waveform

    # Generate stationary signal
    t = np.arange(t0, Nperiods / f0, 1. / Fs)
    x = np.sqrt(2) * np.cos(2 * np.pi * f0 * t)

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
