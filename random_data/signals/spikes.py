'''This module implements a class for detecting and working with
spikes in random signals.

'''


import numpy as np
import matplotlib.pyplot as plt


class SpikeHandler(object):
    '''A class for detecting and working with spikes in random signals.

    Attributes:
    -----------
    spike_start_times - array_like, (`L`,)
        Timestamp corresponding to the start of each detected spike.
        [spike_start_times] = 1 / [Fs], where Fs was the sampling rate
            provided at initialization

    spike_free_start_times - array_like, (`M`,)
        Timestamp corresponding to the start of each spike-free region.
        If the first point in the input signal is identified as a spike,
        then `M` = `L`; otherwise, `M` = `L` + 1.
        [spike_free_start_times] = 1 / [Fs], where Fs was the sampling
            rate provided at initialization

    '''
    def __init__(self, x, Fs=1., t0=0.,
                 sigma_mult=5., debounce_dt=None):
        '''Create an instance of the `SpikeHandler` class.

        Input parameters:
        -----------------
        x - array_like, (`N`,)
            The random signal to be searched for transient spikes.
            It is assumed that `x` has zero-mean and is sampled
            uniformly in time.
            [x] = arbitrary units

        Fs - float
            The sampling rate of `x`.
            If not specified, `Fs` is assigned a value of unity such that
            all frequencies are *normalized* to the sampling rate.
            [Fs] = arbitrary units

        t0 - float
            The initial time corresponding to `x[0]`.
            [t0] = 1 / [Fs]

        sigma_mult - float
            Points for which

                |x| > (sigma_mult * std(x))

            will be considered as "spikes", where `std(x)` is the
            standard deviation of the raw input signal `x`.
            [sigma_mult] = unitless

        debounce_dt - float, or None
            The first point of a spike and the final point of the
            preceding spike will be separated in time by *at least*
            `debounce_dt`. This ensures that peaks belonging to the
            same transient event are identified as a single "spike".
            As this prevents "bouncing" between identification as
            spike vs. non-spike, this is referred to as "debouncing".
            If `None`, do *not* debounce.
            [debounce_dt] = 1 / [Fs]

        '''
        # Note parameters for spike detection & debouncing
        self._Fs = Fs
        self._t0 = t0
        self._sigma_mult = sigma_mult
        self._debounce_dt = debounce_dt

        # Determine spike and spike-free start times
        times = self._getStartTimes(x)
        self.spike_start_times = times[0]
        self.spike_free_start_times = times[1]

    def _getStartTimes(self, x):
        'Get start times for spikes and spike-free portions in `x`.'
        # Determine indices for points exceeding threshold
        threshold = self._sigma_mult * np.std(x)
        ind = np.where(np.abs(x) > threshold)[0]

        if self._debounce_dt is None:
            # Adjacent points in `x` are defined to belong to
            # the same spike, but points with separation of 2
            # or more are defined to belong to different spikes
            debounce_pts = 2
        else:
            # Here, `ceil(...)` ensures that the debounced spikes
            # are separated by *at least* `self._debounce_dt` in time
            debounce_pts = np.int(np.ceil(self._debounce_dt * self._Fs))

        # Determine indices corresponding to the start and stop
        # of each detected spike
        spike_start_ind, spike_stop_ind = _subset_boundary_values(
            ind, min_subset_spacing=debounce_pts)

        # Spike-free region begins immediately after end of spike
        spike_free_start_ind = spike_stop_ind + 1

        # If the first point in `x` does *not* correspond to a spike,
        # however, we need to manually insert the zero index
        # into the spike-free indices
        if spike_start_ind[0] != 0:
            # `len(spike_free_start_ind) >= 1` such that
            # we can always concatenate w/o ValueError
            spike_free_start_ind = np.concatenate((
                [0], spike_free_start_ind))

        # Convert indices into times
        spike_start_times = _index_times(
            spike_start_ind, self._Fs, self._t0)
        spike_free_start_times = _index_times(
            spike_free_start_ind, self._Fs, self._t0)

        return spike_start_times, spike_free_start_times


def _subset_boundary_values(x, min_subset_spacing=2):
    '''Get initial and final values of each subset in `x`,
    where subset boundaries are at least `min_subset_spacing`
    apart.

    Parameters:
    -----------
    x - array_like, (`N`,)
        A monotonically increasing array, consisting of one
        or more subsets. Subsets are determined at runtime
        by comparing the value difference between adjacent
        entries in `x` to the threshold set by keyword
        `min_subset_spacing`. Colloquially, a subset is
        a group of values that are "close" to one another,
        while different subsets are "far" from one another.

    min_subset_spacing - float
        The minimum spacing between the end of one subset
        and the beginning of the next subset. That is, if

            x[i + 1] - x[i] >= min_subset_spacing,

        then `x[i]` is the last value in the current subset,
        and `x[i + 1]` is the first value in the subsequent
        subset.

    Returns:
    --------
    (subset_start_vals, subset_stop_vals) - tuple, where

    subset_start_vals - array_like, (`M`,)
        A monotonically increasing array consisting of
        the *first* value in each subset of `x`.
        `M` <= `N`.

    subset_stop_vals - array_like, (`M`,)
        A monotonically increasing array consisting of
        the *last* value in each subset of `x`.
        `M` <= `N`.

    '''
    delta = np.diff(x)

    if not np.alltrue(delta >= 0):
        raise ValueError('`x` must be monotonically increasing')

    # `bnd` gives the index for the *last* point in each subset of `x`.
    bnd = np.where(delta >= min_subset_spacing)[0]
    subset_stop_vals = x[bnd]

    # Determine the *first* point in each subsequent subset of `x`.
    # Note that each individual element of `bnd` obeys
    #
    #           0 <= bnd[i] <= (len(x) - 2)
    #
    # such that the below indexing should never produce an IndexError.
    # Note that this indexing is also correct if `bnd` is empty;
    # that is, `bnd + 1` is also empty if `bnd` is empty.
    subset_start_vals = x[bnd + 1]

    # As the above subset parsing relies on a differencing method,
    # it misses the first value of the first subset and the last
    # value of the last subset. Add these values manually.
    try:
        subset_start_vals = np.concatenate(([x[0]], subset_start_vals))
    except ValueError:
        # Called if `subset_start_vals` is initially empty;
        # i.e. if there is only one subset in `x`.
        subset_start_vals = np.array([x[0]])

    try:
        subset_stop_vals = np.concatenate((subset_stop_vals, [x[-1]]))
    except ValueError:
        # Called if `subset_stop_vals` is initially empty;
        # i.e. if there is only one subset in `x`.
        subset_start_vals = np.array([x[-1]])

    return subset_start_vals, subset_stop_vals


def _index_times(ind, Fs, t0):
    '''Return times corresponding to indices in `ind` assuming
    uniform sampling at `Fs` beginning at time `t0`.

    '''
    return t0 + (ind / Fs)
