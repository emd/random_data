'''This module implements a class for detecting and working with
spikes in random signals.

'''


import numpy as np
import matplotlib.pyplot as plt


class SpikeHandler(object):
    '''A class for detecting and working with spikes in random signals.

    Attributes:
    -----------

    '''
    def __init__(self, x, Fs=1., t0=0.,
                 times_sigma_thresh=5., debounce_dt=None):
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

        times_sigma_thresh - float
            Points for which

                |x| > (times_sigma_thresh * std(x))

            will be considered as "spikes", where `std(x)` is the
            standard deviation of the raw input signal `x`.
            [times_sigma_thresh] = unitless

        debounce_dt - float, or None
            Detected spikes will be separated in time by *at least*
            `debounce_dt`. This ensures that peaks belonging to the
            same transient event are identified as a single "spike".
            As this prevents "bouncing" between identification as
            spike vs. non-spike, this is referred to as "debouncing".
            If `None`, do *not* debounce.
            [debounce_dt] = 1 / [Fs]

        '''
        self._Fs = Fs
        self._t0 = t0
        self._times_sigma_thresh = times_sigma_thresh
        self._debounce_dt = debounce_dt

    def _getStartTimes(self, x):
        'Get start times for spikes and spike-free portions in `x`.'
        # Determine indices for points exceeding threshold
        threshold = self._times_sigma_thresh * np.std(x)
        ind = np.where(np.abs(x) > threshold)[0]

        if self._debounce_dt is None:
            debounce_pts = 2
        else:
            debounce_pts = np.int(np.ceil(self._debounce_dt * self._Fs))

        # Distinct spikes are separated by *at least* `debounce_pts`.
        new_spike_start = np.where(np.diff(ind) >= debounce_pts)[0] + 1
        old_spike_stop = new_spike_start - 1

        # As the above identification of distinct spikes is based upon
        # a differencing scheme, it misses the start of the
        # first distinct spike and the stop of the last spike;
        # manually insert these values.
        new_spike_start = np.concatenate(([0], new_spike_start))
        old_spike_stop = np.concatenate((old_spike_stop, [len(ind) - 1]))

        spike_start_ind = ind[new_spike_start]
        spike_free_start_ind = ind[old_spike_stop] + 1

        # Boundary cases!!!

        # Get corresponding times

        return


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
