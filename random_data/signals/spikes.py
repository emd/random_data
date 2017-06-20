'''This module implements a class for detecting and working with
spikes in random signals.

'''


import numpy as np
import matplotlib.pyplot as plt


class Spikes(object):
    '''A class for detecting and working with spikes in random signals.

    Attributes:
    -----------

    '''
    def __init__(self, x, Fs=1., t0=0.,
                 times_sigma_thresh=5., debounce_dt=None):
        '''Create an instance of the `Spikes` class.

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

                |x| > (thresh * std(x)),

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

    def _getSpikeTimes(self, x):
        'Get start and end times for spikes in `x`.'
        # Determine points exceeding threshold
        threshold = self._times_sigma_thresh * np.std(x)
        pts = np.where(np.abs(x) > threshold)[0]

        # Use `ceil(...)` to ensure that debounced spikes are separated
        # by *at least* `self._debounce_dt` in time
        debounce_pts = np.int(np.ceil(self._debounce_dt * self._Fs))

        bndry = np.where(np.diff(pts) >= debounce_pts)[0] + 1
        starts = np.concatenate(([0], bndry))
        stops = np.concatenate(bndry - 1, pts[-1])
