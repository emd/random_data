'''This module implements classes for higher-order spectral analysis
e.g. bispectrum, bicoherence, etc.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

# Related 3rd-party imports
from ..ensemble import Ensemble
from ..utilities import get_timebase_indices


class Bispectrum(object):
    '''A class for bispectrum estimation.

    Attributes:
    -----------

    Methods:
    --------
    Type `help(Bispectrum)` in the IPython console for a listing.

    '''
    def __init__(self, x, y, Fs=1.0, t0=0.,
                 tlim=None, Nreal_per_ens=10, fraction_overlap=0.5,
                 Npts_per_real=None, Npts_overlap=None,
                 detrend=None, window=mlab.window_hanning,
                 print_params=True, print_status=True):
        '''Create an instance of the `Bispectrum` class.

        Input Parameters:
        -----------------
        x, y - array_like, (`N`,)
            The signals for which the bispectrum will be computed.
            A ValueError is raised if `x` and `y` contain a different number
            of samples. Note that `x` and `y` must be sampled at the *same*
            rate, `Fs`.
            [x] = arbitrary units
            [y] = arbitrary units, potentially different than [x]

        Fs - float
            The sampling rate of `x` and `y`.
            If not specified, `Fs` is assigned a value of unity such that
            all frequencies are *normalized* to the sampling rate.
            [Fs] = arbitrary units

        t0 - float
            The initial time corresponding to `x[0]` (and `y[0]`).
            [t0] = 1 / [Fs]

        tlim - array_like, (2,) or None
            If not None, then only use the portion of `x` and `y`
            that sit between `min(tlim)` and `max(tlim)`.
            [tlim] = 1 / [Fs]

        Nreal_per_ens - int
            The number of realizations per ensemble. The random error in the
            bispectrum estimate decreases as ~ 1 /Nreal_per_ens.
            The frequency resolution `df` of the bispectrum estimate
            is linearly related to the number of realizations.
            A ValueError is raised if not a positive integer.

        fraction_overlap - float
            The fractional overlap between adjacent realizations.
            0 =< `fraction_overlap` < 1, otherwise a ValueError is raised.

        Npts_per_real - int
            The number of sample points per realization. If None,
            `Tens` is used to compute `Npts_per_real` that is compatible
            with `Nreal_per_ens` and efficient FFT computation.
            If not None, `Tens` is ignored. A ValueError is raised
            if not a positive integer.

        Npts_overlap - int
            The number of overlapping sample points between adjacent
            realizations. If None, `fraction_overlap` sets the
            number of overlapping sample points. If not None,
            `fraction_overlap` is ignored. A ValueError is raised
            if not a positive integer or if greater than or equal to
            the number of points per realization.

        detrend - string
            The function applied to each realization before taking FFT.
            May be [ 'default' | 'constant' | 'mean' | 'linear' | 'none']
            or callable, as specified in :py:func: `csd <matplotlib.mlab.csd>`.

            *Warning*: Naively detrending (even with something as simple as
            `mean` or `linear` detrending) can introduce detrimental artifacts
            into the computed spectrum, so *no* detrending is the default.

        window - callable or ndarray
            The window applied to each realization before taking FFT,
            as specified in :py:func: `csd <matplotlib.mlab.csd>`.

        print_params - bool
            If True, print relevant spectral parameters to screen.

        print_status - bool
            If True, print percentage of ensembles whose spectra
            have been computed.

        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('`x` and `y` must be real-valued signals!')

        if len(x) != len(y):
            raise ValueError('`x` and `y` must have the same length!')

        # Use `tlim` to determine the size of the ensemble
        tind = get_timebase_indices(tlim, Fs, t0, len(x))
        Tens = (tind[-1] - tind[0]) / Fs

        # Determine properties for ensemble averaging
        ens = Ensemble(
            x, Fs=Fs, t0=t0, Tens=Tens,
            Nreal_per_ens=Nreal_per_ens, fraction_overlap=fraction_overlap,
            Npts_per_real=Npts_per_real, Npts_overlap=Npts_overlap)

        # Record important aspects of computation
        self.same_data = x is y

        self.Fs = ens.Fs

        self.Npts_per_real = ens.Npts_per_real
        self.Nreal_per_ens = ens.Nreal_per_ens
        self.Npts_overlap = ens.Npts_overlap
        self.Npts_per_ens = ens.Npts_per_ens

        self.detrend = detrend
        self.window = window

        self.f = ens.f
        self.df = ens.df

        self.t = ens.t
        self.dt = ens.dt

        if print_params:
            self.printSpectralParams()

        # Perform spectral calculations
        self._getBispectra(x, y, ens)

    def printSpectralParams(self):
        print '\ndt: %.6g' % self.dt
        print 'df: %.6g' % self.df
        print 'Npts_per_real: %i' % self.Npts_per_real
        print ('overlap: %.2f'
               % (np.float(self.Npts_overlap) / self.Npts_per_real))
        print 'Nreal_per_ens: %i' % self.Nreal_per_ens
        print 'detrend: %s' % self.detrend
        print 'window: %s' % self.window.func_name

        return

    def _getBispectra(self, x, y, ens):
        'Get bispectrum estimate.'
        X = ens.getFFTs(x, detrend=self.detrend, window=self.window)

        if not self.same_data:
            Y = ens.getFFTs(y, detrend=ens.detrend, window=ens.window)
        else:
            Y = X

        # Initialize bispectrum array
        Nf = len(ens.f)
        shape = (Nf // 2, Nf)
        self.Bxy = np.zeros(shape, dtype='complex')

        # Compute bispectrum
        for j in np.arange(Nf // 2):
            for k in np.arange(j, Nf - j):
                self.Bxy[j, k] = np.mean(
                    np.conj(X[j + k, ...]) * Y[j, ...] * Y[k, ...],
                    axis=-1)

        # Initialize squared-bicoherence array
        self.b2xy = np.zeros(shape)

        # Compute bicoherence
        for j in np.arange(Nf // 2):
            for k in np.arange(j, Nf - j):
                num = (np.abs(self.Bxy[j, k])) ** 2

                denX = np.mean(
                    (np.abs(X[j + k, ...])) ** 2,
                    axis=-1)
                denY = np.mean(
                    (np.abs(Y[j, ...] * Y[k, ...])) ** 2,
                    axis=-1)

                self.b2xy[j, k] = num / (denX * denY)

        return
