'''This module implements a class that conveniently defines *ensembles*
for the analysis of stationary and non-stationary random processes.
It is assumed that all spectral analysis is done with the FFT.

'''


# Standard library imports
import numpy as np
from matplotlib import mlab


class Ensemble(object):
    '''A class that defines ensemble properties for a random process.

    For random process `x`, statistical property `Q` is estimated as

                            Q = E[M{x(t)}]

    where `E[...]` denotes the expectation value operator and
    `M{...}` is the corresponding moment or transform corresponding
    to property `Q`.

    The expectation value is typically computed by averaging over
    several "realizations" of the process. Each realization is
    an independent measurement of the process, and the collection
    of these realizations is known as the "ensemble". Typically,
    random error in estimates of the process' properties decreases
    as the number of realizations per ensemble increases;
    however, for fixed ensemble size, the spectral resolution
    typically decreases as the number of realizations increases.

    For analysis of non-stationary signals, the ensemble time window
    should be chosen small enough such that the statistical properties
    of the process do not change appreciably over the time window
    (this can be thought of as an "adiabatic" condition).

    It is assumed that all spectral analysis is done with the FFT.

    Attributes:
    -----------
    Nreal_per_ens - int
        The number of realizations per ensemble.

    Npts_per_real - int
        The number of sample points per realization.

    Npts_overlap - int
        The number of overlapping points between adjacent realizations.

    Npts_per_ens - int
        The number of sample points per ensemble.

    Fs - float
        The signal sampling rate, as specified at object initialization.
        [Fs] = arbitrary units

    t - array_like, (`M`,)
        The temporal midpoint of each ensemble.
        [t] = 1 / [Fs]

    f - array_like, (`L`,)
        The frequencies at which spectral quantities can be estimated
        with the defined ensemble.
        [f] = [Fs]

    dt - float
        The temporal resolution between ensembles.
        [dt] = 1 / [Fs]

    df - float
        The frequency resolution with which spectral quantities
        can be estimated with the defined ensemble.
        [df] = 1 / [Fs]

    Methods:
    --------
    Type `help(Ensemble)` in the IPython console for a listing.

    '''
    def __init__(self, x, Fs=1.0, t0=0.,
                 Tens=40960., Nreal_per_ens=10, fraction_overlap=0.5,
                 Npts_per_real=None, Npts_overlap=None):
        '''Create an instance of the `Ensemble` class.

        Input Parameters:
        -----------------
        x - array_like, (`N`,)
            The signal that is being split into ensembles.
            [x] = arbitrary units

        Fs - float
            The sampling rate of `x`. If not specified, `Fs` is assigned
            a value of unity such that all frequencies are *normalized*
            to the sampling rate.
            [Fs] = arbitrary units

        t0 - float
            The initial time corresponding to `x[0]`.
            [t0] = 1 / [Fs]

        Tens - float
            The time window defining an ensemble. `Tens` determines the
            time resolution of the spectral density calculations,
            with larger `Tens` corresponding to reduced time resolution
            and increased frequency resolution.
            [Tens] = 1 / [Fs]

        Nreal_per_ens - int
            The number of realizations per ensemble. Typically,
            increasing the number of realizations decreases random error
            but decreases spectral resolution. A ValueError is raised
            if not a positive integer.

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

        '''
        self.Fs = Fs

        # Assign number of realizations per ensemble, if valid
        if Nreal_per_ens > 0 and isinstance(Nreal_per_ens, int):
            self.Nreal_per_ens = Nreal_per_ens
        else:
            raise ValueError('`Nreal_per_ens` must be a positive integer!')

        # Assign number of sample points to use per realization
        if Npts_per_real is None:
            self.Npts_per_real = self.getNumPtsPerReal(
                Fs, Tens, self.Nreal_per_ens, fraction_overlap)
        elif Npts_per_real > 0 and isinstance(Npts_per_real, int):
            self.Npts_per_real = Npts_per_real
        else:
            raise ValueError('`Npts_per_real` must be a positive integer!')

        # Determine number of overlapping points between adjacent realizations
        if Npts_overlap is None:
            if fraction_overlap >= 0 and fraction_overlap < 1:
                self.Npts_overlap = np.int(
                    fraction_overlap * self.Npts_per_real)
            else:
                raise ValueError('`fraction_overlap` must be between 0 and 1!')
        else:
            if Npts_overlap < 0 or not isinstance(Npts_overlap, int):
                raise ValueError('`Npts_overlap` must be an integer >= 0!')
            elif Npts_overlap >= self.Npts_per_real:
                raise ValueError('`Npts_overlap` must be < `Npts_per_real`!')
            else:
                self.Npts_overlap = Npts_overlap

        # Determine number of points per ensemble
        self.Npts_per_ens = self.getNumPtsPerEns()

        # Generate times `t` corresponding to the midpoint of each ensemble,
        # and compute the frequencies `f` at which spectral estimates
        # can be made with the defined ensemble
        self.t = self.getTimes(x, Fs, t0)
        self.f = self.getFrequencies(Fs)

        # Determine resolution in time and frequency, if applicable
        self.dt = self.getTens()

        try:
            self.df = self.f[1] - self.f[0]
        except IndexError:
            self.df = np.nan

    def getNumPtsPerReal(self, Fs, Tens, Nreal_per_ens, fraction_overlap):
        '''Get number of points per realization.

        As the number of points must be a whole number, there will
        generally be round-off error such that the resulting ensemble
        time window is slightly different than the specified `Tens`.
        Further, to ensure efficient FFT computation, the number of
        points per ensemble is required to be a power of two,
        potentially leading to even larger differences between
        the resulting ensemble time window and the spec'd `Tens`.

        This function should be called *before* `self.getNumPtsPerEns(...)`.

        '''
        # If a given ensemble of length `Tens` consists of `Nreal_per_ens`
        # (potentially overlapping) realizations, each of length `Treal`,
        # then
        #
        #  Tens = Treal * {1 + [(Nreal_per_ens - 1) * (1 - fraction_overlap)]}
        #
        # where `fraction_overlap` is the fractional overlap between
        # adjacent realizations. `Treal` is easily solved for.
        denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
        Treal = np.float(Tens) / denominator  # avoid integer division!

        # Ensure the number of points per realization is a power of 2,
        # allowing efficient computation of the FFT. In the past,
        # rounding `Treal * Fs` to the next largest power of 2 led
        # to normalization errors, so we always round down to the
        # largest power of 2 less than `Treal * Fs`.
        return _largest_power_of_2_leq(Treal * Fs)

    def getNumPtsPerEns(self):
        'Get number of points per ensemble.'
        # In a given ensemble, there are `Nreal_per_ens` (potentially
        # overlapping) realizations. Each of these realizations
        # contains `Npts_per_real` sample points. Thus, the first
        # realization contributes `Npts_per_real` sample points.
        Npts_per_ens = self.Npts_per_real

        # If there `Npts_overlap` overlapping sample points between
        # adjacent realizations, the remaining (`Nreal_per_ens` - 1)
        # realizations each contribute (`Npts_per_real` - `Npts_overlap`)
        # distinct sample points.
        distinct_points_per_real = self.Npts_per_real - self.Npts_overlap
        Npts_per_ens += ((self.Nreal_per_ens - 1) * distinct_points_per_real)

        return Npts_per_ens

    def getFrequencies(self, Fs):
        '''Get frequencies at which spectral quantities can be estimated
        with the defined ensemble. Only f >= 0 is returned.

        '''
        return np.fft.rfftfreq(self.Npts_per_real, d=(1. / Fs))

    def getTens(self):
        '''Get temporal length of each ensemble. In general, the
        ensemble time window will slightly differ from that specified
        during object initialization; this is to ensure efficient FFT
        computation. This method returns the true ensemble time length.

        '''
        # avoid integer division!
        return self.Npts_per_ens / np.float(self.Fs)

    def getTimes(self, x, Fs, t0):
        'Get times corresponding to the midpoint of each ensemble.'
        # The ensemble forms the basic unit/discretization of time
        # for the computed spectral density estimate, so determine
        # ensemble time length.
        Tens = self.getTens()

        # Determine the number of *whole* ensembles in the data record
        # (Disregard fractional ensemble at the end of the data, if present)
        Nens = np.int(len(x) / self.Npts_per_ens)

        # The returned time base corresponds to the midpoint of each ensemble
        return t0 + (Tens * np.arange(0.5, Nens, 1))

    def getFFTs(self, x, detrend=mlab.detrend_none,
                window=mlab.window_hanning):
        '''Get array of FFTs corresponding to each realization of `x`.

        Parameters:
        -----------
        x - array_like, (`N`,)
            Signal to be analyzed. Signal is split into several
            realizations, and the FFT of each realization is computed.
            [x] = arbitrary units

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

        Returns:
        --------
        Xk - array_like, (L, M, N) where
                L = `self.Npts_per_real` = `len(self.f)`,
                M = number of whole ensembles in data record `x`, and
                N = `self.Nreal_per_ens`

            The FFTs of each realization in each ensemble.
            The FFTs are indexed by frequency, ensemble, and realization.

            [Xk] = [x]

        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x):
            raise ValueError('`x` must be a real-valued signal!')

        # Determine the number of *whole* ensembles in the data record
        # (Disregard fractional ensemble at the end of the data, if present)
        Nens = np.int(len(x) / self.Npts_per_ens)

        # Determine number of frequencies in 1-sided FFT, noting that
        # `self.Npts_per_real` is constrained to be a power of 2
        Nf = (self.Npts_per_real // 2) + 1

        # Initialize.
        Xk = np.zeros(
            (Nf, Nens, self.Nreal_per_ens),
            dtype='complex')

        # Loop through each ensemble, computing the FFT of each realization
        # via strides for efficient use of memory. (Note that the below
        # procedure closely parallels that of Matplotlib's internal function
        #
        #     :py:func:`_spectral_helper <matplotlib.mlab._spectral_helper>`
        #
        # Here, we use our own implementation so as not to rely on
        # an internal function)
        stride_axis = 0
        for ens in np.arange(Nens):
            # Split the ensemble into realizations
            sl = slice(
                ens * self.Npts_per_ens,
                (ens + 1) * self.Npts_per_ens)

            result = mlab.stride_windows(
                x[sl],
                self.Npts_per_real,
                self.Npts_overlap,
                axis=stride_axis)

            # Detrend each realization
            result = mlab.detrend(
                result,
                detrend,
                axis=stride_axis)

            # Window each realization (power loss compensated outside loop)
            result, windowVals = mlab.apply_window(
                result,
                window,
                axis=stride_axis,
                return_window=True)

            # Finally compute and return the FFT of each realization
            Xk[:, ens, :] = np.fft.rfft(result, axis=stride_axis)

        # Compensate for windowing power loss
        norm = np.sqrt(np.mean((np.abs(windowVals)) ** 2))
        Xk /= norm

        return Xk


def _largest_power_of_2_leq(x):
    'Get the largest power of 2 that is less than or equal to `x`.'
    exponent = np.log2(x)           # exact
    exponent = np.int(exponent)     # next lowest power of 2
    return 2 ** exponent


def closest_index(v, val):
    'Return integer index of entry in `v` closest in value to `val`.'
    delta = np.abs(v - val)
    return np.where(delta == np.min(delta))[0][0]
