'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.mlab import specgram
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter


class Spectrogram(object):
    '''Spectrogram class.

    Input Parameters:
    -----------------
    x - array, (`L`,)
        Time series from which spectrogram is generated.
        The signal is discrete, i.e. x = x(t_n) = x(n / Fs) = x_n
        for integer `n` and sampling frequency `Fs`
        [x] = Arbitrary

    Fs - float
        Sampling frequency
        [Fs] = 1 / [time]

    t0 - float
        Time corresponding to first data point in signal `x`
        [t0] = 1 / [Fs]

    overlap_frac - float
        Fraction of overlap between spectrogram windows
        0 <= overlap_frac < 1
        [overlap_frac] = unitless

    df - float
        Desired frequency resolution of spectrogram. For reasons
        of computational efficiency, the resulting frequency resolution
        may not be *exactly* the value specified
        [df] = [Fs]

    xunits - string
        Units of time series `x`. Default value of `None` prevents
        incorrect method usage.

    Fsunits -string
        Units of sampling frequency `Fs` and spectrogram frequency bin
        spacing `df`. Default value of `None` prevents incorrect method usage.

    funits - string
        Units of spectrogram frequency bins. Default value of `None`
        prevents incorrect method usage.

    Attributes:
    -----------
    Gxx - array, (`M`, `N`)
        One-sided power spectral density as a function of frequency bin `f`
        and time bin `t`.
        [Gxx] = (`xunits`)^2 / `funits`

    f - array, (`M`,)
        Frequency bins
        [f] = `funits`

    t - array, (`N`,)
        Time bins
        [t] = 1 / `Fsunits`

    '''
    def __init__(self, x, Fs, df,
                 t0=0., overlap_frac=0.,
                 xunits=None, Fsunits=None, funits=None):
        '''Create an instance of the Spectrogram class.'''
        # Check that supported units are being used prior to
        # performing any calculations
        if xunits is not None:
            self.xunits = xunits
        else:
            raise ValueError('Units of signal required!')

        if Fsunits == 'Hz':
            self._Fs = Fs
            self._Fsunits = Fsunits
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        if funits == 'kHz':
            self.funits = funits
            Hz_per_kHz = 1e3
        else:
            raise ValueError('Only kHz spectrogram frequency bins supported!')

        # In the absence of zero-padding, the window size `NFFT`
        # determines the size of the FFT frequency bins for
        # a given signal sampled at frequency `Fs` via
        #
        #       FFT frequency bin size = Fs / NFFT
        #
        # However, the FFT is most efficiently computed when
        # `NFFT` is a power of 2. Here, we take `NFFT` to be
        # the power of 2 that yields a frequency bin size
        # *closest* in value to the specified bin size `df`
        exponent = np.log2(Fs / df)            # exact
        exponent = np.int(np.round(exponent))  # for nearest power of 2
        self._NFFT = 2 ** exponent

        # A nonzero overlap decreases spectrogram "graininess"
        # and increases the number of spectrogram time bins
        self._noverlap = int(overlap_frac * self._NFFT)

        # Times of initial (`_t0`) and final (`_tf`) points in signal `x`
        # [_t0] = [_tf] = 1 / [Fs]
        self._t0 = t0
        self._tf = t0 + (len(x) / self._Fs)

        # TODO: detrend

        # Compute spectrogram, where `Gxx` is the one-sided PSD
        Gxx, f, t = specgram(x, Fs=Fs, NFFT=self._NFFT,
                             noverlap=self._noverlap)

        self.Gxx = Gxx * Hz_per_kHz
        self.f = f / Hz_per_kHz
        self.t = t + self._t0

    def plotSpec(self, fignum, cmap='Purples'):
        '''Plot spectrogram in figure `fignum`.'''
        # Check that supported units are being used prior to
        # performing any calculations
        if self.funits is 'kHz':
            yaxisunits = 'kHz'
        else:
            raise ValueError('Only kHz spectrogram frequency bins supported!')

        if self._Fsunits is 'Hz':
            xaxisunits = 's'
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        # Obtain local copy and flip array vertically
        Z = self.Gxx.copy()
        Z = np.flipud(Z)

        # Determine (x, y) extent of plot
        xmin = self._t0
        xmax = self._tf

        ymin = self.f[0]
        ymax = self.f[-1]

        extent = xmin, xmax, ymin, ymax

        plt.figure(fignum)
        plt.clf()
        plt.imshow(Z, norm=LogNorm(), extent=extent, aspect='auto',
                   cmap=cmap)
        plt.colorbar(format=LogFormatter(labelOnlyBase=True))
        plt.xlabel('$t \, [\mathrm{' + xaxisunits + '}]$', fontsize=16)
        plt.ylabel('$f \, [\mathrm{' + yaxisunits + '}]$', fontsize=16)
        plt.title('$|G_{xx}(f)|^2 \, [\mathrm{' + self.xunits +
                  '}^2 / \mathrm{' + yaxisunits + '}]$',
                  fontsize=16)
        plt.show()

        return
