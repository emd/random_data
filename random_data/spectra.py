'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram


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

    df - float
        Desired frequency resolution of spectrogram. For reasons
        of computational efficiency, the resulting frequency resolution
        may not be *exactly* the value specified
        [df] = [Fs]

    xunits - string
        Units of time series `x`. Default value of `None` prevents
        incorrect method usage.

    funits - string
        Units of sampling frequency `Fs` and spectrogram frequency bin
        spacing `df`. Default value of `None` prevents incorrect method usage.

    Attributes:
    -----------
    Gxx - array, (`M`, `N`)
        One-sided power spectral density as a function of frequency bin `f`
        and time bin `t`.
        [Gxx] = [x]^2 / [Fs]

    f - array, (`M`,)
        Frequency bins
        [f] = [Fs]

    t - array, (`N`,)
        Time bins
        [t] = 1 / [Fs]

    '''
    def __init__(self, x, Fs, df, xunits=None, funits=None):
        '''Create an instance of the Spectrogram class.'''
        # Check that supported units are being used prior to
        # performing any calculations
        if xunits is not None:
            self.xunits = xunits
        else:
            raise ValueError('Units of time series required!')

        if funits == 'Hz':
            self.funits = funits
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

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
        NFFT = 2 ** exponent

        # Compute spectrogram, where `Gxx` is the one-sided PSD
        Gxx, f, t = specgram(x, NFFT=NFFT, Fs=Fs)

        self.Gxx = Gxx
        self.f = f
        self.t = t

    def plotSpec(self):
        # Check that supported units are being used prior to
        # performing any calculations
        if self.funits == 'Hz':
            xunits = 's'
            yunits = 'kHz'
            yconv = 1e3
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        # Obtain local copy, calculate power in dB, and flip array vertically
        Z = yconv * self.Gxx.copy()
        Z = 10 * np.log10(Z)
        Z = np.flipud(Z)

        # TODO: (1) xlims not exact currently??
        xmin = self.t[0]
        xmax = self.t[-1]

        ymin = self.f[0] / yconv
        ymax = self.f[-1] / yconv

        extent = xmin, xmax, ymin, ymax

        plt.imshow(Z, cmap='Purples', extent=extent, aspect='auto')
        plt.colorbar()
        plt.xlabel('$t \, [\mathrm{' + xunits + '}]$', fontsize=16)
        plt.ylabel('$f \, [\mathrm{' + yunits + '}]$', fontsize=16)
        plt.title('$|G_{xx}(f)|^2 \, [\mathrm{' + self.xunits +
                  '}^2 / \mathrm{' + yunits + '}]$',
                  fontsize=16)
        plt.show()

        return
