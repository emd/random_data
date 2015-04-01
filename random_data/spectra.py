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
    def __init__(self, x, Fs, df, xunits=None, Fsunits=None, funits=None):
        '''Create an instance of the Spectrogram class.'''
        # Check that supported units are being used prior to
        # performing any calculations
        if xunits is not None:
            self.xunits = xunits
        else:
            raise ValueError('Units of signal required!')

        if Fsunits == 'Hz':
            self.Fsunits = Fsunits
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
        NFFT = 2 ** exponent

        # TODO: noverlap???

        # Compute spectrogram, where `Gxx` is the one-sided PSD
        Gxx, f, t = specgram(x, NFFT=NFFT, Fs=Fs)

        self.Gxx = Gxx * Hz_per_kHz
        self.f = f / Hz_per_kHz
        self.t = t

    def plotSpec(self):
        # Check that supported units are being used prior to
        # performing any calculations
        if self.funits is 'kHz':
            yaxisunits = 'kHz'
        else:
            raise ValueError('Only kHz spectrogram frequency bins supported!')

        if self.Fsunits is 'Hz':
            xaxisunits = 's'
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        # Obtain local copy, calculate power in dB, and flip array vertically
        Z = self.Gxx.copy()
        Z = 10 * np.log10(Z)
        Z = np.flipud(Z)

        # TODO: (1) xlims not exact currently??
        xmin = self.t[0]
        xmax = self.t[-1]

        ymin = self.f[0]
        ymax = self.f[-1]

        extent = xmin, xmax, ymin, ymax

        plt.imshow(Z, cmap='Purples', extent=extent, aspect='auto')
        plt.colorbar()
        plt.xlabel('$t \, [\mathrm{' + xaxisunits + '}]$', fontsize=16)
        plt.ylabel('$f \, [\mathrm{' + yaxisunits + '}]$', fontsize=16)
        plt.title('$|G_{xx}(f)|^2 \, [\mathrm{' + self.xunits +
                  '}^2 / \mathrm{' + yaxisunits + '}]$',
                  fontsize=16)
        plt.show()

        return
