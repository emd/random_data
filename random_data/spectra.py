'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram


class Spectrogram(object):
    '''Spectrogram class.'''
    def __init__(self, x, Fs, df, cmap='Purples'):
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
        # Obtain local copy, calculate power in dB, and flip array vertically
        Z = self.Gxx.copy()
        Z = 10 * np.log10(Z)
        Z = np.flipud(Z)

        # [t] = s
        # TODO: (1) Enforce that time is in seconds; (2) xlims not exact now??
        xmin = self.t[0]
        xmax = self.t[-1]

        # [f] = kHz
        # TODO: (1) Enforce that resulting frequency in kHz; (2) same as above
        units = 1e3
        ymin = self.f[0] / units
        ymax = self.f[-1] / units

        extent = xmin, xmax, ymin, ymax

        plt.imshow(Z, cmap='Purples', extent=extent, aspect='auto')
        plt.colorbar()
        # TODO: Ensure these labels are correct
        plt.xlabel('$t \, [\mathrm{s}]$', fontsize=16)
        plt.ylabel('$f \, [\mathrm{kHz}]$', fontsize=16)
        plt.title('$|G_{xx}(f)|^2$',
                  fontsize=16)
        plt.show()

        # TODO: Return image handle???
        return
