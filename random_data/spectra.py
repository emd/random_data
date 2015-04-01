'''This module implements a routines for analyzing spectra.

'''


import numpy as np
from matplotlib.pylab import specgram


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
        Gxx, f, t, im = specgram(x, NFFT=NFFT, Fs=Fs, cmap=cmap)

        self.Gxx = Gxx
        self.f = f
        self.t = t
        self.im = im
