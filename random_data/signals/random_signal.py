'''This module implements a class for generating random data.

'''


import numpy as np


class RandomSignal(object):
    '''A class for the creation of random signals.

    Attributes:
    -----------
    Fs - float
        Sample rate.
        [Fs] = arbitrary units

    t0 - float
        Initial time (i.e. when sampling begins).
        [t0] = 1 / [Fs]

    fc - float, optional
        Cutoff frequency; frequencies f > fc are "cutoff"
        with a strength determined by `self.pole`
        [fc] = [Fs]

    pole - float, optional
        Order of the pole for f > fc

    '''
    def __init__(self, Fs, t0, T, fc=np.inf, pole=None):
        '''Create instance of the `RandomSignal` class.

        Parameters:
        -----------
        Fs - float
            Sample rate.
            [Fs] = arbitrary units

        t0 - float
            Initial time (i.e. when sampling begins).
            [t0] = 1 / [Fs]

        T - float
            Desired time interval over which signal is measured.
            Because creation of the random signal relies on the FFT
            (which is fastest for powers of two), the realized time
            interval `Treal` will be selected such that

                    Treal * Fs = nearest_power_of_2(T * Fs).

            [T] = 1 / [Fs]

        fc - float
            Cutoff frequency; frequencies f > fc will be "cutoff"
            with a strength determined by `pole`.
            [fc] = [Fs]

        pole - float, optional
            Order of the pole for f > fc.

        '''
        self.Fs = Fs
        self.t0 = t0
        self.fc = fc
        self.pole = pole

        # Construct the random signal in the frequency domain
        #
        # As we are dealing with *real* signals, the Fourier transform
        # is Hermitian, so we can simply look at the one-sided spectrum.
        # This reduces the number of bins in the frequency domain by 2.
        Nfreq_1sided = T * Fs / 2.

        # However, we want the resulting signal to have length equal
        # to a power of 2 for most efficient FFT computations, so
        # we will find the nearest power of 2
        exponent = int(np.round(np.log2(Nfreq_1sided)))
        Nfreq_1sided = (2 ** exponent) + 1

        # Signal phase, random
        ph = 2 * np.pi * np.random.random(Nfreq_1sided)

        # Signal magnitude, random
        Xf = np.abs(np.random.standard_normal(Nfreq_1sided))

        # If desired, weight signal magnitude with a pole of order `pole`
        # above cutoff frequency `fc`
        if pole is not None:
            f = np.linspace(0, Fs / 2., Nfreq_1sided)
            Xf = Xf / (1 + ((f / fc) ** pole))

        # Random signal's frequency domain representation, magnitude and phase
        Xf = Xf * np.exp(1j * ph)

        # Random signal in the time domain
        self.x = np.fft.irfft(Xf)

    def t(self):
        'Get times for points in `self.x`.'
        return self.t0 + (np.arange(len(self.x)) / self.Fs)
