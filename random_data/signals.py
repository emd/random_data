'''This module implements a class for generating random data.

'''


import numpy as np


class RandomSignal(object):
    '''Random signal class.

    Random signal sampled at frequency `Fs` over time interval `T`
    with a pole of order `pole` above the cutoff frequency `fc`.

    Instances have the following attributes as inputs:

        Fs - float, required
            Sample frequency
            [Fs] = Arbitrary Units

        T - float, required
            Time interval over which signal is measured
            [T] = 1 / [Fs]

        fc - float, optional
            Cutoff frequency; frequencies f > fc will be "cutoff"
            with a strength determined by `pole`
            [fc] = 1 / [Fs]

        pole - float, optional
            Order of the pole for f > fc

    In addition, the following attributes are created during initialization

        f - array, (`N`,)
            Frequency bins
            [f] = [Fs]

        Xf - array, (`N`,)
            Fourier transform (i.e. frequency domain representation)
            of signal
            [Xf] = [x] / [Fs]

        t - array, (`M`,) where `M` = 2 * (`N` - 1)
            Sample times
            [t] = 1 / [Fs]

        x - array, (`M`,) where `M` = 2 * (`N` - 1)
            Time domain representation of signal/sampled values
            [x] = Arbitrary Units

    '''
    def __init__(self, Fs, T, fc=None, pole=None):
        '''
        '''
        self.Fs = Fs
        self.fc = fc
        self.pole = pole

        dt = 1. / Fs

        # Number of frequency bins `Nfreq`, specified such that
        # the resulting signal has length equal to a power of 2
        # for most efficient FFT computations
        exponent = np.log2(T / dt)          # exact
        exponent = int(np.round(exponent))  # nearest power of 2
        Nfreq = (2 ** exponent) + 1

        # The realized time, using `Nfreq` as determined above
        self.T = Nfreq * dt
        Ntimes = 2 * (Nfreq - 1)
        self.t = np.linspace(0, self.T, Ntimes)

        # Construct the random signal in the frequency domain
        self.f = np.linspace(0, Fs / 2., Nfreq)

        # Signal phase, random
        ph = 2 * np.pi * np.random.random(Nfreq)

        # Signal magnitude, random
        Xf = np.abs(np.random.standard_normal(Nfreq))

        # If desired, weight signal magnitude with a pole of order `pole`
        # above cutoff frequency `fc`
        if fc is not None:
            Xf = Xf / (1 + ((self.f / fc) ** pole))

        # Random signal's frequency domain representation, magnitude and phase
        self.Xf = Xf * np.exp(1j * ph)

        # Random signal in the time domain
        self.x = np.fft.irfft(self.Xf)
