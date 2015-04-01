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
    def __init__(self, Fs, T, fc=np.inf, pole=None):
        '''
        '''
        self.Fs = Fs
        self.fc = fc
        self.pole = pole

        # As we are dealing with *real* signals, the Fourier transform
        # is Hermitian, so we can simply look at the one-sided spectrum.
        # This reduces the number of bins in the frequency domain by 2.
        Nfreq_1sided = T * Fs / 2.

        # However, we want the resulting signal to have length equal
        # to a power of 2 for most efficient FFT computations, so
        # we will find the nearest power of 2
        exponent = int(np.round(np.log2(Nfreq_1sided)))
        Nfreq_1sided = (2 ** exponent) + 1

        # The realized time, using `Nfreq_1sided` as determined above
        Ntimes = 2 * (Nfreq_1sided - 1)
        self.T = Ntimes / Fs
        self.t = np.linspace(0, self.T, Ntimes)

        # Construct the random signal in the frequency domain
        self.f = np.linspace(0, Fs / 2., Nfreq_1sided)

        # Signal phase, random
        ph = 2 * np.pi * np.random.random(Nfreq_1sided)

        # Signal magnitude, random
        Xf = np.abs(np.random.standard_normal(Nfreq_1sided))

        # If desired, weight signal magnitude with a pole of order `pole`
        # above cutoff frequency `fc`
        if pole is not None:
            Xf = Xf / (1 + ((self.f / fc) ** pole))

        # Random signal's frequency domain representation, magnitude and phase
        self.Xf = Xf * np.exp(1j * ph)

        # Random signal in the time domain
        self.x = np.fft.irfft(self.Xf)
