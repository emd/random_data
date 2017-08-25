'''This module implements classes for generating random data.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt


# Intra-package imports
from ..ensemble import _largest_power_of_2_leq


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


class RandomSignal2d(object):
    '''A class for the creation of 2-dimensional random signals with
    spectral shaping in both the spatial and time domains.

    Attributes:
    -----------
    x - array_like, (`self.Nz`, `self.Nt`)
        The 2-dimensional random signal with the spectral shaping provided
        during initialization. The first axes corresponds to the spatial
        dimension with `self.Nz` spatial points; the second axes corresponds
        to the temporal dimension with `self.Nt` temporal points. The signal
        is constrained to be real.
        [x] = arbitrary units

    Fs - float
        Temporal sampling rate of signal `self.x`.
        [Fs] = arbitrary units

    t0 - float
        Initial time stamp of signal `self.x`.
        [t0] = 1 / [self.Fs]

    Nt - int
        The number of timestamps in `self.x`. Constrained to be
        a power of 2, for fastest FFT computations.
        [Nt] = unitless

    fc - float
        Cutoff frequency of signal `self.x`, wher frequencies f > self.fc
        have been "cutoff" with a strength determined by `self.pole`.
        [fc] = [self.Fs]

    pole - float, optional
        Order of the pole for f > `self.fc`.
        [pole] = unitless

    Fs_spatial - float
        Spatial sampling rate of signal `self.x`.
        [Fs_spatial] = arbitrary units

    z0 - float
        Initial spatial stamp of signal `self.x`.
        [z0] = 1 / [self.Fs_spatial]

    Nz - int
        The number of spatial stamps in `self.x`. Constrained to be
        a power of 2, for fastest FFT computations.
        [Nz] = unitless

    vph - float
        The phase velocity, vph, of the mode, defined as

                    vph = omega / k = f / xi,

        where omega = 2 * pi * f is the angular frequency and
        k = 2 * pi * xi is the wavenumber (xi is the spatial frequency).
        [vph] = [self.Fs] / [self.Fs_spatial]

    Delta - float
        The mode's "width" in the spatial-frequency domain.
        [Delta] = [self.Fs_spatial]

    '''
    def __init__(self,
                 Fs=1., t0=0., T=128., fc=0.1, pole=2,
                 Fs_spatial=1., z0=0., Z=64., vph=1., Delta=0.05,
                 amplitude_noise_floor=1e-4):
        '''Create an instance of the `RandomSignal2d` class.

        Input parameters:
        -----------------
        Fs - float
            Temporal sampling rate.
            [Fs] = arbitrary units

        t0 - float
            Initial time stamp.
            [t0] = 1 / [Fs]

        T - float
            Desired time interval over which signal is measured.
            Because creation of the random signal relies on the FFT
            (which is fastest for powers of two), the realized time
            interval `Treal` will be selected such that

                Treal * Fs = _largest_power_of_2_leq(T * Fs),

            where `_largest_power_of_2_leq(a)` selects the largest
            power of 2 that is less than or equal to `a`.
            [T] = 1 / [Fs]

        fc - float
            Cutoff frequency; frequencies f > fc will be "cutoff"
            with a strength determined by `pole`.
            [fc] = [Fs]

        pole - float, optional
            Order of the pole for f > fc.
            [pole] = unitless

        Fs_spatial - float
            Spatial sampling rate.
            [Fs_spatial] = arbitrary units

        z0 - float
            Initial spatial stamp.
            [z0] = 1 / [Fs_spatial]

        Z - float
            Desired spatial interval over which signal is measured.
            Because creation of the random signal relies on the FFT
            (which is fastest for powers of two), the realized spatial
            interval `Zreal` will be selected such that

                Zreal * Fs_spatial = _largest_power_of_2_leq(Z * Fs_spatial),

            where `_largest_power_of_2_leq(a)` selects the largest
            power of 2 that is less than or equal to `a`.
            [Z] = 1 / [Fs_spatial]

        vph - float
            The phase velocity, vph, of the mode, defined as

                        vph = omega / k = f / xi,

            where omega = 2 * pi * f is the angular frequency and
            k = 2 * pi * xi is the wavenumber (xi is the spatial frequency).
            [vph] = [Fs] / [Fs_spatial]

        Delta - float
            The mode's "width" in the spatial-frequency domain.
            [Delta] = [Fs_spatial]

        amplitude_noise_floor - float
            The amplitude noise floor. Specifying a noise floor ensures
            that plotting routines etc produce reasonable/useful graphics,
            and it is also motivated by the physical situation in which a
            measurement hits a noise floor.
            [amplitude_noise_floor] = [self.x], i.e. the amplitude noise
                floor is specified as a fraction of the signal amplitude

        '''
        # Temporal parameters
        self.Fs = Fs
        self.t0 = t0
        self.fc = fc
        self.pole = pole

        # Spatial parameters
        self.Fs_spatial = Fs_spatial
        self.z0 = z0
        self.vph = vph
        self.Delta = Delta

        # Amplitude noise floor
        self._amplitude_noise_floor = amplitude_noise_floor

        # Determine number of points in temporal and spatial grids,
        # rounded to largest power of 2 that is less than or equal
        # to the requested domain length
        self.Nt = _largest_power_of_2_leq((T - t0) * Fs)
        self.Nz = _largest_power_of_2_leq((Z - z0) * Fs_spatial)

        # Get Fourier representation of signal
        res = self._getFourierRepresentation()
        self._xi = res[0]
        self._f = res[1]
        self._X = res[2]

        # Compute signal as a function of space and time
        self.x = self._getSpaceTimeRepresentation()

    def t(self):
        'Get times for points in `self.x`.'
        return _uniform_grid(self.Nt, self.t0, 1. / self.Fs)

    def z(self):
        'Get spatial coordinates for points in `self.x`.'
        return _uniform_grid(self.Nz, self.z0, 1. / self.Fs_spatial)

    def _getFourierRepresentation(self):
        '''Get Fourier representation of signal.

        Returns:
        --------
        (xi, f, X) - tuple, where

        xi - array_like, (`self.Nz`,)
            The spatial frequency. Note that the spatial frequency
            is related to the wavenumber k via k = 2 * pi * xi.
            Note that `xi` is ordered via the standard `np.fft.fftfreq`
            ordering (that is, the zero-spatial-frequency component
            sits at `xi[0]` etc.).
            [xi] = [self.Fs_spatial]

        f - array_like, (`self.Nt`,)
            The frequency. Note that `f` is ordered via the standard
            `np.fft.fftfreq` ordering (that is, the zero-frequency
            component sits at `f[0]` etc.).
            [f] = [self.Fs]

        X - array_like, (`self.Nz`, `self.Nt`)
            The Fourier representation of the signal. Note that `X`
            is ordered via the standard `np.fft.fft2` ordering (that is,
            the zero-frequency components sit at `X[0, 0]` etc.).
            [X] = [self.x]

        '''
        # Ensure Hermitian symmetry in spectral domain by
        # Fourier transforming a real, random, 2d signal.
        # Of course, this produces a white spectrum, but
        # we will subsequently shape the spectrum according
        # to the parameters provided at initialization.
        X = np.fft.fft2(np.random.randn(self.Nz, self.Nt))

        # Before shaping spectrum, specify amplitude noise floor.
        noise_floor = self._amplitude_noise_floor * X

        # Construct computational grid
        f = np.fft.fftfreq(self.Nt, d=(1. / self.Fs))
        xi = np.fft.fftfreq(self.Nz, d=(1. / self.Fs_spatial))
        ff, xixi = np.meshgrid(f, xi)

        # Given the dispersion relation omega(k), the phase
        # velocity vph is given as
        #
        #       vph = omega(k) / k = f(xi) / xi.
        #
        # Thus, the mode branch lies on the (xi, f) satisfying
        #
        #       xi - (f / vph) == 0.
        #
        branch = xixi - (ff / self.vph)

        # Shape spectrum
        xi_shaping = np.exp(-((branch / self.Delta) ** 2))
        f_shaping = (1. / (1 + ((ff / self.fc) ** self.pole)))
        X *= (xi_shaping * f_shaping)

        # Add noise floor, for reasons discussed in `__init__()` docs.
        X += noise_floor

        return xi, f, X

    def plotFourierRepresentation(
            self, amplitude_cmap='viridis', phase_cmap='RdBu', Nlev=10):
        '''Plot contours of Fourier amplitudes and phases of signal.

        This works best when `self.Nt` and `self.Nz` are not "too big",
        which reduces the load on the contour-plotting routines. Thus,
        it is best to create a `RandomSignal2d` instance with small
        `self.Nt` and `self.Nz` with the desired spectral properties
        (e.g. `self.fc`, `self.pole`, `self.vph`, `self.Delta`) and
        then use this plotting routine; if the spectrum looks as
        desired, then create a new `RandomSignal2d` instance with
        the same spectral parameters but with larger `self.Nt` and
        `self.Nz` for use in spectral calculations.

        '''
        xi = np.fft.fftshift(self._xi)
        f = np.fft.fftshift(self._f)
        X = np.fft.fftshift(self._X)

        fig, axes = plt.subplots(
            2, 1, sharex=True, sharey=True, figsize=(6, 8))

        cmag = axes[0].contourf(
            xi, f, np.log10(np.abs(X.T)), Nlev, cmap=amplitude_cmap)
        cph = axes[1].contourf(
            xi, f, np.angle(X.T), Nlev, cmap=phase_cmap)

        axes[0].set_ylabel(r'$f$')
        axes[1].set_xlabel(r'$\xi$')
        axes[1].set_ylabel(r'$f$')

        cbmag = plt.colorbar(cmag, ax=axes[0])
        cbmag.set_label(r'$|X(\xi, f)|$')

        cbph = plt.colorbar(cph, ax=axes[1])
        cbph.set_label(r'$\angle X(\xi, f)$')

        plt.tight_layout()
        plt.show()

        return

    def _getSpaceTimeRepresentation(self):
        '''Get Fourier representation of signal.

        Returns:
        --------
        x - array_like, (`self.Nz`, `self.Nt`)
            The *real* signal as a function of space and time.
            [x] = arbitrary units

        '''
        return np.real(np.fft.ifft2(self._X))

    def plotSignal(self, cmap='RdBu', Nlev=10):
        '''Plot contours of signal `self.x` as a function of
        space and time.

        This works best when `self.Nt` and `self.Nz` are not "too big",
        which reduces the load on the contour-plotting routines.

        '''
        plt.figure()

        c = plt.contourf(self.z(), self.t(), self.x.T, Nlev, cmap=cmap)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$t$')

        cb = plt.colorbar(c)
        cb.set_label(r'$x(z, t)$')

        plt.tight_layout()
        plt.show()

        return


def _uniform_grid(Npts, x0, dx):
    'Get uniform grid of `Npts` starting at `x0` and spaced by `dx`.'
    return x0 + (np.arange(Npts) * dx)
