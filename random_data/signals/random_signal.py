'''This module implements classes for generating random data.

'''


# Standard library imports
import numpy as np


# Intra-package imports
from ..ensemble import _largest_power_of_2_leq
from ..spectra.nonparametric import _plot_image


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
        The 2-dimensional random signal with the autospectral density
        specified at initialization. The first axis corresponds to the
        spatial dimension with `self.Nz` spatial points; the second axis
        corresponds to the temporal dimension with `self.Nt` temporal
        points. The signal is constrained to be real.
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

    Lz - float
        The spatial correlation length, where a Gaussian correlation
        function has been assumed.
        [Lz] = 1 / [self.Fs_spatial]

    '''
    def __init__(self,
                 Fs=1., t0=0., T=128., fc=0.1, pole=2,
                 Fs_spatial=1., z0=0., Z=64., vph=1., Lz=5,
                 noise_floor=1e-2):
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

        Lz - float
            The spatial correlation length, where a Gaussian correlation
            function has been assumed.
            [Lz] = 1 / [Fs_spatial]

        noise_floor - float
            The noise floor of the random process's autospectral density.
            [noise_floor] = [self.x]^2 / [Fs] / [Fs_spatial], where
                `self.x` is the realization of the random process
                created at object initialization.

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
        self.Lz = Lz

        #  Noise floor of the random process's autospectral density
        self._noise_floor = noise_floor

        # Determine number of points in temporal and spatial grids,
        # rounded to largest power of 2 that is less than or equal
        # to the requested domain length
        self.Nt = _largest_power_of_2_leq((T - t0) * Fs)
        self.Nz = _largest_power_of_2_leq((Z - z0) * Fs_spatial)

        # Get autospectral density of 2d random process
        res = self._getAutoSpectralDensity()
        self._xi = res[0]
        self._f = res[1]
        self._Sxx = res[2]

        # Get a space-time realization of the 2d random process
        self.x = self._getSignal()

    def _getAutoSpectralDensity(self):
        '''Get autospectral density Sxx(xi, f) of the 2d random process.

        Returns:
        --------
        (xi, f, Sxx) - tuple, where

        xi - array_like, (`self.Nz`,)
            The (two-sided) spatial frequency in ascending order.

            Note that the spatial frequency is related to the wavenumber k
            via k = 2 * pi * xi.

            [xi] = [self.Fs_spatial]

        f - array_like, ((`self.Nt` // 2) + 1,)
            The (one-sided) frequency in ascending order.
            [f] = [self.Fs]

        Sxx - array_like, (`self.Nz`, `self.Nt`)
            The autospectral density Sxx(xi, f) of the 2d random process.

            Note that the autospectral density is one-sided in frequency (f)
            and two-sided in spatial frequency (xi).

            [Sxx] = [self.x]^2 / [self.Fs] / [self.Fs_spatial], where
                `self.x` is the realization of the random process
                created at object initialization.

        '''
        # Construct the spectral grid.
        #
        # Typically, we present present the autospectral density Sxx(xi, f)
        # as one-sided in frequency (f) & two-sided in spatial frequency (xi),
        # so we will follow that convention here.
        f = np.fft.rfftfreq(self.Nt, d=(1. / self.Fs))
        xi = np.fft.fftshift(np.fft.fftfreq(self.Nz, d=(1. / self.Fs_spatial)))
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

        # Shape auto-spectral density, Sxx
        f_shaping = (1. / (1 + ((ff / self.fc) ** self.pole)))
        xi_shaping = np.exp(-((np.pi * self.Lz * branch) ** 2))
        Sxx = xi_shaping * f_shaping

        # Incorporate noise floor
        Sxx += (self._noise_floor * np.max(Sxx))

        return xi, f, Sxx

    def _getSignal(self):
        '''Get a space-time realization of the 2d random process.

        Returns:
        --------
        x - array_like, (`self.Nz`, `self.Nt`)
            A realization of the 2d random process in space and time.

            For a given random process, the space-time representation
            will vary from one realization to the next, but the underlying
            autospectral density of each realization will be identical.

            [x] = arbitrary units

        '''
        # Compute *magnitude* of FFT corresponding to autospectral density.
        #
        # The frequency normalization includes an additional factor of 2
        # to account for the one-sided in frequency representation of
        # the autospectral density.
        f_norm = 2. / (self.Nt * self.Fs)
        xi_norm = 1. / (self.Nz * self.Fs_spatial)
        Xmag = np.sqrt(self._Sxx / f_norm / xi_norm)

        # To obtain a realization of the random process, we now need
        # to add a random phase to each point of the FFT.
        #
        # Note that if the random signal is real-valued, as is desired,
        # then the FFT must have Hermitian symmetry, i.e.
        #
        #               X(-xi, -f) = [X(xi, f)]*,
        #
        # where * indicates the complex conjugate.
        #
        # Perhaps the easiest way to satisfy the above Hermitian-symmetry
        # constraint is to steal the phase from a dummy random signal `y`
        # with the desired dimensions, as is done below. Note that care
        # must be exercised in application of one-sided vs. two-sided FFTs,
        # as they do *not* commute; specifically, the forward one-sided FFT
        # will "silently discard" any imaginary component of the input signal.
        # Thus, in computation of the forward FFT, the one-sided FFT (in time)
        # must be applied first, and then the forward two-sided FFT (in space)
        # can be applied. When computing the inverse FFTs, the opposite
        # ordering must be used.
        y = np.random.randn(self.Nz, self.Nt)
        Y = np.fft.fft(np.fft.rfft(y, axis=1), axis=0)
        ph = np.angle(Y)

        # Shift along spatial axis, as the autospectral density's convention
        # is two-sided spatial frequencies in ascending order.
        ph = np.fft.fftshift(ph, axes=0)

        # Construct the complex-valued FFT of the realization by
        # multiplying the FFT magnitude by the set of random phases.
        X = Xmag * np.exp(1j * ph)

        # Inverse the shift along the spatial axis to bring the FFT
        # into the conventional FFT ordering.
        X = np.fft.ifftshift(X, axes=0)

        # As discussed when computing the phase above, the two-sided and
        # one-sided FFTs do *not* commute. Thus, to compute the space-time
        # realization of the random process, we need to first compute the
        # two-sided inverse FFT in space and then compute the one-sided
        # inverse FFT in time.
        return np.fft.irfft(np.fft.ifft(X, axis=0), axis=1)

    def t(self):
        'Get times for points in `self.x`.'
        return _uniform_grid(self.Nt, self.t0, 1. / self.Fs)

    def z(self):
        'Get spatial coordinates for points in `self.x`.'
        return _uniform_grid(self.Nz, self.z0, 1. / self.Fs_spatial)

    def plotSpectralDensity(
            self, xilim=None, flim=None, vlim=None,
            cmap='viridis', interpolation='none',
            fontsize=16,
            title=None, xlabel=r'$\xi$', ylabel=r'$f$',
            cblabel=r'$|S_{xx}(\xi, f)|$', cborientation='horizontal',
            ax=None, fig=None, geometry=111):
        'Plot magnitude of autospectral density on log scale.'
        ax = _plot_image(
            self._xi, self._f, np.abs(self._Sxx).T,
            xlim=xilim, ylim=flim, vlim=vlim,
            norm='log', cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel=cblabel, cborientation=cborientation,
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotSignal(self, cmap='RdBu', interpolation='none', ax=None):
        'Plot image of signal as a function of space and time.'
        ax = _plot_image(
            self.z(), self.t(), self.x.T,
            norm=None, cmap=cmap, interpolation=interpolation,
            title='', xlabel=r'$z$', ylabel=r'$t$',
            cblabel=r'$x(z, t)$', cborientation='vertical',
            ax=ax)

        return ax


def _uniform_grid(Npts, x0, dx):
    'Get uniform grid of `Npts` starting at `x0` and spaced by `dx`.'
    return x0 + (np.arange(Npts) * dx)
