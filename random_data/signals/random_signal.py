'''This module implements classes for generating random data.

'''


# Standard library imports
import numpy as np


# Intra-package imports
from ..ensemble import _largest_power_of_2_leq
from ..spectra.nonparametric import _plot_image


class RandomSignal(object):
    '''A class for the creation of 1-dimensional random signals.

    Note that `M` >= 1 turbulent branches can be specified simultaneously.

    Attributes:
    -----------
    x - array_like, (`self.Nt`,)
        The 1-dimensional random signal with the autospectral density
        specified at initialization. The signal is constrained to be real.
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

    f0 - array_like, (`M`,)
        The dominant temporal frequency of each turbulent branch.
        [f0] = [self.Fs]

    tau - array_like, (`M`,)
        The correlation time of each turbulent branch, where
        a Gaussian correlation function has been assumed.
        [tau] = 1 / [self.Fs]

    G0 - array_like, (`M`,)
        The relative peak one-sided autospectral density of each branch.
        [G0] = unitless

    '''
    def __init__(self,
                 Fs=1., t0=0., T=128.,
                 f0=[0.], tau=[10.], G0=[1.],
                 noise_floor=1e-2, seed=None):
        '''Create instance of the `RandomSignal` class.

        Note that `M` >= 1 turbulent branches can be specified simultaneously.

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

        f0 - array_like, (`M`,)
            The dominant temporal frequency of each turbulent branch
            in the medium's rest frame (i.e. `f0` is *not* attributable
            to a Doppler shift; see `v` for Doppler-shift effects).
            [f0] = [Fs]

        tau - array_like, (`M`,)
            The correlation time of each turbulent branch, where
            a Gaussian correlation function has been assumed.
            [tau] = 1 / [Fs]

        G0 - array_like, (`M`,)
            The relative peak one-sided autospectral density of each branch.
            [G0] = unitless

        noise_floor - float
            The noise floor of the random process's autospectral density.
            [noise_floor] = [self.x]^2 / [Fs] / [Fs_spatial], where
                `self.x` is the realization of the random process
                created at object initialization.

        seed - int or None
            Random seed used to initialize pseudo-random number generator.
            If `None`, generator is seeded from `/dev/urandom` or the clock.

        '''
        # Grid parameters
        self.Fs = Fs
        self.t0 = t0
        self.Nt = _largest_power_of_2_leq(T * Fs)

        # Turbulence parameters
        self.f0 = np.array(f0, dtype='float', ndmin=1)
        self.tau = np.array(tau, dtype='float', ndmin=1)
        self.G0 = np.array(G0, dtype='float', ndmin=1)

        #  Noise floor of the random process's autospectral density
        self._noise_floor = noise_floor

        # Get autospectral density of random process
        res = self._getAutoSpectralDensity()
        self._f = res[0]
        self._Gxx = res[1]

        # Get a temporal realization of the random process
        self.x = self._getSignal(seed=seed)

    def _getAutoSpectralDensity(self):
        '''Get one-sided autospectral density Gxx(f) of the 1d random process.

        Returns:
        --------
        (f, Gxx) - tuple, where

        f - array_like, ((`self.Nt` // 2) + 1,)
            The (one-sided) frequency in ascending order.
            [f] = [self.Fs]

        Gxx - array_like, (`self.Nt`,)
            The one-sided autospectral density Gxx(f) of the 1d random process.

            [Gxx] = [self.x]^2 / [self.Fs], where
                `self.x` is the realization of the random process
                created at object initialization.

        '''
        # Construct the spectral grid.
        f = np.fft.rfftfreq(self.Nt, d=(1. / self.Fs))

        # Initialize autospectral density with zeros
        Gxx = np.zeros(f.shape)

        # Iteratively incorporate autospectral density of each branch
        for branch_ind in np.arange(len(self.f0)):
            # Parse turbulence parameters of branch
            f0 = self.f0[branch_ind]
            tau = self.tau[branch_ind]
            G0 = self.G0[branch_ind]

            # Shape auto-spectral density, Sxx.
            f_shaping = G0 * np.exp(-((np.pi * tau * (f - f0)) ** 2))
            Gxx += f_shaping

        # Define peak autospectral density of turbulence to be unity
        Gxx /= np.max(Gxx)

        # Finally, incorporate noise floor
        Gxx += self._noise_floor

        return f, Gxx

    def _getSignal(self, seed=None):
        '''Get a temporal realization of the 1d random process.

        Input parameters:
        -----------------
        seed - int or None
            Random seed used to initialize pseudo-random number generator.
            If `None`, generator is seeded from `/dev/urandom` or the clock.

        Returns:
        --------
        x - array_like, (`self.Nt`,)
            A realization of the 1d random process in time.

            For a given random process, the temporal representation
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
        Xmag = np.sqrt(self._Gxx / f_norm)

        # To obtain a realization of the random process, we now need
        # to add a random phase to each point of the FFT.
        if seed is not None:
            np.random.seed(seed)

        ph = 2 * np.pi * np.random.rand(len(self._f))

        # Construct the complex-valued FFT of the realization by
        # multiplying the FFT magnitude by the set of random phases.
        X = Xmag * np.exp(1j * ph)

        return np.fft.irfft(X)

    def t(self):
        'Get times for points in `self.x`.'
        return _uniform_grid(self.Nt, self.t0, 1. / self.Fs)


class RandomSignal2d(object):
    '''A class for the creation of 2-dimensional random signals.

    Note that `M` >= 1 turbulent branches can be specified simultaneously.

    Attributes:
    -----------
    x - array_like, (`self.Nz`, `self.Nt`)
        The 2-dimensional random signal with the autospectral density
        specified at initialization. The first axis corresponds to the
        spatial dimension with `self.Nz` spatial points; the second axis
        corresponds to the temporal dimension with `self.Nt` temporal
        points. The signal is constrained to be real.
        [x] = arbitrary units

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

    xi0 - array_like, (`M`,)
        The dominant spatial frequency of each turbulent branch.
        [xi0] = [self.Fs_spatial]

    Lz - array_like, (`M`,)
        The spatial correlation length of each turbulent branch, where
        a Gaussian correlation function has been assumed.
        [Lz] = 1 / [self.Fs_spatial]

    f0 - array_like, (`M`,)
        The dominant temporal frequency of each turbulent branch
        in the medium's rest frame (i.e. `f0` is *not* attributable
        to a Doppler shift; see `v` for Doppler-shift effects).
        [f0] = [self.Fs]

    tau - array_like, (`M`,)
        The correlation time of each turbulent branch, where
        a Gaussian correlation function has been assumed.
        [tau] = 1 / [self.Fs]

    v - array_like, (`M`,)
        The lab-frame velocity of the medium through which
        the turbulent branch is propagating. Note that
        non-zero velocity produces a Doppler-shifted
        lab-frame frequency

                            df = xi * v

        where `xi` is the spatial frequency.
        [v] = [self.Fs] / [self.Fs_spatial]

    S0 - array_like, (`M`,)
        The relative peak autospectral density of each branch.
        [S0] = unitless

    '''
    def __init__(self,
                 Fs_spatial=1., z0=0., Z=64.,
                 Fs=1., t0=0., T=128.,
                 xi0=[0.], Lz=[5.], f0=[0.], tau=[10.], v=[1.], S0=[1.],
                 noise_floor=1e-2, seed=None):
        '''Create an instance of the `RandomSignal2d` class.

        Note that `M` >= 1 turbulent branches can be specified simultaneously.

        Input parameters:
        -----------------
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

        xi0 - array_like, (`M`,)
            The dominant spatial frequency of each turbulent branch.
            [xi0] = [Fs_spatial]

        Lz - array_like, (`M`,)
            The spatial correlation length of each turbulent branch, where
            a Gaussian correlation function has been assumed.
            [Lz] = 1 / [Fs_spatial]

        f0 - array_like, (`M`,)
            The dominant temporal frequency of each turbulent branch
            in the medium's rest frame (i.e. `f0` is *not* attributable
            to a Doppler shift; see `v` for Doppler-shift effects).
            [f0] = [Fs]

        tau - array_like, (`M`,)
            The correlation time of each turbulent branch, where
            a Gaussian correlation function has been assumed.
            [tau] = 1 / [Fs]

        v - array_like, (`M`,)
            The lab-frame velocity of the medium through which
            the turbulent branch is propagating. Note that
            non-zero velocity produces a Doppler-shifted
            lab-frame frequency

                                df = xi * v

            where `xi` is the spatial frequency.
            [v] = [Fs] / [Fs_spatial]

        S0 - array_like, (`M`,)
            The relative peak autospectral density of each branch.
            [S0] = unitless

        noise_floor - float
            The noise floor of the random process's autospectral density.
            [noise_floor] = [self.x]^2 / [Fs] / [Fs_spatial], where
                `self.x` is the realization of the random process
                created at object initialization.

        seed - int or None
            Random seed used to initialize pseudo-random number generator.
            If `None`, generator is seeded from `/dev/urandom` or the clock.

        '''
        # Spatial-grid parameters
        self.Fs_spatial = Fs_spatial
        self.z0 = z0
        self.Nz = _largest_power_of_2_leq(Z * Fs_spatial)

        # Temporal-grid parameters
        self.Fs = Fs
        self.t0 = t0
        self.Nt = _largest_power_of_2_leq(T * Fs)

        # Turbulence parameters
        self.xi0 = np.array(xi0, dtype='float', ndmin=1)
        self.Lz = np.array(Lz, dtype='float', ndmin=1)
        self.f0 = np.array(f0, dtype='float', ndmin=1)
        self.tau = np.array(tau, dtype='float', ndmin=1)
        self.v = np.array(v, dtype='float', ndmin=1)
        self.S0 = np.array(S0, dtype='float', ndmin=1)

        #  Noise floor of the random process's autospectral density
        self._noise_floor = noise_floor

        # Get autospectral density of 2d random process
        res = self._getAutoSpectralDensity()
        self._xi = res[0]
        self._f = res[1]
        self._Sxx = res[2]

        # Get a space-time realization of the 2d random process
        self.x = self._getSignal(seed=seed)

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

        # Initialize autospectral density with zeros
        Sxx = np.zeros(ff.shape)

        # Iteratively incorporate autospectral density of each branch
        for branch_ind in np.arange(len(self.xi0)):
            # Parse turbulence parameters of branch
            xi0 = self.xi0[branch_ind]
            Lz = self.Lz[branch_ind]
            f0 = self.f0[branch_ind]
            tau = self.tau[branch_ind]
            v = self.v[branch_ind]
            S0 = self.S0[branch_ind]

            # Shape auto-spectral density, Sxx.
            xi_shaping = np.exp(-((np.pi * Lz * (xixi - xi0)) ** 2))

            df = v * xixi
            f_shaping = np.exp(-((np.pi * tau * (ff - f0 - df)) ** 2))

            Sxx += (S0 * xi_shaping * f_shaping)

        # Define peak autospectral density of turbulence to be unity
        Sxx /= np.max(Sxx)

        # Finally, incorporate noise floor
        Sxx += self._noise_floor

        return xi, f, Sxx

    def _getSignal(self, seed=None):
        '''Get a space-time realization of the 2d random process.

        Input parameters:
        -----------------
        seed - int or None
            Random seed used to initialize pseudo-random number generator.
            If `None`, generator is seeded from `/dev/urandom` or the clock.

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
        if seed is not None:
            np.random.seed(seed)

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
