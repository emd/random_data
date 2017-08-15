'''This module defines a class for estimating the 2-dimensional
autospectral density of a field. Temporal spectral estimates are
obtained through Welch's method of ensemble averaging overlapped,
windowed FFTs (i.e. nonparametric spectral estimation), while
spatial spectral estimates can be obtained through either
nonparametric (FFT-based) or parametric (Burg autoregression)
means.

'''


# Standard library imports
import string
import numpy as np
from matplotlib import mlab

# Intra-package imports
from .parametric import BurgAutoSpectralDensity
from .nonparametric import _plot_image
from ..array import SpatialCrossCorrelation


default_fourier_params = {
    'window': np.hanning
}

default_burg_params = {
    'p': 5,
    'Nxi': 100
}


class TwoDimensionalAutoSpectralDensity(object):
    '''A class for estimating the 2-dimensional autospectral density
    of a field given the complex-valued spatial correlation function,
    which e.g. can be computed from an array of measurements.

    Attributes:
    -----------

    Methods:
    --------
    Type `help(TwoDimensionalAutoSpectralDensity)` in the IPython console
    for a listing.

    '''
    def __init__(
            self, corr, spatial_method='burg',
            burg_params=default_burg_params,
            fourier_params=default_fourier_params):
        '''Create an instance of `TwoDimensionalAutoSpectralDensity` class.

        Input parameters:
        -----------------
        corr - :py:class:`SpatialCrossCorrelation
                <random_data.array.SpatialCrossCorrelation>` instance
            A `SpatialCrossCorrelation` instance that characterizes
            the complex-valued, spatial correlation function of the
            process under study. Typically, the correlation function
            is derived from cross-correlating an array of sensors
            that are all measuring the same field.

        spatial_method - string
            The method to use when estimating the spatial spectral
            density. Valid methods are:

                - 'fourier': use FFT-based estimation, or
                - 'burg': use Burg AR estimation.

            Specification of other values will raise a ValueError.

        burg_params - dict
            A dictionary containing the parameters of relevance
            for Burg spectral estimation. Valid dictionary keys
            are: {'p', 'Nxi'} where

            - p: int, order of Burg AR spectral-density estimate,
            - Nxi: int, number of points in the two-sided *spatial*
                spectral-density estimate at each frequency.

            See documentation for :py:class:`BurgAutoSpectralDensity
            <random_data.spectra.parametric.BurgAutoSpectralDensity>`
            for more information on these parameters.

        fourier_params - dict
            A dictionary containing the parameters of relevance
            for Fourier spectral estimation. Valid dictionary keys
            are: {'window'} where

            - window: tapering function to be applied to spatial
                dimension of correlation function prior to calculating
                the FFT; should be a function of the window length,
                such as `np.hanning`. If `None`, do not apply a window
                to the spatial dimension of the correlation function.
                Although the correlation function typically smoothly
                tapers to zero on its own, the application of a
                window can still suppress leakage, at the cost of
                marginally decreased resolution.

        '''
        # Check that user-provided `corr` is correct type
        if type(corr) is not SpatialCrossCorrelation:
            raise ValueError(
                '`corr` must be of type %s'
                % SpatialCrossCorrelation)

        # Parse requested spatial method
        self.spatial_method = string.lower(spatial_method)
        implemented_spatial_methods = ['fourier', 'burg']

        if self.spatial_method not in set(implemented_spatial_methods):
            raise ValueError(
                '`spatial_method` must be in %s'
                % implemented_spatial_methods)

        # Record important aspects of the computation
        self.Fs = corr.Fs
        self.Fs_spatial = 1. / (corr.separation[1] - corr.separation[0])

        self.Npts_per_real = corr.Npts_per_real
        self.Nreal_per_ens = corr.Nreal_per_ens
        self.Npts_overlap = corr.Npts_overlap
        self.Npts_per_ens = corr.Npts_per_ens
        self.Nens = corr.Nens

        self.detrend = corr.detrend
        self.window = corr.window

        self.f = corr.f
        self.df = corr.df

        self.t = corr.t
        self.dt = np.nan

        if self.spatial_method == 'burg':
            self.p = burg_params['p']
            self.Nxi = burg_params['Nxi']
        elif self.spatial_method == 'fourier':
            self.window_spatial = fourier_params['window']

        # Estimate autospectral density
        if self.spatial_method == 'burg':
            self._getBurgSpectralDensity(corr)
        elif self.spatial_method == 'fourier':
            self._getFourierSpectralDensity(corr)

    def _getFourierSpectralDensity(self, corr):
        '''Get 2-d autospectral density estimate by Fourier transforming
        the complex-valued, spatial correlation function `corr`. The
        autospectral density estimate is normalized such that integrating
        over the full spectrum yields the total power in the raw signal.

        '''
        Fs_spatial = 1. / (corr.separation[1] - corr.separation[0])
        Npts = len(corr.separation[corr._valid])

        if self.window_spatial is not None:
            w = self.window_spatial(Npts)

            # Normalize the window such that it has
            # equivalent power to a window of ones
            # with the same length.
            power_loss = np.sum(np.abs(w) ** 2) / Npts
            w /= np.sqrt(power_loss)
        else:
            w = np.ones(Npts)

        # The spectral density is the Fourier transform
        # of the correlation function. However, note that
        # the Fourier transform X(f) of signal x(t) is related
        # to the FFT of x(t) via:
        #
        #       X(f) = (1 / Fs) * FFT[x(t)](f),
        #
        # where Fs is the sampling rate of x(t).
        #
        # Further, recall that `corr.Gxy` has already been
        # Fourier analyzed in time, so we only need to
        # Fourier transform in space to obtain the estimated
        # autospectral density.
        self.Sxx = (1. / Fs_spatial) * np.fft.fftshift(
            np.fft.fft(w[:, np.newaxis] * corr.Gxy[corr._valid], axis=0),
            axes=0)

        # Construct grid for spatial spectral density.
        # Note that xi = (1 / wavelength) such that the
        # wavenumber k is related to xi via k = 2 * pi * xi.
        self.xi = np.fft.fftshift(np.fft.fftfreq(
            Npts, d=(1. / Fs_spatial)))

        self.dxi = self.xi[1] - self.xi[0]

        # Normalize spectral density to power in raw signal such that
        # integrating over all `np.abs(self.Sxx)` yields total power.
        self.Sxx *= self._getNormalizationPrefactor(corr)

        return

    def _getBurgSpectralDensity(self, corr):
        '''Get 2-d autospectral density estimate by using a Burg
        autoregression of  the complex-valued, spatial correlation
        function `corr`.

        '''
        # Initialize a real array to hold autospectral-density estimate,
        # as autospectral density should be real-valued
        self.Sxx = np.zeros((self.Nxi, len(self.f)))

        # Determine maximum valid separation, `Delta`, of points
        # in the correlation function
        separation = corr.separation[corr._valid]
        Delta = separation[-1] - separation[0]

        # Loop through frequency, estimating spatial autospectral density
        # at each using Burg autoregression. Note that this is done in a
        # somewhat round-about way. First, we use the Burg AR to estimate
        # the autospectral density of the *correlation function*,
        # S_{corr}(xi), which, by definition, is equal to
        #
        #       S_{corr}(xi) = (1 / Delta) * E[|FDFT(corr, Delta)|^2],
        #
        # in the limit that the maximum separation `Delta` in the correlation
        # function goes to infinity. Here, `E[...]` is the expectation-value
        # operator and `FDFT(x, T)` is the finite duration Fourier transform
        # of signal x(t); explicitly
        #
        #   FDFT(x, T) = \int_{0}^{T} dt [e^{-2j * pi * f * t} * x(t)]
        #
        # Note, however, that our desired autospectral density S_{xx}
        # is simply the Fourier transform of the correlation function, i.e.
        #
        #           S_{xx}(xi) = FT[corr(delta)](xi),
        #
        # where `FT[x(t)](f)` is the Fourier transform of signal `x(t)`.
        # Approximating the Fourier transform by the FDFT, we see that
        #
        #           S_{xx}(xi)    =    FT[corr(delta)](xi),
        #                      \approx FDFT(corr, Delta),
        #                      \approx [Delta * S_{corr}(xi)]^{1/2},
        #
        # where we have selected the positive root, as S_{xx}(xi) is
        # positive semi-definite.
        for find in np.arange(len(self.f)):
            # Burg AR spectral-density estimate for correlation function.
            # Don't waste time with normalization, as it will normalize
            # to power in the correlation function, not the raw signal.
            # We will handle normalization externally.
            asd_burg = BurgAutoSpectralDensity(
                self.p,
                corr.Gxy[corr._valid, find],
                Fs=self.Fs_spatial,
                Nf=self.Nxi,
                normalize=False)

            # Compute corresponding autospectral density of process
            # underlying the correlation function
            self.Sxx[:, find] = np.sqrt(Delta * asd_burg.Sxx)

        # Note that xi = (1 / wavelength) such that the
        # wavenumber k is related to xi via k = 2 * pi * xi.
        self.xi = asd_burg.f
        self.dxi = self.xi[1] - self.xi[0]

        # Normalize spectral density to power in raw signal such that
        # integrating over all `np.abs(self.Sxx)` yields total power.
        self.Sxx *= self._getNormalizationPrefactor(corr)

        return

    def _getNormalizationPrefactor(self, corr):
        '''Get multiplicative prefactor for spectral density such that
        integrating over all of the resulting spectral density yields
        the total power in the raw signal.

        '''
        ind0sep = np.where(corr.separation == 0)[0][0]
        signal_power = np.sum(np.abs(corr.Gxy[ind0sep, :])) * self.df
        integrated_power = np.sum(np.abs(self.Sxx)) * self.df * self.dxi

        return (signal_power / integrated_power)

    def plotSpectralDensity(self, xilim=None, flim=None, vlim=None,
                            cmap='viridis', interpolation='none', fontsize=16,
                            title=None, xlabel=r'$\xi$', ylabel='$f$',
                            cblabel=r'$|G_{xx}(\xi,f)|$',
                            ax=None, fig=None, geometry=111):
        'Plot magnitude of spectral density on log scale.'
        ax = _plot_image(
            self.xi, self.f, np.abs(self.Sxx.T),
            xlim=xilim, ylim=flim, vlim=vlim,
            norm='log', cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, cblabel=cblabel,
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)
