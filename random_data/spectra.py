'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
from matplotlib import mlab
from matplotlib.colors import LogNorm, Colormap
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

# Related 3rd-party imports
from .ensemble import Ensemble


class AutoSpectralDensity(object):
    '''A class for autospectral density characterization.

    For stationary signal `x`, the autospectral density `Sxx`
    is defined as

        Sxx(f) = lim_{T \\rightarrow \infty} (1 / T) E[ |X(f, T)| ^ 2 ]

    where X(f, T) is the finite-time Fourier transform of `x` and
    E[...] denotes the expectation value operator. The autopower
    within a spectral band fmin < f < fmax is given as

        Pxx(fmin < f < fmax) = \int_{fmin}^{fmax} Sxx(f) df

    The total autopower is obtained when fmin = -f_Ny and fmax = f_Ny,
    where f_Ny is the Nyquist frequency of the signals.

    For real-valued signal `x`, X(-f, T) = X*(f, T). Thus,
    only one side of the spectral density is uniquely determined.
    This class assumes that `x` is a real-valued signal such that
    a one-sided spectral density (f >= 0) is returned.
    The one sided spectral density is denoted `Gxx`. As `Gxx`
    is only defined for f >= 0, conservation of signal power requires

                    Gxx(f) = 2 Sxx(f), f >= 0, and
                    Gxx(f) = Sxx(f),   f == 0

    If `x` is a stationary signal, the entire sample record
    is referred to as an "ensemble". The spectral density is then
    *estimated* by splitting the ensemble into a number of
    (potentially overlapping) smaller segments, referred to as
    "realizations". These realizations are detrended, windowed, and
    FFT'd; the resulting FFTs are then averaged to obtain an
    estimate of the spectral density. The ensemble averaging is
    required for a statistically consistent definition of the
    spectral density: that is, as T -> \infty, the estimated
    spectral density only converges to the true spectral density
    when the ensemble average is computed. In particular, the
    random error in the estimate decreases as the number of
    realizations increases.

    This class allows for analysis of *nonstationary* signals.
    `x` is first split into a number of ensembles, where
    `x` is approximately stationary over the ensemble time.
    The estimation of the spectral density during each ensemble
    then proceeds as above.

    Attributes:
    -----------
    Below, `x` refers to the signal for which the autospectral
    density is computed, and `Fs` is the signal sampling rate.

    Gxx - array_like, (`L`, `M`)
        The one-sided (f >= 0) spectral density estimate.
        [Gxx] = [x]^2 / [Fs]

    f - array_like, (`L`,)
        The frequencies at which the spectral density has been estimated.
        [f] = [Fs]

    t - array_like, (`M`,)
        The temporal midpoint of each ensemble.
        [t] = 1 / [Fs]

    Fs - float
        The signal sampling rate, as specified at object initialization.
        [Fs] = arbitrary units

    Nreal_per_ens - int
        The number of realizations per ensemble used in the computation
        of the spectral density estimate `Gxx`. The random error in
        the spectral density estimate decreases ~ 1 / sqrt(`Nreal_per_ens`).

    Npts_per_real - int
        The number of sample points per realization used in the computation
        of the spectral density estimate `Gxx`.

    Npts_overlap - int
        The number of overlapping points between adjacent realizations
        in the computation of the spectral density estimate `Gxx`.

    detrend - string
        The function applied to each realization before taking the FFT.

    window - callable or ndarray
        The window applied to each realization before taking the FFT.

    Methods:
    --------
    Type `help(SpectralDensity)` in the IPython console for a listing.

    '''
    def __init__(self, x, Fs=1.0, t0=0.,
                 Tens=40960., Nreal_per_ens=10, fraction_overlap=0.5,
                 Npts_per_real=None, Npts_overlap=None,
                 detrend=None, window=mlab.window_hanning,
                 print_params=True, print_status=True):
        '''Create an instance of the `SpectralDensity` class.

        Input Parameters:
        -----------------
        x - array_like, (`N`,)
            The signal for which the autospectral density will be computed.
            [x] = arbitrary units

        Fs - float
            The sampling rate of `x`.
            If not specified, `Fs` is assigned a value of unity such that
            all frequencies are *normalized* to the sampling rate.
            [Fs] = arbitrary units

        t0 - float
            The initial time corresponding to `x[0]`.
            [t0] = 1 / [Fs]

        Tens - float
            The time window defining an ensemble. `Tens` determines the
            time resolution of the spectral density calculations,
            with larger `Tens` corresponding to reduced time resolution
            and increased frequency resolution.
            [Tens] = 1 / [Fs]

        Nreal_per_ens - int
            The number of realizations per ensemble. The random error in the
            spectral density estimate decreases as ~ 1 / sqrt(Nreal_per_ens).
            The frequency resolution `df` of the spectral density estimate
            is linearly related to the number of realizations.
            A ValueError is raised if not a positive integer.

        fraction_overlap - float
            The fractional overlap between adjacent realizations.
            0 =< `fraction_overlap` < 1, otherwise a ValueError is raised.

        Npts_per_real - int
            The number of sample points per realization. If None,
            `Tens` is used to compute `Npts_per_real` that is compatible
            with `Nreal_per_ens` and efficient FFT computation.
            If not None, `Tens` is ignored. A ValueError is raised
            if not a positive integer.

        Npts_overlap - int
            The number of overlapping sample points between adjacent
            realizations. If None, `fraction_overlap` sets the
            number of overlapping sample points. If not None,
            `fraction_overlap` is ignored. A ValueError is raised
            if not a positive integer or if greater than or equal to
            the number of points per realization.

        detrend - string
            The function applied to each realization before taking FFT.
            May be [ 'default' | 'constant' | 'mean' | 'linear' | 'none']
            or callable, as specified in :py:func: `csd <matplotlib.mlab.csd>`.

            *Warning*: Naively detrending (even with something as simple as
            `mean` or `linear` detrending) can introduce detrimental artifacts
            into the computed spectrum, so *no* detrending is the default.

        window - callable or ndarray
            The window applied to each realization before taking FFT,
            as specified in :py:func: `csd <matplotlib.mlab.csd>`.

        print_params - bool
            If True, print relevant spectral parameters to screen.

        print_status - bool
            If True, print percentage of ensembles whose spectra
            have been computed.

        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x):
            raise ValueError('`x` must be a real-valued signal!')

        # Determine properties for ensemble averaging
        ens = Ensemble(
            x, Fs=Fs, t0=t0, Tens=Tens,
            Nreal_per_ens=Nreal_per_ens, fraction_overlap=fraction_overlap,
            Npts_per_real=Npts_per_real, Npts_overlap=Npts_overlap)

        # Record important aspects of computation
        self.Fs = ens.Fs

        self.Npts_per_real = ens.Npts_per_real
        self.Nreal_per_ens = ens.Nreal_per_ens
        self.Npts_overlap = ens.Npts_overlap
        self.Npts_per_ens = ens.Npts_per_ens

        self.detrend = detrend
        self.window = window

        self.f = ens.f
        self.df = ens.df

        self.t = ens.t
        self.dt = ens.dt

        if print_params:
            self.printSpectralParams()

        # Perform spectral calculations
        self.Gxx = self.getSpectralDensity(x, print_status=print_status)

    def printSpectralParams(self):
        print '\ndt: %.6g' % self.dt
        print 'df: %.6g' % self.df
        print 'Npts_per_real: %i' % self.Npts_per_real
        print ('overlap: %.2f'
               % (np.float(self.Npts_overlap) / self.Npts_per_real))
        print 'Nreal_per_ens: %i' % self.Nreal_per_ens
        print 'detrend: %s' % self.detrend
        print 'window: %s' % self.window.func_name

        return

    def getSpectralDensity(self, x, print_status=False):
        'Get spectral density of provided signal.'
        return _spectral_density(
            x, x, self.Fs, len(self.f), len(self.t),
            self.Npts_per_real, self.Npts_overlap, self.Npts_per_ens,
            self.detrend, self.window,
            print_status=print_status, status_label='Gxx')

    def plotSpectralDensity(self, tlim=None, flim=None, vlim=None,
                            AC_coupled=True,
                            cmap='viridis', interpolation='none', fontsize=16,
                            title=None, xlabel='$t$', ylabel='$f$',
                            ax=None, fig=None, geometry=111):
        'Plot magnitude of spectral density on log scale.'
        if flim is None and AC_coupled:
            # Don't allow DC signal to influence color mapping
            flim = [self.f[1], self.f[-1]]

        ax = _plot_image(
            self.t, self.f, np.abs(self.Gxx),
            xlim=tlim, ylim=flim, vlim=vlim,
            norm='log', cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, cblabel='$|G_{xx}(f)|$',
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax


class CrossSpectralDensity(object):
    '''A class for cross-spectral density characterization.

    For stationary signals `x` and `y`, the cross-spectral density `Sxy`
    is defined as

        Sxy(f) = lim_{T \\rightarrow \infty} (1 / T) E[ X*(f, T) Y(f, T)]

    where X(f, T) and Y(f, T) are the finite-time Fourier transforms
    of `x` and `y`, respectively, * denotes complex conjugation, and
    E[...] denotes the expectation value operator. The crosspower
    within a spectral band fmin < f < fmax is given as

        Pxy(fmin < f < fmax) = \int_{fmin}^{fmax} Sxy(f) df

    The total crosspower is obtained when fmin = -f_Ny and fmax = f_Ny,
    where f_Ny is the Nyquist frequency of the signals.

    For real-valued signal `x`, X(-f, T) = X*(f, T). Thus, if both
    `x` and `y` are real-valued, Sxy(-f) = [Syx(f)]*, and only
    one side of the spectral density is uniquely determined.
    This class assumes that `x` and `y` are real-valued signals
    such that a one-sided spectral density (f >= 0) is returned.
    The one sided spectral density is denoted `Gxy`. As `Gxy`
    is only defined for f >= 0, conservation of signal power requires

                    Gxy(f) = 2 Sxy(f), f >= 0, and
                    Gxy(f) = Sxy(f),   f == 0

    If `x` and `y` are stationary signals, the entire sample record
    is referred to as an "ensemble". The spectral density is then
    *estimated* by splitting the ensemble into a number of
    (potentially overlapping) smaller segments, referred to as
    "realizations". These realizations are detrended, windowed, and
    FFT'd; the resulting FFTs are then averaged to obtain an
    estimate of the spectral density. The ensemble averaging is
    required for a statistically consistent definition of the
    spectral density: that is, as T -> \infty, the estimated
    spectral density only converges to the true spectral density
    when the ensemble average is computed. In particular, the
    random error in the estimate decreases as the number of
    realizations increases.

    This class allows for analysis of *nonstationary* signals.
    `x` and `y` are first split into a number of ensembles, where
    `x` and `y` are approximately stationary over the ensemble time.
    The estimation of the spectral density during each ensemble
    then proceeds as above.

    Attributes:
    -----------
    Below, `x` and `y` refer to the signals for which the cross-spectral
    density is computed, and `Fs` is the signal sampling rate.

    Gxy - array_like, (`L`, `M`)
        The one-sided (f >= 0) spectral density estimate.
        [Gxy] = [x] [y] / [Fs]

    f - array_like, (`L`,)
        The frequencies at which the spectral density has been estimated.
        [f] = [Fs]

    t - array_like, (`M`,)
        The temporal midpoint of each ensemble.
        [t] = 1 / [Fs]

    Fs - float
        The signal sampling rate, as specified at object initialization.
        [Fs] = arbitrary units

    Nreal_per_ens - int
        The number of realizations per ensemble used in the computation
        of the spectral density estimate `Gxy`. The random error in
        the spectral density estimate decreases ~ 1 / sqrt(`Nreal_per_ens`).

    Npts_per_real - int
        The number of sample points per realization used in the computation
        of the spectral density estimate `Gxy`.

    Npts_overlap - int
        The number of overlapping points between adjacent realizations
        in the computation of the spectral density estimate `Gxy`.

    detrend - string
        The function applied to each realization before taking the FFT.

    window - callable or ndarray
        The window applied to each realization before taking the FFT.

    Methods:
    --------
    Type `help(SpectralDensity)` in the IPython console for a listing.

    '''
    def __init__(self, x, y, Fs=1.0, t0=0.,
                 Tens=40960., Nreal_per_ens=10, fraction_overlap=0.5,
                 Npts_per_real=None, Npts_overlap=None,
                 detrend=None, window=mlab.window_hanning,
                 print_params=True, print_status=True):
        '''Create an instance of the `SpectralDensity` class.

        Input Parameters:
        -----------------
        x, y - array_like, (`N`,)
            The signals for which the cross-spectral density will be computed.
            A ValueError is raised if `x` and `y` contain a different number
            of samples. Note that `x` and `y` must be sampled at the *same*
            rate, `Fs`.
            [x] = arbitrary units
            [y] = arbitrary units, potentially different than [x]

        Fs - float
            The sampling rate of `x` and `y`.
            If not specified, `Fs` is assigned a value of unity such that
            all frequencies are *normalized* to the sampling rate.
            [Fs] = arbitrary units

        t0 - float
            The initial time corresponding to `x[0]` (and `y[0]`).
            [t0] = 1 / [Fs]

        Tens - float
            The time window defining an ensemble. `Tens` determines the
            time resolution of the spectral density calculations,
            with larger `Tens` corresponding to reduced time resolution
            and increased frequency resolution.
            [Tens] = 1 / [Fs]

        Nreal_per_ens - int
            The number of realizations per ensemble. The random error in the
            spectral density estimate decreases as ~ 1 / sqrt(Nreal_per_ens).
            The frequency resolution `df` of the spectral density estimate
            is linearly related to the number of realizations.
            A ValueError is raised if not a positive integer.

        fraction_overlap - float
            The fractional overlap between adjacent realizations.
            0 =< `fraction_overlap` < 1, otherwise a ValueError is raised.

        Npts_per_real - int
            The number of sample points per realization. If None,
            `Tens` is used to compute `Npts_per_real` that is compatible
            with `Nreal_per_ens` and efficient FFT computation.
            If not None, `Tens` is ignored. A ValueError is raised
            if not a positive integer.

        Npts_overlap - int
            The number of overlapping sample points between adjacent
            realizations. If None, `fraction_overlap` sets the
            number of overlapping sample points. If not None,
            `fraction_overlap` is ignored. A ValueError is raised
            if not a positive integer or if greater than or equal to
            the number of points per realization.

        detrend - string
            The function applied to each realization before taking FFT.
            May be [ 'default' | 'constant' | 'mean' | 'linear' | 'none']
            or callable, as specified in :py:func: `csd <matplotlib.mlab.csd>`.

            *Warning*: Naively detrending (even with something as simple as
            `mean` or `linear` detrending) can introduce detrimental artifacts
            into the computed spectrum, so *no* detrending is the default.

        window - callable or ndarray
            The window applied to each realization before taking FFT,
            as specified in :py:func: `csd <matplotlib.mlab.csd>`.

        print_params - bool
            If True, print relevant spectral parameters to screen.

        print_status - bool
            If True, print percentage of ensembles whose spectra
            have been computed.

        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('`x` and `y` must be real-valued signals!')

        if len(x) != len(y):
            raise ValueError('`x` and `y` must have the same length!')

        # Determine properties for ensemble averaging
        ens = Ensemble(
            x, Fs=Fs, t0=t0, Tens=Tens,
            Nreal_per_ens=Nreal_per_ens, fraction_overlap=fraction_overlap,
            Npts_per_real=Npts_per_real, Npts_overlap=Npts_overlap)

        # Record important aspects of computation
        self.Fs = ens.Fs

        self.Npts_per_real = ens.Npts_per_real
        self.Nreal_per_ens = ens.Nreal_per_ens
        self.Npts_overlap = ens.Npts_overlap
        self.Npts_per_ens = ens.Npts_per_ens

        self.detrend = detrend
        self.window = window

        self.f = ens.f
        self.df = ens.df

        self.t = ens.t
        self.dt = ens.dt

        if print_params:
            self.printSpectralParams()

        # Perform spectral calculations
        self.Gxy = self.getSpectralDensity(x, y, print_status=print_status)
        self.gamma2xy = self.getCoherence(x, y, print_status=print_status)
        self.theta_xy = self.getPhaseAngle()

    def printSpectralParams(self):
        print '\ndt: %.6g' % self.dt
        print 'df: %.6g' % self.df
        print 'Npts_per_real: %i' % self.Npts_per_real
        print ('overlap: %.2f'
               % (np.float(self.Npts_overlap) / self.Npts_per_real))
        print 'Nreal_per_ens: %i' % self.Nreal_per_ens
        print 'detrend: %s' % self.detrend
        print 'window: %s' % self.window.func_name

        return

    def getSpectralDensity(self, x, y, print_status=False):
        'Get spectral density of provided signals.'
        return _spectral_density(
            x, y, self.Fs, len(self.f), len(self.t),
            self.Npts_per_real, self.Npts_overlap, self.Npts_per_ens,
            self.detrend, self.window,
            print_status=print_status, status_label='Gxy')

    def getCoherence(self, x, y, print_status=False):
        '''Get (magnitude-squared) coherence of signals `x` and `y`.

        The magnitude-squared coherence function (gamma_{xy})^2
        of signals `x` and `y` is is defined as

            (gamma_{xy})^2 = [ | G_{xy}(f) |^2 ] / [ G_{xx}(f) G_{yy}(f) ]

        where `G_{xy}(f)` is the cross-spectral density of `x` and `y` and
        `G_{xx}(f)` (`G_{yy}(f)`) is the autospectral density of `x` (`y`).

        For real-valued `x` and `y`, `G_{xx}(f)` and `G_{yy}(f)` are
        also real-valued. Thus, the magnitude-squared coherence function
        is real-valued. Further, for all `f`,

                            0 <= (gamma_{xy})^2 <= 1

        '''
        Gxx = _spectral_density(
            x, x, self.Fs, len(self.f), len(self.t),
            self.Npts_per_real, self.Npts_overlap, self.Npts_per_ens,
            self.detrend, self.window,
            print_status=print_status, status_label='Gxx')

        Gyy = _spectral_density(
            y, y, self.Fs, len(self.f), len(self.t),
            self.Npts_per_real, self.Npts_overlap, self.Npts_per_ens,
            self.detrend, self.window,
            print_status=print_status, status_label='Gyy')

        num = (np.abs(self.Gxy) ** 2)
        den = Gxx * Gyy

        return num / den

    def getPhaseAngle(self, unwrap=False):
        '''Get phase angle `theta_xy` (in radians) of spectral density `Gxy`.
        If `unwrap` is False, the returned angle will be between [-pi, pi].

        '''
        if unwrap:
            self.theta_xy = np.unwrap(np.angle(self.Gxy), axis=-1)
        else:
            self.theta_xy = np.angle(self.Gxy)

        return np.copy(self.theta_xy)

    def plotSpectralDensity(self, tlim=None, flim=None, vlim=None,
                            AC_coupled=True,
                            cmap='viridis', interpolation='none', fontsize=16,
                            title=None, xlabel='$t$', ylabel='$f$',
                            ax=None, fig=None, geometry=111):
        'Plot magnitude of spectral density on log scale.'
        if flim is None and AC_coupled:
            # Don't allow DC signal to influence color mapping
            flim = [self.f[1], self.f[-1]]

        ax = _plot_image(
            self.t, self.f, np.abs(self.Gxy),
            xlim=tlim, ylim=flim, vlim=vlim,
            norm='log', cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, cblabel='$|G_{xy}(f)|$',
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotCoherence(self, tlim=None, flim=None, vlim=None,
                      cmap='viridis', interpolation='none', fontsize=16,
                      title=None, xlabel='$t$', ylabel='$f$',
                      ax=None, fig=None, geometry=111):
        'Plot magnitude squared coherence on linear scale.'
        ax = _plot_image(
            self.t, self.f, self.gamma2xy,
            xlim=tlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel='$\gamma_{xy}^2$',
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotPhaseAngle(self, gamma2xy_threshold=0.5, Gxy_threshold=0.,
                       theta_min=-np.pi, dtheta=(np.pi / 4),
                       tlim=None, flim=None,
                       cmap='RdBu', interpolation='none', fontsize=16,
                       title=None, xlabel='$t$', ylabel='$f$',
                       mode_number=False,
                       ax=None, fig=None, geometry=111):
        '''Plot phase angle `theta` if magnitude-squared coherence is
        greater than or equal to `gamma2xy_threshold`, cross-spectral
        density amplitude is greater than or equal to `Gxy_threshold`,
        and `theta` satisfies

                theta_min <= theta < [theta_min + (2 * pi)]

        If `dtheta` divides (2 * pi) into an *integer* number of bins,
        the plotted phase angles will be displayed with resolution `dtheta`;
        that is, plotted phase angles will fall within bins of width `dtheta`
        centered on

                    theta_i = theta_min + (i * dtheta)

        with 0 <= i < N, and N = (2 * pi) / dtheta.

        If `dtheta` does *not* divide (2 * pi) into an integer number
        of bins, `dtheta` will be redefined as the next largest
        value that does divide (2 * pi) into an integer number of bins;
        the above discussion about the bin width and centering for the
        plotted phase angles then applies with this re-defined `dtheta`.

        If `mode_number` is True, the plotted phase angles will be
        normalized to `dtheta`, producing a plot of mode number `n`
        rather than phase angle.

        '''
        theta_max = theta_min + (2 * np.pi)

        # Ensure that `dtheta` divides (2 * pi) into an integer number of bins
        dtheta = _next_largest_divisor_for_integer_quotient(2 * np.pi, dtheta)

        # The plotted phase angles will fall within bins of width `dtheta`
        # centered on
        #
        #           theta_i = theta_min + (i * dtheta)
        #
        # where 0 <= i < N and N = (2 * pi) / dtheta.
        # Each bin centerpoint will have its own colorbar tick in `cbticks`.
        cbticks = np.arange(theta_min, theta_max, dtheta)

        # Get "discrete" colormap, with a distinct color corresponding
        # to each value in `cbticks`
        cmap = plt.get_cmap(cmap, len(cbticks))

        # However, the bins also have width `dtheta` such that
        # the colorbar boundaries should correspond to
        #
        #        lower bound (0):      theta_min - (0.5 * dtheta)
        #        (1):                  theta_min + (0.5 * dtheta)
        #        (2):                  theta_min + (1.5 * dtheta)
        #        ...
        #        upper bound (N + 1):  theta_max - (0.5 * dtheta)
        #
        # This is easily accomplished by setting the minimum and maximum
        # values to represent in the image as follows:
        vlim = np.array([
            theta_min - (0.5 * dtheta),
            theta_max - (0.5 * dtheta)])

        # Now, "wrap" the phase angles onto the specified domain
        theta_xy = wrap(self.theta_xy, vlim[0], vlim[1])

        # Normalize phase angle to `dtheta` to obtain plot of mode number `n`
        if mode_number:
            theta_xy /= dtheta
            vlim /= dtheta
            cbticks = (cbticks / dtheta).astype('int')
            cblabel = '$n$'
        else:
            cblabel='$\\theta_{xy}$'

        # Finally, only consider phase angles from regions whose
        # magnitude-square coherence is greater than or equal to `threshold`
        theta_xy = np.ma.masked_where(
            np.logical_or(
                self.gamma2xy < gamma2xy_threshold,
                np.abs(self.Gxy) < Gxy_threshold),
            theta_xy)

        ax = _plot_image(
            self.t, self.f, theta_xy,
            xlim=tlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel=cblabel, cbticks=cbticks,
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax


def _spectral_density(x, y, Fs, Nf, Nens,
                      Npts_per_real, Npts_overlap, Npts_per_ens,
                      detrend, window,
                      print_status=False, status_label=''):
    'Get spectral density of provided signals.'
    same_data = x is y

    # Initialize spectral density array
    if not same_data:
        # Cross-spectral density is intrinsically complex-valued, so
        # we must initialize the spectral density as a complex-valued
        # array to avoid loss of information
        Gxy = np.zeros([Nf, Nens], dtype=np.complex128)
    else:
        # Autospectral density is intrinsically real-valued
        # (assuming `x` is real-valued), so we don't need the
        # overhead of a complex-valued array
        Gxy = np.zeros([Nf, Nens])

    if print_status:
        print ''

    # Loop over successive ensembles
    for ens in np.arange(Nens):
        # Create a slice corresponding to current ensemble
        ens_start = ens * Npts_per_ens
        ens_stop = (ens + 1) * Npts_per_ens
        sl = slice(ens_start, ens_stop)

        if same_data:
            Gxy[:, ens] = mlab.psd(
                x[sl], Fs=Fs,
                NFFT=Npts_per_real, noverlap=Npts_overlap,
                detrend=detrend, window=window)[0]
        else:
            Gxy[:, ens] = mlab.csd(
                x[sl], y[sl], Fs=Fs,
                NFFT=Npts_per_real, noverlap=Npts_overlap,
                detrend=detrend, window=window)[0]

        if print_status:
            print ('%s percent complete: %.1f \r'
                   % (status_label, (100 * np.float(ens + 1) / Nens))),

    if print_status:
        print ''

    return Gxy


def _plot_image(x, y, z,
                xlim=None, ylim=None, vlim=None,
                norm=None, cmap='viridis', interpolation='none',
                title=None, xlabel=None, ylabel=None, fontsize=16,
                cblabel=None, cbticks=None,
                ax=None, fig=None, geometry=111):
    '''Create an image of z(y, x).

    Parameters:
    -----------
    x - array_like, (`M`,)
        The x-axis of image. It is assumed that the x-values correspond
        to the midpoints of bins in the x-dimension (e.g. the midpoint
        of the ensembles in
            :py:class:`SpectralDensity <random_data.spectra.SpectralDensity>`)

    y - array_like, (`N`,)
        The y-axis of image. It is assumed that the y-values correspond
        to discrete samples of a function, such as the discrete frequencies
        of a discrete Fourier transform.

    z - array_like, (`N`, `M`)
        The array containing the image values.

    xlim - array_like, (2,)
        The minimum and maximum values of `x` to display.

    ylim - array_like, (2,)
        The minimum and maximum values of `y` to display.

    vlim - array_like, (2,)
        The minimum and maximum values of `z` to display.

    norm - string or None
        If `log`, display image on logarithmic scale;
        otherwise, display image on linear scale.

    cmap - string
        Colormap used for image. Default matplotlib colormaps
        are found in :py:module:`cm <matplotlib.cm>`.

    interpolation - string
        Interpolation method to be used by :py:function `imshow.
        <matplotlib.pyplot.imshow>`. Examples of each interpolation
        scheme are displayed here:

            http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html

        and the difference between 'none' and 'nearest' is demonstrated here:

            http://matplotlib.org/examples/images_contours_and_fields/interpolation_none_vs_nearest.html

    title, xlabel, ylabel, cblabel - string
        Titles of respective objects in image.

    fontsize - int
        Size of font in titles, labels, etc.

    ax - :py:class:`AxesSubplot <matplotlib.axes._subplots.AxesSubplot>`
        instance corresponding to the axis (i.e. "subplot") where
        the image will be drawn. `ax` will (obviously) be
        modified by this method. If an axis instance is not provided,
        an axis will automatically be created.

    fig - :py:class:`Figure <matplotlib.figure.Figure>` instance
        If an axis instance is *not* provided, one can provide
        a figure instance (and an axis `geometry`, describing the
        location of the axis instance in the figure) to control
        which window is plotted in. If a figure instance is not
        provided (and axis instance is also not provided),
        a figure instance will be created with the next available
        window number.

    geometry - int, or tuple
        If an axis instance is *not* provided, `geometry` determines
        the location of the axis instance in the provided or created
        figure. The standard matplotlib subplot geometry indexing is
        used (see `<matplotlib.pyplot.subplot>` for more information).

    Returns:
    --------
    ax - :py:class:`AxesSubplot <matplotlib.axes._subplots.AxesSubplot>`
        instance corresponding to the axis (i.e. "subplot") where
        the spectrogram will be drawn. This is either identical to
        the axis instance used during the call or, if an axis instance
        was not provided, the axis instance created during the call.

    '''
    # Determine (x, y) extent of image
    if xlim is not None:
        xlim = np.sort(xlim)
        xind = np.where(np.logical_and(x >= xlim[0], x <= xlim[1]))[0]
    else:
        xind = np.arange(len(x))

    if ylim is not None:
        ylim = np.sort(ylim)
        yind = np.where(np.logical_and(y >= ylim[0], y <= ylim[1]))[0]
    else:
        yind = np.arange(len(y))

    dx = x[1] - x[0]

    extent = (x[xind[0]] - (0.5 * dx),
              x[xind[-1]] + (0.5 * dx),
              y[yind[0]],
              y[yind[-1]])

    # If an axis instance is not provided, create one
    if ax is None:
        # If, in addition, a figure instance is not provided,
        # create a new figure
        if fig is None:
            fig = plt.figure()
        # Create axis with desired subplot geometry
        ax = fig.add_subplot(geometry)

    if vlim is not None:
        vlim = np.sort(vlim)
    else:
        vlim = [np.min(z[yind, :][:, xind]), np.max(z[yind, :][:, xind])]

    if norm == 'log':
        norm = LogNorm()

    # Ensure that specified colormap is available
    if not isinstance(cmap, Colormap) and (cmap not in plt.colormaps()):
        cmap_backup = 'Purples'
        print ("\nThe '%s' colormap is not available; falling back to '%s'\n"
               % (cmap, cmap_backup))
        cmap = cmap_backup

    # Create plot
    im = ax.imshow(np.flipud(z[yind, :][:, xind]),
                   extent=extent, aspect='auto',
                   vmin=vlim[0], vmax=vlim[1],
                   norm=norm, cmap=cmap, interpolation=interpolation)

    # Colorbar
    if norm == 'log':
        format = LogFormatter(labelOnlyBase=True)
    else:
        format = None

    cb = plt.colorbar(im, format=format, ticks=cbticks,
                      ax=ax, orientation='horizontal')

    # Labeling
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if cblabel is not None:
        cb.set_label(cblabel, fontsize=fontsize)

    return ax


def wrap(theta, theta_min, theta_max):
    '''Wrap array `theta` between `theta_min` and `theta_max`.
    This is the inverse operation to :py:function: `unwrap <numpy.unwrap>`.

    '''
    full_cycle = theta_max - theta_min

    return ((theta - theta_min) % full_cycle) + theta_min


def _next_largest_divisor_for_integer_quotient(dividend, divisor):
    '''Return `divisor` or next largest value that yields an
    integer quotient when dividing into `dividend`.

    '''
    integer_quotient = np.int(np.float(dividend) / divisor)
    return dividend / integer_quotient


def _test_phase_angle(
        gamma2xy_threshold=0.5, Gxy_threshold=0.,
        theta_min=-np.pi, dtheta=(np.pi / 4),
        flim=[10e3, 100e3],
        cmap='RdBu',
        mode_number=False,
        Tens=5e-3, Nreal_per_ens=10):
    '''This routine plots the phase angle of several test cases
    to ensure that the phase angle is correctly represented
    by the methods in `CrossSpectralDensity`. Each test case
    lies near the lower or upper boundary of a phase angle bin,
    and good behavior at the bin's extrema implies good behavior
    throughout the rest of the bin's interior. Note that several
    figures are generated!

    '''
    # Create some uncorrelated noise
    from .signals import RandomSignal
    sig1 = RandomSignal(4e6, 0.1, fc=100e3, pole=2)
    sig2 = RandomSignal(4e6, 0.1, fc=100e3, pole=2)

    # Coherent signal amplitude
    A0 = 1e-3

    # Signal will have a linearly *ramping* frequency
    f0 = 50e3
    f1 = 75e3
    m = (f1 - f0) / (2 * (sig1.t[-1] - sig1.t[0]))
    f = f0 + (m * sig1.t)

    # Check that plotted phase angle is correct for specified phase angles
    theta_max = theta_min + (2 * np.pi)
    dtheta = _next_largest_divisor_for_integer_quotient(2 * np.pi, dtheta)
    theta = np.arange(theta_min, theta_max, dtheta)

    # Check lower boundary for each phase angle
    for i, th0 in enumerate(theta):
        # Ideal lower boundary of phase angle is at `theta` - (0.5 * `dtheta`),
        # but we select 0.45 to give a bit of head room due to noise etc.
        th = th0 - (0.45 * dtheta)
        y1 = sig1.x + (A0 * np.cos(2 * np.pi * f * sig1.t))
        y2 = sig2.x + (A0 * np.cos((2 * np.pi * f * sig2.t) + th))

        csd = CrossSpectralDensity(
            y1, y2, Fs=sig1.Fs, t0=sig1.t[0],
            Tens=Tens, Nreal_per_ens=Nreal_per_ens,
            print_params=False, print_status=False)

        # Plot cross-spectral spectral density amplitude *once*
        # so that it is easy to specify relevant alternative values
        # for `Gxy_threshold`
        if i == 0:
            csd.plotSpectralDensity(flim=flim)

        if mode_number:
            title='Lower bound, n = %i' % np.round(th0 / dtheta)
        else:
            title='Lower bound, theta = %.3f' % th0

        csd.plotPhaseAngle(
            gamma2xy_threshold=gamma2xy_threshold,
            Gxy_threshold=Gxy_threshold,
            theta_min=theta_min, dtheta=dtheta,
            flim=flim,
            cmap=cmap,
            title=title,
            mode_number=mode_number)

    # Check upper boundary for each phase angle
    for i, th0 in enumerate(theta):
        # Ideal upper boundary of phase angle is at `theta` + (0.5 * `dtheta`),
        # but we select 0.45 to give a bit of head room due to noise etc.
        th = th0 + (0.45 * dtheta)
        y1 = sig1.x + (A0 * np.cos(2 * np.pi * f * sig1.t))
        y2 = sig2.x + (A0 * np.cos((2 * np.pi * f * sig2.t) + th))

        csd = CrossSpectralDensity(
            y1, y2, Fs=sig1.Fs, t0=sig1.t[0],
            Tens=Tens, Nreal_per_ens=Nreal_per_ens,
            print_params=False, print_status=False)

        if mode_number:
            title='Upper bound, n = %i' % np.round(th0 / dtheta)
        else:
            title='Upper bound, theta = %.3f' % th0

        csd.plotPhaseAngle(
            gamma2xy_threshold=gamma2xy_threshold,
            Gxy_threshold=Gxy_threshold,
            theta_min=theta_min, dtheta=dtheta,
            flim=flim,
            cmap=cmap,
            title=title,
            mode_number=mode_number)

    return
