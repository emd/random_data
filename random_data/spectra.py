'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
from matplotlib import mlab
from scipy.signal import fftconvolve, convolve2d
from matplotlib.colors import LogNorm
from matplotlib.mlab import specgram
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

# Related 3rd-party imports
from event_manager.for_matplotlib import FigureList


class SpectralDensity(object):
    '''A class for spectral density characterization.

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

    If `y` == `x`, substitute y -> x in all of the documentation
    below (e.g. `Gxy` -> `Gxx`). In such cases, the computed spectral
    density is referred to as the autospectral density. For a
    real-valued signal `x`, the corresponding autospectral density
    `Gxx` is, by definition, also real-valued.

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
    realizations increases; explicitly

        random error in estimated Gxy = 1 / sqrt(# of realizations)

    This class allows for analysis of *nonstationary* signals.
    `x` and `y` are first split into a number of ensembles, where
    `x` and `y` are approximately stationary over the ensemble time.
    The estimation of the spectral density during each ensemble
    then proceeds as above.

    Attributes:
    -----------
    Below, `x` (`y`) refers to the signal(s) for which the spectral density
    is computed, and `Fs` is the signal sampling rate.

    Gxy - array_like, (`L`, `M`)
        The one-sided (f >= 0) spectral density estimate.
        [Gxy] = [x] [y] / [Fs]

    f - array_like, (`L`,)
        The frequencies at which the spectral density has been estimated.
        [f] = [Fs]

    t - array_like, (`M`,)
        The temporal midpoint of each ensemble.
        [t] = 1 / [Fs]

    Nreal_per_ens - int
        The number of realizations per ensemble used in the computation
        of the spectral density estimate `Gxy`. The random error in
        the spectral density estimate decreases as 1 / sqrt(`Nreal_per_ens`).

    Npts_per_real - int
        The number of sample points per realization used in the computation
        of the spectral density estimate `Gxy`.

    Npts_overlap - int
        The number of overlapping points between adjacent realizations
        in the computation of the spectral density estimate `Gxy`.

    random_error - float
        The fractional random error in the spectral density estimate `Gxy`,
        as determined from the number of realizations averaged over
        per ensemble.

    kind - string
        The kind of spectral density: {'autospectral', 'cross-spectral'}

    detrend - string
        The function applied to each realization before taking the FFT.

    window - callable or ndarray
        The window applied to each realization before taking the FFT.

    Methods:
    --------
    Type `help(SpectralDensity)` in the IPython console for a listing.

    '''
    def __init__(self, x, y=None, Fs=1.0, t0=0.,
                 Tens=40960., Nreal_per_ens=10, fraction_overlap=0.5,
                 Npts_per_real=None, Npts_overlap=None,
                 detrend='linear', window=mlab.window_hanning):
        '''Create an instance of the `SpectralDensity` class.

        Input Parameters:
        -----------------
        x (, y) - array_like, (`N`,)
            The signal(s) from which the spectral density will be computed.
            If `y` is `None`, the autospectral density of `x` is computed.
            If `y` is not `None`, the cross-spectral density of `x` and `y`
            is computed. Further, if `y` is not `None`, a ValueError is
            raised if `x` and `y` contain a different number of samples.
            Note that `x` and `y` must be sampled at the *same* rate, `Fs`.
            [x] = arbitrary units
            ([y] = arbitrary units, potentially different than [x])

        Fs - float
            The sampling rate of `x` (and `y`, if specified).
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
            spectral density estimate decreases as 1 / sqrt(Nreal_per_ens).
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

        window - callable or ndarray
            The window applied to each realization before taking FFT,
            as specified in :py:func: `csd <matplotlib.mlab.csd>`.

        '''
        # Only real-valued signals are expected/supported at the moment
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('`x` and `y` must be real-valued signals!')

        # Determine if we are computing autospectral density or
        # cross-spectral density. If computing cross-spectral density,
        # ensure that both signals are the same length.
        if y is None:
            self.kind = 'autospectral'
        else:
            if len(x) != len(y):
                raise ValueError('`x` and `y` must have the same length!')

            if y is x:
                self.kind = 'autospectral'
            else:
                self.kind = 'cross-spectral'

        # Ensure specified number of realizations is valid
        if Nreal_per_ens < 1 or not isinstance(Nreal_per_ens, int):
            raise ValueError('`Nreal_per_ens` must be a positive integer!')

        # Determine number of sample points to use per realization
        if Npts_per_real is None:
            Npts_per_real = self._getNumPtsPerReal(
                Fs, Tens, Nreal_per_ens, fraction_overlap)
        elif Npts_per_real < 1 or not isinstance(Npts_per_real, int):
            raise ValueError('`Npts_per_real` must be a positive integer!')

        # Determine number of overlapping points between adjacent realizations
        if Npts_overlap is None:
            if fraction_overlap >= 0 and fraction_overlap < 1:
                Npts_overlap = np.int(fraction_overlap * Npts_per_real)
            else:
                raise ValueError('`fraction_overlap` must be between 0 and 1!')
        else:
            if Npts_overlap < 1 or not isinstance(Npts_overlap, int):
                raise ValueError('`Npts_overlap` must be a positive integer!')
            elif Npts_overlap >= Npts_per_real:
                raise ValueError('`Npts_overlap` must be < `Npts_per_real`!')

        # Record important aspects of computation
        self.Npts_per_real = Npts_per_real
        self.Nreal_per_ens = Nreal_per_ens
        self.Npts_overlap = Npts_overlap
        self.detrend = detrend
        self.window = window
        self.random_error = 1 / np.sqrt(Nreal_per_ens)

        # Generate frequency and time base of spectral density estimate
        self.f = self.getFrequencies(Fs)
        self.t = self.getTimes(x, Fs, t0)

        # Determine resolution in frequency and time, if applicable
        try:
            self.df = self.f[1] - self.f[0]
        except IndexError:
            self.df = np.nan
        try:
            self.dt = self.t[1] - self.t[0]
        except IndexError:
            self.dt = np.nan

        self.Gxy = self.getSpectralDensity(x, y, Fs)

    def _getNumPtsPerReal(self, Fs, Tens, Nreal_per_ens, fraction_overlap):
        '''Get number of points per realization. This directly
        determines the frequency resolution `df` of the resulting
        spectral density estimate.

        As the number of points must be a whole number, there will
        generally be round-off error such that the resulting ensemble
        time window is slightly different than the specified `Tens`.
        Further, to ensure efficient FFT computation, the number of
        points per ensemble is required to be a power of two,
        potentially leading to even larger differences between
        the resulting ensemble time window and the spec'd `Tens`.

        This function should be called *before* utilizing other
        functions related to the FFT, such as `_getNumPtsPerEns(...)`.

        '''
        # If a given ensemble of length `Tens` consists of `Nreal_per_ens`
        # (potentially overlapping) realizations, each of length `Treal`,
        # then
        #
        #  Tens = Treal * {1 + [(Nreal_per_ens - 1) * (1 - fraction_overlap)]}
        #
        # where `fraction_overlap` is the fractional overlap between
        # adjacent realizations. `Treal` is easily solved for.
        denominator = 1 + ((Nreal_per_ens - 1) * (1 - fraction_overlap))
        Treal = np.float(Tens) / denominator  # avoid integer division!

        Npts_per_real = np.int(np.round(Treal * Fs))

        # TODO: Generalize `_closest_power_of_2(...)` to allow for
        # additional small factors (e.g. {2, 3, 5, ...})
        return _closest_power_of_2(Npts_per_real)

    def _getNumPtsPerEns(self):
        'Get number of points per ensemble.'
        # In a given ensemble, there are `Nreal_per_ens` (potentially
        # overlapping) realizations. Each of these realizations
        # contains `Npts_per_real` sample points. Thus, the first
        # realization contributes `Npts_per_real` sample points.
        Npts_per_ens = self.Npts_per_real

        # If there `Npts_overlap` overlapping sample points between
        # adjacent realizations, the remaining (`Nreal_per_ens` - 1)
        # realizations each contribute (`Npts_per_real` - `Npts_overlap`)
        # distinct sample points.
        distinct_points_per_real = self.Npts_per_real - self.Npts_overlap
        Npts_per_ens += ((self.Nreal_per_ens - 1) * distinct_points_per_real)

        return Npts_per_ens

    def getFrequencies(self, Fs):
        '''Get frequencies at which spectral density is estimated.
        It is assumed that a one-sided spectral density is computed.

        '''
        return np.fft.rfftfreq(self.Npts_per_real, d=(1. / Fs))

    def getTimes(self, x, Fs, t0):
        '''Get time base for spectral density estimate that is consistent
        with the number of points per realization. The returned time base
        corresponds to the midpoint of each ensemble.

        '''
        # The ensemble forms the basic unit/discretization of time
        # for the computed spectral density estimate, so determine
        # the number of points in an ensemble and the corresponding
        # time window. In general, this time window will slightly
        # differ from that specified during object initialization;
        # this is to ensure efficient FFT computation.
        Npts_per_ens = self._getNumPtsPerEns()
        Tens = Npts_per_ens / np.float(Fs)  # avoid integer division!

        # Determine the number of *whole* ensembles in the data record
        # (Disregard fractional ensemble at the end of the data, if present)
        Nens = np.int(len(x) / Npts_per_ens)

        # The returned time base corresponds to the midpoint of each ensemble
        return t0 + (Tens * np.arange(0.5, Nens, 1))

    def getSpectralDensity(self, x, y, Fs):
        'Get spectral density of provided signals.'
        # Initialize spectral density array
        if self.kind == 'cross-spectral':
            # Cross-spectral density is intrinsically complex-valued, so
            # we must initialize the spectral density as a complex-valued
            # array to avoid loss of information
            Gxy = np.zeros([len(self.f), len(self.t)], dtype=np.complex128)
        elif self.kind == 'autospectral':
            # Autospectral density is intrinsically real-valued
            # (assuming `x` is real-valued), so we don't need the
            # overhead of a complex-valued array
            Gxy = np.zeros([len(self.f), len(self.t)])
        else:
            raise ValueError('`kind` = %s is not supported.' % self.kind)

        # Number of points per ensemble
        Npts_per_ens = self._getNumPtsPerEns()

        # Loop over successive ensembles
        for ens in np.arange(len(self.t)):
            # Create a slice corresponding to current ensemble
            ens_start = ens * Npts_per_ens
            ens_stop = (ens + 1) * Npts_per_ens
            sl = slice(ens_start, ens_stop)

            # While it would be nice to remove the conditional below
            # and simply use `mlab.csd(...)` for both autospectral and
            # cross-spectral density calculations, `mlab.csd(...)` returns
            # a complex-valued array. If computing the autospectral density,
            # this complex-valued array is is cast as a real value, and
            # Python raises a warning about information loss. For
            # real-valued `x`, the autospectral density is, by definition,
            # also real-valued, and there is no true information loss.
            #
            # However, if users are not aware that this particular warning
            # is moot, they may find it unsettling. Further, and perhaps
            # more important, if the users becomes accustomed to this
            # warning statement, they may ignore a similar warning that
            # is legitimately raised at another point in the code.
            #
            # In contrast, `mlab.psd(...)` explicitly returns a real-valued
            # array. Python realizes this is intentional and does not complain.
            # For these reasons, autospectral and cross-spectral density
            # calculations are treated "differently" below.
            if self.kind == 'autospectral':
                Gxy[:, ens] = mlab.psd(
                    x[sl], Fs=Fs,
                    NFFT=self.Npts_per_real, noverlap=self.Npts_overlap,
                    detrend=self.detrend, window=self.window)[0]
            else:
                Gxy[:, ens] = mlab.csd(
                    x[sl], y[sl], Fs=Fs,
                    NFFT=self.Npts_per_real, noverlap=self.Npts_overlap,
                    detrend=self.detrend, window=self.window)[0]

        return Gxy

    def getPhaseAngle(self):
        'Get phase angle `theta_xy` of spectral density `Gxy`.'
        if self.kind == 'autospectral':
            # By definition, autospectral density is real-valued
            # for real-valued signal `x`
            self.theta_xy = 0
            print '\nAutospectral density of real signal is also real.\n'
        else:
            # Unwrap phase along time dimension to avoid 2 * pi discontinuities
            self.theta_xy = np.unwrap(np.angle(self.Gxy), axis=-1)

        return

    def getCoherence(self):
        'Get the magnitude squared coherence function, `gamma2xy`.'

        pass

    def plotSpectralDensity(self, tlim=None, flim=None, vlim=None,
                            cmap='Purples', fontsize=16,
                            title=None, xlabel='$t$', ylabel='$f$',
                            ax=None, fig=None, geometry=111):
        'Plot magnitude of spectral density on log scale.'
        if self.kind == 'autospectral':
            cblabel = '$G_{xx}(f)$'
        else:
            cblabel = '$|G_{xy}(f)|$'

        ax = _plot_image(
            self.t, self.f, np.abs(self.Gxy),
            xlim=tlim, ylim=flim, vlim=vlim,
            norm='log', cmap=cmap, fontsize=fontsize,
            title=title, xlabel=xlabel, ylabel=ylabel, cblabel=cblabel,
            ax=ax, fig=fig, geometry=geometry)

        return ax


def _closest_power_of_2(x):
    'Get the number expressible as a power of 2 that is closest to `x`.'
    exponent = np.log2(x)                   # exact
    exponent = np.int(np.round(exponent))   # for nearest power of 2
    return 2 ** exponent


class Coherence(object):
    'A class for magnitude squared coherence characterization.'
    def __init__(self, Gxy=None, Gxx=None, Gyy=None, x=None, y=None):
        '''Create an instance of the `Coherence` class.

        Input parameters:
        -----------------
        Gxy - array_like (`M`, `N`)
            The cross-spectral density of signals `x` and `y`
            [Gxy] = [x] [y] / [Fs], where `Fs` is the signal sampling rate

        Gxx, Gyy - array_like (`M`, `N`)
            The autospectral densities of signals `x` and `y`, respectively.
            [Gxx] = [Gyy] = [Gxy]

        '''
        pass

    def getCoherence(self):
        pass


def _plot_image(x, y, z,
                xlim=None, ylim=None, vlim=None,
                norm=None, cmap='Purples', fontsize=16,
                title=None, xlabel=None, ylabel=None, cblabel=None,
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

    title, xlabel, ylabel, cblabel - string
        Titles of respective objects in image.

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
    else:
        norm = None

    # Create plot
    im = ax.imshow(np.flipud(z[yind, :][:, xind]),
                   extent=extent, aspect='auto',
                   vmin=vlim[0], vmax=vlim[1],
                   norm=norm, cmap=cmap)

    # Colorbar
    if norm == 'log':
        format = LogFormatter(labelOnlyBase=True)
    else:
        format = None

    cb = plt.colorbar(im, format=format,
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


class Spectrogram(object):
    '''Spectrogram class.

    Attributes:
    -----------
    Gxx - array, (`M`, `N`)
        Estimates of one-sided power spectral density as a function of
        frequency bin `f` and time bin `t`.
        [Gxx] = (`xunits`)^2 / `funits`

    f - array, (`M`,)
        Frequency bin midpoints.
        [f] = `funits`

    t - array, (`N`,)
        Time bin midpoints.
        [t] = 1 / `Fsunits`

    df - float
        Spacing between frequency bins.
        [df] = `funits`

    dt - float
        Spacing between time bins.
        [dt] = 1 / `Fsunits`

    '''
    def __init__(self, x, Fs, Twindow,
                 t0=0., overlap_frac=0.5, detrend='linear',
                 kernel_df=None, kernel_dt=None,
                 xunits=None, Fsunits=None, funits=None, verbose=True):
        '''Create an instance of the Spectrogram class.

        Input Parameters:
        -----------------
        x - array, (`L`,)
            Time series from which spectrogram is generated.
            The signal is discrete, i.e. x = x(t_n) = x(n / Fs) = x_n
            for integer `n` and sampling frequency `Fs`.
            [x] = Arbitrary

        Fs - float
            Sampling frequency.
            [Fs] = 1 / [time]

        Twindow - float
            Window length over which spectral estimates of `x` are computed.
            [Twindow] = 1 / [Fs]

        t0 - float
            Time corresponding to first data point in signal `x`.
            [t0] = 1 / [Fs]

        overlap_frac - float
            Fraction of overlap between spectrogram windows
            0 <= overlap_frac < 1.
            [overlap_frac] = unitless

        detrend - string
            [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or callable

        kernel_df - float
            Size of convolution window in frequency dimension.
            If `None` is specified, the kernel window size in the
            frequency dimension is set to unity, and no convolution
            in the frequency dimension occurs.
            [kernel_df] = `funits`

        kernel_dt - float
            Size of convolution window in frequency dimension.
            If `None` is specified, the kernel window size in the
            time dimension is set to unity, and no convolution
            in the time dimension occurs.
            [kernel_dt] = 1 / [Fs_units] 

        xunits - string
            Units of time series `x`. Default value of `None` prevents
            incorrect method usage.

        Fsunits -string
            Units of sampling frequency `Fs`. Default value of `None` prevents
            incorrect method usage.

        funits - string
            Units of spectrogram frequency bins. Default value of `None`
            prevents incorrect method usage.

        verbose - bool
            If True, print spectral calculation parameters to screen.

        '''
        # Check that supported units are being used prior to
        # performing any calculations
        if xunits is not None:
            self.xunits = xunits
        else:
            raise ValueError('Units of signal required!')

        if Fsunits == 'Hz':
            # Avoid integer division problems by converting to a float
            self._Fs = float(Fs)
            self._Fsunits = Fsunits
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        if funits == 'kHz':
            self.funits = funits
            Hz_per_kHz = 1e3
        else:
            raise ValueError('Only kHz spectrogram frequency bins supported!')

        # TODO: get Pint to convert this to unitless number!!!
        #
        # The FFT is most efficiently computed when `NFFT` is a power of 2.
        # Here, we take `NFFT` to be the power of 2 that yields a window size
        # *closest* in value to the specified window size `Twindow`.
        exponent = np.log2(self._Fs * Twindow)      # exact
        exponent = np.int(np.round(exponent))       # for nearest power of 2
        self._NFFT = 2 ** exponent

        if verbose:
            print '\nWindow length = ' + str(self._NFFT / self._Fs) + ' s'
            print '# pts. / window = ' + str(self._NFFT)

        # A nonzero overlap decreases spectrogram "graininess"
        # and increases the number of spectrogram time bins.
        # However, increased overlap also leads to increased
        # correlation between time bins.
        self._noverlap = int(overlap_frac * self._NFFT)

        if verbose:
            print 'Window overlap = ' + str(int(overlap_frac * 100)) + '%'

        self._detrend = detrend

        if verbose:
            print 'Detrending = ' + self._detrend

        # Compute spectrogram, where `Gxx` is the one-sided PSD
        Gxx, f, t = specgram(x, Fs=self._Fs, NFFT=self._NFFT,
                             noverlap=self._noverlap, detrend=self._detrend)

        self.Gxx = Gxx * Hz_per_kHz
        self.f = f / Hz_per_kHz
        self.t = t + t0
        self.df = self.f[1] - self.f[0]
        self.dt = self.t[1] - self.t[0]

        if (kernel_df is not None) or (kernel_dt is not None):
            self.Gxx = self._convolve(
                kernel_df=kernel_df, kernel_dt=kernel_dt, verbose=verbose)

    def _convolve(self, kernel_df=None, kernel_dt=None, verbose=True):
        '''Convolve spectrogram with specified kernel.

        Currently, only a boxcar kernel is implemented. Convolution with
        this kernel is equivalent to averaging over the kernel window.
        Additional kernels may be implemented in the future
        (e.g. an edge detection kernel for coherent modes, etc.).

        Parameters:
        -----------
        kernel_df - float
            Size of convolution window in frequency dimension.
            If `None` is specified, the kernel window size in the
            frequency dimension is set to unity, and no convolution
            in the frequency dimension occurs.
            [kernel_df] = [self.f]

        kernel_dt - float
            Size of convolution window in frequency dimension.
            If `None` is specified, the kernel window size in the
            time dimension is set to unity, and no convolution
            in the time dimension occurs.
            [kernel_dt] = [self.t]

        verbose - bool
            If True, print convolution window parameters to screen.

        '''
        # Determine size of kernel, with min size of *one* in each dimension
        if kernel_df is not None:
            Nf = max([int(round(kernel_df / self.df)), 1])
        else:
            Nf = 1

        if kernel_dt is not None:
            Nt = max([int(round(kernel_dt / self.dt)), 1])
        else:
            Nt = 1

        if verbose:
            if Nf > 1:
                print 'Kernel df = ' + str(Nf * self.df) + ' kHz'
            else:
                print 'No convolution in frequency.'

            if Nt > 1:
                print 'Kernel dt = ' + str(Nt * self.dt) + ' s'
            else:
                print 'No convolution in time.'

        # Create boxcar kernel for averaging over kernel window.
        # Note that the integral of kernel should be *unity*
        # in order to preserve spectral power.
        kernel = np.ones([Nf, Nt]) / float(Nf * Nt)

        # When convolving a matrix of shape (w1, w2) with a
        # kernel of shape (k1, k2), convolution is *fastest*
        # via an FFT if
        #
        #           k1 * k2 >= 4 log_2 (w1 * w2)
        #
        # Otherwise, a straightforward convolution is faster.
        # Relevant background information here:
        #
        #       http://programmers.stackexchange.com/a/172839
        if kernel.size >= (4 * np.log2(self.Gxx.size)):
            Gxx_convolved = fftconvolve(self.Gxx, kernel, mode='same')
        else:
            Gxx_convolved = convolve2d(self.Gxx, kernel, mode='same')

        return Gxx_convolved

    def plotSpec(self, fmin=None, fmax=None,
                 vmin=None, vmax=None, cmap='Purples',
                 ax=None, fig=None, geometry=111, title=None):
        '''Plot spectrogram.

        Parameters:
        -----------
        fmin (fmax) - float
            The minimum (maximum) frequency displayed in spectrogram plot.
            If `None`, use the minimum (maximum) frequency in `self.f`.
            [fmin] = [fmax] = [self.f]

        vmin (vmax) - float
            `self.Gxx` <= `vmin` will be mapped to the minimum color
            specified by colormap `cmap`. Similarly, `self.Gxx` >= `vmax`
            will be mapped to the maximum color specified by colormap `cmap`.
            Specification of None for `vmin` (`vmax`) defaults to mapping
            the minimum (maximum) value in `self.Gxx` to the minimum
            (maximum) color specified by colormap `cmap`.
            [vmin] = [vmax] = [self.Gxx]

        ax - :py:class:`AxesSubplot <matplotlib.axes._subplots.AxesSubplot>`
            instance corresponding to the axis (i.e. "subplot") where
            the spectrogram will be drawn. `ax` will (obviously) be
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

        title - string
            The title of the spectrogram to be placed over the subplot
            specified by `axis`.

        cmap - string
            Colormap used for spectrogram. Default matplotlib colormaps
            are found in :py:module:`cm <matplotlib.cm>`.

        Returns:
        --------
        ax - :py:class:`AxesSubplot <matplotlib.axes._subplots.AxesSubplot>`
            instance corresponding to the axis (i.e. "subplot") where
            the spectrogram will be drawn. This is either identical to
            the axis instance used during the call or, if an axis instance
            was not provided, the axis instance created during the call.

        '''
        # If an axis instance is not provided, create one
        if ax is None:
            # If, in addition, a figure instance is not provided,
            # create a figure with the next lowest consecutive window number
            if fig is None:
                fig = plt.figure(FigureList().getNext())
            # Create axis with desired subplot geometry
            ax = fig.add_subplot(geometry)

        # Check that supported units are being used prior to
        # performing any calculations
        if self.funits is 'kHz':
            yaxisunits = 'kHz'
        else:
            raise ValueError('Only kHz spectrogram frequency bins supported!')

        if self._Fsunits is 'Hz':
            xaxisunits = 's'
        else:
            raise ValueError('Only sampling frequencies in Hz supported!')

        # Determine (x, y) extent of plot; time on x-axis, frequency on y-axis
        tmin = self.t[0]
        tmax = self.t[-1]

        if fmin is None:
            fmin = self.f[0]
        if fmax is None:
            fmax = self.f[-1]

        extent = tmin, tmax, fmin, fmax

        # Find frequencies `f` satisfying fmin <= f <= fmax
        find = np.where(np.logical_and(self.f >= fmin, self.f <= fmax))[0]

        # Create plot
        im = ax.imshow(np.flipud(self.Gxx[find, :]), norm=LogNorm(),
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=extent, aspect='auto')

        # Labeling
        ax.set_xlabel('$t \, [\mathrm{' + xaxisunits + '}]$', fontsize=16)
        ax.set_ylabel('$f \, [\mathrm{' + yaxisunits + '}]$', fontsize=16)
        if title is not None:
            ax.set_title(title, fontsize=16)

        # Colorbar
        cb = plt.colorbar(im, format=LogFormatter(labelOnlyBase=True),
                          ax=ax, orientation='horizontal')
        cb.set_label('$|G_{xx}(f)|^2 \, [\mathrm{' + self.xunits +
                     '}^2 / \mathrm{' + yaxisunits + '}]$',
                     fontsize=16)

        return ax


def compare_spectrograms(S1, S2, title1=None, title2=None,
                         fmin=None, fmax=None, fig=None, cmap='Purples'):
    '''Plot spectrograms `S1` and `S2` side-by-side.

    Parameters:
    -----------
    S1 (S2) - :py:class:`Spectrogram <random_data.spectra.Spectrogram>`
        Spectrogram instance corresponding to signal 1 (2).

    title1 (title2) - string
        The title to be placed over `S1` (`S2`).

    fmin (fmax) - float
        The minimum (maximum) frequency displayed in spectrogram plot.
        If `None`, use the minimum (maximum) frequency in `self.f`.
        [fmin] = [fmax] = [S1.f] = [S2.f]

    fig - :py:class:`Figure <matplotlib.figure.Figure>` instance
        in which the spectrograms `S1` and `S2` will be plotted.
        If a figure instance is not provided, a figure instance
        will be created with the next available window number.

    cmap - string
        Colormap used for spectrogram. Default matplotlib colormaps
        are found in :py:module:`cm <matplotlib.cm>`.

    Returns:
    --------
    fig - :py:class:`Figure <matplotlib.figure.Figure>` instance
        in which the spectrograms `S1` and `S2` are plotted.
        If a figure instance was provided during the function call,
        the returned figure instance will be identical to `fig`.
        If a figure instance was not provided during the function call,
        the returned figure instance will correspond to the instance
        automatically created during the function call.

        This is useful as one can (for example) then get the subplot axes
        via

                    axes = fig.get_axes()

        and subsequently manipulate the subplots.

    '''
    # If a figure instance is not provided, create a figure
    # with the next lowest consecutive window number
    if fig is None:
        fig = plt.figure(FigureList().getNext())

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    S1.plotSpec(fmin=fmin, fmax=fmax, ax=ax1, title=title1, cmap=cmap)
    S2.plotSpec(fmin=fmin, fmax=fmax, ax=ax2, title=title2, cmap=cmap)

    return fig
