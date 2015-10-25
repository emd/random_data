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
    '''A class for characterizing spectral densities.

    Attributes:
    -----------
    Gxy - array_like, (`L`, `M`)
        The spectral density estimate. If a single signal `x` is provided
        at object initialization, this corresponds to an autospectral density.
        If two distinct signals `x` and `y` are provided at initialization,
        the estimate corresponds to the cross-spectral density of `x` and `y`.
        [Gxy] = [x] [y] / [Fs], where `x` (`y`) is the signal(s) for
            which the spectral density has been computed and `Fs`
            is the sampling rate of `x` (and `y`). Note that if
            `Gxy` corresponds to an autospectral density estimate,
            we substitute [y] -> [x] in the above units expression.

    f - array_like, (`L`,)
        The frequencies at which the spectral density has been estimated.
        [f] = [Fs], where `Fs` is the signal sampling rate provided
            at object initialization.

    t - array_like, (`M`,)
        The temporal midpoint of each ensemble.
        [t] = 1 / [Fs], where `Fs` is the signal sampling rate provided
            at object initialization.

    random_error - float
        The fractional random error in the spectral density estimate `Gxy`,
        as determined from the number of realizations averaged over
        per ensemble.

    kind - string
        The kind of spectral density: {'autospectral', 'cross-spectral'}

    fraction_overlap = fraction_overlap
        The fractional overlap between adjacent realizations
        of a given ensemble used when computing the spectral density.

    detrend - string
        The function applied to each realization before taking the FFT.

    window - callable or ndarray
        The window applied to each realization before taking the FFT.

    '''
    def __init__(self, x, y=None, Fs=1.0, t0=0,
                 Tens=40960., Nreal_per_ens=10, fraction_overlap=0.5,
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
            [t0 = 1 / [Fs]

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

        fraction_overlap - float
            The fractional overlap between adjacent realizations.
            0 < `fraction_overlap` < 1, otherwise a ValueError is raised.

        detrend - string
            The function applied to each realization before taking FFT. 
            May be [ 'default' | 'constant' | 'mean' | 'linear' | 'none']
            or callable, as specified in :py:func: `csd <matplotlib.mlab.csd>`.

        window - callable or ndarray
            As specified in :py:func: `csd <matplotlib.mlab.csd>`.

        '''
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

        # Determine number of sample points to use per realization and
        # the number of overlapping points between adjacent realizations
        Npts_per_real = self._getNumPtsPerReal(Fs, Tens, Nreal_per_ens)
        Npts_overlap = np.int(fraction_overlap * Npts_per_real)

        # Record important aspects of computation
        self.fraction_overlap = np.float(Npts_overlap) / Npts_per_real
        self.detrend = detrend
        self.window = window
        self.random_error = 1 / np.sqrt(Nreal_per_ens)

        # Generate frequency and time base of spectral density estimate
        self.f = self.getFrequencies(Npts_per_real, Fs)
        self.t = self.getTimes(x, Fs, t0, Npts_per_real, Nreal_per_ens)

        # Determine resolution in frequency and time, if applicable
        try:
            self.df = self.f[1] - self.f[0]
        except IndexError:
            self.df = np.nan
        try:
            self.dt = self.t[1] - self.t[0]
        except IndexError:
            self.dt = np.nan

        self.Gxy = self.getSpectralDensity(
            x, y, Fs, Npts_per_real, Nreal_per_ens, Npts_overlap)

    def _getNumPtsPerReal(self, Fs, Tens, Nreal_per_ens):
        '''Get number of points per realization. This directly
        determines the frequency resolution `df` of the resulting
        spectral density estimate.

        '''
        # TODO: Ensure `Npts_per_real` is *not* a large prime number
        # for which the FFT will run slowly...
        return np.int(np.round(Tens * Fs / Nreal_per_ens))

    def getFrequencies(self, Npts_per_real, Fs):
        '''Get frequencies at which spectral density is estimated.
        It is assumed that a one-sided spectral density is computed.

        '''
        return np.fft.rfftfreq(Npts_per_real, d=(1. / Fs))

    def getTimes(self, x, Fs, t0, Npts_per_real, Nreal_per_ens):
        '''Get time base for spectral density estimate that is consistent
        with the number of points per realization. The returned time base
        corresponds to the midpoint of each ensemble.

        '''
        # The ensemble forms the basic unit/discretization of time
        # for the computed spectral density estimate, so let's determine
        # the number of points in an ensemble and the corresponding
        # time window that is *consistent* with `Npts_per_real` and
        # `Nreal_per_ens`.
        Npts_per_ensemble = Npts_per_real * Nreal_per_ens
        Tens = Npts_per_ensemble / np.float(Fs)  # avoid integer division!

        # Determine the number of *whole* ensembles in the data record
        # (Disregard fractional ensemble at the end of the data, if present)
        Nens = np.int(len(x) / Npts_per_ensemble)

        # The returned time base corresponds to the midpoint of each ensemble
        return t0 + (Tens * np.arange(0.5, Nens, 1))

    def getSpectralDensity(self, x, y, Fs,
                           Npts_per_real, Nreal_per_ens, Npts_overlap):
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
        Npts_per_ens = Npts_per_real * Nreal_per_ens

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
                    NFFT=Npts_per_real, noverlap=Npts_overlap,
                    detrend=self.detrend, window=self.window)[0]
            else:
                Gxy[:, ens] = mlab.csd(
                    x[sl], y[sl], Fs=Fs,
                    NFFT=Npts_per_real, noverlap=Npts_overlap,
                    detrend=self.detrend, window=self.window)[0]

        return Gxy

    def getPhaseAngle(self):
        pass

    def getCoherence(self):
        pass


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
