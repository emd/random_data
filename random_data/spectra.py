'''This module implements a routines for analyzing spectra.

'''


# Standard library imports
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.mlab import specgram
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

# Related 3rd-party imports
from event_manager.for_matplotlib import FigureList


class Spectrogram(object):
    '''Spectrogram class.

    Input Parameters:
    -----------------
    x - array, (`L`,)
        Time series from which spectrogram is generated.
        The signal is discrete, i.e. x = x(t_n) = x(n / Fs) = x_n
        for integer `n` and sampling frequency `Fs`
        [x] = Arbitrary

    Fs - float
        Sampling frequency
        [Fs] = 1 / [time]

    df - float
        Desired frequency resolution of spectrogram. For reasons
        of computational efficiency, the resulting frequency resolution
        may not be *exactly* the value specified
        [df] = [Fs]

    t0 - float
        Time corresponding to first data point in signal `x`
        [t0] = 1 / [Fs]

    overlap_frac - float
        Fraction of overlap between spectrogram windows
        0 <= overlap_frac < 1
        [overlap_frac] = unitless

    detrend - string
        [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or callable

    xunits - string
        Units of time series `x`. Default value of `None` prevents
        incorrect method usage.

    Fsunits -string
        Units of sampling frequency `Fs` and spectrogram frequency bin
        spacing `df`. Default value of `None` prevents incorrect method usage.

    funits - string
        Units of spectrogram frequency bins. Default value of `None`
        prevents incorrect method usage.

    Attributes:
    -----------
    Gxx - array, (`M`, `N`)
        One-sided power spectral density as a function of frequency bin `f`
        and time bin `t`.
        [Gxx] = (`xunits`)^2 / `funits`

    f - array, (`M`,)
        Frequency bins
        [f] = `funits`

    df - float
        Frequency bin spacing
        [df] = `funits`

    t - array, (`N`,)
        Time bins
        [t] = 1 / `Fsunits`

    dt - float
        Time bin size
        [dt] = 1 / `Fsunits`

    '''
    def __init__(self, x, Fs, df,
                 t0=0., overlap_frac=0.5, detrend='linear',
                 xunits=None, Fsunits=None, funits=None):
        '''Create an instance of the Spectrogram class.'''
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

        # In the absence of zero-padding, the window size `NFFT`
        # determines the size of the FFT frequency bins for
        # a given signal sampled at frequency `Fs` via
        #
        #       FFT frequency bin size = Fs / NFFT
        #
        # However, the FFT is most efficiently computed when
        # `NFFT` is a power of 2. Here, we take `NFFT` to be
        # the power of 2 that yields a frequency bin size
        # *closest* in value to the specified bin size `df`
        exponent = np.log2(self._Fs / df)            # exact
        exponent = np.int(np.round(exponent))  # for nearest power of 2
        self._NFFT = 2 ** exponent
        self.df = self._Fs / self._NFFT

        # A nonzero overlap decreases spectrogram "graininess"
        # and increases the number of spectrogram time bins
        self._noverlap = int(overlap_frac * self._NFFT)

        # Times of initial (`_t0`) and final (`_tf`) points in signal `x`
        # [_t0] = [_tf] = 1 / [Fs]
        self._t0 = t0
        self._tf = t0 + ((len(x) - 1) / self._Fs)

        self._detrend = detrend

        # Compute spectrogram, where `Gxx` is the one-sided PSD
        Gxx, f, t = specgram(x, Fs=Fs, NFFT=self._NFFT,
                             noverlap=self._noverlap, detrend=detrend)

        self.Gxx = Gxx * Hz_per_kHz
        self.f = f / Hz_per_kHz
        self.df /= Hz_per_kHz
        self.t = t + self._t0
        self.dt = np.mean(np.diff(self.t))

    def plotSpec(self, ax=None, fig=None, geometry=111,
                 title=None, cmap='Purples'):
        '''Plot spectrogram.

        Parameters:
        -----------
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

        # Determine (x, y) extent of plot
        xmin = self._t0
        xmax = self._tf

        ymin = self.f[0]
        ymax = self.f[-1]

        extent = xmin, xmax, ymin, ymax

        # Create plot
        im = ax.imshow(np.flipud(self.Gxx), norm=LogNorm(),
                       extent=extent, aspect='auto', cmap=cmap)

        # Labeling
        ax.set_xlabel('$t \, [\mathrm{' + xaxisunits + '}]$', fontsize=16)
        ax.set_ylabel('$f \, [\mathrm{' + yaxisunits + '}]$', fontsize=16)
        if title is not None:
            ax.set_title(title, fontsize=16)

        # Colorbar
        cb = plt.colorbar(im, format=LogFormatter(labelOnlyBase=True),
                          orientation='horizontal')
        cb.set_label('$|G_{xx}(f)|^2 \, [\mathrm{' + self.xunits +
                     '}^2 / \mathrm{' + yaxisunits + '}]$',
                     fontsize=16)

        return ax
