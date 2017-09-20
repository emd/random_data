'''This module defines several inter-related classes for analyzing
an array of measurements, where an "array" is defined as three or
more measurements.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from fractions import gcd

# Related 3rd-party imports
from .ensemble import closest_index
from .spectra.nonparametric import CrossSpectralDensity, _plot_image, wrap
from .errors import cross_phase_std_dev


class ArrayStencil(object):
    '''A class for analyzing the geometric and spectral properties
    of arbitrary 1-dimensional sampling "stencils". Point sampling
    is assumed at each stencil point.

    Background:
    -----------
    Uniform sampling (e.g. sampling in time with a constant sampling
    rate) is perhaps the most common stencil, and, as a result, its
    geometric and spectral properties are "well-known". Non-uniform
    stencils can have beneficial or deleterious characteristics that
    result from the underlying geometric structure of the stencil.

    Several of this class's attributes are motivated by and are
    consistent with the definition of the cross-correlation function

            R_{xy}(delta) = E[x_k(z) * y_k(z + delta)],

    where

        - E[...] is the expectation-value operator,

        - x_k (y_k) is the kth realization of random process {x_k} ({y_k})
          (i.e. measurement at stencil point x (y)), and

        - delta is the separation between stencil point y and
          stencil point x.

    Attributes:
    -----------
    locations - array_like, (`N`,)
        Location of each measurement point in the 1-dimensional
        stencil. Provided during initialization.
        [locations] = arbitrary units

    include_autocorrelations - bool
        If True, autocorrelations have been included as unique
        correlation pairs.

    separation - array_like, (`M`,)
        The separation between each unique correlation pair
        in the stencil as determined by

            separation = locations[yind] - locations[xind],

        where `locations`, `yind`, and `xind` are additional
        attributes of `ArrayStencil`.
        [separation] = [locations]

    unique_separation - array_like, (`L`,)
        The unique set of values in `self.separation`.
        [unique_separation] = [locations]

    xind (yind) - array_like, (`M`,)
        The indices of `self.locations` corresponding to the
        "x" ("y") stencil points in each unique correlation pair.
        [xind] = [yind] = unitles

    '''
    def __init__(self, locations, include_autocorrelations=True):
        '''Create an instance of the `ArrayStencil` class.

        Input parameters:
        -----------------
        locations - array_like, (`N`,)
            Location of each measurement point in the 1-dimensional
            stencil. Note that methods belonging to this class are
            most robust if `locations` are *integers*; floating point
            values can lead to round-off errors that lead to strange
            behavior. If possible, it is recommended to convert
            convert a floating point stencil into its equivalent
            integer stencil, e.g. if the stencil is [0.5, 1, 2],
            an equivalent integer stencil would be [1, 2, 4].
            [locations] = arbitrary units

        include_autocorrelations - bool
            If True, include autocorrelations as unique correlation pairs.

        '''
        self.locations = np.asarray(locations)
        self.include_autocorrelations = include_autocorrelations

        self.getUniqueCorrelationPairs()
        self.unique_separation = self.getUniqueSeparation()
        self.separation_gcd = self.getSeparationGCD()

    def getUniqueCorrelationPairs(self):
        'Determine unique correlation pairs in the stencil.'
        # Number of measurement locations
        N = len(self.locations)

        # Number of *unique* cross-correlations provided `N` measurements
        Ncorr = (N * (N - 1)) // 2

        # If desired, also account for autocorrelations
        if self.include_autocorrelations:
            Ncorr += N

        # Initialize
        self.xind = np.zeros(Ncorr, dtype=int)
        self.yind = np.zeros(Ncorr, dtype=int)

        # Correlating a signal at a given location against itself
        # (i.e. zero offset in the indexing of `self.locations`)
        # produces the autocorrelation.
        if self.include_autocorrelations:
            minimum_offset = 0
        else:
            minimum_offset = 1

        # Determine each *unique* correlation pair
        cind = 0  # correlation index
        for x in np.arange(N - minimum_offset):
            for y in np.arange(x + minimum_offset, N):
                # `xind` and `yind` are the indices of `self.locations` that
                # correspond to each correlation pair
                self.xind[cind] = x
                self.yind[cind] = y

                cind += 1

        # Consistency with definition of the cross-correlation function
        # (in the Background of the class documentation) motivates our
        # definition of `self.separation`.
        self.separation = (self.locations[self.yind]
                           - self.locations[self.xind])

        # Sort correlation pairs based upon their spatial separation
        sind = np.argsort(self.separation)
        self.separation = self.separation[sind]
        self.xind = self.xind[sind]
        self.yind = self.yind[sind]

        return

    def getUniqueSeparation(self):
        'Get array of unique values in `self.separation`.'
        return np.array(list(set(self.separation)))

    def getSeparationGCD(self):
        '''Get greatest common divisor of `self.separation`.

        If the stencil `self.locations` is a subset of an underlying
        uniform grid, the greatest common divisor corresponds to the
        spacing between adjacent points on this uniform grid.

        '''
        return reduce(gcd, np.abs(self.separation), 0)

    def getMask(self):
        '''Get "mask" of stencil points that are on (1) or off (0),
        assuming that `self.locations` is a subset of an underlying
        uniform grid.

        '''
        locations = np.sort(self.locations)

        underlying_grid = np.arange(
            locations[0],
            locations[-1] + self.separation_gcd,
            self.separation_gcd)

        mask = np.zeros(len(underlying_grid))
        ind_on = np.searchsorted(underlying_grid, locations)
        mask[ind_on] = 1

        return mask

    def getMaskGapSizes(self):
        '''Get size of each gap in the mask returned by `self.getMask()`,
        where a "gap" is defined as one or more adjacent zero values.
        The returned array `gap_sizes` will have the same size as `mask`,
        with `gap_sizes[i]` corresponding to the size of the gap at
        `mask[i]`. For example, if

                mask      = [1, 0, 0, 1, 1, 1, 0, 1, 1], then
                gap_sizes = [0, 2, 2, 0, 0, 0, 1, 0, 0]

        '''
        mask = self.getMask()

        # The start of a gap (i.e. where mask is 0) is preceded by unity
        gap_start = 1 + np.where(np.logical_and(
            mask[:-1] == 1,
            mask[1:] == 0))[0]

        # A gap (i.e. where mask is 0) is terminated by unity
        gap_stop = 1 + np.where(np.logical_and(
            mask[:-1] == 0,
            mask[1:] == 1))[0]

        gap_sizes = np.zeros(len(mask))

        for gap in np.arange(len(gap_start)):
            sl = slice(gap_start[gap], gap_stop[gap])
            gap_sizes[sl] = gap_stop[gap] - gap_start[gap]

        return gap_sizes

    def plotMask(self):
        'Plot mask of stencil points in both real and Fourier space.'
        mask = self.getMask()

        k = 2 * np.pi * np.fft.rfftfreq(len(mask))
        mask_hat = np.fft.rfft(mask)

        fig, axes = plt.subplots(2, 1)

        axes[0].plot(mask, 'o')
        axes[0].set_xlabel('underlying uniform grid points')
        axes[0].set_ylabel('mask')

        axes[1].semilogy(k, np.abs(mask_hat))
        axes[1].set_xlabel('k [1 / (grid-point spacing)]')
        axes[1].set_ylabel('|FT(mask)|')

        plt.tight_layout()
        plt.show()

        return

    def plotSeparationDistribution(self, bin_width=1):
        'Plot distribution of separations.'
        Nbins = (self.separation[-1] - self.separation[0]) // bin_width
        Nbins = np.int(Nbins)

        plt.figure()
        plt.hist(self.separation, bins=Nbins)
        plt.xlabel('separation')
        plt.ylabel('count')
        plt.show()

        return

    def getCrossCorrelation(self, signal):
        '''Get cross correlation of `signal`.

        Parameters:
        -----------
        signal - array_like, (`N`,)
            The 1-dimensional signal realization to correlate, where
            `signal[i]` corresponds to a measurement made at
            `self.locations[i]`.
            [signal] = arbitrary units

        Returns:
        --------
        cross_correlation - array_like, (`L`,)
            Cross correlation of 1-dimensional realization `signal`.
            Note that this is *not* a statistically consistent definition
            of the underlying cross-correlation function; to obtain such an
            estimate, we must compute `cross_correlation` for numerous
            realizations of `signal` and then average. This is most easily
            accomplished by calling `getCrossCorrelation(...)` numerous
            times on independent realizations of `signal`.
            [cross_correlation] = [signal]^2

        '''
        cross_correlation = np.zeros(len(self.unique_separation))

        for sind, separation in enumerate(self.unique_separation):
            ind = np.where(self.separation == separation)[0]

            cross_correlation[sind] = np.mean(
                signal[self.xind[ind]] * signal[self.yind[ind]],
                axis=-1)

        return cross_correlation

    def getAverageForEachSeparation(self, A):
        '''Get average value of `A` at each `self.unique_separation`.

        Parameters:
        -----------
        A - array_like, (`M`, ...)
            Array to average at each `self.unique_separation`. The
            first index of `A` should correspond to the separations
            in `self.separation`.
            [A] = arbitrary units

        Returns:
        --------
        uniform_separation - array_like, (`K`,)
            The uniform array of separations values that spans
            `self.unique_separation`.
            [unique_separation] = [locations]

        A_avg - array_like, (`K`, ...)
            The average value of input array `A` at each
            `uniform_separation`. If `A` has no values
            corresponding to `uniform_separation[i]`, then
            `A_avg[i, ...] = np.nan`.
            [A_avg] = [A]

        '''
        # Create a *uniform* array of separation values that
        # spans `self.unique_separation`.
        uniform_separation = np.arange(
            self.unique_separation[0],
            self.unique_separation[-1] + self.separation_gcd,
            self.separation_gcd)

        # Initialize array to hold averages
        dims = np.concatenate(([len(uniform_separation)], A.shape[1:]))
        dims = dims.astype('int')

        Adtype = A.dtype

        if np.issubdtype(A.dtype, np.integer):
            # Integer arrays cannot store `np.nan`, so
            # convert from integer data type to float
            Adtype = float

        A_avg = np.zeros(dims, dtype=Adtype)

        # Loop through each separation
        for sind, separation in enumerate(uniform_separation):
            ind = np.where(separation == self.separation)[0]

            if len(ind) > 0:
                A_avg[sind, ...] = np.mean(A[ind, ...], axis=0)
            else:
                # If `A.dtype` is complex, the `*=` operator ensures
                # that both the real and imaginary components of
                # `A_avg` are assigned `np.nan` values.
                A_avg[sind, ...] *= np.nan

        return uniform_separation, A_avg


class CrossSpectralDensityArray(object):
    '''A class for computing the cross-spectral densities corresponding
    to unique correlation pairs from an array of measurements. This class
    forms the base class for several derived classes that characterize
    the spatial structure of signals measured by a sensor array.

    Attributes:
    -----------
    Gxy - array_like, (`L`, `Nf`, `Nt`)
        An array of the cross-spectral-density estimate as a function of

            - measurement separation (1st index, `L`),
            - frequency (2nd index, `Nf`), and
            - time (3rd index, `Nt`).

        Explicitly,

            L = {"N choose 2" if autocorrelations are *not* included, or
                 ("N choose 2" + N) if autocorrelations are included},
            Nf = len(self.f)  # number of frequency bins, and
            Nt = len(self.t)  # number of time bins.

        The indexing in `L` is such that cross-spectral density estimates
        are ordered sequentially from smallest separation of measurement
        locations to largest separation of measurement locations.

        [Gxy] = [signal]^2 / [self.Fs], where `signal` is provided
            at initialization

    theta_xy - array_like, (`L`, `Nf`, `Nt`)
        An array of the cross-phase angle estimate, with indexing
        identical to that of `self.Gxy`.

        [theta_xy] = rad

    gamma2xy - array_like, (`L`, `Nf`, `Nt`)
        An array of the magnitude-squared-coherence estimate, with
        indexing identical to that of `self.Gxy`.

        [gamma2xy] = unitless

    xloc (yloc) - array_like, (`L`,)
        The measurement location of signal "x" ("y"), from which
        the cross-spectral density Gxy is computed. As with the
        cross-spectral density objects, `xloc` (`yloc`) is ordered
        sequentially from smallest separation of measurement locations
        to largest separation of measurement locations.

        [xloc] = [yloc]  = [locations], where `locations` is provided
        at object initialization

    separation - array_like, (`L`,)
        The separation (yloc - xloc) of measurements y and x,
        from which the cross-spectral density Gxy is computed.
        The measurement locations (`self.xloc` and `self.yloc`)
        and the first index of the cross-spectral density
        (`self.Gxy[i, ...]`) are ordered sequentially from smallest
        separation to largest separation of measurement locations.

        [separation] = [locations], where `locations` is provided
        at object initialization

    stencil - :py:class:`ArrayStencil <random_data.array.ArrayStencil>`
        The stencil corresponding to the measurement `locations` provided
        at object initialization.

    equalize - bool
        If True, the powers in the signals provided at initialization
        were equalized prior to performing any spectral calculations.

    The additional attributes:

        {`detrend`, `df`, `dt`, `f`, `Fs`, `Npts_overlap`,
        `Npts_per_ens`, `Npts_per_real`, `Nreal_per_ens`, `t`}

    are described in the documentation for :py:class:`CrossSpectralDensity
    <random_data.spectra.CrossSpectralDensity>`.

    Methods:
    --------
    Type `help(CrossSpectralDensityArray)` in the IPython console
    for a listing.

    '''
    def __init__(self, signals, locations, equalize=False,
                 include_autocorrelations=True,
                 print_status=True, **csd_kwargs):
        '''Create an instance of the `CrossSpectralDensityArray` class.

        Input parameters:
        -----------------
        signals - array_like, (`N`, `M`)
            Measurements of length `M` made at `N` locations.
            [signals] = arbitrary units

        locations - array_like, (`N`,)
            Location of each measurement in `signals`.
            [locations] = arbitrary units

        equalize - bool
            If True, for each location in `locations`, scale the power
            in `signals` to that of the location with the most power.
            Note that this may e.g. increase the effective bit noise
            in channels with initially low power.

        include_autocorrelations - bool
            If True, also compute autospectral densities corresponding
            to autocorrelation of each signal against itself.

        print_status - bool
            If True, print status of computations.

        csd_kwargs - any valid keyword arguments for
            :py:class:`CrossSpectralDensity
                <random_data.spectra.CrossSpectralDensity>`.

            For example, use

                    A = Array(signals,..., Fs=200e3, t0=0.)

            to indicate that the measurements in `signals` were
            sampled at a rate `Fs` beginning at time `t0`.

            Note that the spectral-estimation parameters (such as
            the number of realizations per ensemble, the fractional
            overlap between adjacent realizations, etc.) are
            specified via the keyword packing `csd_kwargs`.
            See the `CrossSpectralDensity` documentation for
            further details.

        '''
        # Ensure that number of signals `N` matches number of locations
        if signals.shape[0] != locations.shape[0]:
            raise ValueError(
                'Number of signals must match number of locations!')

        self.equalize = equalize
        if self.equalize:
            signals = _equalize(signals)

        csd_kwargs['print_params'] = print_status
        csd_kwargs['print_status'] = print_status

        self.getSpectralDensities(
            signals, locations,
            include_autocorrelations=include_autocorrelations,
            **csd_kwargs)

    def getSpectralDensities(
            self, signals, locations,
            include_autocorrelations=True,
            **csd_kwargs):
        'Compute cross-spectral density for each unique measurement pairing.'
        # Sort `locations` in ascending order and shift `signals` accordingly.
        # Enforcing `locations` to be sorted in ascending order removes
        # some later complications.
        #
        # Note that `lind` is a `numpy.ndarray` of integer type, which
        # triggers "advanced indexing". Advanced indexing always returns
        # a copy of the indexed array (rather than a view). Thus, the
        # below sorting does *not* alter the values of `locations` and
        # `signals` passed at object initialization.
        lind = np.argsort(locations)
        locations = locations[lind]
        signals = signals[lind, :]

        # Determine unique cross-correlation pairs
        self.stencil = ArrayStencil(
            locations,
            include_autocorrelations=include_autocorrelations)

        # Parse the unique cross-correlation pairs
        Ncorr = len(self.stencil.separation)
        self.separation = self.stencil.separation
        self.xloc = self.stencil.locations[self.stencil.xind]
        self.yloc = self.stencil.locations[self.stencil.yind]

        # Compute cross-spectral density of first correlation pair.
        #
        # Note that the "advanced indexing" of `signals` sends
        # unique copies of signal "x" and signal "y" to the
        # cross-spectral density routine such that a true,
        # complex-valued cross-spectral density is returned,
        # even if computing an autospectral density.
        if csd_kwargs['print_status']:
            self._printCorrelationParameters(0)

        csd = CrossSpectralDensity(
            signals[self.stencil.xind[0], :],
            signals[self.stencil.yind[0], :],
            **csd_kwargs)

        # Record important aspects of the computation
        self.Fs = csd.Fs

        self.Npts_per_real = csd.Npts_per_real
        self.Nreal_per_ens = csd.Nreal_per_ens
        self.Npts_overlap = csd.Npts_overlap
        self.Npts_per_ens = csd.Npts_per_ens

        self.detrend = csd.detrend
        self.window = csd.window

        self.f = csd.f
        self.df = csd.df

        self.t = csd.t
        self.dt = csd.dt

        # Initialize arrays for all correlation pairs and
        # insert values from first correlation pair
        Nf = len(self.f)
        Nt = len(self.t)
        dims = (Ncorr, Nf, Nt)

        self.Gxy = np.zeros(dims, dtype=csd.Gxy.dtype)
        self.theta_xy = np.zeros(dims, dtype=csd.theta_xy.dtype)
        self.gamma2xy = np.zeros(dims, dtype=csd.gamma2xy.dtype)

        self.Gxy[0, ...] = csd.Gxy
        self.theta_xy[0, ...] = csd.theta_xy
        self.gamma2xy[0, ...] = csd.gamma2xy

        # Compute cross-spectral density for each remaining
        # correlation pair.
        for cind in (np.arange(Ncorr - 1) + 1):
            if csd_kwargs['print_status']:
                self._printCorrelationParameters(cind)

            csd = CrossSpectralDensity(
                signals[self.stencil.xind[cind], :],
                signals[self.stencil.yind[cind], :],
                **csd_kwargs)

            self.Gxy[cind, ...] = csd.Gxy
            self.theta_xy[cind, ...] = csd.theta_xy
            self.gamma2xy[cind, ...] = csd.gamma2xy

        return

    def _printCorrelationParameters(self, cind):
        'Print parameters of correlation pair `cind`.'
        print '\npair %i of %i' % (cind + 1, len(self.separation))
        print 'x-loc: %.3f' % self.xloc[cind]
        print 'y-loc: %.3f' % self.yloc[cind]
        print 'separation (y - x): %.3f' % self.separation[cind]

        return

    def plotSlice(self, attr, tind=None, find=None, t=None, f=None,
                  error_bars=True, return_indices=False):
        '''Plot slice of cross-spectral-density attribute `attr`
        at specified time and frequency.

        Parameters:
        -----------
        attr - string
            May be {'Gxy', 'gamma2xy', 'theta_xy'}, in reference
            to the object attribute of the same name.

        tind - int
            Time index of requested slice. If both `tind` and `t`
            are specified, `tind` takes precedent.

        find - int
            Frequency index of requested slice. If both `find` and `f`
            are specified, `find` takes precedent.

        t - float
            The time of requested slice. The returned slice will
            correspond to the time nearest to `t`. If both `tind`
            and `t` are specified, `tind` takes precedent.
            [t] = [self.t] = time

        f - float
            The frequency of requested slice. The returned slice will
            correspond to the frequency nearest to `f`.  If both `find`
            and `f` are specified, `find` takes precedent.
            [f] = [self.f] = 1 / time

        error_bars - bool
            If True, plot error bars corresponding to the
            random error in estimate of `attr`.

        return_indices - bool
            If True, return `find` and `tind`.

        '''
        # Ensure valid attribute has been requested
        valid_attr = ['Gxy', 'gamma2xy', 'theta_xy']

        if attr not in set(valid_attr):
            raise ValueError("Valid attributes are %s" % valid_attr)

        # Determine time index for slice, if needed
        if (t is not None):
            if tind is not None:
                print '\nBoth `tind` and `t` specified; slicing at `tind`.'
            else:
                tind = closest_index(self.t, t)

        # Determine frequency index for slice, if needed
        if (f is not None):
            if find is not None:
                print '\nBoth `find` and `f` specified; slicing at `find`.'
            else:
                find = closest_index(self.f, f)

        # Grab requested slice
        if attr is 'theta_xy':
            sl = self.theta_xy[:, find, tind]
        elif attr is 'gamma2xy':
            sl = self.gamma2xy[:, find, tind]
        else:
            sl = self.Gxy[:, find, tind]

        # Error bars only currently implemented for cross-phase
        if error_bars and (attr is 'theta_xy'):
            gamma2xy = self.gamma2xy[:, find, tind]
            yerr = cross_phase_std_dev(gamma2xy, self.Nreal_per_ens)
        else:
            yerr = None

        plt.figure()

        # Plot only real component if complex
        plt.errorbar(self.separation, np.real(sl), yerr=yerr, fmt='o')

        if attr is 'gamma2xy':
            # Enforce physical bounds of magnitude-squared coherence
            plt.ylim([0, 1])

        if attr is 'Gxy':
            plt.errorbar(self.separation, np.imag(sl), yerr=yerr, fmt='s')
            plt.legend(['Re', 'Im'], loc='lower right')

        plt.xlabel('measurement separation')
        plt.ylabel(attr)

        plt.show()

        if return_indices:
            return find, tind
        else:
            return


class FittedCrossPhaseArray(CrossSpectralDensityArray):
    '''A class for fitting the cross-phase angles of an array
    of measurements vs. measurement separation to a linear model
    for the purposes of determining the corresponding mode number
    as a function of frequency and time.

    This class is derived from :py:class:`CrossSpectralDensityArray
    <random_data.array.CrossSpectralDensityArray>` and thus shares
    most of its attributes and methods. Attributes and properties
    *unique* to this class are discussed below.

    Attributes:
    -----------
    R2 - array_like, (`Nf`, `Nt`)
        The coefficient of determination (R^2) of linear fit
        as a function of frequency and time, with R^2 = 1
        indicating a perfect fit and lesser values indicating
        progressively worse fits.

        [R2] = unitless

    mode_number - array_like, (`Nf`, `Nt`)
        The mode number as a function of frequency and time,
        as determined by fitting the cross-phase angle vs.
        measurement location to a linear model. The higher
        the corresponding R^2, the more believable the
        mode number. The fit at frequency `f0` and time `t0`
        can also be qualitatively and easily visualized using

            self.plotSlice('theta_xy', f=f0, t=t0)

        [mode_number] = radians / [locations], where `locations`
        is provided at object initialization. For example,
        if `locations` corresponds to the toroidal (poloidal)
        locations of an array of sensors, then `mode_number`
        will correspond to the toroidal mode number, n
        (poloidal mode number, m). In contrast, if `locations`
        is given in units of length, `mode_number` will instead
        have the units of wavenumber.

    gamma2xy_max - float
        The maximum allowed value of magnitude-squared coherence.
        The phase-angle fitting weights vary as

                [gamma2xy / (1 - gamma2xy)]^{0.5}

        To prevent singular weights, enforce a ceiling
        on the magnitude-squared coherence of `gamma2xy_max`.

        [gamma2xy_max] = unitless

    Methods:
    --------
    Type `help(Array)` in the IPython console for a listing.

    '''
    def __init__(self, signals, locations, equalize=False,
                 gamma2xy_max=0.95, print_status=True, **csd_kwargs):
        '''Create an instance of the `Array` class.

        Input parameters:
        -----------------
        signals - array_like, (`N`, `M`)
            Measurements of length `M` made at `N` locations.
            [signals] = arbitrary units

        locations - array_like, (`N`,)
            Location of each measurement in `signals`.
            [locations] = arbitrary units

        equalize - bool
            If True, for each location in `locations`, scale the power
            in `signals` to that of the location with the most power.
            Note that this may e.g. increase the effective bit noise
            in channels with initially low power.

        gamma2xy_max - float
            Maximum allowed value of magnitude-squared coherence.
            The phase-angle fitting weights vary as

                    [gamma2xy / (1 - gamma2xy)]^{0.5}

            To prevent singular weights, enforce a ceiling
            on the magnitude-squared coherence of `gamma2xy_max`.

        print_status - bool
            If True, print status of computations.

        csd_kwargs - any valid keyword arguments for
            :py:class:`CrossSpectralDensity
                <random_data.spectra.CrossSpectralDensity>`.

            For example, use

                    A = Array(signals,..., Fs=200e3, t0=0.)

            to indicate that the measurements in `signals` were
            sampled at a rate `Fs` beginning at time `t0`.

            Note that the spectral-estimation parameters (such as
            the number of realizations per ensemble, the fractional
            overlap between adjacent realizations, etc.) are
            specified via the keyword packing `csd_kwargs`.
            See the `CrossSpectralDensity` documentation for
            further details.

        '''
        # By definition, the autospectral densities have zero cross phase
        # and zero separation. Thus, including them in the computation
        # biases the fit to have zero y-intercept at zero separation.
        # However, the cross-phase angles are *already* fit to a linear
        # model with zero y-intercept, so the autospectral densities
        # provide no new phase-angle information. To save computational
        # time, don't calculate autospectral densities.
        CrossSpectralDensityArray.__init__(
            self, signals, locations, equalize=equalize,
            include_autocorrelations=False,
            print_status=print_status,
            **csd_kwargs)

        # Fit phase angles to linear model, setting a maximum value
        # for magnitude-squared coherence in weighting of points
        self.gamma2xy_max = gamma2xy_max
        self.fitPhaseAngles(print_status=print_status)

    def fitPhaseAngles(self, print_status=True):
        '''Fit cross-phase angle vs. measurement location to a
        linear, zero-intercept model using weighted, linear least-squares.

        Parameters:
        -----------
        print_status - bool
            If true, print status of computations.

        '''
        # Initialize
        dims = self.Gxy[0, ...].shape
        self.mode_number = np.zeros(dims)
        self.R2 = np.zeros(dims)

        # Compute unweighted coefficient matrix, `A0`: array_like, (`N`, 1),
        # where `N` is number of measurements, and the second dimension
        # is needed for compatibility with numpy's least-squares algorithm.
        #
        # The fact that the coefficient matrix has only one column means
        # that we are fitting the measured phase angles at a given frequency
        # and time to a model of the form
        #
        #   phase-angle change = (mode number) * (change in location)
        #
        # with *zero* intercept -- by definition, the phase angle at
        # a given frequency and time cannot change if we do not change
        # locations, and the above model strictly enforces this constraint.
        # (Note that one should *not* naively create a second column
        # full of zeros for this purpose, as described here:
        #
        #       http://stackoverflow.com/a/28157066/5469497
        #
        # as this results in poor numerical properties).
        A0 = (np.atleast_2d(self.separation)).T

        if print_status:
            print ''

        # Fit cross-phase angle vs. measurement separation by
        # looping through time and frequency
        for tind in np.arange(len(self.t)):
            for find in np.arange(len(self.f)):
                # Get cross-phase-angle slice and unwrap to prevent
                # phase jumps.
                theta_xy = np.unwrap(np.squeeze(self.theta_xy[:, find, tind]))

                # Get magnitude-squared coherence, enforcing ceiling
                gamma2xy = np.squeeze(self.gamma2xy[:, find, tind])
                gamma2xy = np.minimum(gamma2xy, self.gamma2xy_max)

                # Get standard deviation `sigma` of phase-angle estimates
                sigma = cross_phase_std_dev(
                    gamma2xy, self.Nreal_per_ens)

                # Solve weighted, linear, least-squares problem A * x = b,
                # where `A` is the weighted coefficient matrix and
                # `b` is the weighted cross-phase angles.
                #
                # Following the discussion here:
                #
                #   https://en.wikipedia.org/wiki/Least_squares#Weighted_least_squares
                #
                # we see that the "BLUE" fit is obtained when the least-squares
                # residuals are weighted by the *inverse* of the measurement
                # variance. However, proceeding with the derivation of the
                # normal equations, we see that this is equivalent to
                # weighting `A` and `b` by `diag(1. / sigma)`.
                A = np.dot(np.diag(1. / sigma), A0)
                b = np.dot(np.diag(1. / sigma), theta_xy)
                soln = np.linalg.lstsq(A, b)

                # Unpack solution and relevant metrics
                self.mode_number[find, tind] = soln[0][0]
                self.R2[find, tind] = coefficient_of_determination(
                    soln[1][0], np.var(b))

            # Only print status when moving to a new point in time
            # to avoid excessive printing (which could slow the fitting);
            # these updates should be more than rapid enough for user.
            if print_status:
                print ('Phase-angle fitting percent complete: %.1f \r'
                       % (100 * np.float(tind + 1) / len(self.t))),

        if print_status:
            print ''

        return

    def plotR2(self, tlim=None, flim=None, vlim=[0, 1],
               cmap='viridis', interpolation='none', fontsize=16,
               title=None, xlabel='$t$', ylabel='$f$',
               ax=None, fig=None, geometry=111):
        '''Plot coefficient of determination as a function of frequency
        and time on linear scale.

        '''
        ax = _plot_image(
            self.t, self.f, self.R2,
            xlim=tlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel='$R^2$',
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotModeNumber(self, R2_threshold=0.9,
                       mode_number_lim=[-5, 5],
                       tlim=None, flim=None,
                       cmap='RdBu', interpolation='none', fontsize=16,
                       title=None, xlabel='$t$', ylabel='$f$',
                       cblabel='mode number',
                       ax=None, fig=None, geometry=111):
        '''Plot mode number as a function of frequency and time
        provided that the corresponding coefficient of determination R^2
        exceeds `R2_threshold`.

        '''
        cbticks = np.arange(mode_number_lim[0], mode_number_lim[-1] + 1)

        # Get "discrete" colormap, with a distinct color corresponding
        # to each value in `cbticks`
        cmap = plt.get_cmap(cmap, len(cbticks))

        # However, the bins also have unity width such that
        # the colorbar boundaries should correspond to
        #
        #        lower bound (0):      cbticks[0] - 0.5
        #        (1):                  cbticks[0] + 0.5
        #        (2):                  cbticks[0] + 1.5
        #        ...
        #        upper bound (N + 1):  cbticks[-1] + 0.5
        #
        # This is easily accomplished by setting the minimum and maximum
        # values to represent in the image as follows:
        vlim = np.array([
            cbticks[0] - 0.5,
            cbticks[-1] + 0.5])

        # Only consider phase angles from regions whose coefficient of
        # determination R^2 are greater-than-or-equal-to specified threshold
        mode_number = np.ma.masked_where(
            self.R2 < R2_threshold,
            self.mode_number)

        ax = _plot_image(
            self.t, self.f, mode_number,
            xlim=tlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel=cblabel, cbticks=cbticks,
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotSlice(self, attr, tind=None, find=None,
                  t=None, f=None, error_bars=True,
                  loc_span=(2 * np.pi)):
        '''Plot slice of cross-spectral-density attribute `attr`
        at specified time and frequency.

        Parameters:
        -----------
        attr - string
            Any time-varying spectral attribute of

                :py:class:`CrossSpectralDensity
                    <random_data.spectra.CrossSpectralDensity>`,

            i.e. {'Gxy', 'gamma2xy', 'theta_xy'}.

        tind - int
            Time index of requested slice. If both `tind` and `t`
            are specified, `tind` takes precedent.

        find - int
            Frequency index of requested slice. If both `find` and `f`
            are specified, `find` takes precedent.

        t - float
            The time of requested slice. The returned slice will
            correspond to the time nearest to `t`. If both `tind`
            and `t` are specified, `tind` takes precedent.
            [t] = [self.t] = time

        f - float
            The frequency of requested slice. The returned slice will
            correspond to the frequency nearest to `f`.  If both `find`
            and `f` are specified, `find` takes precedent.
            [f] = [self.f] = 1 / time

        error_bars - bool
            If True, plot error bars corresponding to the
            random error in estimate of `attr`.

        loc_span - float
            The relevant span in measurement locations. For example,
            if the measurements are in an angular coordinate system,
            the span is 2 * pi radians.
            [loc_span] = [self.xloc] = [self.yloc]

        '''
        # Call method from base class
        find, tind = CrossSpectralDensityArray.plotSlice(
            self, attr, tind=tind, find=find, t=t, f=f,
            error_bars=error_bars, return_indices=True)

        # Plot over full span of measurement locations
        xlim = [0, loc_span]

        # Plot linear fit wrapped onto [-pi, pi)
        if attr is 'theta_xy':
            xfit = np.arange(xlim[0], xlim[1], np.pi / 180)
            yfit = self.mode_number[find, tind] * xfit
            yfit = wrap(yfit, -np.pi, np.pi)
            plt.plot(xfit, yfit)

        plt.xlim(xlim)
        plt.show()

        return


class SpatialCrossCorrelation(object):
    '''A class for computing the complex-valued, spatial cross-correlation
    function corresponding to an array of measurements.

    Note that this class can analyze signals with both uniform and nonuniform
    spatial sampling. In both cases, unique correlation pairs are identified,
    and the corresponding temporal cross-spectral density is computed. If there
    are `N` spatial samples, there are `N * (N - 1) // 2` unique correlation
    pairs, requiring computation of O(N^2) cross-spectral densities. If,
    however, the spatial samples are *uniform*, it will be much more efficient
    to use FFT methods (O(N * log(N)) in the spatial dimension; such methods
    are *not* implemented in this class, as non-uniform spatial sampling is
    the case of immediate concern, but more efficient algorithms for uniform
    spatial sampling could be implemented in future releases.

    Attributes:
    -----------
    Gxy - array_like, (`L`, `Nf`)
        An array of the average cross-spectral-density estimate as a
        function of:

            - measurement separation (1st index, `L`), and
            - frequency (2nd index, `Nf`),

        where ensemble averaging has been done

            - in time (i.e. application of ergodic theorem), and
            - at each measurement *separation*.

        The indexing in `L` is such that cross-spectral density estimates
        are ordered sequentially from most-negative separation of measurement
        locations to most-positive separation of measurement locations.

        [Gxy] = [signal]^2 / [Fs], where `signal` and `Fs` are provided
            at initialization

    separation - array_like, (`L`,)
        The measurement separation.
        [separation] = [locations], where `locations` is provided
        at object initialization

    equalize - bool
        If True, the powers in the signals provided at initialization
        were equalized prior to performing any spectral calculations.

    The additional attributes:

        {`detrend`, `df`, `dt`, `f`, `Fs`, `Npts_overlap`,
        `Npts_per_ens`, `Npts_per_real`, `Nreal_per_ens`, `t`}

    are described in the documentation for :py:class:`CrossSpectralDensity
    <random_data.spectra.CrossSpectralDensity>`.

    Methods:
    --------
    Type `help(SpatialCrossCorrelation)` in the IPython console
    for a listing.

    '''
    def __init__(self, signals, locations, tlim=None, equalize=False,
                 print_status=True, **csd_kwargs):
        '''Create an instance of the `SpatialCrossCorrelation` class.

        Input parameters:
        -----------------
        signals - array_like, (`N`, `M`)
            Measurements of length `M` made at `N` locations.
            [signals] = arbitrary units

        locations - array_like, (`N`,)
            Location of each measurement in `signals`.
            [locations] = arbitrary units

        tlim - array_like, (2,) or None
            If not None, then only use the portion of `signals`
            that sits between `min(tlim)` and `max(tlim)`.
            [tlim] = 1 / [csd_kwargs['Fs']]

        equalize - bool
            If True, for each location in `locations`, scale the power
            in `signals` to that of the location with the most power.
            Note that this may e.g. increase the effective bit noise
            in channels with initially low power.

        print_status - bool
            If True, print status of computations.

        csd_kwargs - any valid keyword arguments for
            :py:class:`CrossSpectralDensity
                <random_data.spectra.CrossSpectralDensity>`.

            The signal sample rate `Fs` and initial timestamp `t0`
            must be specified; spectral-estimation parameters can
            also be specified. For example, use

                corr = SpatialCrossCorrelation(
                    signals, ...,
                    Fs=200e3, t0=0., Nreal_per_ens=100)

            to indicate that the measurements in `signals` were
            sampled at a rate `Fs` beginning at time `t0` and that
            `Nreal_per_ens` realizations should be averaged over
            to obtain the ensemble spectral estimate.

            Note that additional parameters of relevance to spectral
            estimation (such as windowing, window overlap, etc.) are
            specified via the keyword packing `**csd_kwargs`. See the
            `CrossSpectralDensity` documentation for further details.

        '''
        self.Fs = np.float(csd_kwargs['Fs'])

        # Use `tlim` to determine the size of the ensemble
        tind = _get_timebase_indices(
            tlim, self.Fs, csd_kwargs['t0'], signals.shape[-1])

        csd_kwargs['Tens'] = (tind[-1] - tind[0]) / self.Fs

        # Compute cross-spectral density for each measurement pair
        csdArray = CrossSpectralDensityArray(
            signals[..., tind], locations, equalize=equalize,
            include_autocorrelations=True,
            print_status=print_status,
            **csd_kwargs)

        # Record important aspects of the computation
        self.equalize = csdArray.equalize

        self.Npts_per_real = csdArray.Npts_per_real
        self.Nreal_per_ens = csdArray.Nreal_per_ens
        self.Npts_overlap = csdArray.Npts_overlap
        self.Npts_per_ens = csdArray.Npts_per_ens

        self.detrend = csdArray.detrend
        self.window = csdArray.window

        self.f = csdArray.f
        self.df = csdArray.df

        # There should only be *one* index in the time dimension
        # corresponding to the midpoint of the single ensemeble;
        # `dt` gives the ensemble window length (+/-0.5 * `dt`
        # about the ensemble midpoint `t`).
        self.t = csdArray.t
        self.dt = csdArray.dt

        # Note that each *unique* measurement separation
        # (i.e. `csdArray.stencil.unique_separation`) does
        # *not* necessarily map to a unique cross-spectral
        # density, as various correlation pairs may be
        # separated by the same distance.
        #
        # To create a cross-spectral density that is truly
        # a function of each unique measurement separation,
        # compute average cross-spectral density at each
        # unique separation.
        #
        # Also, "squeeze" `csdArray.Gxy` to remove the
        # length-1 time dimension.
        sep, Gxy_av = csdArray.stencil.getAverageForEachSeparation(
            np.squeeze(csdArray.Gxy, axis=-1))

        # Reflect and conjugate `Gxy_av` to obtain correlation function
        # for both positive and negative separations.
        self.separation = np.concatenate((
            -sep[1:][::-1],
            sep))
        self.Gxy = np.concatenate((
            np.conj(Gxy_av[1:, ...][::-1, ...]),
            Gxy_av))

        self._central_block = self._getCentralBlock()

    def _getCentralBlock(self):
        '''Get slice for viewing central, non-nan values of `self.Gxy`.
        It is assumed that the nans are distributed identically for
        both positive and negative separations, as required for a
        stationary random process where

                    R_{xx}(-delta) = conj(R_{xx}(delta)).

        '''
        ind0sep = np.where(self.separation == 0)[0][0]
        indnan = _find_first_nan(self.Gxy[ind0sep:, 0])

        if indnan is not None:
            return slice(ind0sep - indnan + 1, ind0sep + indnan)
        else:
            return slice(None, None)

    def interpolate(self, max_gap_size):
        '''Interpolate `self.Gxy` across spatial gaps that are
        less than or equal to `max_gap_size`. Interpolation is
        linear and can be undone by calling `self.unInterpolate()`.

        '''
        # First, make a hidden copy of the original signal, in case
        # we don't like the interpolation and want to revert to original
        self._Gxy_original = self.Gxy.copy()

        # Gaps are identified by presence of `np.nan` values in `self.Gxy`.
        # Because gaps only occur in space, we only need to look at
        # a slice at a single frequency, e.g. `self.Gxy[:, 0]`.
        not_nan_ind = np.where(np.logical_not(np.isnan(self.Gxy[:, 0])))[0]

        # Determine size of gaps
        stencil = ArrayStencil(not_nan_ind)
        gap_sizes = stencil.getMaskGapSizes()

        # Do not interpolate across gaps exceeding `max_gap_size`, and,
        # to save some computation, do not interpolate where there are
        # no gaps
        interp_ind = np.where(np.logical_and(
            gap_sizes > 0,
            gap_sizes <= max_gap_size))[0]

        # Linearly interpolate in space at each frequency.
        for find in np.arange(len(self.f)):
            if np.__version__ >= '1.13.1':
                # `np.interp` works for complex functions
                self.Gxy[interp_ind, find] = np.interp(
                    self.separation[interp_ind],
                    self.separation[not_nan_ind],
                    self.Gxy[not_nan_ind, find])
            else:
                # `np.interp` does *not* work for complex functions,
                # so perform interpolation individually on real &
                # imaginary components
                self.Gxy[interp_ind, find] = np.interp(
                    self.separation[interp_ind],
                    self.separation[not_nan_ind],
                    self.Gxy[not_nan_ind, find].real)
                self.Gxy[interp_ind, find] += (1j * np.interp(
                    self.separation[interp_ind],
                    self.separation[not_nan_ind],
                    self.Gxy[not_nan_ind, find].imag))

        # Find new central block, as some `np.nan` values
        # should have been replaced by interpolated values
        self._central_block = self._getCentralBlock()

        return

    def unInterpolate(self):
        '''Inverts the actions of `self.interpolate()`, restoring
        `self.Gxy` to its value prior to the last `self.interpolate()`
        call. Thus, if `self.interpolate()` has been called more than
        once, the restored `self.Gxy` may not correspond to the raw,
        un-interpolated value.

        '''
        self.Gxy = self._Gxy_original
        del self._Gxy_original

        self._central_block = self._getCentralBlock()

        return

    def plotNormalizedCorrelationFunction(
            self, xlim=None, flim=None, vlim=[-1, 1],
            cmap='RdBu', interpolation='none', fontsize=16,
            xlabel='$\delta$', ylabel='$f$', no_nan=False):
        'Plot normalized correlation function, Gxy(delta, f) / Gxy(0, f).'
        # At each frequency, normalize correlation function
        # by it's magnitude at zero separation.
        ind0sep = np.where(self.separation == 0)[0][0]
        Gxy_norm = self.Gxy / np.abs(self.Gxy[ind0sep, :])

        separation = self.separation.copy()

        if no_nan:
            separation = separation[self._central_block]
            Gxy_norm = Gxy_norm[self._central_block, ...]

        fig, axes = plt.subplots(
            2, 1, sharex=True, sharey=True, figsize=(7, 9))

        # Plot real component
        axes[0] = _plot_image(
            separation, self.f, Gxy_norm.T.real,
            xlim=xlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            xlabel='', ylabel=ylabel,
            cborientation='vertical',
            cblabel='$\mathrm{Re}[G_{xy}(\delta, f) / G_{xy}(0, f)]$',
            fontsize=fontsize,
            ax=axes[0])

        # Plot imaginary component
        axes[1] = _plot_image(
            separation, self.f, Gxy_norm.T.imag,
            xlim=xlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            xlabel=xlabel, ylabel=ylabel,
            cborientation='vertical',
            cblabel='$\mathrm{Im}[G_{xy}(\delta, f) / G_{xy}(0, f)]$',
            fontsize=fontsize,
            ax=axes[1])

        plt.tight_layout()
        plt.show()

        return


def coefficient_of_determination(ssresid, sstot):
    '''Get the coefficient of determination, "R^2", for a given fit.

    Parameters:
    -----------
    ssresid - array_like, (`L`, `M`, ...)
        The squared sum of the fit residuals.

    ssrtot - array_like, (`L`, `M`, ...)
        The total sum of squares.

    Returns:
    --------
    R2 - array_like, (`L`, `M`, ...)
        The coefficient of determination.

    For an explanation of terms, see the article on Wikipedia:

        https://en.wikipedia.org/wiki/Coefficient_of_determination

    Further, computing R^2 from numpy outputs is discussed here:

        http://stackoverflow.com/a/3057858/5469497

    '''
    # To avoid divide-by-zero warnings at boundary points with
    # little-to-no physical importance
    if sstot == 0:
        sstot = np.finfo('float64').eps

    return 1 - (ssresid / sstot)


def _test_plotModeNumber(
        R2_threshold=0.9,
        cmap='RdBu',
        Tens=5e-3, Nreal_per_ens=10):
    '''This routine plots the mode number of several test cases
    to ensure that the mode number is correctly represented
    by the methods in `Array`. Each test case lies near the lower
    or upper boundary of a mode-number bin, and good behavior at
    the bin's extrema implies good behavior throughout the rest of
    the bin's interior. Note that several figures are generated!

    '''
    from .signals import RandomSignal

    # Measurement locations
    locations = np.arange(0, 2 * np.pi)
    Nsig = len(locations)

    # For a uniform spacing of `dzeta` radian between measurement locations,
    # the Nyquist mode number is floor(pi / dzeta). For a spacing
    # of 1 radian, the Nyquist mode number is 3.
    n_Ny = np.int(np.floor(np.pi / (locations[1] - locations[0])))

    # Signal parameters
    Fs = 200e3
    t0 = 0
    T = 0.1
    s = RandomSignal(Fs, t0, T)
    Npts = len(s.x)
    t = s.t()
    f0 = 50e3
    A0 = 1e-2
    n0 = 1  # fitting performs poorly when n0 = 0, so use finite mode number
    omega0_t = 2 * np.pi * f0 * t

    # Initialize
    signals = np.zeros((Nsig, Npts))

    # Signal generation
    for i in np.arange(Nsig):
        # Create some uncorrelated noise
        signals[i, :] = (RandomSignal(Fs, t0, T)).x

        # Add coherent mode
        dtheta = n0 * (locations[i] - locations[0])
        signals[i, :] += (A0 * np.cos(omega0_t + dtheta))

    # Perform fit
    A = FittedCrossPhaseArray(
        signals, locations, Fs=Fs,
        Tens=Tens, Nreal_per_ens=Nreal_per_ens)

    # Check lower boundary for each mode-number bin.
    # As the Nyquist mode number is 3, we do not check
    # the lower boundary of bins for n <= -n_Ny or n > n_Ny.
    n_lower = np.arange(-n_Ny + 1, n_Ny + 1) - 0.45

    for n in n_lower:
        # Artificially insert mode number -- we are just testing
        # the *visualization* here, not the computation, so
        # we are free to impose any mode number we wish.
        # This will result in a much more efficient test,
        # as we will not have to recompute cross-spectral densities
        # and perform fits for each mode-number bin.
        A.mode_number[:] = n

        A.plotModeNumber(
            R2_threshold=R2_threshold,
            mode_number_lim=[-n_Ny, n_Ny],
            cmap=cmap,
            title='Lower boundary of mode number %i' % np.int(np.ceil(n)))

    # Check upper boundary for each mode-number bin.
    # As the Nyquist mode number is 3, we do not check
    # the upper boundary of bins for n < -n_Ny or n >= n_Ny.
    n_upper = np.arange(-n_Ny, n_Ny) + 0.45

    for n in n_upper:
        # Artificially insert mode number -- we are just testing
        # the *visualization* here, not the computation, so
        # we are free to impose any mode number we wish.
        # This will result in a much more efficient test,
        # as we will not have to recompute cross-spectral densities
        # and perform fits for each mode-number bin.
        A.mode_number[:] = n

        A.plotModeNumber(
            R2_threshold=R2_threshold,
            mode_number_lim=[-n_Ny, n_Ny],
            cmap=cmap,
            title='Upper boundary of mode number %i' % np.int(np.floor(n)))

    return


def _find_first_nan(x):
    '''Find location of first `np.nan` in array `x`; return `None`
    if there are no `np.nan` values in `x`.

    '''
    try:
        return np.where(np.isnan(x))[0][0]
    except IndexError:
        return None


def _get_timebase_indices(tlim, Fs, t0, Npts):
    '''For a timebase determined by {Fs, t0, Npts}, get the indices
    corresponding to `tlim`.

    Input parameters:
    -----------------
    tlim - array_like, (2,) or None
        The initial and final times, respectively. If `None`,
        return indices corresponding to the full timebase.
        [tlim] = 1 / [Fs]

    Fs - float
        The sample rate.
        [Fs] = arbitrary units

    t0 - float
        The timestamp of the first point in the timebase.
        [t0] = 1 / [Fs]

    Npts - int
        The number of points in the timebase.
        [Npts] = unitless

    Returns:
    --------
    ind - array_like, (`M`,) with `M <= Npts`
        The indices of the timebase determined by {Fs, t0, Npts} that are
        both (a) greater than or equal to `min(tlim)` and (b) less than or
        equal to `min(tlim)`.
        [ind] = unitless

    '''
    if tlim is not None:
        # Construct the timebase of raw signal
        t = t0 + (np.arange(Npts) / np.float(Fs))

        ind = np.where(np.logical_and(
            t >= np.min(tlim),
            t <= np.max(tlim)))[0]
    else:
        ind = np.arange(Npts)

    return ind


def _equalize(x):
    '''Equalize the power in the channels of `x`.

    Input parameters:
    -----------------
    x - array_like, (`M`, `N`)
        A 2-dimensional signal of position/channel (`M`) and time (`N`).
        [x] = arbitrary units

    Returns:
    --------
    xeq - array_like, (`M`, `N`)
        A channel-equalized representation of `x`. All the channels in `x`
        are scaled to have the same power as the most powerful channel in `x`.
        [xeq] = [x]

    '''
    P = np.var(x, axis=-1)

    maxind = np.where(P == np.max(P))[0]
    amplitude_scaling = np.sqrt(P[maxind] / P)
    xeq = amplitude_scaling[:, np.newaxis] * x

    return xeq
