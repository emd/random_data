'''This module defines a class for analyzing an array of measurements,
where an "array" is defined as three or more measurements.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from fractions import gcd

# Related 3rd-party imports
from .ensemble import closest_index
from .spectra import CrossSpectralDensity, _plot_image, wrap
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
            stencil.
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


class Array(object):
    '''A class for fitting the cross-phase angles of an array
    of measurements vs. measurement separation to a linear model
    for the purposes of determining the corresponding mode number
    as a function of frequency and time.

    Attributes:
    -----------
    csd - list, length L
        A list of :py:class:`CrossSpectralDensity
        <random_data.spectra.CrossSpectralDensity>` objects, where
        each cross-spectral density object corresponds to a
        *unique* correlation pair of the `N` inputs in `signals`
        (explicitly, there will be "N choose 2" cross-spectral density
        objects in the list).

        Each cross-spectral density object is given as
        a function of frequency and time; below

            Nf = len(csd[0].f)  # number of frequency bins, and
            Nt = len(csd[0].t)  # number of time bins.

        The cross-spectral density objects are ordered
        sequentially from smallest separation of measurement
        locations to largest separation of measurement locations.
        The cross-phase angle vs. measurement separation is
        then fit to a linear model to determine the corresponding
        mode number as a function of frequency and time.

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
        The cross-spectral density objects (`self.csd`) and
        the corresponding measurement locations (`self.xloc` and
        `self.yloc`) are ordered sequentially from smallest
        separation to largest separation of measurement locations.

        [separation] = [locations], where `locations` is provided
        at object initialization

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
    def __init__(self, signals, locations,
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
        # Ensure that number of signals `N` matches number of locations
        if signals.shape[0] != locations.shape[0]:
            raise ValueError(
                'Number of signals must match number of locations!')

        csd_kwargs['print_params'] = print_status
        csd_kwargs['print_status'] = print_status

        self.gamma2xy_max = gamma2xy_max

        self.getSpectralDensities(
            signals, locations, **csd_kwargs)

        self.fitPhaseAngles(print_status=print_status)

    def getSpectralDensities(
            self, signals, locations, **csd_kwargs):
        'Compute cross-spectral density for each unique measurement pairing.'
        # Sort `locations` in ascending order and shift `signals` accordingly.
        # Enforcing `locations` to be sorted in ascending order removes
        # some later complications.
        lind = np.argsort(locations)
        locations = locations[lind]
        signals = signals[lind, :]

        # Determine unique cross-correlation pairs
        stencil = ArrayStencil(locations, include_autocorrelations=False)

        # Parse the unique cross-correlation pairs
        Ncorr = len(stencil.separation)
        self.separation = stencil.separation
        self.xloc = stencil.locations[stencil.xind]
        self.yloc = stencil.locations[stencil.yind]

        # Initialize list to store cross-spectral density instance
        # for each correlation pair
        self.csd = [None] * Ncorr

        # Compute cross-spectral density of each correlation pair
        for cind in np.arange(len(self.csd)):
            if csd_kwargs['print_status']:
                print '\npair %i of %i' % (cind + 1, Ncorr)
                print 'x-loc: %.3f' % self.xloc[cind]
                print 'y-loc: %.3f' % self.yloc[cind]
                print 'separation (y - x): %.3f' % self.separation[cind]

            self.csd[cind] = CrossSpectralDensity(
                signals[stencil.xind[cind], :],
                signals[stencil.yind[cind], :],
                **csd_kwargs)

        return

    def fitPhaseAngles(self, print_status=True):
        '''Fit cross-phase angle vs. measurement location to a
        linear, zero-intercept model using weighted, linear least-squares.

        Parameters:
        -----------
        print_status - bool
            If true, print status of computations.

        '''
        # Initialize
        self.mode_number = np.zeros(self.csd[0].Gxy.shape)
        self.R2 = np.zeros(self.csd[0].Gxy.shape)

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

        # Fit cross-phase angle vs. measurement location by
        # looping through time and frequency
        for tind in np.arange(len(self.csd[0].t)):
            for find in np.arange(len(self.csd[0].f)):
                # Get cross-phase angles and unwrap to prevent phase jumps
                theta_xy = self.getSlice('theta_xy', tind=tind, find=find)
                theta_xy = np.unwrap(theta_xy)

                # Get magnitude-squared coherence and enforce ceiling
                gamma2xy = self.getSlice('gamma2xy', tind=tind, find=find)
                gamma2xy = np.minimum(gamma2xy, self.gamma2xy_max)

                # Get standard deviation `sigma` of phase-angle estimates
                sigma = cross_phase_std_dev(
                    gamma2xy, self.csd[0].Nreal_per_ens)

                # Solve weighted, linear, least-squares problem A * x = b,
                # where `A` is the weighted coefficient matrix and
                # `b` is the weighted cross-phase angles.
                A = np.dot(np.diag(sigma), A0)
                b = np.dot(np.diag(sigma), theta_xy)
                soln = np.linalg.lstsq(A, b)

                # Unpack solution and relevant metrics
                self.mode_number[find, tind] = soln[0][0]
                self.R2[find, tind] = coefficient_of_determination(
                    soln[1][0], np.var(theta_xy))

            # Only print status when moving to a new point in time
            # to avoid excessive printing (which could slow the fitting);
            # these updates should be more than rapid enough for user.
            if print_status:
                print ('Phase-angle fitting percent complete: %.1f \r'
                       % (100 * np.float(tind + 1) / len(self.csd[0].t))),

        if print_status:
            print ''

        return

    def _preProcessSliceRequest(
            self, attr, tind=None, find=None, t=None, f=None):
        '''Pre-process slice request (a) checking that `attr` is
        a valid attribute and (b) converting time and frequency
        values into corresponding indices, if needed.

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
                # dt = np.abs(t - self.csd[0].t)
                # tind = np.where(dt == np.min(dt))[0][0]
                tind = closest_index(self.csd[0].t, t)

        # Determine frequency index for slice, if needed
        if (f is not None):
            if find is not None:
                print '\nBoth `find` and `f` specified; slicing at `find`.'
            else:
                # df = np.abs(f - self.csd[0].f)
                # find = np.where(df == np.min(df))[0][0]
                find = closest_index(self.csd[0].f, f)

        return tind, find

    def getSlice(self, attr, tind=None, find=None, t=None, f=None):
        '''Get slice of cross-spectral-density attribute `attr`
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
            [t] = [self.csd[0].t] = time

        f - float
            The frequency of requested slice. The returned slice will
            correspond to the frequency nearest to `f`.  If both `find`
            and `f` are specified, `find` takes precedent.
            [f] = [self.csd[0].f] = 1 / time

        Returns:
        --------
        slice - array_like, (`N`,)
            Slice of requested attribute, where `N` is the
            number of cross-spectral-density objects.

            Note that this slice is *not* a "view" of
            the underlying cross-spectral-density object --
            that is, the returned slice is its own distinct
            object and can be manipulated without influencing
            the state of the underlying cross-spectral densities.

        '''
        tind, find = self._preProcessSliceRequest(
            attr, tind=tind, find=find, t=t, f=f)

        # Initialize
        dtype = (getattr(self.csd[0], attr)).dtype
        sl = np.zeros(len(self.csd), dtype=dtype)

        # Loop through each cross-spectral density object
        for cind in np.arange(len(self.csd)):
            sl[cind] = getattr(self.csd[cind], attr)[find, tind]

        return sl

    def plotR2(self, tlim=None, flim=None, vlim=[0, 1],
               cmap='viridis', interpolation='none', fontsize=16,
               title=None, xlabel='$t$', ylabel='$f$',
               ax=None, fig=None, geometry=111):
        '''Plot coefficient of determination as a function of frequency
        and time on linear scale.

        '''
        ax = _plot_image(
            self.csd[0].t, self.csd[0].f, self.R2,
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
            self.csd[0].t, self.csd[0].f, mode_number,
            xlim=tlim, ylim=flim, vlim=vlim,
            norm=None, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel,
            cblabel=cblabel, cbticks=cbticks,
            fontsize=fontsize,
            ax=ax, fig=fig, geometry=geometry)

        return ax

    def plotSlice(self, attr, error_bars=True,
                  loc_span=(2 * np.pi), **getSlice_kwargs):
        '''Plot slice of cross-spectral-density attribute `attr`
        at specified time and frequency.

        Parameters:
        -----------
        attr - string
            Any time-varying spectral attribute of

                :py:class:`CrossSpectralDensity
                    <random_data.spectra.CrossSpectralDensity>`,

            i.e. {'Gxy', 'gamma2xy', 'theta_xy'}.

        error_bars - bool
            If True, plot error bars corresponding to the
            random error in estimate of `attr`.

        loc_span - float
            The relevant span in measurement locations. For example,
            if the measurements are in an angular coordinate system,
            the span is 2 * pi radians.
            [loc_span] = [self.xloc] = [self.yloc]

        getSlice_kwargs - any valid keyword arguments for
            :py:method:`getSlice <random_data.array.Array.getSlice>`.

            For example, use

                self.plotSlice(t=1. f=50e3)

            to plot slice of `attr` at time nearest to `t` and
            frequency nearest to `f`.

        '''
        tind, find = self._preProcessSliceRequest(attr, **getSlice_kwargs)
        sl = self.getSlice(attr, tind=tind, find=find)

        # Error bars only currently implemented for cross-phase
        if error_bars and (attr is 'theta_xy'):
            gamma2xy = self.getSlice('gamma2xy', tind=tind, find=find)
            gamma2xy = np.minimum(gamma2xy, self.gamma2xy_max)
            yerr = cross_phase_std_dev(gamma2xy, self.csd[0].Nreal_per_ens)
        else:
            yerr = None

        plt.figure()

        # Plot over full span of measurement locations
        xlim = [0, loc_span]

        # Plot only real component if complex
        plt.errorbar(self.separation, np.real(sl), yerr=yerr, fmt='o')

        if attr is 'theta_xy':
            # Plot linear fit wrapped onto [-pi, pi)
            xfit = np.arange(xlim[0], xlim[1], np.pi / 180)
            yfit = self.mode_number[find, tind] * xfit
            yfit = wrap(yfit, -np.pi, np.pi)
            plt.plot(xfit, yfit)

        if attr is 'gamma2xy':
            # Enforce physical bounds of magnitude-squared coherence
            plt.ylim([0, 1])

        if attr is 'Gxy':
            plt.errorbar(self.separation, np.imag(sl), yerr=yerr, fmt='s')
            plt.legend(['Re', 'Im'], loc='lower right')

        plt.xlim(xlim)
        plt.xlabel('measurement separation')
        plt.ylabel(attr)

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
    A = 1e-2

    # Initialize
    signals = np.zeros((Nsig, Npts))

    # Signal generation: use purely coherent modes because
    # if mode-number extraction does not work in this ideal case,
    # it certainly won't work in the presence of noise.
    for i in np.arange(Nsig):
        # Create some uncorrelated noise
        signals[i, :] = (RandomSignal(Fs, t0, T)).x

        # Add coherent mode
        signals[i, :] += A * np.cos(2 * np.pi * f0 * t)

    # Perform fit
    A = Array(signals, locations, Fs=Fs,
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
