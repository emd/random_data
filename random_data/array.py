'''This module defines a class for analyzing an array of measurements,
where an "array" is defined as three or more measurements.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt

# Related 3rd-party imports
from .spectra import CrossSpectralDensity, _plot_image
from .errors import cross_phase_std_dev


class Array(object):
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
        # Number of measurements
        N = signals.shape[0]

        # Number of *unique* correlations provided `N` measurements
        Ncorr = (N * (N - 1)) // 2

        # Initialize
        self.xloc = np.zeros(Ncorr)
        self.yloc = np.zeros(Ncorr)
        self.csd = [None] * Ncorr

        # Loop through each *unique* correlation pair
        cind = 0  # correlation index
        for xind in np.arange(N - 1):
            for yind in np.arange(xind + 1, N):
                # Note location for signal "x" and signal "y"
                self.xloc[cind] = locations[xind]
                self.yloc[cind] = locations[yind]

                if csd_kwargs['print_status']:
                    print '\nx-loc: %.3f' % self.xloc[cind]
                    print 'y-loc: %.3f' % self.yloc[cind]

                # Compute cross-spectral density
                self.csd[cind] = CrossSpectralDensity(
                    signals[xind, :], signals[yind, :], **csd_kwargs)

                cind += 1

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

        # Compute spatial separation for each probe pair and sort
        delta = self.yloc - self.xloc
        dind = np.argsort(delta)
        delta = delta[dind]

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
        A0 = (np.atleast_2d(delta)).T

        if print_status:
            print ''

        # Fit cross-phase angle vs. measurement location by
        # looping through time and frequency
        for tind in np.arange(len(self.csd[0].t)):
            for find in np.arange(len(self.csd[0].f)):
                # Get cross-phase angles, sort by `dind` so as to align
                # with spatial separations `delta`, and unwrap to get
                # a nice line
                theta_xy = self.getSlice('theta_xy', tind=tind, find=find)
                theta_xy = theta_xy[dind]
                theta_xy = np.unwrap(theta_xy)

                # Get magnitude-squared coherence, enforce ceiling, and
                # sort by `dind` to align with spatial separations `delta`
                gamma2xy = self.getSlice('gamma2xy', tind=tind, find=find)
                gamma2xy = np.minimum(gamma2xy, self.gamma2xy_max)
                gamma2xy = gamma2xy[dind]

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
        # Ensure valid attribute has been requested
        valid_attr = ['Gxy', 'gamma2xy', 'theta_xy']

        if attr not in set(valid_attr):
            raise ValueError("Valid attributes are %s" % valid_attr)

        # Determine time index for slice, if needed
        if (t is not None):
            if tind is not None:
                print '\nBoth `tind` and `t` specified; slicing at `tind`.'
            else:
                dt = np.abs(t - self.csd[0].t)
                tind = np.where(dt == np.min(dt))[0][0]

        # Determine frequency index for slice, if needed
        if (f is not None):
            if find is not None:
                print '\nBoth `find` and `f` specified; slicing at `find`.'
            else:
                df = np.abs(f - self.csd[0].f)
                find = np.where(df == np.min(df))[0][0]

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
        and timeon linear scale.

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

    def plotSlice(self, attr, error_bars=True, **getSlice_kwargs):
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

        getSlice_kwargs - any valid keyword arguments for
            :py:method:`getSlice <random_data.array.Array.getSlice>`.

            For example, use

                self.plotSlice(t=1. f=50e3)

            to plot slice of `attr` at time nearest to `t` and
            frequency nearest to `f`.

        '''
        delta = self.yloc - self.xloc
        dind = np.argsort(delta)
        delta = delta[dind]

        sl = self.getSlice(attr, **getSlice_kwargs)
        sl = sl[dind]

        # Error bars only currently implemented for cross-phase
        if error_bars and (attr is 'theta_xy'):
            gamma2xy = self.getSlice('gamma2xy', **getSlice_kwargs)
            gamma2xy = np.minimum(gamma2xy, self.gamma2xy_max)
            gamma2xy = gamma2xy[dind]
            yerr = cross_phase_std_dev(gamma2xy, self.csd[0].Nreal_per_ens)
        else:
            yerr = None

        plt.figure()
        plt.errorbar(delta, sl, yerr=yerr, fmt='o')

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
