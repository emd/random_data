'''This module defines a class for analyzing an array of measurements,
where an "array" is defined as three or more measurements.

'''


# Standard library imports
import numpy as np

# Related 3rd-party imports
from .spectra import CrossSpectralDensity
from .errors import cross_phase_std_dev


class Array(object):
    def __init__(self, signals, locations, print_locations=True, **csd_kwargs):
        '''Create an instance of the `Array` class.

        Input parameters:
        -----------------
        signals - array_like, (`N`, `M`)
            Measurements of length `M` made at `N` locations.
            [signals] = arbitrary units

        locations - array_like, (`N`,)
            Location of each measurement in `signals`.
            [locations] = arbitrary units

        print_locations - bool
            If True, print signal locations prior to spectral computations.

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

        self.getSpectralDensities(
            signals, locations, print_locations=print_locations, **csd_kwargs)

        # self.fitPhaseAngles()

    def getSpectralDensities(
            self, signals, locations, print_locations=True, **csd_kwargs):
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
        for xind in np.arange(Ncorr - 1):
            for yind in np.arange(xind + 1, Ncorr):
                # Correlation index
                cind = xind + yind - 1

                # Note location for signal "x" and signal "y"
                self.xloc[cind] = locations[xind]
                self.yloc[cind] = locations[yind]

                if print_locations:
                    print '\nx-loc: %.3f' % self.xloc[cind]
                    print 'y-loc: %.3f' % self.yloc[cind]

                # Compute cross-spectral density
                self.csd[cind] = CrossSpectralDensity(
                    signals[xind, :], signals[yind, :], **csd_kwargs)

        return

    def fitPhaseAngles(self, gamma2xy_max=0.95):
        '''Fit cross-phase angle vs. measurement location using
        weighted, linear least-squares.

        Parameters:
        -----------
        gamma2xy_max - float
            Maximum allowed value of magnitude-squared coherence.
            The fitting weights vary as

                    [gamma2xy / (1 - gamma2xy)]^{0.5}

            To prevent singular weights, enforce a ceiling
            on the magnitude-squared coherence of `gamma2xy_max`.

        '''
        # Initialize
        self.mode_number = np.zeros(self.csd[0].Gxy.shape)
        self.theta0 = np.zeros(self.csd[0].Gxy.shape)
        self.R2 = np.zeros(self.csd[0].Gxy.shape)
        self.kappa = np.zeros(self.csd[0].Gxy.shape)

        # Compute unweighted coefficient matrix
        # `A0`: array_like, (`N`, 2), where `N` is number of measurements
        delta = self.yloc - self.xloc
        A0 = (np.vstack([delta, np.ones(len(delta))])).T

        # Loop through time
        for tind in np.arange(len(self.csd[0].t)):
            # Get cross-phase angles
            theta_xy = self.getTimeSlice('theta_xy', tind)

            # Get magnitude-squared coherence and enforce ceiling
            gamma2xy = self.getTimeSlice('gamma2xy', tind)
            gamma2xy = np.minimum(gamma2xy, gamma2xy_max)

            # Get standard deviation `sigma` of phase-angle estimates
            sigma = cross_phase_std_dev(gamma2xy, self.csd[0].Nreal_per_ens)

            # Solve weighted, linear, least-squares problem A * x = b,
            # where `A` is the weighted coefficient matrix and
            # `b` is the weighted cross-phase angles.
            A = np.dot(np.diag(sigma), A0)
            b = np.dot(np.diag(sigma), theta_xy)
            soln = np.linalg.lstsq(A, b)

            # Unpack solution and relevant metrics
            self.mode_number[:, tind] = soln[0][0, :]
            self.theta0[:, tind] = soln[0][1, :]
            self.R2[:, tind] = coefficient_of_determination(
                soln[1], np.var(theta_xy, axis=0))
            self.kappa[:, tind] = np.linalg.cond(A)

        return

    def getTimeSlice(self, attr, tind):
        '''Get time slice of cross-spectral-density attribute `attr`.

        Parameters:
        -----------
        attr - string
            Any time-varying spectral attribute of

                :py:class:`CrossSpectralDensity
                    <random_data.spectra.CrossSpectralDensity>`,

            such as {'Gxy', 'gamma2xy', 'theta_xy'}.

        tind - int
            Index of requested time slice.

        Returns:
        --------
        slice - array_like, (`N`, `L`)
            Time slice of requested attribute, where `N` is the
            number of measurements and `L` is the number of
            frequency bins in the cross-spectral-density object.

            Note that this time slice is *not* a "view" of
            the underlying cross-spectral-density object --
            that is, the returned time slice is its own distinct
            object and can be manipulated without influencing
            the state of the underlying cross-spectral densities.

        '''
        valid_attr = ['Gxy', 'gamma2xy', 'theta_xy']

        if attr not in set(valid_attr):
            raise ValueError("Valid attributes are %s" % valid_attr)

        # Initialize
        dtype = (getattr(self.csd[0], attr)).dtype
        sl = np.zeros((len(self.csd), len(self.csd[0].f)), dtype=dtype)

        # Loop through each cross-spectral density object
        for cind in np.arange(len(self.csd)):
            sl[cind, :] = getattr(self.csd[cind], attr)[:, tind]

        return sl


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
    return 1 - (ssresid / sstot)
