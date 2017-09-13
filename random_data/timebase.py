'''This module implements a class for estimating the so-called
"trigger offset" between random-data series.

'''


# Standard library imports
import numpy as np
import matplotlib.pyplot as plt

# Intra-package imports
from .errors import cross_phase_std_dev
from .spectra import CrossSpectralDensity
from .spectra.nonparametric import _plot_image


class TriggerOffset(object):
    '''

    Attributes:
    -----------

    '''
    def __init__(self, x, Fs=1.0, Nreal_per_ens=1000,
                 gamma2xy_max=0.95, shifts=np.arange(-5, 6, 1)):
        '''Create an instance of the `TriggerOffset` class.

        Input parameters:
        -----------------
        x - array_like, `(2, N)` or `(4, N)`
            The set of data records from which the trigger offset will be
            estimated. The estimation algorithm uses a least-squares
            minimization of the record's cross phase against an *assumed*
            form for the true cross phase. The form of the assumed cross
            phase depends on whether `x` has shape `(2, N) or `(4, N)`.
            If `x` does *not* have shape `(2, N) or `(4, N)`, a ValueError
            is raised.

            For the `(2, N)` case, let `x = np.array([x1, y1])`, where `x1`
            and `y1` are `N`-length digital samples of random processes
            {x1} and {y1}, respectively. It is assumed that the true,
            physical cross phase between {x1} and {y1} is *zero* and that
            any cross phase between the digital records `x1` and `y1`results
            from a constant timebase offset (i.e. "trigger offset") between
            the records. Such trigger offsets may arise if two different
            types of transducers measure a given field; for example, if a
            camera and a photodiode both measure light emitted from the
            same physical location, differing electronics and connections
            to the digitizer may result in a timebase offset between the
            camera and photodiode measurements.

            For the `(4, N)` case, let `x = np.array([x1, y1, x2, y2])`,
            where `x1`, `y1`, `x2`, and `y2` are `N`-length digital samples
            of random processes {x1}, {y1}, {x2}, and {y2}, respectively.
            It is assumed that the true, physical cross phase between
            {x1} and {y1} is *equal* to that between {y1} and {x2}, which
            is in turn *equal* to that between {x2} and {y2}; deviations
            from this assumed structure result from a constant timebase
            offset (i.e. a "trigger offset") between the measurements
            made in domain 1 (i.e. `x1` and `y1`) and the measurements
            made in domain 2 (i.e. `x2` and `y2`). Such trigger offsets
            arise if four adjacent transducers {t1, t2, t3, t4} measure
            an isotropic field but digitization is performed with two
            distinct digitizer "boards", as drawn below

                    transducer:         t1 | t2 | t3 | t4
                    ----------------    -----------------
                    digitizer board:       1    |    2
                    ----------------    -----------------
                    digital record:     x1 | y1 | x2 | y2

            This above is precisely the configuration envisioned for
            the `(4, N)` case. In particular, `x1` and `y1` should be
            digitized on the same board (board 1), making any associated
            trigger offset negligible and allowing an accurate estimate
            of the true cross phase between {x1} and {y1}; `x2` and `y2`
            should similarly be digitized on the same board (board 2),
            allowing an accurate estimate of the cross phase between
            {x2} and {y2}. If the cross phase between `y1` and `x2`
            differs from that purely board 1 or purely board 2 estimates,
            the discrepancy is attributed to a trigger offset between
            board 1 and board 2.

            [x] = arbitrary units

        Fs - float
            The sampling rate of signal `x`.
            [Fs] = arbitrary units

        **csd_kwargs ???

        '''
        x1, y1, x2, y2 = self._parseSignals(x)

        # Record important aspects of computation
        self.Fs = np.float(Fs)
        self._Tens = len(x1) / self.Fs
        self.Nreal_per_ens = Nreal_per_ens
        self.gamma2xy_max = gamma2xy_max
        self.shifts = shifts

        self._getExpectedCrossPhase(x1, y1, x2, y2)

        if (x2 is None) and (y2 is None):
            self._getShiftedCrossPhase(x1, y1)
        else:
            self._getShiftedCrossPhase(y1, x2)

        self._fitTotalCrossPhaseError()

    def _parseSignals(self, x):
        'Check that `x` has acceptable dimensions; if so, parse `x`.'
        if x.ndim != 2:
            raise ValueError('`x` must be 2-dimensional')

        if x.shape[0] == 2:
            x1 = x[0, :].copy()
            y1 = x[1, :].copy()
            x2 = None
            y2 = None
        elif x.shape[0] == 4:
            x1 = x[0, :].copy()
            y1 = x[1, :].copy()
            x2 = x[2, :].copy()
            y2 = x[3, :].copy()
        else:
            raise ValueError('`x` must have shape (2, `N`) or (4, `N`)')

        return x1, y1, x2, y2

    def _getExpectedCrossPhase(self, x1, y1, x2, y2):
        'Get "expected" cross phase from signals `x1`, `y1`, `x2`, & `y2`.'
        if (x2 is None) and (y2 is None):
            csd1 = CrossSpectralDensity(
                x1, y1, Fs=self.Fs, t0=0,
                Tens=self._Tens, Nreal_per_ens=self.Nreal_per_ens)

            # Expect zero cross phase between `x1` and `y1` in this case,
            # but needed to compute cross-spectral density above to obtain
            # desired number of spectral components
            self._expected_cross_phase = np.zeros(len(csd1.f))
        else:
            # Compute cross-spectral densities for the (`x1`, `y1`) and
            # (`x2`, `y2`) correlation pairs
            csd1 = CrossSpectralDensity(
                x1, y1, Fs=self.Fs, t0=0,
                Tens=self._Tens, Nreal_per_ens=self.Nreal_per_ens)
            csd2 = CrossSpectralDensity(
                x2, y2, Fs=self.Fs, t0=0,
                Tens=self._Tens, Nreal_per_ens=self.Nreal_per_ens)

            # Extract cross phase and magnitude-squared coherence
            theta_xy_1 = np.squeeze(csd1.theta_xy)
            theta_xy_2 = np.squeeze(csd1.theta_xy)
            gamma2xy_1 = np.minimum(
                np.squeeze(csd1.gamma2xy),
                self.gamma2xy_max)
            gamma2xy_2 = np.minimum(
                np.squeeze(csd2.gamma2xy),
                self.gamma2xy_max)

            # Use random error from cross-phase estimates to define
            # a set of weights
            sigma1 = cross_phase_std_dev(gamma2xy_1, self.Nreal_per_ens)
            sigma2 = cross_phase_std_dev(gamma2xy_2, self.Nreal_per_ens)
            w1 = 1. / (sigma1 ** 2)
            w2 = 1. / (sigma2 ** 2)

            # Compute *weighted* average of cross phase
            num = (w1 * theta_xy_1) + (w2 * theta_xy_2)
            den = w1 + w2
            self._expected_cross_phase = num / den

        self.f = csd1.f

        return

    def _getShiftedCrossPhase(self, x, y):
        'Get cross phase between `x` and `self.shifts` shifted `y`.'
        Nf = len(self._expected_cross_phase)
        Nshifts = len(self.shifts)

        # Initialize
        self.theta_xy = np.zeros((Nshifts, Nf))
        self.gamma2xy = np.zeros((Nshifts, Nf))

        # Loop through each offset, computing cross phase and coherence
        for i, shift in enumerate(self.shifts):
            csd = CrossSpectralDensity(
                x, np.roll(y, shift), Fs=self.Fs, t0=0,
                Tens=self._Tens, Nreal_per_ens=self.Nreal_per_ens)

            # Unwrap from f = 0 to obtain smoothly varying phase
            self.theta_xy[i, :] = np.unwrap(np.squeeze(csd.theta_xy))

            # Enforce ceiling on values of magnitude-squared coherence
            self.gamma2xy[i, :] = np.minimum(
                np.squeeze(csd.gamma2xy),
                self.gamma2xy_max)

        # Unwrap cross phase
        #
        #   (a) from zero shift to most positive shift, and
        #   (b) from zero shift to most negative shift.
        #
        # This ensures that the cross phase varies smoothly with shift.
        pos_shift_ind = np.where(self.shifts >= 0)[0]
        neg_shift_ind = np.where(self.shifts <= 0)[0]

        self.theta_xy[pos_shift_ind, :] = np.unwrap(
            self.theta_xy[pos_shift_ind, :], axis=0)
        self.theta_xy[neg_shift_ind, :] = np.unwrap(
            self.theta_xy[neg_shift_ind, :][::-1], axis=0)[::-1]

        # Use random error from cross-phase estimates to define
        # a set of weights.
        sigma = cross_phase_std_dev(self.gamma2xy, self.Nreal_per_ens)
        self.weight = 1. / (sigma ** 2)

        return

    def _fitTotalCrossPhaseError(self):
        '''Fit difference between expected and shifted cross phases via
        weighted, linear least squares.

        '''
        # Compute integrated error `self.E` for each offset
        self.E = np.sum(self.weight * self.theta_xy, axis=-1)

        # Put error in standard A * x = b form for least-squares fitting
        A0 = np.array([self.shifts, np.ones(len(self.shifts))]).T

        w = np.sqrt(np.sum(self.weight, axis=-1))  # effective weight per shift
        A = np.dot(np.diag(w), A0)
        b = np.dot(np.diag(w), self.E)

        self.Efit = np.linalg.lstsq(A, b)[0]

        return

    def plotCrossPhaseError(self):
        'Plot cross-phase error as a function of shift and frequency.'
        e = self.weight * self.theta_xy
        emax = np.max(np.abs(e))

        _plot_image(
            self.shifts, self.f, e.T,
            vlim=[-emax, emax], cmap='RdBu')

        return

    def plotTotalCrossPhaseError(self):
        '''Plot total cross-phase error and its fit as a function
        of `self.shifts`.

        '''
        plt.figure()
        plt.plot(self.shifts, self.E, 'o')
        plt.plot(self.shifts, (self.Efit[0] * self.shifts) + self.Efit[1])
        plt.xlabel('shift')
        plt.ylabel('total error, E')
        plt.show()

        return

    @property
    def tau(self):
        '''Get trigger offset from linear least-squares fit of
        total cross-phase error.

        '''
        return -self.Efit[1] / self.Efit[0] / self.Fs
