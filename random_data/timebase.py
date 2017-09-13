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

    Background:
    -----------
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

    Attributes:
    -----------
    The additional attributes:

        {`f`, `Fs`}

    are described in the documentation for :py:class:`CrossSpectralDensity
    <random_data.spectra.CrossSpectralDensity>`.

    '''
    def __init__(self, x, shifts=np.arange(-5, 6, 1),
                 gamma2xy_max=0.95, **csd_kwargs):
        '''Create an instance of the `TriggerOffset` class.

        Input parameters:
        -----------------
        x - array_like, `(2, N)` or `(4, N)`
            The set of data records from which the trigger offset will be
            estimated. The estimation algorithm uses a least-squares
            minimization of the record's cross phase against an *assumed*
            form for the true cross phase. The form of the assumed cross
            phase depends on whether `x` has shape `(2, N) or `(4, N)`,
            as is discussed in the "Background" section of the class
            Docstring. If `x` does *not* have shape `(2, N) or `(4, N)`,
            a ValueError is raised.
            [x] = arbitrary units

        shifts - array_like, `(M,)`
            The number of timestamps by which the digital records should
            be "rolled" relative to one another in order to probe the
            trigger offset.
            [shifts] = unitless

        gamma2xy_max - float
            Maximum allowed value of magnitude-squared coherence.
            The weights used in the least-squares minimization
            vary as

                    [gamma2xy / (1 - gamma2xy)]^{0.5}

            To prevent singular weights, enforce a ceiling
            on the magnitude-squared coherence of `gamma2xy_max`.
            [gamma2xy_max] = unitless

        csd_kwargs - any valid keyword arguments for
            :py:class:`CrossSpectralDensity
                <random_data.spectra.CrossSpectralDensity>`.

            The signal sample rate `Fs` must be specified;
            spectral-estimation parameters can also be specified.
            For example, use

                trig = TriggerOffset(
                    x, ...,
                    Fs=4e6, Nreal_per_ens=1000)

            to indicate that the measurements in `x` were sampled at
            a rate `Fs` and that `Nreal_per_ens` realizations should
            be averaged over to obtain the ensemble spectral estimate.
            (The full record length is treated as a *single* ensemble).

            Note that additional parameters of relevance to spectral
            estimation (such as windowing, window overlap, etc.) are
            specified via the keyword packing `**csd_kwargs`. See the
            `CrossSpectralDensity` documentation for further details.

        '''
        x1, y1, x2, y2 = self._parseSignals(x)

        self.shifts = np.sort(shifts)
        self.gamma2xy_max = gamma2xy_max
        self.Fs = np.float(csd_kwargs['Fs'])

        # Treat the full digital record as a single ensemble
        csd_kwargs['Tens'] = len(x1) / self.Fs

        self._getExpectedCrossPhase(x1, y1, x2, y2, **csd_kwargs)

        if (x2 is None) and (y2 is None):
            self._getShiftedCrossPhase(x1, y1, **csd_kwargs)
        else:
            self._getShiftedCrossPhase(y1, x2, **csd_kwargs)

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

    def _getExpectedCrossPhase(self, x1, y1, x2, y2, **csd_kwargs):
        'Get "expected" cross phase from signals `x1`, `y1`, `x2`, & `y2`.'
        csd1 = CrossSpectralDensity(x1, y1, **csd_kwargs)

        # Record important aspects of the computation
        self._Npts_per_real = csd1.Npts_per_real
        self._Nreal_per_ens = csd1.Nreal_per_ens
        self._Npts_overlap = csd1.Npts_overlap
        self._Npts_per_ens = csd1.Npts_per_ens

        self._detrend = csd1.detrend
        self._window = csd1.window

        self.f = csd1.f
        self._df = csd1.df

        self._t = csd1.t
        self._dt = csd1.dt

        if (x2 is None) and (y2 is None):
            # Expect zero cross phase between `x1` and `y1` in this case
            self._expected_cross_phase = np.zeros(len(csd1.f))
        else:
            csd2 = CrossSpectralDensity(x2, y2, **csd_kwargs)

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
            sigma1 = cross_phase_std_dev(gamma2xy_1, self._Nreal_per_ens)
            sigma2 = cross_phase_std_dev(gamma2xy_2, self._Nreal_per_ens)
            w1 = 1. / (sigma1 ** 2)
            w2 = 1. / (sigma2 ** 2)

            # Compute *weighted* average of cross phase
            num = (w1 * theta_xy_1) + (w2 * theta_xy_2)
            den = w1 + w2
            self._expected_cross_phase = num / den

        return

    def _getShiftedCrossPhase(self, x, y, **csd_kwargs):
        'Get cross phase between `x` and `self.shifts` shifted `y`.'
        Nf = len(self.f)
        Nshifts = len(self.shifts)

        # Initialize
        self.theta_xy = np.zeros((Nshifts, Nf))
        self.gamma2xy = np.zeros((Nshifts, Nf))

        # Loop through each offset, computing cross phase and coherence
        for i, shift in enumerate(self.shifts):
            csd = CrossSpectralDensity(x, np.roll(y, shift), **csd_kwargs)

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
        sigma = cross_phase_std_dev(self.gamma2xy, self._Nreal_per_ens)
        self.weight = 1. / (sigma ** 2)

        return

    def _fitTotalCrossPhaseError(self):
        '''Fit difference between expected and shifted cross phases via
        weighted, linear least squares.

        '''
        # Local error as a function of shift and frequency
        self._e = self.weight * self.theta_xy

        # Integrate error over frequency to obtain a "total" error
        # at each value in `self.shifts`
        self.E = np.sum(self._e, axis=-1)

        # Put error in standard A * x = b form for least-squares fitting
        A0 = np.array([self.shifts, np.ones(len(self.shifts))]).T

        w = np.sqrt(np.sum(self.weight, axis=-1))  # effective weight per shift
        A = np.dot(np.diag(w), A0)
        b = np.dot(np.diag(w), self.E)

        self.Efit = np.linalg.lstsq(A, b)[0]

        return

    def plotLocalCrossPhaseError(
            self, xlim=None, ylim=None, vlim=None,
            norm=None, cmap='RdBu', interpolation='none',
            title=None, xlabel='shift', ylabel='f', fontsize=16,
            cblabel='local error, e', cbticks=None, cborientation='vertical',
            ax=None, fig=None, geometry=111):
        'Plot cross-phase error as a function of shift and frequency.'
        if vlim is None:
            emax = np.max(np.abs(self._e))
            vlim = [-emax, emax]

        _plot_image(
            self.shifts, self.f, self._e.T,
            xlim=xlim, ylim=ylim, vlim=vlim,
            norm=norm, cmap=cmap, interpolation=interpolation,
            title=title, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
            cblabel=cblabel, cbticks=cbticks, cborientation=cborientation,
            ax=ax, fig=fig, geometry=geometry)

        plt.tight_layout()
        plt.show()

        return

    def plotIntegratedCrossPhaseError(self):
        '''Plot cross-phase error (integrated over frequency) and its fit
        as a function of `self.shifts`.

        '''
        plt.figure()

        # Integrated error and fit
        plt.plot(self.shifts, self.E, 'o')
        plt.plot(self.shifts, (self.Efit[0] * self.shifts) + self.Efit[1])

        # Place crosshair at fit's zero crossing, which indicates
        # the trigger offset giving minimal integrated error
        plt.axhline(0, c='k')
        plt.axvline(self.tau * self.Fs, c='k')

        plt.xlabel('shift')
        plt.ylabel('integrated error, E')

        plt.tight_layout()
        plt.show()

        return

    @property
    def tau(self):
        '''Get trigger offset from linear least-squares fit of
        total cross-phase error.

        '''
        return -self.Efit[1] / self.Efit[0] / self.Fs
