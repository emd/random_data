'''This module defines a class for analyzing an array of measurements,
where an "array" is defined as three or more measurements.

'''


# Standard library imports
import numpy as np

# Related 3rd-party imports
from .spectra import CrossSpectralDensity


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
