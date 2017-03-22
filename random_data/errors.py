'''This module implements routines for characterizing the random error
in the estimates of various spectral quantities.

'''


# Standard library imports
import numpy as np


def cross_phase_std_dev(gamma2xy, Nreal_per_ens):
    '''Standard deviation of cross-phase estimate.

    The formula for the standard deviation in the cross-phase
    estimate is taken from:

        Bendat & Piersol, "Random Data", 4th ed., (2010), pg. 298

    Parameters:
    -----------
    gamma2xy - array_like, (`M`, `N`, ...)
        The magnitude-squared coherence.
        [gamma2xy] = unitless

    Nreal_per_ens - int
        The number of realizations per ensemble used in the computation
        of the cross-spectral-density estimate, from which
        an estimate of the cross-phase is derived.
        [Nreal_per_ens] = unitless

    Returns:
    --------
    sigma - array_like, (`M`, `N`, ...)
        The standard deviation of the random error associated with
        the estimate of the cross-phase angle.
        [sigma] = rad

    '''
    return np.sqrt(1 - gamma2xy) / np.sqrt(2 * Nreal_per_ens * gamma2xy)
