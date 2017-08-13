'''A module for autoregressive (AR) spectral-density estimation of
signal `x` via Burg's algorithm, as described in:

    S. Lawrence Marple Jr, "A tutorial overview of modern
    spectral estimation", IEEE, 1989

'''


import numpy as np
from scipy.signal import fftconvolve


class BurgAutoSpectralDensity(object):
    '''A class for autoregressive (AR) spectral-density estimation of
    signal `x` via Burg's algorithm.

    The Burg algorithm is described in:

        S. Lawrence Marple Jr, "A tutorial overview of modern
        spectral estimation", IEEE, 1989,

    whereby the forward-prediction AR coefficients `a` are obtained
    by minimizing the sum of forward- and backward-prediction error
    energies subject to the constraint that the AR coefficients
    satisfy the Levinson recursion. The autospectral-density estimate
    is computed from the resulting AR coefficients.

    Attributes:
    -----------
    Ep - float
        The sum of forward- and backward-prediction error energies for
        autoregression (AR) of `x` with forward-prediction AR coefficients
        `self.a`.
        [Ep] = [x]^2

    Fs - float
        The sampling rate of signal `x`.
        [Fs] = arbitrary units

    Sxx - array_like, (`Nf`,)
        The *two-sided* autospectral-density estimate of signal `x` at
        frequencies `self.f` using a Burg autoregression of order `self.p`.
        [Sxx] = [x]^2 / [self.Fs] if `self.normalize` is True
              = [x]^2             if `self.normalize` is False

    a - array_like, (`self.p + 1`,)
        The `self.p`-order forward-prediction AR coefficients for `x`
        as determined by the Burg algorithm.
        [a] = unitless

    f - array_like, (`Nf`,)
        The frequencies corresponding to the autospectral-density estimate
        `self.Sxx`. As `self.Sxx` is two-sided, the frequencies are uniformly
        spaced by `df = self.Fs / Nf` and satisfy

                -(0.5 * self.Fs) <= self.f < (0.5 * self.Fs)

        [f] = [self.Fs]

    normalize - bool
        If True, Burg-estimated autospectral density `self.Sxx` has been
        normalized to the power in signal `x`.

    p - int
        The order of the Burg autoregression.
        [p] = unitless

    Methods:
    --------
    Type `help(BurgAutoSpectralDensity)` in the IPython console for a listing.

    '''
    def __init__(self, p, x, Fs=1.0, Nf=128, normalize=True):
        '''Create an instance of the `BurgAutoSpectralDensity` class.

        Input parameters:
        -----------------
        p - int
            The order of the Burg autoregression. Computational time scales
            roughly as `p ** 2` for "typical" orders (p < 100, or so) and
            scales roughly as `p` for larger orders. Larger orders are often
            subject to numerical artifacts.
            [p] = unitless

        x - array_like, (`N`,)
            The signal whose autospectral density will be estimated.
            The signal should be a zero-mean, stationary process
            uniformly sampled at sampling rate `Fs`. Unlike FFT-based
            methods, AR spectral-density estimates do *not* exhibit
            sidelobes due to windowing (see Marple's discussion following
            Eqs. (4.8) & (4.9)), so it is *not* necessary to smoothly
            taper/window `x` to zero at its edges.
            [x] = arbitrary units

        Fs - float
            The sampling rate of signal `x`.
            [Fs] = arbitrary units

        Nf - int
            The number of frequencies for which the autospectral density
            will be estimated. The frequencies `f` will be uniformly spaced
            by `df = Fs / Nf` and will satisfy

                            -(0.5 * Fs) <= f < (0.5 * Fs)

            Unlike FFT-based methods, there is nothing special about using
            a power of 2 for the Burg algorithm; larger values of `Nf`
            marginally increase the computational time.
            [Nf] = unitless

        normalize - bool
            If True, normalize Burg-estimated autospectral density to
            the power in signal `x`.

        '''
        # Record spectral-estimation parameters
        self.p = p
        self.Fs = np.float(Fs)
        self.f = self.Fs * np.arange(-0.5, 0.5, 1. / Nf)
        self.normalize = normalize
        self._sigma_x = np.std(x)

        # Compute Burg AR coefficients and total energy in resulting error
        self.a = burg_coefficients(x, self.p)
        self.Ep = total_error_energy(x, self.a)

        # Compute two-sided autospectral density
        self.Sxx = self._getSpectralDensity()

    def _getSpectralDensity(self):
        'Get the AR autospectral-density estimate via Eq. (4.6) of Marple.'
        # Numerator `num` of autospectral-density estimate. Note that
        # Marple's Eq. (4.6) has `sigma ^ 2` in the numerator, where
        # `sigma` is the standard deviation of the driving white noise.
        # From the discussion following Marple's Eq. (4.19), it is implied
        # that `sigma ^ 2` -> `Ep`, where `Ep` is the sum of the forward-
        # and backward-prediction error energies.
        num = self.Ep

        # Compute denominator `den` of autospectral-density estimate
        k = np.arange(1, self.p + 1)
        ph = 2 * np.pi * np.outer(k, self.f / self.Fs)
        den = (np.abs(1 + np.dot(self.a[1:], np.exp(-1j * ph)))) ** 2

        psd = num / den

        if self.normalize:
            df = self.f[1] - self.f[0]
            psd *= ((self._sigma_x ** 2) / (np.sum(psd) * df))

        return psd


def forward_prediction_error(x, a):
    '''Get forward-prediction error for autoregression (AR) of `x`
    with forward-prediction AR coefficients `a`.

    Note that this is Marple's Eq. (4.13) with p <= n < N.

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    a - array_like, (`p + 1`,)
        The AR coefficients for the `p`-order forward-prediction
        autoregression. By definition, `a[0]` is unity for all orders.
        [a] = unitless

    Returns:
    --------
    error - array_like, (`N - p`,)
        The forward-prediction error, restricted to the available data.
        That is, `error[i]` will be the error from the forward prediction
        of `x[n]` from the `p` previous values (i.e. `x[(n - p):n]`) and
        AR coefficients `a`, where `i = n - p`.
        [error] = [x]

    '''
    N = len(x)
    p = len(a) - 1

    if p > (N - 1):
        raise ValueError('`len(a)` must be less than or equal to `len(x)`')

    # When convolving a vector of shape (`N`,) with a
    # kernel of shape (`k`,) convolution is *fastest*
    # via an FFT if
    #
    #               k >= 4 log_2 (N)
    #
    # Otherwise, a straightforward convolution is faster.
    # Relevant background information here:
    #
    #       http://programmers.stackexchange.com/a/172839
    #
    mode = 'valid'

    if len(a) >= (4 * np.log2(len(x))):
        return fftconvolve(x, a, mode=mode)
    else:
        return np.convolve(x, a, mode=mode)


def backward_prediction_error(x, a):
    '''Get backward-prediction error for autoregression (AR) of `x`
    with *forward-prediction* AR coefficients `a`.

    Note that this is Marple's Eq. (4.20) with p <= n < N.

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    a - array_like, (`p + 1`,)
        The AR coefficients for the `p`-order *forward-prediction*
        autoregression. Assuming a stationary process, the corresponding
        backward-prediction AR coefficients, needed for the computation here,
        are simply the forward-prediction coefficients conjugated and
        reversed in time.
        [a] = unitless

    Returns:
    --------
    error - array_like, (`N - p`,)
        The backward-prediction error, restricted to the available data.
        See documentation for `forward_prediction_error` for more information.
        [error] = [x]

    '''
    return forward_prediction_error(x, np.conjugate(a[::-1]))


def total_error_energy(x, a):
    '''Get sum of forward- and backward-prediction error energies for
    autoregression (AR) of `x` with forward-prediction AR coefficients `a`.

    Note that this is Marple's Eq. (4.21).

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    a - array_like, (`p + 1`,)
        The AR coefficients for the `p`-order forward-prediction
        autoregression. By definition, `a[0]` is unity for all orders.
        [a] = unitless

    Returns:
    --------
    Ep - float
        The sum of the forward- and backward-prediction error energies
        [Ep] = [x]^2

    '''
    # Obtain forward- and backward-prediction errors
    ef = forward_prediction_error(x, a)
    eb = backward_prediction_error(x, a)

    return np.sum((np.abs(ef) ** 2) + (np.abs(eb) ** 2))


def next_order_a_pp(x, a):
    '''Generate pth forward-prediction AR coefficient for the `p`-order
    autoregression of `x` using `p - 1`-order coefficients `a`.

    Note that this is Marple's Eq. (4.23).

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    a - array_like, (`p`,)
        The AR coefficients for the `p - 1`-order forward-prediction
        autoregression. By definition, `a[0]` is unity for all orders.
        [a] = unitless

    Returns:
    --------
    a_pp - float
        The pth forward-prediction AR coefficient for the `p`-order
        autoregression. The remainder of the remainder of the `p`-order
        forward-prediction AR coefficients can be generated from the
        Levinson recursion. Thus, using the `p - 1`-order forward-prediction
        AR coefficients `a` provided as input, we can generate all of the
        `p`-order AR coefficients.
        [a_pp] = unitless

    '''
    # Obtain forward- and backward-prediction errors from
    # AR model of order `p - 1`, and store only the slices
    # needed for the computation
    ef = forward_prediction_error(x, a)[1:]
    eb = backward_prediction_error(x, a)[:-1]

    num = -2 * np.dot(np.conj(eb), ef)
    den = np.real(np.dot(np.conj(eb), eb) + np.dot(np.conj(ef), ef))

    return num / den


def levinson_recursion(a, anext_pp):
    '''Generate 1st through (p - 1)th forward-prediction AR coefficients
    for the `p`-order autoregression.

    Note that this is Marple's Eq. (4.22).

    Input parameters:
    -----------------
    a - array_like, (`p`,)
        The AR coefficients for the previous order (i.e. `p - 1`)
        forward-prediction autoregression. By definition, `a[0]`
        is unity for all orders.
        [a_prev] = unitles

    anext_pp - float
        The pth forward-prediction AR coefficient for the `p`-order
        autoregression.
        [anew_pp] = unitles

    Returns:
    --------
    a_levinson - array_like, (`p - 1`,)
        The 1st through (p - 1)th forward-prediction AR coefficients
        for the `p`-order autoregression, generated via the Levinson
        recursion.
        [a_levinson] = unitles

    '''
    return a[1:] + (anext_pp * np.conj(a[1:][::-1]))


def next_order_coefficients(x, a):
    '''Generate all `p`-order forward-prediction AR coefficients for
    autoregression of `x` using `p - 1`-order coefficients `a`.

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    a - array_like, (`p`,)
        The AR coefficients for the `p - 1`-order forward-prediction
        autoregression. By definition, `a[0]` is unity for all orders.
        [a] = unitless

    Returns:
    --------
    anext - array_like, (`p + 1`)
        The `p`-order forward-prediction coefficients for autoregression
        of `x`.
        [anext] = unitless

    '''
    # Initialize
    anext = np.zeros(len(a) + 1, dtype='complex')

    # By definition...
    anext[0] = 1

    # By Burg's algorithm of minimizing total energy in forward and
    # backward errors, subject to constraint of Levinson recursion
    anext[-1] = next_order_a_pp(x, a)

    # By Levinson recursion
    anext[1:-1] = levinson_recursion(a, anext[-1])

    return anext


def burg_coefficients(x, p):
    '''Use the Burg algorithm to generate the forward-prediction AR
    coefficients for the `p`-order autoregression of `x`.

    Input parameters:
    -----------------
    x - array_like, (`N`,)
        The signal to be autoregressed.
        [x] = arbitrary units

    p - int
        The AR order.
        [p] = unitless

    Returns:
    --------
    a - array_like, (`p + 1`,)
        The `p`-order forward-prediction AR coefficients for `x`.
        [a] = unitless

    '''
    # Initialize
    a = np.array([1])

    # Generate requested coefficients recursively. Each recursion
    # increases the length of `a` by one, and `len(a) = p + 1` for
    # a `p`-order autoregression.
    while len(a) <= p:
        a = next_order_coefficients(x, a)

    return a
