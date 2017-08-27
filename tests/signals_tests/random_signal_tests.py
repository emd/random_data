import numpy as np
import random_data as rd


def test__uniform_grid():
    # Simple grid
    Npts = 10
    x0 = 0
    dx = 1
    desired_grid = np.arange(Npts)

    # Base test case
    np.testing.assert_equal(
        rd.signals.random_signal._uniform_grid(Npts, x0, dx),
        desired_grid)

    # Change sample rate
    np.testing.assert_equal(
        rd.signals.random_signal._uniform_grid(Npts, x0, 2 * dx),
        2 * desired_grid)

    # Change initial point
    np.testing.assert_equal(
        rd.signals.random_signal._uniform_grid(Npts, x0 - 1, dx),
        desired_grid - 1)

    # Change sample rate and initial point
    np.testing.assert_equal(
        rd.signals.random_signal._uniform_grid(Npts, x0 - 1, 2 * dx),
        (2 * desired_grid) - 1)

    return


def test_RandomSignal2d():
    # `RandomSignal2d` neglects the imaginary component of `self.x`.
    # Ensure that the Fourier representation of `self.x` is a close
    # approximation to the original Fourier representation used
    # to create `self.x` (i.e. check that the neglected imaginary
    # component of `self.x` is negligible).
    sig2d = rd.signals.RandomSignal2d()

    # Compute Fourier transforms
    X_derived = np.fft.fftshift(np.fft.fft2(sig2d.x))
    X_expected = np.fft.fftshift(sig2d._X)

    # Don't include points at the Nyquist frequencies, as these points
    # seem to be especially susceptible to artifacts. We are almost
    # never interested in the behavior just at the Nyquist frequency,
    # though, so this should not be anything to worry about.
    X_derived = X_derived[1:, 1:]
    X_expected = X_expected[1:, 1:]

    np.testing.assert_almost_equal(
        X_derived,
        X_expected)

    return
