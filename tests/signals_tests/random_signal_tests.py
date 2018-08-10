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
