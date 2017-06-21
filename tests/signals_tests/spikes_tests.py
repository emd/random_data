from nose import tools
import numpy as np
from random_data.signals.spikes import _subset_boundary_values


def test__subset_boundary_values():
    # Monotonicity test:
    # ==================
    x = np.arange(10)
    x[5] = 0                # `x` is now nonmonotonic
    tools.assert_raises(
        ValueError,
        _subset_boundary_values, *[x])

    # Trivial subsets:
    # ================

    # Trivial subset test 1: only one subset in `x`
    # ---------------------------------------------
    x = np.arange(10)

    vals = _subset_boundary_values(x, min_subset_spacing=2)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        np.array([x[0]]))

    np.testing.assert_equal(
        subset_stop_vals,
        np.array([x[-1]]))

    # Trivial subset test 2: uniform spacing by `min_subset_spacing`
    # such that each point in `x` is a distinct subset
    # --------------------------------------------------------------
    x = np.array([0, 1, 2, 3])

    vals = _subset_boundary_values(x, min_subset_spacing=1)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        x)

    np.testing.assert_equal(
        subset_stop_vals,
        x)

    # Trivial subset test 3: non-uniform spacing by >= `min_subset_spacing`
    # such that each point in `x` is a distinct subset
    # ---------------------------------------------------------------------
    x = np.array([0, 1, 10, 33])

    vals = _subset_boundary_values(x, min_subset_spacing=1)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        x)

    np.testing.assert_equal(
        subset_stop_vals,
        x)

    # Non-trivial subsets:
    # ====================

    # Non-trivial subset test 1: all subsets have length > 1
    # -------------------------------------------------------
    x = np.array([
        0, 1, 2, 3,         # subset 1
        5, 6, 7, 8,         # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(x, min_subset_spacing=2)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 13])

    # Non-trivial subset test 2: middle subset has unity length
    # ---------------------------------------------------------
    x = np.array([
        0, 1, 2, 3,         # subset 1
        5,                  # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(x, min_subset_spacing=2)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 5, 13])

    # Non-trivial subset test 3: first subset has unity length
    # --------------------------------------------------------
    x = np.array([
        3,                  # subset 1
        5, 6, 7, 8,         # subset 2
        10, 11, 12, 13])    # subset 3

    vals = _subset_boundary_values(x, min_subset_spacing=2)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [3, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 13])

    # Non-trivial subset test 4: last subset has unity length
    # -------------------------------------------------------
    x = np.array([
        0, 1, 2, 3,     # subset 1
        5, 6, 7, 8,     # subset 2
        10])            # subset 3

    vals = _subset_boundary_values(x, min_subset_spacing=2)
    subset_start_vals = vals[0]
    subset_stop_vals = vals[1]

    np.testing.assert_equal(
        subset_start_vals,
        [0, 5, 10])

    np.testing.assert_equal(
        subset_stop_vals,
        [3, 8, 10])

    return
