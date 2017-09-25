from nose import tools
import numpy as np
import random_data as rd


def test_get_uniform_spacing():
    # Non-uniformly spaced array should raise `ValueError`
    x = np.arange(10)
    x[4] = 10
    tools.assert_raises(ValueError, rd.utilities.get_uniform_spacing, x)

    # Uniformly spaced array of integers
    x = np.arange(10)
    tools.assert_equal(rd.utilities.get_uniform_spacing(x), 1)

    # Uniformly spaced array of floats
    x = np.arange(10.) / 10
    tools.assert_equal(rd.utilities.get_uniform_spacing(x), 0.1)

    return


def test_val2ind():
    offset = 1
    grid = offset + np.arange(10)

    # Valid indices:
    # ==============

    # In domain:
    # ----------
    np.testing.assert_equal(
        rd.utilities.val2ind(4, grid, valid_index=True),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(4.25, grid, valid_index=True),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(4.75, grid, valid_index=True),
        5 - offset)

    # Out of domain:
    # --------------
    np.testing.assert_equal(
        rd.utilities.val2ind(-1, grid, valid_index=True),
        0)

    np.testing.assert_equal(
        rd.utilities.val2ind(11, grid, valid_index=True),
        9)

    # Generalized indices:
    # ===================

    # In domain:
    # ----------
    np.testing.assert_equal(
        rd.utilities.val2ind(4, grid, valid_index=False),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(4.25, grid, valid_index=False),
        4.25 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(4.75, grid, valid_index=False),
        4.75 - offset)

    # Out of domain:
    # --------------
    np.testing.assert_equal(
        rd.utilities.val2ind(-1, grid, valid_index=False),
        -1 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(-2.25, grid, valid_index=False),
        -2.25 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(-2.75, grid, valid_index=False),
        -2.75 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(11, grid, valid_index=False),
        11 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(11.25, grid, valid_index=False),
        11.25 - offset)

    np.testing.assert_equal(
        rd.utilities.val2ind(11.75, grid, valid_index=False),
        11.75 - offset)

    return


def test_ind2val():
    offset = 1
    grid = offset + np.arange(10)

    # In domain:
    # ----------
    np.testing.assert_equal(
        rd.utilities.ind2val(4, grid),
        4 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(4.25, grid),
        4.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(4.75, grid),
        4.75 + offset)

    # Out of domain:
    # --------------
    np.testing.assert_equal(
        rd.utilities.ind2val(-1, grid),
        -1 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(-2.25, grid),
        -2.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(-2.75, grid),
        -2.75 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(10, grid),
        10 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(10.25, grid),
        10.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2val(10.75, grid),
        10.75 + offset)

    return
