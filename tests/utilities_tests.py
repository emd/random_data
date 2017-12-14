from nose import tools
import numpy as np
import random_data as rd
import scipy as sp


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


def test_coord2ind():
    offset = 1
    grid = offset + np.arange(10)

    # Valid indices:
    # ==============

    # In domain, scalar value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(4, grid, valid_index=True),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(4.2, grid, valid_index=True),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(4.7, grid, valid_index=True),
        5 - offset)

    # In domain, array value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4, 5]), grid, valid_index=True),
        np.array([4, 5]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4.2, 5.2]), grid, valid_index=True),
        np.array([4, 5]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4.7, 5.7]), grid, valid_index=True),
        np.array([5, 6]) - offset)

    # Out of domain, scalar value:
    # ----------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(-1, grid, valid_index=True),
        0)

    np.testing.assert_equal(
        rd.utilities.coord2ind(11, grid, valid_index=True),
        9)

    # Out of domain, array value:
    # ---------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([-2, -1]), grid, valid_index=True),
        np.array([0, 0]))

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([11, 12]), grid, valid_index=True),
        np.array([9, 9]))

    # Generalized indices:
    # ===================

    # In domain, scalar value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(4, grid, valid_index=False),
        4 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(4.2, grid, valid_index=False),
        4.2 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(4.7, grid, valid_index=False),
        4.7 - offset)

    # In domain, array value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4, 5]), grid, valid_index=False),
        np.array([4, 5]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4.2, 5.2]), grid, valid_index=False),
        np.array([4.2, 5.2]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([4.7, 5.7]), grid, valid_index=False),
        np.array([4.7, 5.7]) - offset)

    # Out of domain, scalar value:
    # ----------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(-1, grid, valid_index=False),
        -1 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(-2.2, grid, valid_index=False),
        -2.2 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(-2.7, grid, valid_index=False),
        -2.7 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(11, grid, valid_index=False),
        11 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(11.2, grid, valid_index=False),
        11.2 - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(11.7, grid, valid_index=False),
        11.7 - offset)

    # Out of domain, array value:
    # ---------------------------
    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([-2, -1]), grid, valid_index=False),
        np.array([-2, -1]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(-np.array([2.2, 1.2]), grid, valid_index=False),
        -np.array([2.2, 1.2]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(-np.array([2.7, 1.7]), grid, valid_index=False),
        -np.array([2.7, 1.7]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(np.array([11, 12]), grid, valid_index=False),
        np.array([11, 12]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(
            np.array([11.2, 12.2]), grid, valid_index=False),
        np.array([11.2, 12.2]) - offset)

    np.testing.assert_equal(
        rd.utilities.coord2ind(
            np.array([11.7, 12.7]), grid, valid_index=False),
        np.array([11.7, 12.7]) - offset)

    return


def test_ind2coord():
    offset = 1
    grid = offset + np.arange(10)

    # In domain, scalar value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.ind2coord(4, grid),
        4 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(4.25, grid),
        4.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(4.75, grid),
        4.75 + offset)

    # In domain, array value:
    # ------------------------
    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([4, 5]), grid),
        np.array([4, 5]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([4.25, 5.25]), grid),
        np.array([4.25, 5.25]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([4.75, 5.75]), grid),
        np.array([4.75, 5.75]) + offset)

    # Out of domain, scalar value:
    # ----------------------------
    np.testing.assert_equal(
        rd.utilities.ind2coord(-1, grid),
        -1 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(-2.25, grid),
        -2.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(-2.75, grid),
        -2.75 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(10, grid),
        10 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(10.25, grid),
        10.25 + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(10.75, grid),
        10.75 + offset)

    # Out of domain, array value:
    # ---------------------------
    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([-2, -1]), grid),
        np.array([-2, -1]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([-2.25, -1.25]), grid),
        np.array([-2.25, -1.25]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([-2.75, -1.75]), grid),
        np.array([-2.75, -1.75]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([10, 11]), grid),
        np.array([10, 11]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([10.25, 11.25]), grid),
        np.array([10.25, 11.25]) + offset)

    np.testing.assert_equal(
        rd.utilities.ind2coord(np.array([10.75, 11.75]), grid),
        np.array([10.75, 11.75]) + offset)

    return


def test_get_timebase_indices():
    # (a) timebase: 0 <= t <= 9 w/ dt = 1:
    # ------------------------------------
    Fs = 1.
    t0 = 0.
    Npts = 10

    # `tlim` specifies a subset of timebase
    tlim = [3, 7]
    np.testing.assert_equal(
        rd.utilities.get_timebase_indices(tlim, Fs, t0, Npts),
        [3, 4, 5, 6, 7])

    # `tlim` specifies full timebase
    tlim = [0, 9]
    np.testing.assert_equal(
        rd.utilities.get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # `tlim` beyond full timebase
    tlim = [-1, 11]
    np.testing.assert_equal(
        rd.utilities.get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # `tlim` is `None`
    tlim = None
    np.testing.assert_equal(
        rd.utilities.get_timebase_indices(tlim, Fs, t0, Npts),
        np.arange(Npts))

    # (b) timebase: 1 <= t <= 5 w/ dt = 0.5:
    # --------------------------------------
    Fs = 2.
    t0 = 1.
    Npts = 9

    # The tests in (a) checked that we have appropriate behavior
    # at the boundary cases. Here, we just want to check that
    # things work when specifying non-zero `t0` and non-unity `Fs`,
    # so we'll just perform a single check with `tlim` specifying
    # a subset of the timebase.
    tlim = [3, 4]
    np.testing.assert_equal(
        rd.utilities.get_timebase_indices(tlim, Fs, t0, Npts),
        [4, 5, 6])

    return


def test_line_profile_coordinates():
    # Specifying both `lwr` and `lwc` should raise a ValueError:
    # ----------------------------------------------------------
    src = [0, 0]
    dst = [10, 10]
    N = 11
    lwr = 1.
    lwc = 1.
    L = 5

    tools.assert_raises(
        ValueError,
        rd.utilities.line_profile_coordinates,
        *[src, dst],
        **{'N':N, 'lwr':lwr, 'lwc':lwc, 'L':L})

    # Single 1:1 line:
    # ----------------
    src = [0, 0]
    dst = [10, 10]
    N = 11

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N)

    np.testing.assert_equal(
        np.squeeze(lines[0, ...]),
        np.linspace(src[0], dst[0], N))
    np.testing.assert_equal(
        np.squeeze(lines[1, ...]),
        np.linspace(src[1], dst[1], N))

    # Single 2:1 line:
    # ----------------
    src = [0, 0]
    dst = [20, 10]
    N = 11

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N)

    np.testing.assert_equal(
        np.squeeze(lines[0, ...]),
        np.linspace(src[0], dst[0], N))
    np.testing.assert_equal(
        np.squeeze(lines[1, ...]),
        np.linspace(src[1], dst[1], N))

    # Odd number of lines, spread in row-space:
    # -----------------------------------------
    src = [0, 0]
    dst = [10, 10]
    N = 11
    lwr = 1.
    L = 5
    offsets = np.array([-0.5, -0.25, 0, 0.25, 0.5])

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N, lwr=lwr, L=L)

    for lind in np.arange(lines.shape[-1]):
        np.testing.assert_almost_equal(  # `almost` needed for roundoff errors
            np.squeeze(lines[0, :, lind]),
            np.linspace(src[0], dst[0], N) + offsets[lind])
        np.testing.assert_equal(
            np.squeeze(lines[1, :, lind]),
            np.linspace(src[1], dst[1], N))

    # Even number of lines, spread in row-space:
    # ------------------------------------------
    src = [0, 0]
    dst = [10, 10]
    N = 11
    lwr = 1.
    L = 6
    offsets = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N, lwr=lwr, L=L)

    for lind in np.arange(lines.shape[-1]):
        print lind
        np.testing.assert_almost_equal(  # `almost` needed for roundoff errors
            np.squeeze(lines[0, :, lind]),
            np.linspace(src[0], dst[0], N) + offsets[lind])
        np.testing.assert_equal(
            np.squeeze(lines[1, :, lind]),
            np.linspace(src[1], dst[1], N))

    # Odd number of lines, spread in column-space:
    # --------------------------------------------
    src = [0, 0]
    dst = [10, 10]
    N = 11
    lwc = 1.
    L = 5
    offsets = np.array([-0.5, -0.25, 0, 0.25, 0.5])

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N, lwc=lwc, L=L)

    for lind in np.arange(lines.shape[-1]):
        np.testing.assert_equal(
            np.squeeze(lines[0, :, lind]),
            np.linspace(src[0], dst[0], N))
        np.testing.assert_almost_equal(  # `almost` needed for roundoff errors
            np.squeeze(lines[1, :, lind]),
            np.linspace(src[1], dst[1], N) + offsets[lind])

    # Even number of lines, spread in row-space:
    # ------------------------------------------
    src = [0, 0]
    dst = [10, 10]
    N = 11
    lwc = 1.
    L = 6
    offsets = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])

    lines = rd.utilities.line_profile_coordinates(
        src, dst, N=N, lwc=lwc, L=L)

    for lind in np.arange(lines.shape[-1]):
        np.testing.assert_equal(
            np.squeeze(lines[0, :, lind]),
            np.linspace(src[0], dst[0], N))
        np.testing.assert_almost_equal(  # `almost` needed for roundoff errors
            np.squeeze(lines[1, :, lind]),
            np.linspace(src[1], dst[1], N) + offsets[lind])

    return


def test_line_profile():
    # Construct grid and banded matrix:
    # =================================

    # (Non-trivial) grid parameters, to stress test algorithm:
    # --------------------------------------------------------
    Nr = 10
    dr = 2.
    r0 = 10

    Nc = 13
    dc = 3.
    c0 = -5

    # Construct grid:
    # ---------------
    row = r0 + (dr * np.arange(Nr))
    col = c0 + (dc * np.arange(Nc))

    # Construct banded, non-square matrix:
    # ------------------------------------
    ind_offset = 2  # positive offset requires working w/ columns below
    diagonal = np.ones(Nr)
    img = sp.sparse.diags(
        diagonal, offsets=ind_offset, shape=(Nr, Nc)).toarray()

    # Measure various average profiles:
    # =================================
    mc_kwargs = {'order': 0}  # Nearest-neighbor interp. -> easy comparisons

    # Minimize boundary effects by taking points "well away" from boundaries
    ind = 2

    r1 = row[ind]
    r2 = row[-ind]

    r1ind = rd.utilities.coord2ind(r1, row, valid_index=False)
    r2ind = rd.utilities.coord2ind(r2, row, valid_index=False)

    c1 = col[0] + (dc * (ind + ind_offset))
    c2 = c1 + (dc * (r2ind - r1ind))

    src = (r1, c1)
    dst = (r2, c2)

    # Single line passing through `src` and `dst`:
    # --------------------------------------------
    N = 10
    lpc_kwargs = {'N': N, 'lwr': None, 'lwc': None, 'L': 1}
    img_prof, row_prof, col_prof = rd.utilities.line_profile(
        img, row, col, src, dst,
        lpc_kwargs=lpc_kwargs,
        mc_kwargs=mc_kwargs)

    np.testing.assert_equal(img_prof, np.ones(N))

    # Two lines passing *outside* of non-zero region:
    # -----------------------------------------------
    N = 10
    lpc_kwargs = {'N': N, 'lwr': 2 * dr, 'lwc': None, 'L': 2}
    img_prof, row_prof, col_prof = rd.utilities.line_profile(
        img, row, col, src, dst,
        lpc_kwargs=lpc_kwargs,
        mc_kwargs=mc_kwargs)

    np.testing.assert_equal(img_prof, np.zeros(N))

    # Three lines with only central line passing through non-zero region:
    # -------------------------------------------------------------------
    N = 10
    lpc_kwargs = {'N': N, 'lwr': 2 * dr, 'lwc': None, 'L': 3}
    img_prof, row_prof, col_prof = rd.utilities.line_profile(
        img, row, col, src, dst,
        lpc_kwargs=lpc_kwargs,
        mc_kwargs=mc_kwargs)

    np.testing.assert_almost_equal(img_prof, np.ones(N) / 3.)

    return
