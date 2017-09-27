import numpy as np
import skimage.measure


def get_uniform_spacing(x):
    '''Get spacing of uniformly spaced array `x`.

    Input parameters:
    -----------------
    x - array_like, `(N,)`
        The uniformly spaced array. If `x` is *not* uniformly spaced,
        a `ValueError` is raised.
        [x] = arbitrary units

    Returns:
    --------
    dx - float
        The uniform spacing of `x`.
        [dx] = [x]

    '''
    # Due to round-off errors, `diff(x)` will not always be zero, even if `x`
    # is uniformly spaced. Instead, check that deviations from uniform spacing
    # are acceptably low (i.e. up to numerical resolution)
    if np.allclose(np.var(np.diff(x)), 0):
        return np.float(x[1] - x[0])
    else:
        raise ValueError('`x` must be uniformly spaced')


def coord2ind(coord, grid, valid_index=True):
    '''Map coordinates `coord` to the "index space" of `grid`.

    Input parameters:
    -----------------
    coord - array_like, `(M,)`
        The coordinate(s) to be mapped to the "index space" of `grid`.
        [coord] = arbitrary units

    grid - array_like, `(N,)`
        The grid against which `coord` is indexed. A `ValueError` is raised
        if `grid` is *not* uniformly spaced.
        [grid] = [coord]

    valid_index - bool
        If True, the returned indices are constrained to be a valid indices
        of `grid`; that is, `grid[ind]` will *not* raise an `IndexError`.

    Returns:
    --------
    ind - array_like, `(M,)`, where the data type is int if `valid_index`
            is True; float otherwise
        The "indices" of `coord` relative to `grid`. Note that `ind` can
        only be used to index `grid` *if* `valid_index` is True.
        [ind] = unitless

    '''
    grid_spacing = get_uniform_spacing(grid)
    ind = (coord - grid[0]) / grid_spacing

    if valid_index:
        # Enforce floor and ceiling of valid indices for `grid`
        ind = np.maximum(ind, 0)
        ind = np.minimum(ind, len(grid) - 1)

        # Round to nearest integer
        ind = (np.round(ind)).astype('int')

    return ind


def ind2coord(ind, grid):
    '''Map generalized index `ind` to corresponding coordinate on `grid`.

    Input parameters:
    -----------------
    ind - array_like, `(M,)`
        The generalized indices that should be mapped to their corresponding
        coordinates on `grid`. Note that `ind` are "generalized indices" in
        that `grid[ind]` may raise an `IndexError`; however, as `grid` must
        be uniformly spaced, linear interpolation or extrapolation yields
        the correct coordinates of `grid` at generalized indices `ind`.
        [ind] = unitless

    grid - array_like, `(N,)`
        The grid against which `ind` is evaluated. A `ValueError` is raised
        if `grid` is *not* uniformly spaced.
        [grid] = arbitrary units

    Returns:
    --------
    coord - array_like, `(M,)`
        The coordinates of `grid` corresponding to generalized indices `ind`.
        [coord] = [grid]

    '''
    grid_spacing = get_uniform_spacing(grid)
    return grid[0] + (ind * grid_spacing)


def line_profile_coordinates(src, dst, N=100, lwr=None, lwc=None, L=5):
    '''Get coordinates of image profile along specified scan line.

    Input parameters:
    -----------------
    src - array_like, (2,)
        The starting point (i.e. source) of the scan line.
        [src[0]] = arbitrary units
        [src[1]] = arbitrary units

    dst - array_like, (2,)
        The end point (i.e. destination) of the scan line.
        [dst[0]] = [src[0]]
        [dst[1]] = [src[1]]

    N - int
        The number of points along the scan line.

    lwr - float or None
        If not `None`, return coordinates for `L` lines parallel to the
        specified scan line spanning row-coordinate linewidth `lwr` and
        centered on the specified scan line. Note that `lwr` and `lwc`
        *cannot* both be simultaneously specified.
        [lwr] = [src[0]]

    lwc - float or None
        If not `None`, return coordinates for `L` lines parallel to the
        specified scan line spanning column-coordinate linewidth `lwc` and
        centered on the specified scan line. Note that `lwr` and `lwc`
        *cannot* both be simultaneously specified.
        [lwc] = [src[1]]

    L - int
        The number of lines to return coordinates for if `lwr` or `lwc`
        is *not* None.

    Returns:
    --------
    coords - array_like, `(2, N, L)`
        The coordinates of image profile along the specified scan line(s).
        `coords[0, ...]` gives the row coordinates, and `coords[1, ...]`
        gives the column coordinates. `N` is the number of points in the
        profile, as specified during input. Finally, if `lwr` or `lwc` is
        *not* None, `L` corresponds to the number of scan lines specified
        during input; if `lwr` and `lwc` are both None, then `L = 1` and
        the single returned line begins at `src` and ends at `dst`.

    Note:
    -----
    This routine is inspired by scikit-image's

        :py:function:`skimage.measure.profile._line_profile_coordinates`

    However, as that function is *internal* to scikit-image, subject to
    change without notice, it seemed unwise to rely on it for our needs
    here. Thus, we've written a similar function for our own needs.

    '''
    # Get row and column components of scan line connecting `src` to `dst`
    line_row = np.linspace(src[0], dst[0], N)
    line_col = np.linspace(src[1], dst[1], N)

    if (lwr is not None) and (lwc is not None):
        raise ValueError('`lwr` and `lwc` *cannot* both be specified')
    if (lwr is not None) or (lwc is not None):
        # Initialize
        line_rows = np.zeros(line_row.shape + (L,))
        line_cols = np.zeros(line_col.shape + (L,))

        if lwr is not None:
            for lind, delta in enumerate(lwr * np.linspace(-0.5, 0.5, L)):
                line_rows[..., lind] = line_row + delta
                line_cols[..., lind] = line_col
        else:
            for lind, delta in enumerate(lwc * np.linspace(-0.5, 0.5, L)):
                line_rows[..., lind] = line_row
                line_cols[..., lind] = line_col + delta
    else:
        line_rows = line_row[..., np.newaxis]
        line_cols = line_col[..., np.newaxis]

    return np.array([line_rows, line_cols])


def profile_line(z, row, col, start, stop, lwr=None, lwc=None, **profile_kwargs):
    '''Get profile of `z` along specified line.

    Input parameters:
    -----------------
    z - array_like, `(M, N)`
        The array to be profiled along the specified line.
        [z] = arbitrary units

    row - array_like, `(M,)`
        The coordinates corresponding to the rows of `z`.
        [row] = arbitrary units

    col - array_like, `(N,)`
        The coordinates corresponding to the columns of `z`.
        [col] = arbitrary units

    start - 2-tuple of floats
        The start point of the line scan.
        [start[0]] = [row], [start[1]] = [col]

    stop - 2-tuple of floats
        The end point of the line scan.
        [stop[0]] = [row], [stop[1]] = [col]

    lwr - float or None
        If not `None`, average the line scan over `lwr` in row-direction.
        Note that `lwr` and `lwc` *cannot* both be simultaneously specified
        as non-None.
        [lwr] = [row]

    lwc - float or None
        If not `None`, average the line scan over `lwc` in column-direction.
        Note that `lwr` and `lwc` *cannot* both be simultaneously specified
        as non-None.
        [lwc] = [col]

    profile_kwargs - any valid keyword arguments for
        :py:function:`profile_line <skimage.measure.profile_line>`

        For example, use

                rd.utilities.profile_line(..., order=3)

        to use a 3rd order spline interpolation to compute image values
        at non-integer coordinates.

    Returns:
    --------
    (zp, xp, yp, bnd_lines) - tuple, where

    zp - array_like, `(L,)`
        The (potentially averaged) profile of `z` along the specified line.
        [zp] = [z]

    rp - array_like, `(L,)`
        The `row`-like coordinates corresponding to profile `zp`
        [rp] = [row]

    cp - array_like, `(L,)`
        The `col`-like coordinates corresponding to profile `zp`
        [cp] = [col]

    bnd_lines - array_like, `(2, L, a)` where `a in {1, 2}`

    '''
    # Map `start` and `stop` to index space
    r1ind = val2ind(start[0], row, valid_index=False)
    r2ind = val2ind(stop[0], row, valid_index=False)
    c1ind = val2ind(start[1], col, valid_index=False)
    c2ind = val2ind(stop[1], col, valid_index=False)

    src = np.array([r1ind, c1ind])
    dst = np.array([r2ind, c2ind])

    # Determine `linewidth` parameter
    if (lwr is not None) and (lwc is not None):
        raise ValueError('`lwr` and `lwc` *cannot* both be specified')
    elif (lwr is not None) or (lwc is not None):
        theta = np.arctan2(dst[1] - src[1], dst[0] - src[0])

        if lwr is not None:
            ilwr = val2ind(lwr, row, valid_index=False)
            linewidth = np.int(np.abs(ilwr * np.sin(theta)))
        else:
            ilwc = val2ind(lwc, col, valid_index=False)
            linewidth = np.int(np.abs(ilwc * np.cos(theta)))

        # Enforce minimum linewidth
        if linewidth < 1:
            linewidth = 1
    else:
        linewidth = 1

    # Get profile of `z`
    zp = skimage.measure.profile_line(
        z, src, dst, linewidth=linewidth, **profile_kwargs)

    # Get values of `x` and `y` corresponding to profile `zp`
    rind = np.linspace(r1ind, r2ind, len(zp))
    cind = np.linspace(c1ind, c2ind, len(zp))

    rp = ind2val(rind, row)
    cp = ind2val(cind, col)

    # Finally, get bounding lines
    lines = skimage.measure.profile._line_profile_coordinates(
        src, dst, linewidth=linewidth)

    bnd_rows = np.array([
        ind2val(lines[0, :, 0], row),
        ind2val(lines[0, :, -1], row)])
    bnd_cols = np.array([
        ind2val(lines[1, :, 0], col),
        ind2val(lines[1, :, -1], col)])

    bnd_lines = np.array([bnd_rows.T, bnd_cols.T])

    return zp, rp, cp, bnd_lines
