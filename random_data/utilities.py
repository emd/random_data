import numpy as np
import scipy.ndimage as ndi


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


def line_profile(img, row, col, src, dst, lpc_kwargs={}, mc_kwargs={}):
    '''Get profile of `img` along line specified by `src` and `dst`.

    Input parameters:
    -----------------
    img - array_like, `(M, K)`
        The image to be profiled along the specified line.
        [img] = arbitrary units

    row -array_like, `(M,)`
        The coordinates corresponding to the rows of `img`.
        [row] = arbitrary units

    col - array_like, `(K,)`
        The coordinates corresponding to the columns of `img`.
        [col] = arbitrary units

    src - array_like, `(2,)`
        The coordinates of the starting point (i.e. the source)
        of the line scan.
        [src[0]] = [row]
        [src[1]] = [col]

    dst - array_like, `(2,)`
        The coordinates of the ending point (i.e. the destination)
        of the line scan.
        [dst[0]] = [row]
        [dst[1]] = [col]

    lpc_kwargs - dictionary, where keys are any valid keyword arguments for
            :py:func:`random_data.utilities.line_profile_coordinates`

        For example, use

                prof = line_profile(..., lpc_kwargs={'lwr': 5, 'L': 7})

        to indicate that the line profile should be averaged over `L`
        equispaced lines parallel to the line specified by `src` and
        `dst` with the total point-to-point span of the parallel lines
        corresponding to `lwr` units in the row-coordinate direction.

    mc_kwargs - dictionary, where keys are any valid keyword arguments for
            :py:func:`scipy.ndimage.map_coordinates`

        For example, use

                prof = line_profile(..., mc_kwargs={'order': 3})

        to indicate that the line profile should be obtained by using
        an order-3 spline interpolation of the image's pixels.

    Returns:
    --------
    (img_prof, row_prof, col_prof) - tuple, where

    img_prof - array_like, `(N,)`
        The (potentially averaged) profile of `img` along the specified line.
        [img_prof] = [img]

    row_prof - array_like, `(N,)`
        The `row`-like coordinates corresponding to profile `img_prof`
        [row_prof] = [row]

    col_prof - array_like, `(N,)`
        The `col`-like coordinates corresponding to profile `img_prof`
        [col_prof] = [col]

    '''
    # Obtain lines in coordinate space
    coord_lines = line_profile_coordinates(src, dst, **lpc_kwargs)

    # Initialize array to hold index-space representation of lines
    ind_lines = np.zeros(coord_lines.shape)

    # Loop through each line, converting from the coordinate space
    # to index space
    for lind in np.arange(coord_lines.shape[-1]):
        ind_lines[0, :, lind] = coord2ind(
            coord_lines[0, :, lind], row, valid_index=False)
        ind_lines[1, :, lind] = coord2ind(
            coord_lines[1, :, lind], col, valid_index=False)

    # Get average profile of `img` along lines
    img_prof = np.mean(ndi.map_coordinates(
        img, ind_lines, **mc_kwargs), axis=-1)

    # Get values of `row` and `col` corresponding to profile image profile
    row_prof = np.mean(coord_lines[0, ...], axis=-1)
    col_prof = np.mean(coord_lines[1, ...], axis=-1)

    return img_prof, row_prof, col_prof
