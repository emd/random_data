import numpy as np


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


def val2ind(val, grid, valid_index=True):
    '''Map value `val` to the "index space" of `grid`.

    Input parameters:
    -----------------
    val - float
        The value to be mapped to the "index space" of `grid`.
        [val] = arbitrary units

    grid - array_like, `(N,)`
        The grid against which `val` is indexed. A `ValueError` is raised
        if `grid` is *not* uniformly spaced.
        [grid] = [val]

    valid_index - bool
        If True, the returned index is constrained to be a valid index
        of `grid`; that is, `grid[ind]` will *not* raise an `IndexError`.

    Returns:
    --------
    ind - int if `valid_index` is True; float otherwise
        The "index" of `val` relative to `grid`. Note that `ind` can
        only be used to index `grid` *if* `valid_index` is True.
        [ind] = unitless

    '''
    grid_spacing = get_uniform_spacing(grid)

    delta = val - grid
    ind = np.where(np.abs(delta) == np.min(np.abs(delta)))[0][0]

    if valid_index:
        return ind
    else:
        return ind + (delta[ind] / grid_spacing)


def ind2val(ind, grid):
    '''Map generalized index `ind` to corresponding value on `grid`.

    Input parameters:
    -----------------
    ind - float
        The generalized index that should be mapped to its corresponding
        value on `grid`. Note that `ind` is a "generalized index" in that
        `grid[ind]` may raise an `IndexError`; however, as `grid` must
        be uniformly spaced, linear interpolation or extrapolation yields
        the correct value of `grid` at generalized index `ind`.
        [ind] = unitless

    grid - array_like, `(N,)`
        The grid against which `ind` is evaluated. A `ValueError` is raised
        if `grid` is *not* uniformly spaced.
        [grid] = arbitrary units

    Returns:
    --------
    val - float
        The value of `grid` corresponding to generalized index `ind`.
        [val] = [grid]

    '''
    grid_spacing = get_uniform_spacing(grid)
    return grid[0] + (ind * grid_spacing)
