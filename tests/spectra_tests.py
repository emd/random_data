from nose import tools
import numpy as np
from random_data.spectra import SpectralDensity


def test_SpectralDensity__init__():
    # Different length signals should fail
    x = np.random.randn(1000)
    y = np.random.randn(1001)
    tools.assert_raises(ValueError, SpectralDensity, x, {'y': y})

    # If `x` and `y` are different, we are computing cross-spectral density
    tools.assert_equal(SpectralDensity(x, y=y[:-1]).kind, 'cross-spectral')

    # If `x` and `y` are equal, we are computing autospectral density
    y = x
    tools.assert_equal(SpectralDensity(x, y=y).kind, 'autospectral')

    # If `y` is not specified, we are computing autospectral density
    tools.assert_equal(SpectralDensity(x).kind, 'autospectral')
