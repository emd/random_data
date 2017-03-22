from nose import tools
import numpy as np
from random_data.errors import cross_phase_std_dev


def test_cross_phase_std_dev():
    # Random error vanishes for perfectly coherent process
    gamma2xy = 1
    tools.assert_equal(0, cross_phase_std_dev(gamma2xy, 10))

    # Random error should decrease as (Nreal_per_ens)^{-0.5}
    Nreal_per_ens_1 = 10
    Nreal_per_ens_2 = 30
    gamma2xy = 0.5
    sigma1 = cross_phase_std_dev(gamma2xy, Nreal_per_ens_1)
    sigma2 = cross_phase_std_dev(gamma2xy, Nreal_per_ens_2)

    tools.assert_almost_equal(
        np.sqrt(Nreal_per_ens_2 / Nreal_per_ens_1),
        sigma1 / sigma2)

    # ... and a comparison to an expected value
    Nreal_per_ens = 2
    gamma2xy = 0.5
    tools.assert_almost_equal(
        0.5,
        cross_phase_std_dev(gamma2xy, Nreal_per_ens))
