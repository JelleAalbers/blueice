import numpy as np
from blueice.utils import arrays_to_grid, InterpolateAndExtrapolate1D


def test_arrays_to_grid():
    np.testing.assert_array_equal(arrays_to_grid([np.array([0, 1]), np.array([0, 1])]),
                                  np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]))

    np.testing.assert_array_equal(arrays_to_grid([np.array([1, 2]), np.array([3, 4])]),
                                  np.array([[[1, 3], [1, 4]], [[2, 3], [2, 4]]]))

def test_interpolate():
    # Test single point
    itp = InterpolateAndExtrapolate1D(0, 42)
    assert itp(3) == 42
    assert itp([3]) == [42]

    itp = InterpolateAndExtrapolate1D([0], [42])
    assert itp(3) == 42
    assert itp([3]) == [42]

    # Test multiple points
    itp = InterpolateAndExtrapolate1D([0, 1], [0, 42])
    assert itp(3) == 42
    assert itp([3]) == [42]
    assert itp(0.5) == 21
