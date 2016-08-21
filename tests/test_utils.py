from blueice.utils import *


def test_arrays_to_grid():
    np.testing.assert_array_equal(arrays_to_grid([np.array([0,1]), np.array([0,1])]),
                                  np.array([[[0,0],[0,1]], [[1,0], [1,1]]]))

    np.testing.assert_array_equal(arrays_to_grid([np.array([1,2]), np.array([3,4])]),
                                  np.array([[[1,3],[1,4]], [[2,3], [2,4]]]))

