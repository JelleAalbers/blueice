from collections import OrderedDict

import blueice.exceptions
import pytest
import numpy as np
from blueice import pdf_morphers


def test_morpher_api():
    conf = dict(hypercube_shuffle_steps=2,
                r_sample_points=2)

    for name, morph_class in pdf_morphers.MORPHERS.items():
        print("Testing %s" % name)

        with pytest.raises(blueice.exceptions.NoShapeParameters):
            morph_class(config=conf, shape_parameters=OrderedDict())

        shape_pars = OrderedDict([('bla', ({-1: -1, 0: 0, 1: 1}, None, None))])
        mr = morph_class(config=conf, shape_parameters=shape_pars)
        aps = mr.get_anchor_points(bounds=[(-1, 1)], n_models=3)
        assert isinstance(aps, list)
        assert isinstance(aps[0], tuple)

        def scalar_f(_):
            return 0

        scalar_itp = mr.make_interpolator(scalar_f, extra_dims=[], anchor_models={z: None for z in aps})
        assert scalar_itp([0]) == 0

        def matrix_f(_):
            return np.zeros(2, 2)

        matrix_itp = mr.make_interpolator(scalar_f, extra_dims=[2, 2], anchor_models={z: None for z in aps})
        np.testing.assert_array_equal(matrix_itp([0]), np.zeros((2, 2)))

if __name__ == '__main__':
    pytest.main()