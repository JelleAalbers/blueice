from blueice.test_helpers import *
from blueice.inference import *

def test_fit_scipy():
    # Single rate parameter
    lf = LogLikelihood(test_conf())
    lf.add_rate_parameter('s0')
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_scipy(lf)
    assert isinstance(fit_result, dict)
    assert 's0_rate_multiplier' in fit_result

    # Don't fit
    res, ll = bestfit_scipy(lf, s0_rate_multiplier=1)
    assert len(res) == 0
    assert ll == lf(s0_rate_multiplier=1)

    # Single shape parameter
    lf = LogLikelihood(test_conf())
    lf.add_shape_parameter('some_multiplier', (0.5, 1, 1.5, 2))
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_scipy(lf)
    assert 'some_multiplier' in fit_result

    # Shape and rate parameter
    lf = LogLikelihood(test_conf())
    lf.add_rate_parameter('s0')
    lf.add_shape_parameter('some_multiplier', (0.5, 1, 1.5, 2))
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_scipy(lf)
    assert 'some_multiplier' in fit_result
    assert 's0_rate_multiplier' in fit_result

    # Non-numeric shape parameter
    lf = LogLikelihood(test_conf())
    lf.add_shape_parameter('strlen_multiplier', {1: 'x', 2: 'hi', 3:'wha'}, base_value=1)
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_scipy(lf)
    assert 'strlen_multiplier' in fit_result
