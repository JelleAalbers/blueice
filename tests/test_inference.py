import pytest

from blueice.test_helpers import *
from blueice.inference import *
from blueice.likelihood import UnbinnedLogLikelihood as LogLikelihood


def test_fit_minuit():
    import iminuit
    if not iminuit.__version__.startswith('1.'):
        pytest.skip("Blueice's minuit wrappers assume minuit 1.x")

    # Single rate parameter
    lf = LogLikelihood(test_conf())
    lf.add_rate_parameter('s0')
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_minuit(lf)
    assert isinstance(fit_result, dict)
    assert 's0_rate_multiplier' in fit_result

    # Don't fit
    res, ll = bestfit_minuit(lf, s0_rate_multiplier=1)
    assert len(res) == 0
    assert ll == lf(s0_rate_multiplier=1)

    # Single shape parameter
    lf = LogLikelihood(test_conf())
    lf.add_shape_parameter('some_multiplier', (0.5, 1, 1.5, 2))
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_minuit(lf)
    assert 'some_multiplier' in fit_result

    # Shape and rate parameter
    lf = LogLikelihood(test_conf())
    lf.add_rate_parameter('s0')
    lf.add_shape_parameter('some_multiplier', (0.5, 1, 1.5, 2))
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_minuit(lf)
    assert 'some_multiplier' in fit_result
    assert 's0_rate_multiplier' in fit_result

    # Non-numeric shape parameter
    lf = LogLikelihood(test_conf())
    lf.add_shape_parameter('strlen_multiplier', {1: 'x', 2: 'hi', 3:'wha'}, base_value=1)
    lf.prepare()
    lf.set_data(lf.base_model.simulate())
    fit_result, ll = bestfit_minuit(lf)
    assert 'strlen_multiplier' in fit_result

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


# def test_plot():
#     """Tests the plot_likelihood_space code.
#     For now just test that it doesn't crash -- image comparison tests are tricky...
#     """
#     import matplotlib.pyplot as plt
#     lf = LogLikelihood(test_conf())
#     lf.add_rate_parameter('s0')
#     lf.add_shape_parameter('some_multiplier', (0.5, 1, 1.5, 2))
#     lf.prepare()
#     lf.set_data(lf.base_model.simulate())
#
#     plot_likelihood_ratio(lf, ('s0_rate_multiplier', np.linspace(0.5, 2, 3)))
#     plt.close()
#     plot_likelihood_ratio(lf,
#                           ('s0_rate_multiplier', np.linspace(0.5, 2, 3)),
#                           ('some_multiplier', np.linspace(0.5, 2, 3)))
#     plt.close()


def test_limit():
    """Test the limit setting code
    For now just tests if it runs, does not test whether the results are correct...
    """
    lf = LogLikelihood(test_conf(n_sources=2))
    lf.add_rate_parameter('s0')
    lf.prepare()
    lf.set_data(lf.base_model.simulate())

    # Test upper limits
    one_parameter_interval(lf, target='s0_rate_multiplier', kind='upper', bound=40)
    one_parameter_interval(lf, target='s0_rate_multiplier', kind='lower', bound=0.1)
    one_parameter_interval(lf, target='s0_rate_multiplier', kind='central', bound=(0.1, 20))

    # Bit tricky to test multiple params, in these simple examples they can compensate completely for each other
    # so all values in a subspace seem equally likely once two of them are floating.
