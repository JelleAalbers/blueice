from blueice.test_helpers import *
from blueice.likelihood import UnbinnedLogLikelihood, BinnedLogLikelihood
from blueice.exceptions import NotPreparedException, InvalidParameterSpecification, InvalidParameter
import pytest
import scipy.stats as sps


def test_likelihood_value():
    """Just a sanity check to show we get the right likelihood values"""
    lf = UnbinnedLogLikelihood(conf_for_test(events_per_day=1))
    lf.add_rate_parameter('s0')

    # Make a single event at x=0
    lf.set_data(np.zeros(1,
                         dtype=[('x', float), ('source', int)]))

    assert lf() == -1 + stats.norm.logpdf(0)
    assert lf(s0_rate_multiplier=2) == -2 + np.log(2 * stats.norm.pdf(0))


def test_no_shape_params():
    lf = UnbinnedLogLikelihood(conf_for_test())
    d = lf.base_model.simulate()
    lf.prepare()
    lf.set_data(d)
    lf()

    # Test a MonteCarloSource, which should trigger a pdf computation
    lf = UnbinnedLogLikelihood(conf_for_test(mc=True))
    d = lf.base_model.simulate()
    lf.prepare()
    lf.set_data(d)
    lf()


def test_shape_params():
    lf = UnbinnedLogLikelihood(conf_for_test(n_sources=1))
    lf.add_rate_parameter('s0')
    with pytest.raises(InvalidParameterSpecification):
        lf.add_shape_parameter('strlen_multiplier', {1: 'x', 2: 'hi', 3:'wha'})
    lf.add_shape_parameter('strlen_multiplier', {1: 'q', 2: 'hi', 3: 'wha'}, base_value=1)
    d = lf.base_model.simulate()
    lf.prepare()
    lf.set_data(d)
    assert len(lf.anchor_models) == 3

    # Can't call with the raw setting
    with pytest.raises(ValueError):
        lf(strlen_multiplier='hi')

    # But can call with representative number:
    lf(strlen_multiplier=1.5)

    # Test for correct use of base_value
    assert lf() == lf(strlen_multiplier=1)

    # Test for interpolating non-numeric values by their representative settings
    assert lf(strlen_multiplier=1.5) < lf()


def test_rate_uncertainty():
    lf = UnbinnedLogLikelihood(conf_for_test(events_per_day=1))
    lf.add_rate_uncertainty('s0', 0.5)

    # Make a single event at x=0
    lf.set_data(np.zeros(1,
                         dtype=[('x', float), ('source', int)]))

    log_prior = stats.norm(1, 0.5).logpdf
    assert lf() == -1 + stats.norm.logpdf(0) + log_prior(1)
    assert lf(s0_rate_multiplier=2) == -2 + np.log(2 * stats.norm.pdf(0)) + log_prior(2)


def test_shape_uncertainty():
    lf = UnbinnedLogLikelihood(conf_for_test(events_per_day=1))

    with pytest.raises(InvalidParameterSpecification):
        lf.add_shape_uncertainty('strlen_multiplier', 0.5, {1: 'x', 2: 'hi', 3: 'wha'})

    lf.add_shape_uncertainty(setting_name='strlen_multiplier',
                             fractional_uncertainty=0.5,
                             anchor_zs={1: 'x', 2: 'hi', 3: 'wha'},
                             base_value=1)

    # Make a single event at x=0
    lf.prepare()
    lf.set_data(np.zeros(1,
                         dtype=[('x', float), ('source', int)]))

    log_prior = stats.norm(1, 0.5).logpdf
    assert lf() == -1 + stats.norm.logpdf(0) + log_prior(1)
    assert lf(strlen_multiplier=2) == -2 + np.log(2 * stats.norm.pdf(0)) + log_prior(2)


def test_source_wise_interpolation():
    data = np.zeros(5, dtype=[('x', float), ('source', int)])
    data['x'] = np.linspace(0, 1, 5)

    config = conf_for_test(events_per_day=1)

    lf = UnbinnedLogLikelihood(config)
    lf.add_shape_parameter("mu", anchors={-2:-2, 0:0, 2:2})
    lf.prepare()
    lf.set_data(data)
    ret_0 = lf(full_output=True)
    ret_1 = lf(full_output=True, mu=1)

    config["source_wise_interpolation"] = True
    lf_source_wise = UnbinnedLogLikelihood(config)
    lf_source_wise.add_shape_parameter("mu", anchors={-2:-2, 0:0, 2:2})
    lf_source_wise.prepare()
    lf_source_wise.set_data(data)
    ret_source_wise_0 = lf_source_wise(full_output=True)
    ret_source_wise_1 = lf_source_wise(full_output=True, mu=1)

    assert ret_0[0] == ret_source_wise_0[0]
    assert (ret_0[1] == ret_source_wise_0[1]).all()
    assert (ret_0[2] == ret_source_wise_0[2]).all()
    assert ret_1[0] == ret_source_wise_1[0]
    assert (ret_1[1] == ret_source_wise_1[1]).all()
    assert (ret_1[2] == ret_source_wise_1[2]).all()


def test_multisource_likelihood():
    lf = UnbinnedLogLikelihood(conf_for_test(n_sources=2))

    lf.add_shape_parameter('some_multiplier', (0.5, 1, 2, 4))
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    lf.prepare()

    d = lf.base_model.simulate()
    lf.set_data(d)

    assert lf(s0_rate_multiplier=1, s1_rate_multiplier=1, some_multiplier=1) == lf()
    assert lf(s0_rate_multiplier=1, s1_rate_multiplier=1) == lf()
    assert lf(s0_rate_multiplier=1) == lf()
    assert lf(some_multiplier=1) == lf()

    # Equivalence of rate parameters
    assert almost_equal(lf(s0_rate_multiplier=2), lf(s1_rate_multiplier=2))
    assert almost_equal(lf(s0_rate_multiplier=4), lf(s0_rate_multiplier=2.5, s1_rate_multiplier=2.5))

    # Equivalence of rate and shape parameters
    assert lf(s0_rate_multiplier=2, s1_rate_multiplier=2) == lf(some_multiplier=2)

    # Likelihood goes in right direction
    assert lf(some_multiplier=2) < lf()


def test_error_handling():
    lf = UnbinnedLogLikelihood(conf_for_test())
    d = lf.base_model.simulate()
    lf.add_shape_parameter('some_multiplier', (0.5, 1, 2))

    with pytest.raises(NotPreparedException):
        lf.set_data(d)
    with pytest.raises(NotPreparedException):
        lf()

    lf.prepare()

    with pytest.raises(NotPreparedException):
        lf()

    lf.set_data(d)

    lf()

    with pytest.raises(InvalidParameter):
        lf(blargh=41)


def test_noninterpolated_pdf():
    #

    conf = conf_for_test(n_sources=1)
    conf['some_multiplier']=3e-3
    lf = UnbinnedLogLikelihood(conf)
    lf.add_shape_parameter('mu',(0.,1.))
    lf.add_shape_parameter('sigma',(1.,2.))
    lf.prepare()

    d = np.zeros(1,dtype=[('x',float)])
    lf.set_data(d)

    assert almost_equal(lf(compute_pdf=True,mu=0.5,sigma=1.5),sps.poisson(3).logpmf(1)+sps.norm(0.5,1.5).logpdf(0),1e-5)
    assert not almost_equal(lf(compute_pdf=False,mu=0.5,sigma=1.5),sps.poisson(3).logpmf(1)+sps.norm(0.5,1.5).logpdf(0),1e-5)


def test_zero_bin():
    conf = conf_for_test(mc=True, analysis_space=[['x', [-40,40 ]]])

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()

    # Make a single event at x=0
    lf.set_data(np.zeros(0, dtype=[('x', float), ('source', int)]))

    assert lf(s0_rate_multiplier=0.) == stats.poisson(0).logpmf(0)


def test_livetime_scaling():
    conf = conf_for_test()

    # Test we cannot scale without a base lifetime
    lf = UnbinnedLogLikelihood(conf)
    lf.prepare()
    d = np.zeros(1,dtype=[('x',float)])
    lf.set_data(d)
    orig_l = lf()
    with pytest.raises(ValueError):
        lf(livetime_days=1)

    # Livetime scaling has the same effect as the rate multiplier
    # (because there is just one source)
    conf['livetime_days'] = 1
    lf = UnbinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()
    lf.set_data(d)
    assert lf(livetime_days=1) == orig_l
    assert lf(livetime_days=2) == lf(s0_rate_multiplier=2)
    assert lf(livetime_days=0) == lf(s0_rate_multiplier=0)

    # Can't scale with a zero livetime
    conf['livetime_days'] = 0
    lf_zero = UnbinnedLogLikelihood(conf)
    lf_zero.prepare()
    lf_zero.set_data(d)
    with pytest.raises(ValueError):
        lf_zero(livetime_days=1)
    # ... but we can still evaluate the lf:
    assert lf_zero() == lf(s0_rate_multiplier=0)
