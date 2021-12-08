from blueice.test_helpers import *
from blueice.likelihood import UnbinnedLogLikelihood, LogLikelihoodReParam
import numpy as np
from scipy import stats
from copy import deepcopy


def test_likelihood_value():
    """Just a sanity check to show we get the right likelihood values"""
    config = test_conf_reparam(events_per_day=1)
    conv_config = deepcopy(BASE_CONV_CONFIG)
    # initialize the old likelihood first
    # this is an input for the reparameterized likelihood
    lf_old = UnbinnedLogLikelihood(config)

    lf_old.add_rate_parameter("op0")
    lf_old.add_rate_parameter("op1")
    lf_old.add_rate_parameter("op2")

    lf_old.prepare()

    # get new likelihood
    lf_reparam = LogLikelihoodReParam(lf_old, conv_config)

    # dummy data
    d = np.zeros(3, dtype=[('x', np.float), ('source', np.int)])
    lf_reparam.set_data(d)

    def compute_lf(np0=1, np1=1):
        """likelihood under the dummy data"""
        op0 = np0 ** 2
        op1 = np1 ** 2
        op2 = np0 * np1
        sum_s = op0 + op1 + op2
        return -sum_s + 3 * np.log(sum_s) + 3 * stats.norm.logpdf(0)

    # check if the likelihood is computed correctly
    np0_s, np1_s = [1, 2, 3], [1, 2, 3]
    for np0, np1 in zip(np0_s, np1_s):
        assert np.isclose(lf_reparam(np0=np0, np1=np1), compute_lf(np0=np0, np1=np1), atol=1e-08)


def test_likelihoods_before_after_reparam():
    """Compare the likelihood before and after reparameterization"""
    config = test_conf_reparam(events_per_day=1)
    conv_config = deepcopy(BASE_CONV_CONFIG)
    # initialize the old likelihood first
    # this is an input for the reparameterized likelihood
    lf_old = UnbinnedLogLikelihood(config)

    lf_old.add_rate_parameter("op0")
    lf_old.add_rate_parameter("op1")
    lf_old.add_rate_parameter("op2")

    lf_old.prepare()

    # get new likelihood
    lf_reparam = LogLikelihoodReParam(lf_old, conv_config)

    # set data
    d = lf_reparam.base_model.simulate()
    lf_reparam.set_data(d)
    lf_old.set_data(d)

    # check
    assert np.isclose(lf_reparam(), lf_old())
    assert np.isclose(lf_reparam(np0=2), lf_old(op0_rate_multiplier=4, op2_rate_multiplier=2))
    assert np.isclose(lf_reparam(np1=2), lf_old(op1_rate_multiplier=4, op2_rate_multiplier=2))
    assert np.isclose(lf_reparam(np0=2, np1=2),
                      lf_old(op0_rate_multiplier=4, op1_rate_multiplier=4, op2_rate_multiplier=4))


def test_consistency_new_params(use_wrong_config=False, use_wrong_conv_config=False):
    """Check if the input new parameters are consistent:
    1) inside the conv_config
    2) between conv_config and config
    """
    config = test_conf_reparam(events_per_day=1)
    conv_config = deepcopy(BASE_CONV_CONFIG)

    if use_wrong_config:
        config.pop("np0")
        config.pop("np1")

    if use_wrong_conv_config:
        conv_config["np2"] = (np.linspace(1e-12, 10, 2), None, None)

    # initialize the old likelihood first
    # this is an input for the reparameterized likelihood
    lf_old = UnbinnedLogLikelihood(config)

    lf_old.add_rate_parameter("op0")
    lf_old.add_rate_parameter("op1")
    lf_old.add_rate_parameter("op2")

    lf_old.prepare()

    # get new likelihood
    lf_reparam = LogLikelihoodReParam(lf_old, conv_config)