from blueice.test import *
import pytest

def test_multisource_likelihood():
    lf = LogLikelihood(test_conf(n_sources=2))

    lf.add_shape_parameter('some_multiplier', (0.5, 1, 2))
    lf.add_rate_parameter('s0')
    lf.add_rate_parameter('s1')
    lf.prepare()

    d = lf.base_model.simulate()
    lf.set_data(d)

    assert lf(s0_rate_multiplier=1, s1_rate_multiplier=1, some_multiplier=1) == lf()
    assert lf(s0_rate_multiplier=1, s1_rate_multiplier=1) == lf()
    assert lf(s0_rate_multiplier=1) == lf()
    assert lf(some_multiplier=1) == lf()


def test_early_call():
    lf = LogLikelihood(test_conf())
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
