from blueice.test_helpers import *
from blueice.likelihood import BinnedLogLikelihood
from blueice.source import DensityEstimatingSource

from scipy import stats
import numpy as np
import pytest


def test_single_bin():
    conf = test_conf(mc=True, analysis_space=[['x', [-40,40 ]]])

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()

    # Make a single event at x=0
    lf.set_data(np.zeros(1,
                         dtype=[('x', float), ('source', int)]))

    assert lf() == stats.poisson(1000).logpmf(1)
    assert lf(s0_rate_multiplier=5.4) == stats.poisson(5400).logpmf(1)


def test_twobin_mc():
    conf = test_conf(mc=True, analysis_space=[['x', [-40, 0, 40]]])

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()

    # Make 100 events at x=1
    lf.set_data(np.ones(100,
                        dtype=[('x', float), ('source', int)]))

    assert almost_equal(lf(),
                        stats.poisson(500).logpmf(100) + stats.poisson(500).logpmf(0),
                        1e-2)


def test_multi_bin_single_dim():
    instructions_mc = [dict(n_events=24, x=0.5),
                       dict(n_events=56, x=1.5)]
    data, n_mc = make_data(instructions_mc)

    conf = test_conf(events_per_day=42,
                     analysis_space=[['x', [0, 1, 5]]], default_source_class=FixedSampleSource, data=data)

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')

    instructions_data = [dict(n_events=18, x=0.5),
                         dict(n_events=70, x=1.5)]
    data, _ = make_data(instructions_data)
    lf.set_data(data)

    mus = [42 / n_mc * instructions_mc[i]['n_events']
           for i in range(len(instructions_mc))]
    seen = [instructions_data[i]['n_events']
            for i in range(len(instructions_data))]

    assert almost_equal(lf(),
                        np.sum([stats.poisson(mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]),
                        1e-6)


def test_multi_bin():
    instructions_mc = [dict(n_events=24, x=0.5, y=0.5),
                       dict(n_events=56, x=1.5, y=0.5),
                       dict(n_events=6, x=0.5, y=2),
                       dict(n_events=14, x=1.5, y=2)]
    data, n_mc = make_data(instructions_mc)

    conf = test_conf(events_per_day=42, default_source_class=FixedSampleSource, data=data,
                     analysis_space=[['x', [0, 1, 5]], ['y', [0, 1, 4]]])

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')

    lf.add_shape_parameter('strlen_multiplier', {1: 'x', 2: 'hi', 3:'wha'},base_value=1)
    lf.prepare()

    instructions_data = [dict(n_events=18, x=0.5, y=0.5),
                         dict(n_events=70, x=1.5, y=0.5),
                         dict(n_events=4,  x=0.5, y=2),
                         dict(n_events=10, x=1.5, y=2)]
    data, _ = make_data(instructions_data)
    lf.set_data(data)

    mus = [42 / n_mc * instructions_mc[i]['n_events']
           for i in range(len(instructions_mc))]
    seen = [instructions_data[i]['n_events']
            for i in range(len(instructions_data))]

    assert almost_equal(lf(strlen_multiplier=1),
                        np.sum([stats.poisson(mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))

    print("Expected events: ", lf.base_model.expected_events())
    with pytest.raises(NotImplementedError):
        lf(compute_pdf=True,strlen_multiplier=2), np.sum([stats.poisson(2 * mu).logpmf(seen_in_bin) for mu, seen_in_bin in zip(mus, seen)])

    assert almost_equal(lf(compute_pdf=False,strlen_multiplier=2),
                        np.sum([stats.poisson(2 * mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))

    assert almost_equal(lf(strlen_multiplier=2.3),
                        np.sum([stats.poisson(2.3*mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))




