from blueice.test_helpers import *
from blueice.likelihood import BinnedLogLikelihood
from blueice.source import DensityEstimatingSource

from scipy import stats
import numpy as np


def test_single_bin():
    conf = test_conf(mc=True)
    conf['sources'][0]['events_per_day'] = 1
    conf['analysis_space'] = [['x', [-40,40 ]]]

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()

    # Make a single event at x=0
    lf.set_data(np.zeros(1,
                         dtype=[('x', np.float), ('source', np.int)]))

    assert lf() == stats.poisson(1000).logpmf(1)
    assert lf(s0_rate_multiplier=5.4) == stats.poisson(5400).logpmf(1)


def test_twobin_mc():
    conf = test_conf(mc=True)
    conf['analysis_space'] = [['x', [-40, 0, 40]]]

    lf = BinnedLogLikelihood(conf)
    lf.add_rate_parameter('s0')
    lf.prepare()

    # Make 100 events at x=1
    lf.set_data(np.ones(100,
                        dtype=[('x', np.float), ('source', np.int)]))

    assert almost_equal(lf(),
                        stats.poisson(500).logpmf(100) + stats.poisson(500).logpmf(0),
                        1e-2)


def make_data(instructions):
    n_tot = sum([x['n_events'] for x in instructions])

    d = np.zeros(n_tot, dtype=[('source', np.int),
                               ('x', np.float),
                               ('y', np.float)])

    n_done = 0
    for instr in instructions:
        n_new = instr['n_events']
        sl = slice(n_done, n_done + n_new)
        for k in instr.keys():
            if k == 'n_events':
                continue
            d[sl][k] = instr[k]
        n_done += instr['n_events']

    return d, n_tot



def test_multi_bin_single_dim():
    instructions_mc = [dict(n_events=24, x=0.5),
                       dict(n_events=56, x=1.5)]
    data, n_mc = make_data(instructions_mc)

    class TestSource(DensityEstimatingSource):
        events_per_day = 42

        def get_events_for_density_estimate(self):
            return data, n_mc


    conf = test_conf()
    conf['default_source_class'] = TestSource
    conf['analysis_space'] = [['x', [0, 1, 5]]]

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

    class TestSource(DensityEstimatingSource):
        events_per_day = 42

        def __init__(self,*args, **kwargs):
            super().__init__(*args,**kwargs)
            print(self.config.get('strlen_multiplier','x'))
            print(len(self.config.get('strlen_multiplier','x')))
            if (self.events_per_day == 42):
                self.events_per_day *=len(self.config.get('strlen_multiplier','x'))
            
        def get_events_for_density_estimate(self):
            return data, n_mc


    conf = test_conf()
    conf['default_source_class'] = TestSource
    conf['analysis_space'] = [['x', [0, 1, 5]],
                              ['y', [0, 1, 4]]]

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

    print("before compute_pdf")
    assert almost_equal(lf(compute_pdf=True,strlen_multiplier=2),
                        np.sum([stats.poisson(2 * mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))
#                        
    assert almost_equal(lf(compute_pdf=False,strlen_multiplier=2),
                        np.sum([stats.poisson(2 * mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))
#
#
#
    assert almost_equal(lf(strlen_multiplier=2.3),
                        np.sum([stats.poisson(2.3*mu).logpmf(seen_in_bin)
                                for mu, seen_in_bin in zip(mus, seen)]))


