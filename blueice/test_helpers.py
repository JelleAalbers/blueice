"""
Common code for tests. The tests themselves are located in ../tests, but need to import this, so...
"""
from copy import deepcopy

from .source import Source, MonteCarloSource, DensityEstimatingSource
from .utils import combine_dicts

import numpy as np
from scipy import stats


class GaussianSourceBase(Source):
    """Analog of GaussianSource which generates its events by PDF
    """
    def simulate(self, n_events):
        d = np.zeros(n_events, dtype=[('x', np.float), ('source', np.int)])
        d['x'] = stats.norm(self.config['mu'], self.config['sigma']).rvs(n_events)
        return d


class GaussianSource(GaussianSourceBase):
    """A 1d source with a Gaussian PDF -- useful for testing
    If your sources are as simple as this, you probably don't need blueice!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_pdf(self):
        self.events_per_day *= self.config.get('some_multiplier', 1)
        self.events_per_day *= len(self.config.get('strlen_multiplier', 'x'))

    def pdf(self, *args):
        return stats.norm(self.config['mu'], self.config['sigma']).pdf(args[0])


class GaussianMCSource(GaussianSourceBase, MonteCarloSource):
    """Analog of GaussianSource which generates its PDF from MC
    """
    pass


class FixedSampleSource(DensityEstimatingSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events_per_day *= len(self.config.get('strlen_multiplier', 'x'))

    def get_events_for_density_estimate(self):
        return self.config['data'], len(self.config['data'])


BASE_CONFIG = dict(
    sources=[{'name': 's0', 'events_per_day': 1000.}],
    mu=0,
    strlen_multiplier='q',
    events_per_day=1000,
    n_events_for_pdf=int(1e6),
    sigma=1,
    default_source_class=GaussianSource,
    some_multiplier=1,
    force_pdf_recalculation=True,
    analysis_space=[['x', np.linspace(-10, 10, 100)]]
)


def test_conf(n_sources=1, mc=False, **kwargs):
    conf = deepcopy(BASE_CONFIG)
    conf['sources'] = [{'name': 's%d' % i} for i in range(n_sources)]
    if mc:
        conf['default_source_class'] = GaussianMCSource
    return combine_dicts(conf, kwargs)


def almost_equal(a, b, fraction=1e-6):
    return abs((a - b)/a) <= fraction


def make_data(instructions):
    """
    make_data([dict(n_events=24, x=0.5),
               dict(n_events=56, x=1.5)]):
    produces 25 events with x=0.5 and 56 events with x=1.5
    :return: numpy record array accepted by set_data
    """
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
