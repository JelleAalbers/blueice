"""
Common code for tests. The tests themselves are located in ../tests, but need to import this, so...
"""
from blueice.source import *
from blueice.model import *
from blueice.likelihood import *

class GaussianSource(Source):
    events_per_day = 1000
    """A 1d source with a Gaussian PDF -- useful for testing
    If your sources are as simple as this, you probably don't need blueice!
    """
    fraction_in_range = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events_per_day *= self.model.config.get('some_multiplier', 1)
        (self.mu, self.sigma) = (self.model.config.get('mu'), self.model.config.get('sigma', 1))

    def pdf(self, *args):
        return stats.norm(self.mu, self.sigma).pdf(args[0])

    def simulate(self, n_events):
        d = np.zeros(n_events, dtype=[('x',np.float), ('source',np.int)])
        d['x']= stats.norm(self.mu, self.sigma).rvs(n_events)
        return d


BINS = np.linspace(-10, 10, 100)

BASE_SOURCE_CONF = {'class': GaussianSource,
                    'events_per_day': 1000,
                    'name': 's'}

BASE_CONFIG = dict(
    sources = [BASE_SOURCE_CONF],
    mu = 0,
    sigma = 1,
    some_multiplier = 1,
    force_pdf_recalculation=True,
    analysis_space=[['x', BINS]]
)

def test_conf(n_sources=1):
    conf = deepcopy(BASE_CONFIG)
    conf['sources'] = []
    for i in range(n_sources):
        s = BASE_SOURCE_CONF.copy()
        s['name'] = "s%d" % i
        conf['sources'].append(s)
    return conf
