import numpy as np

from blueice.model import Model
from blueice.source import GaussianSource

BINS = np.linspace(-10, 10, 100)

BASE_SOURCE = {'class': GaussianSource,
               'events_per_day': 1000,
               'name': 'foo'}

BASE_CONFIG = dict(
    sources = [BASE_SOURCE],
    mu = 0,
    sigma = 1,
    some_multiplier = 1,
    force_pdf_recalculation=True,
    analysis_space=[['x', BINS]]
)

def test_rates():
    m = Model(BASE_CONFIG)
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))

    m.config['livetime_days'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))
    m.config['livetime_days'] = 1

    m.sources[0].fraction_in_range = 0.5
    np.testing.assert_array_equal(m.expected_events(), np.array([500]))

    m = Model(BASE_CONFIG, sources=[BASE_SOURCE.copy(), BASE_SOURCE.copy()])
    np.testing.assert_array_equal(m.expected_events(), np.array([1000, 1000]))
