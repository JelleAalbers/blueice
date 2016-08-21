from blueice.test import *


def test_rates():
    m = Model(test_conf(n_sources=1))
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))

    m.config['livetime_days'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))
    m.config['livetime_days'] = 1

    m.sources[0].fraction_in_range = 0.5
    np.testing.assert_array_equal(m.expected_events(), np.array([500]))

    m = Model(test_conf(n_sources=2))
    np.testing.assert_array_equal(m.expected_events(), np.array([1000, 1000]))
