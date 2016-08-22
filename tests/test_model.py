from blueice.test_helpers import *
from blueice.model import Model
import matplotlib.pyplot as plt

def test_rates():
    m = Model(test_conf(n_sources=1))
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))

    m.config['livetime_days'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))
    m.config['livetime_days'] = 1

    m.sources[0].fraction_in_range = 0.5
    np.testing.assert_array_equal(m.expected_events(), np.array([500]))
    m.sources[0].fraction_in_range = 1

    # Modifying some_multiplier after the fact has no effect, it's effect is set on GaussianSource.__init__
    m.config['some_multiplier'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))
    m.config['some_multiplier'] = 1

    # Creating a new model, however, will do the trick:
    conf = test_conf(n_sources=2)
    conf['some_multiplier'] = 2
    m = Model(conf)
    np.testing.assert_array_equal(m.expected_events(), np.array([2000, 2000]))

    # Test non-numeric settings
    conf = test_conf(n_sources=1)
    conf['strlen_multiplier'] = 'hi'
    m = Model(conf)
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))

    # Test the "Model.show" function
    m.show(m.simulate())
    plt.close()