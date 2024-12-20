from blueice.test_helpers import *
from blueice.model import Model

def test_rates():
    m = Model(conf_for_test(n_sources=1))
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))

    for source in m.sources:
        source.config['livetime_days'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))
    for source in m.sources:
        source.config['livetime_days'] = 1

    m.sources[0].fraction_in_range = 0.5
    np.testing.assert_array_equal(m.expected_events(), np.array([500]))
    m.sources[0].fraction_in_range = 1

    # Modifying some_multiplier after the fact has no effect, it's effect is set on GaussianSource.__init__
    m.config['some_multiplier'] = 2
    np.testing.assert_array_equal(m.expected_events(), np.array([1000]))
    m.config['some_multiplier'] = 1

    # Creating a new model, however, will do the trick:
    conf = conf_for_test(n_sources=2)
    conf['some_multiplier'] = 2
    m = Model(conf)
    np.testing.assert_array_equal(m.expected_events(), np.array([2000, 2000]))

    # Test get source
    assert m.get_source(1) == m.sources[1]
    assert m.get_source_i(1) == 1
    assert m.get_source_i('s1') == 1
    assert m.get_source('s1') == m.sources[1]

    # Test non-numeric settings
    conf = conf_for_test(n_sources=1)
    conf['strlen_multiplier'] = 'hi'
    m = Model(conf)
    np.testing.assert_array_equal(m.expected_events(), np.array([2000]))

    # Test the "Model.show" function, if we have matplotlib in the test environment
    try:
        import matplotlib.pyplot as plt
        can_plot = True
    except ImportError:
        can_plot = False
    if can_plot:
        m.show(m.simulate())
        plt.close()
