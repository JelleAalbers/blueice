from blueice.test_helpers import *
from blueice.likelihood import BinnedLogLikelihood
from blueice.likelihood import beeston_barlow_root2

from scipy import stats
import numpy as np
import numpy.testing as npt

# Test results of simultaneous analytical fit to a bckg determined from calibration/MC


def test_BeestonBarlowSingleBin():

    instructions_mc = [dict(n_events=32, x=0.5)]
    data, n_mc = make_data(instructions_mc)

    conf = test_conf(default_source_class=FixedSampleSource,
                     events_per_day=32/5,
                     analysis_space=[['x', [0, 1]]],
                     data=data)

    likelihood_config = {'model_statistical_uncertainty_handling': 'bb_single',
                         'bb_single_source': 0}
    lf = BinnedLogLikelihood(conf, likelihood_config=likelihood_config)
    lf.prepare()
    assert lf.n_model_events is not None

    # Make a single event at x=0
    lf.set_data(np.zeros(2, dtype=[('x', float), ('source', int)]))

    assert lf.n_model_events is not None
    assert almost_equal(28.0814209, beeston_barlow_root2(np.array([32]), 0.2, np.array([1]), np.array([2])))

    # A = beeston_barlow_root2(np.array([32]), 0.2, np.array([0]), np.array([2]))
    A = (2+32)/(1+0.2)

    assert almost_equal(lf(), stats.poisson(0.2*A).logpmf(2))


def test_BeestonBarlowMultiBin():
    instructions_mc = [dict(n_events=16, x=0.5),
                       dict(n_events=30, x=1.5),
                       dict(n_events=32, x=2.5),
                       dict(n_events=27, x=3.5),]
    data, n_mc = make_data(instructions_mc)

    conf = test_conf(default_source_class=FixedSampleSource,
                     events_per_day=105/5,
                     analysis_space=[['x', [0, 1, 2, 3, 4]]],
                     data=data)

    likelihood_config = {'model_statistical_uncertainty_handling': 'bb_single',
                         'bb_single_source': 0}
    lf = BinnedLogLikelihood(conf, likelihood_config=likelihood_config)
    lf.prepare()
    assert lf.n_model_events is not None

    # Make events:
    instructions_mc = [dict(n_events=3, x=0.5),
                       dict(n_events=5, x=1.5),
                       dict(n_events=2, x=2.5),
                       dict(n_events=7, x=3.5),]
    data, _ = make_data(instructions_mc)
    lf.set_data(data)

    assert lf.n_model_events is not None

    A_minimized = np.array([15.833,29.166,28.333,28.333])
    A_BB = beeston_barlow_root2(np.array([16,30,32,27]), 0.2, np.array([0.]), np.array([3,5,2,7]))

    npt.assert_almost_equal(A_minimized,A_BB,decimal=2);

    dbin = np.array([3,5,2,7])


    assert almost_equal(lf(), np.sum(stats.poisson(0.2*A_BB).logpmf(dbin)))


def test_BeestonBarlow_second_source():
    instructions_mc_calibration = [dict(n_events=16, x=0.5),
                                   dict(n_events=30, x=1.5),
                                   dict(n_events=32, x=2.5),
                                   dict(n_events=27, x=3.5),]

    data_calibration, n_mc = make_data(instructions_mc_calibration)

    instructions_mc_other = [dict(n_events=5, x=0.5),
                                   dict(n_events=7, x=1.5),
                                   dict(n_events=1, x=2.5),
                                   dict(n_events=3, x=3.5),]
    data_other, n_mc = make_data(instructions_mc_other)

    # in sum, 105 calibration/mc evts.

    conf = test_conf(default_source_class=FixedSampleSource,
                     analysis_space=[['x', [0, 1, 2, 3, 4]]],
                     dummy=1)

    conf['sources']=[{'name': 's0','events_per_day':105/5.,'data': data_calibration},
                     {'name': 's1','events_per_day':16.   ,"data": data_other}]

    likelihood_config = {'model_statistical_uncertainty_handling': 'bb_single',
                         'bb_single_source': 0}
    lf = BinnedLogLikelihood(conf, likelihood_config=likelihood_config)

    lf.add_shape_parameter('dummy', (0, 1))
    lf.prepare()
    assert lf.n_model_events is not None

    # Make events:
    instructions_mc = [dict(n_events=3, x=0.5),
                       dict(n_events=5, x=1.5),
                       dict(n_events=2, x=2.5),
                       dict(n_events=7, x=3.5),]
    data, _ = make_data(instructions_mc)
    lf.set_data(data)

    assert lf.n_model_events is not None

    A_minimized = np.array([14.24, 26.8070, 28.08, 26.21])
    A_BB = beeston_barlow_root2(np.array([16, 30, 32, 27]), 0.2, np.array([5, 7, 1, 3]), np.array([3, 5, 2, 7]))

    npt.assert_almost_equal(A_minimized,A_BB,decimal=2)

    dbin = np.array([3, 5, 2, 7])

    U_bin = np.array([5, 7, 1, 3])

    print("A_BB*0.2: %s" % str(A_BB*0.2))

    assert almost_equal(lf(), np.sum(stats.poisson(0.2*A_BB+U_bin).logpmf(dbin)))







