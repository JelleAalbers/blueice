# from blueice.test_helpers import *
# from blueice.likelihood import BinnedLogLikelihood
#
# from scipy import stats
#
#
# def test_single_bin():
#     conf = test_conf()
#     conf['sources'][0]['events_per_day'] = 1
#     conf['analysis_space'] = [['x', [-10, 10]]]
#
#     lf = BinnedLogLikelihood(conf)
#     lf.add_rate_parameter('s0')
#
#     # Make a single event at x=0
#     lf.set_data(np.zeros(1,
#                          dtype=[('x', np.float), ('source', np.int)]))
#
#     assert lf() == stats.poisson(1).pmf(1)
#     assert lf(s0_rate=5.4) == stats.poisson(5.4).pmf(1)
