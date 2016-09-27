from blueice.test_helpers import *
from blueice.likelihood import BinnedLogLikelihood

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
