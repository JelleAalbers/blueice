from blueice.test_helpers import *
from blueice.model import Model
from multihist import Histdd

def test_mcsource():
    conf = test_conf(mc=True)
    m = Model(conf)
    s = m.sources[0]
    bins = conf['analysis_space'][0][1]
    assert s.events_per_day == 1000
    assert s.fraction_in_range > 0.9999    # Ten sigma events happen sometimes..
    assert isinstance(m.sources[0].pdf_histogram, Histdd)
    assert isinstance(m.sources[0].pdf_errors, Histdd)
    assert abs(s.pdf([0]) - stats.norm.pdf(0)) < 0.01

    # Verify linear interpolation
    assert (s.pdf([bins[0]]) + s.pdf([bins[1]])) / 2 == s.pdf([(bins[0] + bins[1])/2])
