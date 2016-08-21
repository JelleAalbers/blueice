import numpy as np
import matplotlib.pyplot as plt

from . import utils


class Model(object):
    """Model for dataset simulation and analysis
    """

    def __init__(self, config, ipp_client=None, **kwargs):
        """
        :param config: Dictionary specifying detector parameters, source info, etc.
        :param kwargs: Overrides for the config (optional)
        :param ipp_client: ipyparallel client to use for parallelizing initial computations (optional)
        :return:
        """
        defaults = dict(livetime_days=1,
                        data_dirs=1,
                        nohash_settings=['data_dirs', 'pdf_sampling_batch_size',
                                         'force_pdf_recalculation'])
        self.config = utils.combine_dicts(defaults, config, kwargs)

        # Initialize the sources. Each gets passed the entire config (without the 'sources' field)
        # with the settings in their entry in the sources field added to it.
        self.sources = []
        for source_config in self.config['sources']:
            if 'class' in source_config:
                source_class = source_config['class']
            else:
                source_class = self.config['default_source_class']
            conf = utils.combine_dicts(self.config,
                                       source_config,
                                       exclude=['sources', 'default_source_class', 'class'])
            self.sources.append(source_class(conf, ipp_client=ipp_client))
        del self.config['sources']  # So nobody gets the idea to modify it, which won't work after this

    def get_source(self, source_id):
        return self.sources[self.get_source_i(source_id)]

    def get_source_i(self, source_id):
        if isinstance(source_id, (int, float)):
            return int(source_id)
        else:
            for s_i, s in enumerate(self.sources):
                if source_id in s.name:
                    break
            else:
                raise ValueError("Unknown source %s" % source_id)
            return s_i

    def range_cut(self, d):
        """Return events from dataset d which are inside the bounds of the analysis space"""
        mask = np.ones(len(d), dtype=np.bool)
        for dimension, bin_edges in self.config['analysis_space']:
            mask = mask & (d[dimension] >= bin_edges[0]) & (d[dimension] <= bin_edges[-1])
        return d[mask]

    def simulate(self, restrict=True, rate_multipliers=None):
        """Makes a toy dataset.
        if restrict=True, return only events inside analysis range
        """
        if rate_multipliers is None:
            rate_multipliers = dict()
        ds = []
        for s_i, source in enumerate(self.sources):
            d = source.simulate(np.random.poisson(source.events_per_day * self.config['livetime_days'] *
                                                  rate_multipliers.get(source.name, 1)))
            d['source'] = s_i
            ds.append(d)
        d = np.concatenate(ds)
        if restrict:
            d = self.range_cut(d)
        return d

    def to_analysis_dimensions(self, d):
        """Given a dataset, returns list of arrays of coordinates of the events in the analysis dimensions"""
        return utils._events_to_analysis_dimensions(d, self.config['analysis_space'])

    def score_events(self, d):
        """Returns array (n_sources, n_events) of pdf values for each source for each of the events"""
        return np.vstack([s.pdf(*self.to_analysis_dimensions(d)) for s in self.sources])

    def expected_events(self, s=None):
        """Return the total number of events expected in the analysis range for the source s.
        If no source specified, return an array of results for all sources.
        """
        if s is None:
            return np.array([self.expected_events(s) for s in self.sources])
        return s.events_per_day * self.config['livetime_days'] * s.fraction_in_range

    def show(self, d, ax=None, dims=None):
        """Plot the events from dataset d in the analysis range
        ax: plot on this Axes
        Dims: numbers of dimension(s) to plot in. Can be up to two dimensions.
        """
        bins, dim_names = zip(*self.config['analysis_space'])

        if dims is None:
            if len(bins) == 1:
                dims = tuple([0])
            else:
                dims = (0, 1)
        if ax is None:
            ax = plt.gca()

        # d = self.range_cut(d)   # Not needed, simulate already does this
        for s_i, s in enumerate(self.sources):
            q = d[d['source'] == s_i]
            q_in_space = self.to_analysis_dimensions(q)
            ax.scatter(q_in_space[dims[0]],
                       q_in_space[dims[1]] if len(dims) > 1 else np.zeros(len(q)),
                       color=s.color, s=5, label=s.label)

        ax.set_xlabel(dim_names[dims[0]])
        ax.set_xlim(bins[dims[0]][0], bins[dims[0]][-1])

        if len(dims) > 1:
            ax.set_ylabel(dim_names[dims[1]])
            ax.set_ylim(bins[dims[1]][0], bins[dims[1]][-1])
