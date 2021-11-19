import numpy as np

from . import utils

__all__ = ['Model']


class Model(object):
    """Model for dataset simulation and analysis: collects several Sources, which do the actual work
    """

    def __init__(self, config, **kwargs):
        """
        :param config: Dictionary specifying detector parameters, source info, etc.
        :param kwargs: Overrides for the config (optional)
        """
        defaults = dict(livetime_days=1,
                        data_dirs=1,
                        nohash_settings=['data_dirs', 'pdf_sampling_batch_size',
                                         'force_recalculation'])
        self.config = utils.combine_dicts(defaults, config, kwargs, deep_copy=True)

        if 'rate_multiplier' in self.config:
            raise ValueError("Don't put a setting named rate_multiplier in the model config please...")

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

            # Special handling for the _rate_multiplier settings
            source_name = conf.get('name', 'WHAAAAAA_YOUDIDNOTNAMEYOURSOURCETHIS')
            conf['rate_multiplier'] = conf.get('%s_rate_multiplier' % source_name, 1)
            conf = {k:v for k,v in conf.items() if not k.endswith('_rate_multiplier')}

            s = source_class(conf)
            self.sources.append(s)

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
        mask = np.ones(len(d), dtype=bool)
        for dimension, bin_edges in self.config['analysis_space']:
            mask = mask & (d[dimension] >= bin_edges[0]) & (d[dimension] <= bin_edges[-1])
        return d[mask]

    def simulate(self, rate_multipliers=None, livetime_days=None):
        """Makes a toy dataset, poisson sampling simulated events from all sources.
        :param rate_multipliers: dict {source name: multiplier} to change rate of individual sources
        :param livetime_days: days of exposure to simulate (affects rate of all sources)
        """
        if rate_multipliers is None:
            rate_multipliers = dict()
        ds = []
        for s_i, source in enumerate(self.sources):
            # We have to divide by the fraction in range (increasing the number of events)
            # since we're going to call simulate, which will produce also events out of range.
            mu = self.expected_events(source) * rate_multipliers.get(source.name, 1) / source.fraction_in_range
            if livetime_days is not None:
                # Adjust exposure to custom livetime-days
                mu *= livetime_days / self.config['livetime_days']

            d = source.simulate(np.random.poisson(mu))
            d['source'] = s_i
            ds.append(d)

        d = np.concatenate(ds)
        d = self.range_cut(d)
        return d

    def to_analysis_dimensions(self, d):
        """Given a dataset, returns list of arrays of coordinates of the events in the analysis dimensions"""
        return utils._events_to_analysis_dimensions(d, self.config['analysis_space'])

    def score_events(self, d):
        """Returns array (n_sources, n_events) of pdf values for each source for each of the events"""
        return np.vstack([s.pdf(*self.to_analysis_dimensions(d)) for s in self.sources])

    def pmf_grids(self):
        """Return array (n_sources, *analysis_space_shape) of integrated PDFs in the analysis space for each source"""
        return (np.stack([s.get_pmf_grid()[0] for s in self.sources]),
                np.stack([s.get_pmf_grid()[1] for s in self.sources]))

    def expected_events(self, s=None):
        """Return the total number of events expected in the analysis range for the source s.
        If no source specified, return an array of results for all sources.
        # TODO: Why is this not a method of source?
        """
        if s is None:
            return np.array([self.expected_events(s) for s in self.sources])
        return s.events_per_day * self.config['livetime_days'] * s.fraction_in_range * s.config['rate_multiplier']

    def show(self, d, ax=None, dims=None, **kwargs):
        """Plot the events from dataset d in the analysis range
        ax: plot on this Axes
        Dims: numbers of dimension(s) to plot in. Can be up to two dimensions.
        """
        kwargs.setdefault('s', 5)

        import matplotlib.pyplot as plt
        dim_names, bins = zip(*self.config['analysis_space'])

        if dims is None:
            if len(bins) == 1:
                dims = tuple([0])
            else:
                dims = (0, 1)
        if ax is None:
            ax = plt.gca()

        for s_i, s in enumerate(self.sources):
            q = d[d['source'] == s_i]
            q_in_space = self.to_analysis_dimensions(q)
            ax.scatter(q_in_space[dims[0]],
                       q_in_space[dims[1]] if len(dims) > 1 else np.zeros(len(q)),
                       color=s.config['color'], label=s.config['label'], **kwargs)

        ax.set_xlabel(dim_names[dims[0]])
        ax.set_xlim(bins[dims[0]][0], bins[dims[0]][-1])

        if len(dims) > 1:
            ax.set_ylabel(dim_names[dims[1]])
            ax.set_ylim(bins[dims[1]][0], bins[dims[1]][-1])
