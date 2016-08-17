import collections
from copy import deepcopy
from tqdm import tqdm

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
        :param cache: Saves the model after initialization. Will has the config to provide identifier.
        :return:
        """
        # Copy the config: we're going to modify it and don't want user to be surprised
        self.config = deepcopy(config)

        # Defaults for settings that are used in several places.
        # Settings used in one place have their default coded there (using .get)
        self.config.setdefault('livetime_days', 1)
        self.config.setdefault('data_dirs', 1)
        self.config.setdefault('nohash_settings',
                               tuple(['data_dirs', 'pdf_sampling_batch_size', 'force_pdf_recalculation']))

        self.config.update(kwargs)
        self.inert_config = deepcopy(self.config)       # Copy without file names -> loaded objects conversion

        # Load objects specified by file name into the config dictionary.
        utils.process_files_in_config(self.config, self.config['data_dirs'])

        self.space = collections.OrderedDict(self.config['analysis_space'])
        self.dims = list(self.space.keys())
        self.bins = list(self.space.values())

        self.sources = []
        for source_config in self.config['sources']:
            if 'class' in source_config:
                source_class = source_config['class']
                del source_config['class']    # Don't want this stored in source config
            else:
                source_class = self.config['default_source_class']
            self.sources.append(source_class(self, source_config, ipp_client=ipp_client))

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

    def range_cut(self, d, ps=None):
        """Return events from dataset d which are in the analysis space
        Also removes events for which all of the source PDF's are zero (which cannot be meaningfully interpreted)
        """
        mask = np.ones(len(d), dtype=np.bool)
        for dimension, bin_edges in self.space.items():
            mask = mask & (d[dimension] >= bin_edges[0]) & (d[dimension] <= bin_edges[-1])

        # Ignore events to which no source pdf assigns a positive probability.
        # These would cause log(sum_sources (mu * p)) in the loglikelihood to become -inf.
        if ps is None:
            ps = self.score_events(d)
        mask &= ps.sum(axis=0) != 0

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

    def to_space(self, d):
        """Given a dataset, returns list of arrays of coordinates of the events in the analysis dimensions"""
        return [d[self.dims[i]] for i in range(len(self.dims))]

    def score_events(self, d):
        """Returns array (n_sources, n_events) of pdf values for each source for each of the events"""
        return np.vstack([s.pdf(*self.to_space(d)) for s in self.sources])

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
        Dims: tuple of numbers indicating which dimension(s) to plot in. Can be one or two dimensions.
        """
        if dims is None:
            if len(self.bins) == 1:
                dims = tuple([0])
            else:
                dims = (0, 1)
        if ax is None:
            ax = plt.gca()

        # d = self.range_cut(d)   # Not needed, simulate already does this
        for s_i, s in enumerate(self.sources):
            q = d[d['source'] == s_i]
            q_in_space = self.to_space(q)
            ax.scatter(q_in_space[dims[0]],
                       q_in_space[dims[1]] if len(dims) > 1 else np.zeros(len(q)),
                       color=s.color, s=5, label=s.label)

        ax.set_xlabel(self.dims[dims[0]])
        ax.set_xlim(self.bins[dims[0]][0], self.bins[dims[0]][-1])

        if len(dims) > 1:
            ax.set_ylabel(self.dims[dims[1]])
            ax.set_ylim(self.bins[dims[1]][0], self.bins[dims[1]][-1])


def create_models_in_parallel(configs, ipp_client=None, block=False):
    """Return Models for each configuration in configs.
    :param ipp_client: ipyparallel client to use for parallelized computation, or None (in which case models will be
                       computed serially. For now only engines running in the same directory as the main code
                       are supported, see #1.
    :param configs: list of Model configuration dictionaries
    :param block: passed to the async map of ipyparallel. Useful for debugging, but disables progress bar.
    :return: list of Models.
    """
    if ipp_client is not None:
        # Fully fledged blueice Models don't pickle, so we have to construct them again later in the main process
        # (but then we can just grab their PDFs from cache, so it's quick)

        def compute_model(conf):
            Model(conf)
            return None

        asyncresult = ipp_client.load_balanced_view().map(compute_model, configs, ordered=False, block=block)
        for _ in tqdm(asyncresult,
                      desc="Computing models in parallel",
                      smoothing=0,   # Show average speed, instantaneous speed is extremely variable
                      total=len(configs)):
            pass

    # (Re)make the models in the main process; hopefully PDFs use the cache...
    return [Model(conf) for conf in configs]
