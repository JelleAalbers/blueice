import pickle
import os
import collections
import json
from hashlib import sha1
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from multihist import Histdd
from .source import Source
from . import utils

# Features I would like to add:
#  - Non-asymptotic limit setting
#  - General (shape) uncertainties
#  - Fit parameters other than source strength


def load_pickle(filename):
    """Loads a pickle from filename"""
    with open(utils.data_file_name(filename), mode='rb') as infile:
        return pickle.load(infile)

def save_pickle(stuff, filename):
    """Saves stuff in a pickle at filename"""
    with open(filename, mode='wb') as outfile:
        pickle.dump(stuff, outfile)
#
# def hash_dict(d):
#     return tuple(sorted(d.items()))


class Model(object):
    """Model for XENON1T dataset simulation and analysis
    """
    config = None            # type: dict
    exposure_factor = 1      # Increase this to quickly change the exposure of the model
    from_cache = False       # If true, this model's pdfs were loaded from the cache

    def __init__(self, config, ipp_client=None, **kwargs):
        """
        :param config: Dictionary specifying detector parameters, source info, etc.
        :param ipp_client: ipyparallel client to use for parallelizing pdf computation (optional)
        :param kwargs: Overrides for the config (optional)
        :param cache: Saves the model after initialization. Will has the config to provide identifier.
        :return:
        """
        self.config = deepcopy(config)
        self.config.update(kwargs)

        # Compute a hash of the config dictionary now that it is still "unpimped"
        self.config_hash = sha1(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        # hashable = deepcopy(self.config)
        # hashable['sources'] = tuple([hash_dict(x) for x in hashable['sources']])
        # self.config_hash = hash(hash_dict(hashable))
        cache_filename = 'pdfs_%s.pklz' % self.config_hash

        # Have we computed the pdfs for this configuration before? If so, load them.
        if not self.config['force_pdf_recalculation'] and os.path.exists(cache_filename):
            self.from_cache = True
            cached_source_data = load_pickle(cache_filename)
            for i, source_spec in enumerate(self.config['sources']):
                self.config['sources'][i].update(cached_source_data[source_spec['name']])

        # "pimp" the configuration by turning file name settings into the objects they represent
        # After this we can no longer compute a hash of the config.
        self.config['s1_relative_ly_map'] = load_pickle(self.config['s1_relative_ly_map'])
        for s_i, source_spec in enumerate(self.config['sources']):
            self.config['sources'][s_i]['energy_distribution'] = \
                load_pickle(source_spec['energy_distribution'])

        self.space = collections.OrderedDict(self.config['analysis_space'])
        self.dims = list(self.space.keys())
        self.bins = list(self.space.values())

        # Intialize the sources
        self.sources = []
        for source_spec in self.config['sources']:
            source = Source(self.config, source_spec)
            if source.pdf_histogram is None or self.config['force_pdf_recalculation']:
                self.compute_source_pdf(source, ipp_client=ipp_client)
            self.sources.append(source)

        if not self.from_cache and self.config.get('save_pdfs', True):
            # Save the source PDFs, rate, etc. for later use
            save_pickle({s.name: {k: getattr(s, k) for k in ('pdf_histogram', 'pdf_errors', 'fraction_in_range',
                                                             'events_per_day')}
                         for s in self.sources},
                        cache_filename)

    def compute_source_pdf(self, source, ipp_client=None):
        """Computes the PDF of the source in the analysis space.
        Returns nothing, modifies source in-place.
        :param ipp_client: ipyparallel client to use for parallelizing pdf computation (optional)
        """
        # Not a method of source, since it needs analysis space definition...
        # To be honest I'm not sure where this method (and source.simulate) actually would fit best.

        # Simulate batches of events at a time (to avoid memory errors, show a progressbar, and split up among machines)
        # Number of events to simulate will be rounded up to the nearest batch size
        batch_size = self.config['pdf_sampling_batch_size']
        n_batches = int((source.n_events_for_pdf * self.config['pdf_sampling_multiplier']) // batch_size + 1)
        n_events = n_batches * batch_size
        mh = Histdd(bins=self.bins)

        if ipp_client is not None:
            # We need both a directview and a load-balanced view: the latter doesn't have methods like push.
            directview = ipp_client[:]
            lbview = ipp_client.load_balanced_view()

            # Get the necessary objects to the engines
            # For some reason you can't directly .push(dict(bins=self.bins)),
            # it will fail with 'bins' is not defined error. When you first assign bins = self.bins it works.
            bins = self.bins
            dims = self.dims

            def to_space(d):
                """Standalone counterpart of Model.to_space, needed for parallel simulation. Ugly!!"""
                return [d[dims[i]] for i in range(len(dims))]

            def do_sim(_):
                """Run one simulation batch and histogram it immediately (so we don't pass gobs of data around)"""
                return Histdd(*to_space(source.simulate(batch_size)), bins=bins).histogram

            directview.push(dict(source=source, bins=bins, dims=dims, batch_size=batch_size, to_space=to_space),
                                  block=True)

            amap_result = lbview.map(do_sim, [None for _ in range(n_batches)], ordered=False,
                                     block=self.config.get('block_during_simulation', False))
            for r in tqdm(amap_result, total=n_batches, desc='Sampling PDF of %s' % source.name):
                mh.histogram += r

        else:
            for _ in tqdm(range(n_batches),
                          desc='Sampling PDF of %s' % source.name):
                mh.add(*self.to_space(source.simulate(batch_size)))

        source.fraction_in_range = mh.n / n_events

        # Convert the histogram to a PDF
        # This means we have to divide by
        #  - the number of events histogrammed
        #  - the bin sizes (particularly relevant for non-uniform bins!)
        source.pdf_histogram = mh.similar_blank_hist()
        source.pdf_histogram.histogram = mh.histogram.astype(np.float) / mh.n
        source.pdf_histogram.histogram /= np.outer(*[np.diff(self.bins[i]) for i in range(len(self.bins))])

        # Estimate the MC statistical error. Not used for anything, but good to inspect.
        source.pdf_errors = source.pdf_histogram / np.sqrt(np.clip(mh.histogram, 1, float('inf')))
        source.pdf_errors[source.pdf_errors == 0] = float('nan')

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

    def simulate(self, wimp_strength=0, restrict=True):
        """Makes a toy dataset.
        if restrict=True, return only events inside analysis range
        """
        ds = []
        for s_i, source in enumerate(self.sources):
            n = np.random.poisson(source.events_per_day *
                                  self.config['livetime_days'] *
                                  (10**wimp_strength if source.name.startswith('wimp') else 1) *
                                  self.exposure_factor)
            d = source.simulate(n)
            d['source'] = s_i
            ds.append(d)
        d = np.concatenate(ds)
        if restrict:
            d = self.range_cut(d)
        return d

    def show(self, d, ax=None, dims=(0, 1)):
        """Plot the events from dataset d in the analysis range
        ax: plot on this Axes
        Dims: tuple of numbers indicating which two dimensions to plot in.
        """
        if ax is None:
            ax = plt.gca()

        d = self.range_cut(d)
        for s_i, s in enumerate(self.sources):
            q = d[d['source'] == s_i]
            q_in_space = self.to_space(q)
            ax.scatter(q_in_space[dims[0]],
                       q_in_space[dims[1]],
                       color=s.color, s=5, label=s.label)

        ax.set_xlabel(self.dims[dims[0]])
        ax.set_ylabel(self.dims[dims[1]])
        ax.set_xlim(self.bins[dims[0]][0], self.bins[dims[0]][-1])
        ax.set_ylim(self.bins[dims[1]][0], self.bins[dims[1]][-1])

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
        return s.events_per_day * self.config['livetime_days'] * s.fraction_in_range * self.exposure_factor