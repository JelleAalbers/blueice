from functools import reduce
import os
import inspect

import numpy as np
from multihist import Histdd
from scipy.interpolate import RegularGridInterpolator

from . import utils
from .data_reading import read_if_is_filename


class Source(object):
    """Base class for a source of events."""

    def __init__(self, config, *args, **kwargs):
        defaults = dict(name='unnamed_source',
                        label='Unnamed source',
                        color='black',            # Color to use in plots
                        events_per_day=0,         # Events per day this source produces (detected or not).
                        fraction_in_range=1,      # Fraction of simulated events that fall in analysis space.

                        # List of names of settings which are not included in the hash. These should be all settings
                        # that have no impact on the pdf (e.g. whether to show progress bars or not).
                        dont_hash_settings=[],

                        # List of attributes you want to be stored in cache. When the same config is passed later
                        # (ignoreing the dont_hash_settings), these attributes will be set from the cached file.
                        cache_attributes=[],

                        # If true, never retrieve things from the cache. Saving to cache still occurs.
                        force_recalculation=False,
                        cache_dir='pdf_cache',
                        )
        c = utils.combine_dicts(defaults, config)
        c['dont_hash_settings'] += ['force_recalculation', 'dont_hash_settings', 'label', 'color']

        # Name becomes an attribute.
        self.name = c['name']
        del c['name']

        # Compute a hash to uniquely identify the relevant settings for this source.
        hash_config = utils.combine_dicts(c, exclude=c['dont_hash_settings'])
        self.hash = utils.deterministic_hash(hash_config)

        # What filename would a source with this config have in the cache?
        if not os.path.exists(c['cache_dir']):
            os.makedirs(c['cache_dir'])
        self._cache_filename = os.path.join(c['cache_dir'], self.hash)

        # Can we load this source from cache? If so, do so: we don't even need to load any files...
        if not c['force_recalculation'] and os.path.exists(self._cache_filename):
            self.from_cache = True
            for k, v in utils.read_pickle(self._cache_filename).items():
                if k not in c['cache_attributes']:
                    raise ValueError("%s found in cached file, but you only wanted %s from cache. "
                                     "Old cache?" % (k, c['cache_attributes']))
                setattr(self, k, v)
        else:
            self.from_cache = False

        # Convert any filename-valued settings to whatever is in those files.
        c = {k: read_if_is_filename(v) for k, v in c.items()}

        self.config = c

    def save_to_cache(self):
        """Save attributes in self.config['cache_attributes'] of this source to cache."""
        if not self.from_cache:
            utils.save_pickle({k: getattr(self, k) for k in self.config['cache_attributes']}, self._cache_filename)
        return self._cache_filename

    def pdf(self, *args):
        raise NotImplementedError

    def simulate(self, n_events):
        raise NotImplementedError


class PDFNotComputedException(Exception):
    pass


class DensityEstimatingSource(Source):
    """A source which estimates its PDF by some events you give to it.
    Child classes need to implement get_events_for_density_estimate, and call compute_pdf when they are ready
    (usually at the end of their own init).
    """

    def __init__(self, config, *args, **kwargs):
        """Prepares the PDF of this source for use.
        """
        defaults = dict(n_events_for_pdf=1e6,
                        pdf_sampling_multiplier=1,
                        pdf_interpolation_method='linear',)
        config = utils.combine_dicts(defaults, config)
        config['cache_attributes'] = config.get('cache_attributes', []) + \
            ['_pdf_histogram', '_pdf_errors', 'events_per_day', 'fraction_in_range']
        self.pdf_has_been_computed = False
        Source.__init__(self, config)

    def compute_pdf(self):
        if not self.from_cache:
            # Get the events to estimate the PDF
            dimnames, bins = zip(*self.config['analysis_space'])
            mh = Histdd(bins=bins, axis_names=dimnames)

            # Get a generator function which will give us the events
            get = self.get_events_for_density_estimate
            if not inspect.isgeneratorfunction(get):
                def get():
                    yield self.get_events_for_density_estimate()

            n_events = 0
            for events in get():
                n_events += len(events)
                mh.add(*utils._events_to_analysis_dimensions(events, self.config['analysis_space']))

            self.fraction_in_range = mh.n / n_events

            # Convert the histogram to a density estimate
            # This means we have to divide by
            #  - the number of events IN RANGE received
            #    (fraction_in_range keeps track of how many events were not in range)
            #  - the bin sizes
            self._pdf_histogram = mh.similar_blank_hist() / n_events
            self._pdf_histogram.histogram = mh.histogram.astype(np.float) / mh.n

            # For the bin widths we need to take an outer product of several vectors, for which numpy has no builtin
            # This reduce trick does the job instead, see http://stackoverflow.com/questions/17138393
            self._pdf_histogram.histogram /= reduce(np.multiply, np.ix_(*[np.diff(bs) for bs in bins]))

            # Estimate the MC statistical error. Not used for anything, but good to inspect.
            self._pdf_errors = self._pdf_histogram / np.sqrt(np.clip(mh.histogram, 1, float('inf')))
            self._pdf_errors[self._pdf_errors == 0] = float('nan')

        # Construct a linear interpolator between the histogram bins
        if self.config['pdf_interpolation_method'] == 'linear':
            self._pdf_interpolator = RegularGridInterpolator(self._pdf_histogram.bin_centers(),
                                                             self._pdf_histogram.histogram)

        self.save_to_cache()
        self.pdf_has_been_computed = True

    def pdf(self, *args):
        if not self.pdf_has_been_computed:
            raise PDFNotComputedException("Attempt to call a PDF that has not been computed")
        method = self.config['pdf_interpolation_method']
        if method == 'linear':
            # The interpolator works only within the bin centers region: clip the input data to that.
            # Assuming you've cut the data to the analysis space first (which you should have!)
            # this is equivalent to assuming constant density in the outer half of boundary bins
            clipped_data = []
            for dim_i, x in enumerate(args):
                bcs = self._pdf_histogram.bin_centers(dim_i)
                clipped_data.append(np.clip(x, bcs.min(), bcs.max()))

            return self._pdf_interpolator(np.transpose(clipped_data))

        elif method == 'piecewise':
            return self._pdf_histogram.lookup(*args)

        else:
            raise NotImplementedError("PDF Interpolation method %s not implemented" % self.pdf_interpolation_method)

    def get_events_for_density_estimate(self):
        """Return, or yield in batches, events for use in density estimation"""
        raise NotImplementedError


class MonteCarloSource(DensityEstimatingSource):
    """A DensityEstimatingSource which gets the events for the density estimator from its own simulate() method.
    Child classes have to implement simulate, and call compute_pdf when they are ready
    (usually at the end of their own init)
    """
    def __init__(self, config, *args, **kwargs):
        defaults = dict(n_events_for_pdf=1e6,
                        pdf_sampling_multiplier=1,
                        pdf_sampling_batch_size=1e6)
        config = utils.combine_dicts(defaults, config)
        config['dont_hash_settings'] = config.get('dont_hash_settings', []) + ['pdf_sampling_batch_size']
        DensityEstimatingSource.__init__(self, config)

    def get_events_for_density_estimate(self):
        # Simulate batches of events at a time (to avoid memory errors, show a progressbar, and split up among machines)
        # Number of events to simulate will be rounded up to the nearest batch size
        n_events = self.config['n_events_for_pdf'] * self.config['pdf_sampling_multiplier']
        batch_size = self.config['pdf_sampling_batch_size']
        if n_events <= batch_size:
            batch_size = n_events

        yield self.simulate(n_events=batch_size)
