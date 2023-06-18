"""Built-in Source baseclasses. In order of increasing functionality and decreasing generality:

 * Source: only sets up default arguments and helper functions for caching.
   Use e.g. if you have an analytic pdf

 * HistogramPdfSource:  + fetch/interpolate the PDF/PMF from a (multihist) histogram
   Use e.g. if you have a numerically computable pdf (e.g. using convolution of some known functions)

 * DensityEstimatingSource: + create that histogram by binning some sample of events
   Use e.g. if you want to estimate density from a calibration data sample.

 * MonteCarloSource: + get that sample from the source's own simulate method.
   Use if you have a Monte Carlo to generate events. This was the original 'niche' for which blueice was created.

Parent methods (e.g. Source.compute_pdf) are meant to be called at the end of the child methods
that override them (e.g. HistogramPdfSource.compute_pdf).
"""
import inspect
import os
from functools import reduce

import numpy as np
from blueice.exceptions import PDFNotComputedException
from multihist import Histdd
from scipy.interpolate import RegularGridInterpolator

from . import utils
from .data_reading import read_files_in

__all__ = ['Source', 'HistogramPdfSource', 'DensityEstimatingSource', 'MonteCarloSource']


class Source(object):
    """Base class for a source of events."""

    # Class-level cache for loaded sources
    # Useful so we don't create possibly expensive objects
    _data_cache = dict()

    def __repr__(self):
        return "%s[%s]" % (self.name, self.hash if hasattr(self, 'hash') else 'nohashknown')

    def __init__(self, config, *args, **kwargs):
        defaults = dict(name='unnamed_source',
                        label='Unnamed source',
                        color='black',            # Color to use in plots

                        # Defaults for events_per_day and fraction_in_range. These immediately get converted into
                        # attributes, which can be modified dynamically (e.g. Not only can these be overriden in config,
                        # some child classes set them dynamically (eg DensityEstimatingSource will set them based on
                        # the sample events you pass in).
                        events_per_day=0,         # Events per day this source produces (detected or not).
                        rate_multiplier=1,        # Rate multiplier (independent of loglikelihood's rate multiplier)
                        fraction_in_range=1,      # Fraction of simulated events that fall in analysis space.

                        # List of attributes you want to be stored in cache. When the same config is passed later
                        # (ignoring the dont_hash_settings), these attributes will be set from the cached file.
                        cache_attributes=[],

                        # Set to True if you want to call compute_pdf at a time of your convenience, rather than
                        # at the end of init.
                        delay_pdf_computation=False,

                        # List of names of settings which are not included in the hash. These should be all settings
                        # that have no impact on the pdf (e.g. whether to show progress bars or not).
                        dont_hash_settings=[],
                        extra_dont_hash_settings=[],

                        # If true, never retrieve things from the cache. Saving to cache still occurs.
                        force_recalculation=False,
                        # If true, never save things to the cache. Loading from cache still occurs.
                        never_save_to_cache=False,
                        cache_dir='pdf_cache',
                        task_dir='pdf_tasks')
        c = utils.combine_dicts(defaults, config)
        c['cache_attributes'] += ['fraction_in_range', 'events_per_day', 'pdf_has_been_computed']
        c['dont_hash_settings'] += ['hash', 'rate_multiplier',
                                    'force_recalculation', 'never_save_to_cache', 'dont_hash_settings',
                                    'label', 'color', 'extra_dont_hash_settings', 'delay_pdf_computation',
                                    'cache_dir', 'task_dir']

        # Merge the 'extra' (per-source) dont hash settings into the normal dont_hash_settings
        c['dont_hash_settings'] += c['extra_dont_hash_settings']
        del c['extra_dont_hash_settings']

        self.name = c['name']
        del c['name']

        # events_per_day and fraction_in_range may be modified / set properly for the first time later (see comments
        # in 'defaults' above)
        if hasattr(self, 'events_per_day'):
            raise ValueError("events_per_day defaults should be set via config!")
        self.events_per_day = c['events_per_day']
        self.fraction_in_range = c['fraction_in_range']
        self.pdf_has_been_computed = False

        # What is this source's unique id?
        if 'hash' in c:
            # id already given in config: probably because config has already been 'pimped' with loaded objects
            self.hash = c['hash']
        else:
            # Compute id from config
            hash_config = utils.combine_dicts(c, exclude=c['dont_hash_settings'])
            self.hash = c['hash'] = utils.deterministic_hash(hash_config)

        # What filename would a source with this config have in the cache?
        if not os.path.exists(c['cache_dir']):
            os.makedirs(c['cache_dir'])
        self._cache_filename = os.path.join(c['cache_dir'], self.hash)

        # Can we load this source from cache? If so, do so: we don't even need to load any files...
        if not c['force_recalculation'] and os.path.exists(self._cache_filename):
            self.from_cache = True

            if self.hash in self._data_cache:
                # We already loaded this from cache sometime in this process
                stuff = self._data_cache[self.hash]
            else:
                # Load it from disk, and store in the class-level cache
                stuff = self._data_cache[self.hash] = utils.read_pickle(self._cache_filename)

            for k, v in stuff.items():
                if k not in c['cache_attributes']:
                    raise ValueError("%s found in cached file, but you only wanted %s from cache. "
                                     "Old cache?" % (k, c['cache_attributes']))
                setattr(self, k, v)
        else:
            self.from_cache = False

        # Convert any filename-valued settings to whatever is in those files.
        c = read_files_in(c, config['data_dirs'])

        self.config = c

        if self.from_cache:
            assert self.pdf_has_been_computed

        else:
            if self.config['delay_pdf_computation']:
                self.prepare_task()
            else:
                self.compute_pdf()

    def compute_pdf(self):
        """Initialize, then cache the PDF. This is called
         * AFTER the config initialization and
         * ONLY when source is not already loaded from cache. The caching mechanism exists to store the quantities you
           need to compute here.
        """
        if self.pdf_has_been_computed:
            raise RuntimeError("compute_pdf called twice on a source!")
        self.pdf_has_been_computed = True
        self.save_to_cache()

    def save_to_cache(self):
        """Save attributes in self.config['cache_attributes'] of this source to cache."""
        if not self.from_cache and not self.config['never_save_to_cache']:
            utils.save_pickle({k: getattr(self, k) for k in self.config['cache_attributes']},
                              self._cache_filename)
        return self._cache_filename

    def prepare_task(self):
        """Create a task file in the task_dir for delayed/remote computation"""
        task_filename = os.path.join(self.config['task_dir'], self.hash)
        utils.save_pickle((self.__class__, self.config), task_filename)

    def pdf(self, *args):
        raise NotImplementedError

    def get_pmf_grid(self, *args):
        """Returns pmf_grid, n_events:
         - pmf_grid: pmf per bin in the analysis space
         - n_events: if events were used for density estimation: number of events per bin (for DensityEstimatingSource)
           otherwise float('inf')
        This is used by binned likelihoods. if you have an unbinned density estimator, you'll have to write
        some integration / sampling routine!
        """
        raise NotImplementedError

    def simulate(self, n_events):
        """Simulate n_events according to the source. It's ok to return less than n_events events,
        if you decide some events are not detectable.
        """
        raise NotImplementedError


class HistogramPdfSource(Source):
    """A source which takes its PDF values from a multihist histogram.
    """
    _pdf_histogram = None
    _bin_volumes = None
    _n_events_histogram = None

    def __init__(self, config, *args, **kwargs):
        """Prepares the PDF of this source for use.
        """
        defaults = dict(pdf_sampling_multiplier=1,
                        pdf_interpolation_method='linear',)
        config = utils.combine_dicts(defaults, config)
        config['cache_attributes'] = config.get('cache_attributes', []) + \
            ['_pdf_histogram', '_n_events_histogram', '_bin_volumes']
        Source.__init__(self, config, *args, **kwargs)

    def build_histogram(self):
        """Set the _pdf_histogram (Histdd), _n_events_histogram (Histdd) and _bin_volumes (numpy array) attributes
        """
        raise NotImplementedError

    def compute_pdf(self):
        # Fill the histogram with either events or an evaluated pdf
        self.build_histogram()
        Source.compute_pdf(self)

    def pdf(self, *args):
        if not self.pdf_has_been_computed:
            raise PDFNotComputedException("%s: Attempt to call a PDF that has not been computed" % self)

        method = self.config['pdf_interpolation_method']

        if method == 'linear':
            if not hasattr(self, '_pdf_interpolator'):
                # First call:
                # Construct a linear interpolator between the histogram bins
                self._pdf_interpolator = RegularGridInterpolator(self._pdf_histogram.bin_centers(),
                                                                 self._pdf_histogram.histogram)

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
            raise NotImplementedError("PDF Interpolation method %s not implemented" % method)

    def simulate(self, n_events):
        """Simulate n_events from the PDF histogram"""
        if not self.pdf_has_been_computed:
            raise PDFNotComputedException("%s: Attempt to simulate events from a PDF that has not been computed" % self)

        events_per_bin = self._pdf_histogram * self._bin_volumes
        q = events_per_bin.get_random(n_events)

        # Convert to numpy record array
        d = np.zeros(n_events,
                     dtype=[('source', int)] +
                           [(x[0], float)
                            for x in self.config['analysis_space']])
        for i, x in enumerate(self.config['analysis_space']):
            d[x[0]] = q[:, i]

        return d

    def get_pmf_grid(self):
        return self._pdf_histogram.histogram * self._bin_volumes, self._n_events_histogram.histogram


class DensityEstimatingSource(HistogramPdfSource):
    """A source which estimates its PDF by some events you give to it.
    Child classes need to implement get_events_for_density_estimate, and call compute_pdf when they are ready
    (usually at the end of their own init).
    """

    def __init__(self, config, *args, **kwargs):
        """Prepares the PDF of this source for use.
        """
        defaults = dict(n_events_for_pdf=1e6)
        config = utils.combine_dicts(defaults, config)
        config['cache_attributes'] = config.get('cache_attributes', [])
        HistogramPdfSource.__init__(self, config, *args, **kwargs)

    def build_histogram(self):
        # Get the events to estimate the PDF
        dimnames, bins = zip(*self.config['analysis_space'])
        mh = Histdd(bins=bins, axis_names=dimnames)

        # Get a generator function which will give us the events
        get = self.get_events_for_density_estimate
        if not inspect.isgeneratorfunction(get):
            def get():
                return [self.get_events_for_density_estimate()]

        n_events = 0
        for events, n_simulated in get():
            n_events += n_simulated
            mh.add(*utils._events_to_analysis_dimensions(events, self.config['analysis_space']))

        self.fraction_in_range = mh.n / n_events

        # Convert the histogram to a density estimate
        # This means we have to divide by
        #  - the number of events IN RANGE received
        #    (fraction_in_range keeps track of how many events were not in range)
        #  - the bin sizes
        self._pdf_histogram = mh.similar_blank_hist()
        self._pdf_histogram.histogram = mh.histogram.astype(float) / mh.n

        # For the bin widths we need to take an outer product of several vectors, for which numpy has no builtin
        # This reduce trick does the job instead, see http://stackoverflow.com/questions/17138393
        self._bin_volumes = reduce(np.multiply, np.ix_(*[np.diff(bs) for bs in bins]))
        self._pdf_histogram.histogram /= self._bin_volumes

        self._n_events_histogram = mh

        return mh

    def get_events_for_density_estimate(self):
        """Return, or yield in batches, (events for use in density estimation, events simulated/read)
        Passing the count is necessary because you sometimes work with simulators that already cut some events.
        """
        raise NotImplementedError


class MonteCarloSource(DensityEstimatingSource):
    """A DensityEstimatingSource which gets the events for the density estimator from its own simulate() method.
    Child classes have to implement simulate.
    """
    def __init__(self, config, *args, **kwargs):
        defaults = dict(n_events_for_pdf=1e6,
                        pdf_sampling_multiplier=1,
                        pdf_sampling_batch_size=1e6)
        config = utils.combine_dicts(defaults, config)
        config['dont_hash_settings'] = config.get('dont_hash_settings', []) + ['pdf_sampling_batch_size']
        DensityEstimatingSource.__init__(self, config, *args, **kwargs)

    def get_events_for_density_estimate(self):
        # Simulate batches of events at a time (to avoid memory errors, show a progressbar, and split up among machines)
        # Number of events to simulate will be rounded up to the nearest batch size
        n_events = self.config['n_events_for_pdf'] * self.config['pdf_sampling_multiplier']
        batch_size = self.config['pdf_sampling_batch_size']
        if n_events <= batch_size:
            batch_size = n_events

        for _ in range(int(n_events // batch_size)):
            result = self.simulate(n_events=batch_size)
            yield result, batch_size
