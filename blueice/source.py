import os
import numpy as np
from multihist import Histdd
from tqdm import tqdm
from functools import reduce

from . import utils

class Source(object):
    """Base class for a source of events.
    Child classes must implement 'pdf' and 'simulate'.
    Child classes need to ensure events_per_day and fraction_in_range is set.
    """
    name = 'unspecified'
    label = 'Catastrophic irreducible noise'
    color = 'black'                 # Color to use in plots
    events_per_day = 0
    fraction_in_range = 0           # Fraction of simulated events that fall in analysis space.
    ignore_settings = tuple()       # Settings from the model config that don't impact this source at all

    def __init__(self, model, config, *args, **kwargs):
        """
        config: dict with general parameters (same as Model.config)
        spec: dict with source-specific stuff, e.g. name, label, etc. Used to set attributes defined above.
        args and kwargs are ignored. Accepted so children can pass their args/kwargs on to parent without fear.
        """
        self.model = model
        utils.process_files_in_config(config, self.model.config['data_dirs'])
        for k, v in config.items():       # Store source specs as attributes
            setattr(self, k, v)

    def pdf(self, *args):
        raise NotImplementedError

    def simulate(self, n_events):
        raise NotImplementedError


class MonteCarloSource(Source):
    """A source which computes its PDF from MC simulation
    PDFs are cached in cache_dir.
    """
    n_events_for_pdf = 1e6
    pdf_histogram = None            # Histdd
    pdf_errors = None               # Histdd
    from_cache = False              # If True, the pdf was loaded from cache instead of computed on init
    hash = None
    pdf_interpolation_method = 'linear'
    pdf_interpolator = None         # scipy.interpolate.RegularGridInterpolator

    def __init__(self, model, config, ipp_client=None, **kwargs):
        """Prepares the PDF of this source for use.
        :param ipp_client: ipyparallel client to parallelize computation (optional)
        """
        # Compute a hash to uniquely identify the relevant settings for this source.
        # Must do this before filename arguments are converted to objects from those files in Source.__init__.
        self.hash = utils.deterministic_hash({k: v for k, v in model.inert_config.items()
                                              if k not in config.get('ignore_settings', []) +
                                                          model.config['nohash_settings']})
        self.hash += utils.deterministic_hash(config)

        super().__init__(model, config, **kwargs)
        self.setup()

        # What filename would a source with this config have in the cache?
        cache_dir = self.model.config.get('pdf_cache_dir', 'pdf_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_filename = os.path.join(cache_dir, self.hash)

        # Have we computed the pdfs etc. for this configuration before? If so, load the information.
        if not self.model.config.get('force_pdf_recalculation', False) and os.path.exists(cache_filename):
            self.from_cache = True
            for k, v in utils.load_pickle(cache_filename).items():
                setattr(self, k, v)
        else:
            self.compute_pdf(ipp_client)

        # Construct a linear interpolator between the histogram bins
        from scipy.interpolate import RegularGridInterpolator
        self.pdf_interpolator = RegularGridInterpolator(self.pdf_histogram.bin_centers(),
                                                        self.pdf_histogram.histogram)

        # Should we save the source PDFs, rate, etc. for later use?
        if not self.from_cache and self.model.config.get('save_pdfs', True):
            utils.save_pickle({k: getattr(self, k) for k in ('pdf_histogram', 'pdf_errors',
                                                             'fraction_in_range', 'events_per_day')},
                               cache_filename)

    def setup(self):
        """Called just after converting config arguments, but before calculating pdf"""
        pass

    def compute_pdf(self, ipp_client=None):
        # Simulate batches of events at a time (to avoid memory errors, show a progressbar, and split up among machines)
        # Number of events to simulate will be rounded up to the nearest batch size
        n_events = self.n_events_for_pdf * self.model.config.get('pdf_sampling_multiplier', 1)
        batch_size = self.model.config.get('pdf_sampling_batch_size', n_events)
        bins = self.model.bins
        # Round to the nearest number of full batches
        n_batches = int(np.round(n_events/batch_size))
        n_events = n_batches * batch_size
        mh = Histdd(bins=bins)

        def simulate_batch(args):
            """Run one simulation batch and histogram it immediately (so we don't pass gobs of data around)"""
            source, batch_size = args
            d = source.simulate(batch_size)
            d = source.model.to_space(d)
            return Histdd(*d, bins=source.model.bins).histogram

        map_args = (simulate_batch, [(self, batch_size) for _ in range(n_batches)])

        if ipp_client is not None:
            # If this gives you a strange unintelligible error, turn on block_during_simulation
            map_result = ipp_client.load_balanced_view().map(*map_args,
                                                             ordered=False,
                                                             block=self.model.config.get('block_during_simulation',
                                                                                         False))
        else:
            map_result = map(*map_args)

        if self.model.config.get('show_pdf_sampling_progress', True):
            map_result = tqdm(map_result, total=n_batches, desc='Sampling PDF of %s' % self.name)

        for r in map_result:
            mh.histogram += r

        self.fraction_in_range = mh.n / n_events

        # Convert the histogram to a PDF estimate
        # This means we have to divide by
        #  - the number of events histogrammed
        #  - the bin sizes (particularly relevant for non-uniform bins!)
        self.pdf_histogram = mh.similar_blank_hist()
        self.pdf_histogram.histogram = mh.histogram.astype(np.float) / mh.n
        # For the bin widths we need to take an outer product of several vectors, for which numpy has no builtin
        # This reduce trick does the job instead, see http://stackoverflow.com/questions/17138393
        self.pdf_histogram.histogram /= reduce(np.multiply, np.ix_(*[np.diff(bs) for bs in bins]))

        # Estimate the MC statistical error. Not used for anything, but good to inspect.
        self.pdf_errors = self.pdf_histogram / np.sqrt(np.clip(mh.histogram, 1, float('inf')))
        self.pdf_errors[self.pdf_errors == 0] = float('nan')

    def simulate(self, n_events):
        raise NotImplementedError

    def pdf(self, *args):
        if self.pdf_interpolation_method == 'linear':
            # Clip the data to lay inside the bin centers region.
            # Assuming you've cut the data to the analysis space first (which you should have!)
            # this is equivalent to assuming constant density in the outer half of boundary bins
            clipped_data = []
            for dim_i, x in enumerate(args):
                bcs = self.pdf_histogram.bin_centers(dim_i)
                clipped_data.append(np.clip(x, bcs.min(), bcs.max()))

            return self.pdf_interpolator(np.transpose(clipped_data))

        elif self.pdf_interpolation_method == 'piecewise':
            return self.pdf_histogram.lookup(*args)

        else:
            raise NotImplementedError("PDF Interpolation method %s not implemented" % self.pdf_interpolation_method)
