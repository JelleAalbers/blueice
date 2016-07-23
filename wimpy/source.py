import os
import numpy as np
from multihist import Histdd
from tqdm import tqdm

from .utils import load_pickle

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

    def __init__(self, config, spec, *args, **kwargs):
        """
        config: dict with general parameters (same as Model.config)
        spec: dict with source-specific stuff, e.g. name, label, etc. Used to set attributes defined above.
        args and kwargs are ignored. Accepted so children can pass their args/kwargs on to parent without fear.
        """
        self.config = config
        for k, v in spec.items():       # Store source specs as attributes
            setattr(self, k, v)

    def pdf(self, *args):
        raise NotImplementedError

    def simulate(self, n_events):
        raise NotImplementedError


class MonteCarloSource(Source):
    """A source which computes its PDF from MC simulation
    """
    n_events_for_pdf = 1e6
    pdf_histogram = None            # Histdd
    pdf_errors = None               # Histdd
    from_cache = False              # If True, the pdf was loaded from cache instead of computed on init

    def __init__(self, *args, **kwargs):
        """Prepares the PDF of this source for use.
        :param ipp_client: ipyparallel client to parallelize computation (optional)
        """
        ipp_client = kwargs.get('ipp_client')
        super().__init__(*args, **kwargs)

        # hash = self.config_hash    # TODO: add source info in hash

        # cache_dir = self.config.get('pdf_cache_dir', 'pdf_cache')
        # if not os.path.exists(cache_dir):
        #     os.makedirs(cache_dir)
        # cache_filename = cache_dir + str(hash)
        #
        # # Have we computed the pdfs etc. for this configuration before? If so, load the information.
        # if not self.config['force_pdf_recalculation'] and os.path.exists(cache_filename):
        #     self.from_cache = True
        #     for k, v in load_pickle(cache_filename):
        #         setattr(self, k, v)
        #     return
        self.compute_pdf(ipp_client)

        # # Should we save the source PDFs, rate, etc. for later use?
        # if not self.from_cache and self.config.get('save_pdfs', True):
        #     save_pickle({k: getattr(s, k) for k in ('pdf_histogram', 'pdf_errors', 'fraction_in_range',
        #                                             'events_per_day')},
        #                 cache_filename)

    def compute_pdf(self, ipp_client=None):
        # Simulate batches of events at a time (to avoid memory errors, show a progressbar, and split up among machines)
        # Number of events to simulate will be rounded up to the nearest batch size
        batch_size = self.config['pdf_sampling_batch_size']
        analysis_space = self.config['analysis_space']
        bins = list([x[1] for x in analysis_space])   # TODO: Repetition with Model...
        dims = list([x[0] for x in analysis_space])   # TODO: Repetition with Model...
        n_batches = int((self.n_events_for_pdf * self.config['pdf_sampling_multiplier']) // batch_size + 1)
        n_events = n_batches * batch_size
        mh = Histdd(bins=bins)

        def simulate_batch(args):
            """Run one simulation batch and histogram it immediately (so we don't pass gobs of data around)"""
            source, bins, dims, batch_size = args
            d = source.simulate(batch_size)
            d = [d[dims[i]] for i in range(len(dims))]          # TODO: repetition with Model...
            return Histdd(*d, bins=bins).histogram

        map_args = (simulate_batch, [(self, bins, dims, batch_size) for _ in range(n_batches)])

        if ipp_client is not None:
            # If this gives you a strange unintelligible error, turn on block_during_simulation
            map_result = ipp_client.load_balanced_view().map(*map_args,
                                                             ordered=False,
                                                             block=self.config.get('block_during_simulation', False))
        else:
            map_result = map(*map_args)

        if self.config.get('show_pdf_sampling_progress', True):
            map_result = tqdm(map_result, total=n_batches, desc='Sampling PDF of %s' % self.name)

        for r in map_result:
            mh.histogram += r

        self.fraction_in_range = mh.n / n_events

        # Convert the histogram to a PDF
        # This means we have to divide by
        #  - the number of events histogrammed
        #  - the bin sizes (particularly relevant for non-uniform bins!)
        self.pdf_histogram = mh.similar_blank_hist()
        self.pdf_histogram.histogram = mh.histogram.astype(np.float) / mh.n
        if len(bins) == 1:
            self.pdf_histogram.histogram /= np.diff(bins[0])
        else:
            self.pdf_histogram.histogram /= np.outer(*[np.diff(bins[i]) for i in range(len(bins))])

        # Estimate the MC statistical error. Not used for anything, but good to inspect.
        self.pdf_errors = self.pdf_histogram / np.sqrt(np.clip(mh.histogram, 1, float('inf')))
        self.pdf_errors[self.pdf_errors == 0] = float('nan')

    def simulate(self, n_events):
        raise NotImplementedError

    def pdf(self, *args):
        return self.pdf_histogram.lookup(*args)
