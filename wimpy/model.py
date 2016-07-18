import pickle
import collections

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from scipy import stats
from tqdm import tqdm
from copy import deepcopy

from multihist import Histdd
from .source import Source


class Model(object):
    """Model for XENON1T dataset simulation and analysis
    """
    config = None       # type: dict
    no_wimp_strength = -10
    max_wimp_strength = 2
    exposure_factor = 1  # Increase this to quickly change the exposure of the model

    def __init__(self, config, ipp_client=None, **kwargs):
        """
        :param config: Dictionary specifying detector parameters, source info, etc.
        :param ipp_client: ipyparallel client to use for parallelizing pdf computation (optional)
        :param kwargs: Overrides for the config (optional)
        :return:
        """
        c = self.config = deepcopy(config)
        self.config.update(kwargs)

        self.space = collections.OrderedDict(c['analysis_space'])
        self.dims = list(self.space.keys())
        self.bins = list(self.space.values())

        self.sources = []
        for source_spec in tqdm(c['sources'], desc='Initializing sources'):
            source = Source(self.config, source_spec)

            # Has the PDF in the analysis space already been provided in the spec?
            # Usually not, do so now.
            if source.pdf_histogram is None or c['force_pdf_recalculation']:
                self.compute_source_pdf(source, ipp_client=ipp_client)
            self.sources.append(source)

    def compute_source_pdf(self, source, ipp_client=None):
        """Computes the PDF for the source. Returns nothing but modifies source.
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


    def range_cut(self, d):
        """Return events from dataset d which are in the analysis space"""
        mask = np.ones(len(d), dtype=np.bool)
        for dimension, bin_edges in self.space.items():
            mask = mask & (d[dimension] >= bin_edges[0]) & (d[dimension] <= bin_edges[-1])
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
        # TODO: Handle outliers in a better way than just putting loglikelihood = -20...
        # In particular, you need to handle events which are outside any source pdf, they are now considered
        # equally likely to be signal or background!
        return np.array([np.clip(s.pdf(*self.to_space(d)),
                                 1e-20,
                                 float('inf')) for s in self.sources])

    def get_source_i(self, source_id):
        if isinstance(source_id, (int, float)):
            return int(source_id)
        else:
            for s_i, s in enumerate(self.sources):
                if source_id in s.name:
                    break
            return s_i

    def loglikelihood(self, d, wimp_strength, ps=None, rate_modifiers=None):
        """Gives the loglikelihood of the dataset d
        under the hypothesis that the source source_id's rate is multiplied by 10**strength.
        rate_modifiers: array of rate adjustments for each source.
                        Rate will be increaed by source.rate * source.rate_uncertainty * rate_modifier events/day,
                        Penalty term in likelihood is normal(0, 1).logpdf(rate_modifier)
        NB: Assumes d is already restricted to analysis space. If not, you will get into trouble!!
        """
        if rate_modifiers is None:
            rate_modifiers = np.zeros(len(self.sources))
        else:
            rate_modifiers = np.asarray(rate_modifiers)

        # Compute expected number of events for each source
        mu = np.array([s.events_per_day * self.config['livetime_days'] * s.fraction_in_range * self.exposure_factor
                       for s in self.sources])

        # Set the correct rate for the WIMP signal
        mu[-1] *= 10**wimp_strength

        # Apply the rate modifiers
        mu *= 1 + rate_modifiers * np.asarray([s.rate_uncertainty for s in self.sources])
        mu = np.clip(mu, 0, float('inf'))

        # Compute p(event) for each source. ps has shape (n_sources, n_events).
        if ps is None:
            ps = self.score_events(d)

        # Return the extended log likelihood (without tedious normalization constant that anyway drops out of
        # likelihood ratio computations).
        result = -mu.sum() + np.sum(np.log(np.sum(mu[:,np.newaxis] * ps, axis=0)))
        result += stats.norm.logpdf(rate_modifiers).sum()  # - len(m.sources) * stats.norm.logpdf(0)
        return result

    def bestfit(self, d, guess_strength=0, fit_uncertainties=False, fit_strength=True, ps=None):
        """Returns best-fit wimp strength for dataset d, loglikelihood value at that point.
        Throws exception if optimization is unsuccesfull
        if float_uncertainties = True, fits and returns the rate uncertainties for each source

        """
        if ps is None:
            ps = self.score_events(d)

        def add_zeros_if_not_uncertain(_rate_mods_temp):
            # Get rate modifiers from params: fold in zeros for sources without rate uncertainties
            _rate_mods_temp = _rate_mods_temp.tolist()
            return [_rate_mods_temp.pop(0) if s.rate_uncertainty else 0 for s in self.sources]

        n_uncertain_sources = len([s for s in self.sources if s.rate_uncertainty != 0])

        # Determine what kind of minimization we should do.
        if fit_uncertainties and fit_strength:
            guess = [guess_strength] + [0] * n_uncertain_sources

            def objective(params):
                return -self.loglikelihood(d, params[0], ps=ps,
                                           rate_modifiers=add_zeros_if_not_uncertain(params[1:]))


        elif fit_uncertainties:
            guess = [0] * n_uncertain_sources

            def objective(params):
                return -self.loglikelihood(d, guess_strength, ps=ps,
                                           rate_modifiers=add_zeros_if_not_uncertain(params))

        elif fit_strength:
            guess = [guess_strength]

            def objective(wimp_strength):
                return -self.loglikelihood(d, wimp_strength, ps=ps)

        else:
            # Fit nothing: just calculate!
            return guess_strength, self.loglikelihood(d, guess_strength, ps=ps)

        # For some reason the default BGFS minimization fails hard... Powell does much better
        optresult = minimize(objective, guess, method='Powell')
        if not optresult.success:
            raise RuntimeError("Optimization failure: ", optresult)
        return optresult.x, -optresult.fun

        # Sometimes a data is very background-like, and the best fit is a silly low value (-700 or something).
        # Maybe we should regard this as 'unphysical' and return self.no_wimp_strength instead?
        # if bestfit < model.no_wimp_strength:
        #     return self.no_wimp_strength, self.loglikelihood(d, self.no_wimp_strength)

    def interval(self, d, confidence_level=0.9, kind='limit', profile=False):
        """Return an interval of kind and confidence_level on dataset d
         - d is assumed to be restricted to the analysis range.
         - confidence level is the probability content of the interval (1 - the false positive rate)
         - kind can be 'central' (for a two-sided central CI) or 'limit' (for upper limits)
         - If profile=True, will compute a profile likelihood interval, i.e. will consider the rate uncertainty on each
           source.

        The interval is set using the likelihood ratio method, assuming the asymptotic distribution (Wilk's theorem)
        for the likelihood ratio test statistic.
        """
        ps = self.score_events(d)

        # Find the best-fit wimp strength
        optresult, max_likelihood = self.bestfit(d, ps=ps, fit_uncertainties=profile, fit_strength=True)
        best_strength = optresult[0] if profile else optresult.item()

        def f(wimp_strength, z):
            fit, conditional_max_ll = self.bestfit(d, wimp_strength, fit_strength=False, fit_uncertainties=profile)
            return 2*(max_likelihood - conditional_max_ll) - z**2

        if kind == 'limit':
            return brentq(f,
                          best_strength, self.max_wimp_strength,
                          args=(stats.norm(0, 1).ppf(confidence_level),))

        elif kind == 'central':
            if best_strength <= self.no_wimp_strength:
                a = self.no_wimp_strength
            else:
                a = brentq(f,
                           self.no_wimp_strength, best_strength,
                           args=(stats.norm(0, 1).ppf(1/2 - confidence_level/2),))
            b = brentq(f,
                       best_strength, self.max_wimp_strength,
                       args=(stats.norm(0, 1).ppf(1/2 + confidence_level/2),))
            return a, b

        else:
            raise ValueError("Invalid interval kind %s" % kind)

    # Utilities
    @staticmethod
    def load(filename):
        with open(filename, mode='rb') as infile:
            return pickle.load(infile)

    def save(self, filename=None):
        if filename is None:
            filename = 'model_' + str(np.random.random())
        with open(filename, mode='wb') as outfile:
            pickle.dump(self, outfile)
        return filename

    def copy(self):
        return deepcopy(self)
