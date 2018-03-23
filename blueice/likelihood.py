"""Log likelihood constructors: the heart of blueice.


"""

import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import wraps

import numpy as np
from multihist import Histdd
from scipy import stats
from scipy.special import gammaln
from tqdm import tqdm

from .exceptions import NotPreparedException, InvalidParameterSpecification, InvalidParameter
from .model import Model
from .parallel import create_models_ipyparallel, compute_many
from .pdf_morphers import MORPHERS
from .utils import combine_dicts, inherit_docstring_from
from . import inference

__all__ = ['LogLikelihoodBase', 'BinnedLogLikelihood', 'UnbinnedLogLikelihood', 'LogLikelihoodSum']


##
# Decorators for methods which have to be run after prepare or set_data
##

def _needs_preparation(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.is_prepared:
            if not len(self.shape_parameters):
                # preparation is going to be trivial, just do it
                self.prepare()
            else:
                raise NotPreparedException("%s requires you to first prepare the likelihood function using prepare()" %
                                           f.__name__)
        return f(self, *args, **kwargs)
    return wrapper


def _needs_data(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.is_data_set:
            raise NotPreparedException("%s requires you to first set the data using set_data()" % (f.__name__))
        return f(self, *args, **kwargs)
    return wrapper


class LogLikelihoodBase(object):
    """Log likelihood function with several rate and/or shape parameters

    likelihood_config options:
        unphysical_behaviour
        outlier_likelihood
        parallelize_models: True (default) or False
        block_during_paralellization: True or False (default)
    """

    def __init__(self, pdf_base_config, likelihood_config=None, **kwargs):
        """
        :param pdf_base_config: dictionary with configuration passed to the Model
        :param likelihood_config: dictionary with options for LogLikelihood itself
        :param kwargs: Overrides for pdf_base_config, not likelihood config!
        :return:
        """
        self.pdf_base_config = combine_dicts(pdf_base_config, kwargs, deep_copy=True)

        if likelihood_config is None:
            likelihood_config = {}
        self.config = likelihood_config
        self.config.setdefault('morpher', 'GridInterpolator')

        # Base model: no variations of any settings
        self.base_model = Model(self.pdf_base_config)
        self.source_name_list = [s.name for s in self.base_model.sources]
        self.source_allowed_negative = [s.config.get("allow_negative",False)
                                        for s in self.base_model.sources]
        self.source_apply_efficiency = np.array([s.config.get("apply_efficiency", False)
                                                 for s in self.base_model.sources])

        self.rate_parameters = OrderedDict()     # sourcename_rate -> logprior
        self.shape_parameters = OrderedDict()    # settingname -> (anchors, logprior, base_z).
        # where anchors is dict: representative number -> actual setting
        # From here on representative number will be called 'z-score'.
        # base_value is the default z-score that will be used.
        # We'll take care of sorting the keys in self.prepare()

        self.is_prepared = False
        self.is_data_set = False
        self._has_non_numeric = False

        # If there are NO shape parameters:
        self.ps = None                # ps of the data

        # If there are shape parameters:
        self.anchor_models = OrderedDict()  # dictionary mapping model zs -> actual model
        # Interpolators created by morphers. These map zs to...
        self.mus_interpolator = None                          # rates for each source
        self.ps_interpolator = None                          # (source, event) p-values (unbinned), or pmf grid (binned)
        # number of events per bin observed in Monte Carlo / calibration data that gave rise to the model.
        self.n_model_events_interpolator = lambda x: None
        self.n_model_events = None

    def prepare(self, n_cores=1, ipp_client=None):
        """Prepares a likelihood function with shape parameters for use.
        This will compute the models for each shape parameter anchor value combination.
        """
        if len(self.shape_parameters):
            self.morpher = MORPHERS[self.config['morpher']](self.config.get('morpher_config', {}),
                                                            self.shape_parameters)
            zs_list = self.morpher.get_anchor_points(bounds=self.get_bounds())

            # Create the configs for each new model
            configs = []
            for zs in zs_list:
                config = deepcopy(self.pdf_base_config)
                for i, (setting_name, (anchors, _, _)) in enumerate(self.shape_parameters.items()):
                    # Translate from zs to settings using the anchors dict. Maybe not all settings are numerical.
                    config[setting_name] = anchors[zs[i]]
                if ipp_client is None and n_cores != 1:
                    # We have to compute in parallel: must have delayed computation on
                    config['delay_pdf_computation'] = True
                configs.append(config)

            # Create the new models
            if n_cores == 1:
                models = [Model(c) for c in tqdm(configs, desc="Computing/loading models on one core")]

            elif ipp_client is not None:
                models = create_models_ipyparallel(configs, ipp_client,
                                                   block=self.config.get('block_during_paralellization', False))

            else:
                models = [Model(c) for c in tqdm(configs, desc="Preparing model computation tasks")]

                hashes = set()
                for m in models:
                    for s in m.sources:
                        hashes.add(s.hash)

                compute_many(hashes, n_cores)

                # Reload models so computation takes effect
                models = [Model(c) for c in tqdm(configs, desc="Loading computed models")]

            # Add the new models to the anchor_models dict
            for zs, model in zip(zs_list, models):
                self.anchor_models[tuple(zs)] = model

            # Build the interpolator for the rates of each source.
            self.mus_interpolator = self.morpher.make_interpolator(f=lambda m: m.expected_events(),
                                                                   extra_dims=[len(self.source_name_list)],
                                                                   anchor_models=self.anchor_models)

        self.is_data_set = False
        self.is_prepared = True

    @_needs_preparation
    def set_data(self, d):
        """Prepare the dataset d for likelihood function evaluation
        :param d: Dataset, must be an indexable object that provides the measurement dimensions
        For example, if your models are on 's1' and 's2', d must be something for which d['s1'] and d['s2'] give
        the s1 and s2 values of your events as numpy arrays.
        """
        self._data = d
        self.is_data_set = True

    def add_rate_parameter(self, source_name, log_prior=None):
        """Add a rate parameter names source_name + "_rate_multiplier" to the likelihood function..
        The values of this parameter will MULTIPLY the expected rate of events for the source.
        The rates of sources can also vary due to shape parameters.
        :param source_name: Name of the source for which you want to vary the rate
        :param log_prior: prior logpdf function on rate multiplier (not on rate itself!)
        """
        self.rate_parameters[source_name] = log_prior

    def add_shape_parameter(self, setting_name, anchors, log_prior=None, base_value=None):
        """Add a shape parameter to the likelihood function
        :param setting_name: Name of the setting to vary
        :param anchors: a list/tuple/array of setting values (if they are numeric)
                        OR a dictionary with some numerical value -> setting values (for non-numeric settings).
        :param base_value: for non-numeric settings, the number which represents the base model value of the setting.
        For example, if you have LCE maps with varying reflectivities, use
            add_shape_variation('s1_relative_ly_map', {0.98: 'lce_98%.pklz', 0.99: 'lce_99%.pklz, ...})
        then the argument s1_relative_ly_map of the likelihood function takes values between 0.98 and 0.99.
        """
        is_numeric = isinstance(self.pdf_base_config.get(setting_name), (float, int))
        if not isinstance(anchors, dict):
            # Convert anchors list to a dictionary
            if not is_numeric:
                raise InvalidParameterSpecification("When specifying anchors only by setting values, "
                                                    "base setting must have a numerical default.")
            anchors = {z: z for z in anchors}

        if not is_numeric:
            self._has_non_numeric = True
        if not is_numeric and base_value is None:
            raise InvalidParameterSpecification("For non-numeric settings, you must specify what number will represent "
                                                "the default value (the base model setting)")
        if is_numeric and base_value is not None:
            raise InvalidParameterSpecification("For numeric settings, base_value is an unnecessary argument.")

        self.shape_parameters[setting_name] = (anchors, log_prior, base_value)

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters.keys()]
        if parameter_name in self.shape_parameters:
            anchor_settings = list(self.shape_parameters[parameter_name][0].keys())
            return min(anchor_settings), max(anchor_settings)
        elif parameter_name.endswith('_rate_multiplier'):
            for source_name, allow_negative in zip(self.source_name_list,self.source_allowed_negative):
                if parameter_name.startswith(source_name) and allow_negative==True:
                    return float('-inf'), float('inf')
            return 0, float('inf')
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)

    @_needs_data
    def __call__(self, livetime_days=None, compute_pdf=False, full_output=False, **kwargs):
        """Evaluate the likelihood function. Pass any values for parameters as keyword arguments.
        For values not passed, their base values will be assumed.
        For rate uncertainties, pass sourcename_rate_multiplier.
        :param lifetime_days: lifetime in days to use, will affect rates of all sources.
        :param full_output: instead of returning just the loglikelihood, return also the adjusted mus and ps as well.
        :param compute_pdf: compute new PDFs instead of interpolating the PDF at the requested parameters.
        """
        result = 0
        rate_multipliers, shape_parameter_settings = self._kwargs_to_settings(**kwargs)

        if len(self.shape_parameters):
            if compute_pdf:
                if self._has_non_numeric:
                    raise NotImplementedError("compute_pdf only works for numerical values")

                mus, ps, n_model_events = self._compute_single_pdf(**kwargs)

            else:
                # We can use the interpolators. They require the settings to come in order:
                zs = []
                for setting_name, (_, log_prior, _) in self.shape_parameters.items():
                    z = shape_parameter_settings[setting_name]
                    zs.append(z)

                    # Test if the z value is out of range; if so, return -inf (since we can't extrapolate)
                    minbound, maxbound = self.get_bounds(setting_name)
                    if not minbound <= z <= maxbound:
                        return -float('inf')

                    if log_prior is not None:
                        result += log_prior(z)

                # The RegularGridInterpolators want numpy arrays: give it to them...
                zs = np.asarray(zs)

                mus = self.mus_interpolator(zs)
                ps = self.ps_interpolator(zs)
                n_model_events = self.n_model_events_interpolator(zs)

        else:
            # No shape parameters
            mus = self.base_model.expected_events()
            ps = self.ps
            n_model_events = self.n_model_events

        # Apply the rate multipliers
        for source_i, source_name in enumerate(self.source_name_list):
            mult = rate_multipliers[source_i]
            mus[source_i] *= mult
            log_prior = self.rate_parameters.get(source_name, None)
            if log_prior is not None:
                result += log_prior(mult)

        # Apply the lifetime scaling
        if livetime_days is not None:
            mus *= livetime_days / self.pdf_base_config['livetime_days']
        
        # Apply efficiency to those sources that use it:
        if 'efficiency' in self.shape_parameters:
            mus[self.source_apply_efficiency] *= shape_parameter_settings['efficiency']

        # Perform fits to background calibration data if needed:
        # Currently only performed (analytically) for Binned likelihood via the Beeston-Barlow method
        mus, ps = self.adjust_expectations(mus, ps, n_model_events)

        # Check for negative rates. Depending on the config, either error or return -float('inf') as loglikelihood
        # If any source is allowed to be negative, check the sources one by one
        if not any(self.source_allowed_negative):
            if not np.all((mus >= 0) & (mus < float('inf'))):
                if self.config.get('unphysical_behaviour') == 'error':
                    raise ValueError("Unphysical rates: %s" % str(mus))
                else:
                    return -float('inf')
        else:
            if (not any(mus < float('inf'))) or (np.sum(mus) < 0):
                if self.config.get('unphysical_behaviour') == 'error':
                    raise ValueError("Unphysical rates: %s" % str(mus))
                else:
                    return -float('inf')

            for mu,allowed_negative in zip(mus,self.source_allowed_negative):
                if not (0 <= mu) and (not allowed_negative):
                    if self.config.get('unphysical_behaviour') == 'error':
                        raise ValueError("Unphysical rates: %s" % str(mus))
                    else:
                        return -float('inf')

        # Get the loglikelihood. At last!
        result += self._compute_likelihood(mus, ps)

        if full_output:
            return result, mus, ps
        else:
            return result

    def adjust_expectations(self, mus, ps, n_model_events):
        """Adjust uncertain (mus, pmfs) based on the observed data.

        If the density is derived from a finite-statistics sample (n_model_events array of events per bin),
        we can take into account this uncertainty by modifying the likelihood function.

        For a binned likelihood, this means adding the expected number of events for each bin for each source as
        nuisance parameters constrained by Poisson terms around the number of events observed in the model.
        While these nuisance parameters could be optimized numerically along with the main parameters,
        for a given value of the main parameters these per-bin nuisance parameters can often be estimated analytically,
        as shown by Beeston & Barlow (1993).
        """
        return mus, ps

    def _kwargs_to_settings(self, **kwargs):
        """Return shape parameters, rate_multipliers from kwargs.
          shape_parmeters is a dict mapping setting name -> value | representative number
          rate_multipliers is a list of rate multipliers for each source in self.source_name_list
        """
        # Validate the kwargs: must be either shape parameters, or <known_source>_rate_multiplier
        for k in kwargs.keys():
            if k in self.shape_parameters:
                continue
            if k.endswith('_rate_multiplier'):
                s_name = k[:-16]
                if s_name in self.source_name_list:
                    continue
            raise InvalidParameter("%s is not a known shape or rate parameter!" % k)

        shape_parameter_settings = dict()
        for setting_name, (_, _, base_value) in self.shape_parameters.items():
            z = kwargs.get(setting_name)

            if z is None:
                # Parameter was not given: get the default value of (the number representing) this shape parameter
                base_setting = self.pdf_base_config.get(setting_name)
                is_numeric = isinstance(base_setting, (float, int))
                if is_numeric:
                    assert base_value is None
                    z = base_setting
                else:
                    z = base_value

            if not isinstance(z, (int, float)):
                raise ValueError("Arguments to likelihood function must be numeric, not %s" % type(z))

            shape_parameter_settings[setting_name] = z

        rate_multipliers = []
        for source_i, source_name in enumerate(self.source_name_list):
            rate_multipliers.append(kwargs.get(source_name + '_rate_multiplier', 1))

        return rate_multipliers, shape_parameter_settings

    ##
    # Convenience functions for uncertainties.
    # Adding more general priors is the user's responsibility
    # (either provide prior argument to add_x_parameter, or wrap the loglikelihood function)
    ##
    def add_rate_uncertainty(self, source_name, fractional_uncertainty):
        """Adds a rate parameter to the likelihood function with Gaussian prior"""
        self.add_rate_parameter(source_name, log_prior=stats.norm(1, fractional_uncertainty).logpdf)

    def add_shape_uncertainty(self, setting_name, fractional_uncertainty, anchor_zs=(-2, -1, 0, 1, 2), base_value=None):
        """Adds a shape parameter to the likelihood function, with Gaussian prior around the default value.
        :param fractional uncertainty: Relative uncertainty on the default value.
        Other parameters as in add_shape_parameter.
        """
        # Call add_shape_parameter without a prior first, then inject the prior later.
        # It's a bit of a hack, but there is some validation / default-setting code for base_value we don't want to
        # replicate.
        self.add_shape_parameter(setting_name, anchor_zs, base_value=base_value)
        anchors, log_prior, base_value = self.shape_parameters[setting_name]
        self.shape_parameters[setting_name] = (anchors,
                                               stats.norm(base_value, base_value * fractional_uncertainty).logpdf,
                                               base_value)

    def _compute_single_model(self, **kwargs):
        """Return a model formed using the base config, using kwargs as overrides"""
        _, shape_parameter_settings = self._kwargs_to_settings(**kwargs)
        config = combine_dicts(self.pdf_base_config, shape_parameter_settings, deep_copy=True)

        config['never_save_to_cache'] = True
        return Model(config, **shape_parameter_settings)

    ##
    # Methods to override
    ##
    def _compute_single_pdf(self, **kwargs):
        """Return likelihood arguments for a single newly computed model,
        formed using the base config, using kwargs as overrides.
        Returns mus, ps, n_model_events
        """
        raise NotImplementedError


class UnbinnedLogLikelihood(LogLikelihoodBase):

    @inherit_docstring_from(LogLikelihoodBase)
    def set_data(self, d):
        LogLikelihoodBase.set_data(self, d)
        if len(self.shape_parameters):
            self.ps_interpolator = self.morpher.make_interpolator(f=lambda m: m.score_events(d),
                                                                  extra_dims=[len(self.source_name_list), len(d)],
                                                                  anchor_models=self.anchor_models)
        else:
            self.ps = self.base_model.score_events(d)

    @inherit_docstring_from(LogLikelihoodBase)
    def _compute_single_pdf(self, **kwargs):
        model = self._compute_single_model(**kwargs)
        mus = model.expected_events()
        ps = model.score_events(self._data)
        return mus, ps, None

    def _compute_likelihood(self, mus, pdf_values_at_events):
        return extended_loglikelihood(mus, pdf_values_at_events,
                                      outlier_likelihood=self.config.get('outlier_likelihood', 1e-12))


class LogLikelihood(UnbinnedLogLikelihood):
    """Deprecated alias for UnbinnedLogLikelihood"""

    def __init__(self, *args, **kwargs):
        warnings.warn("Unbinned log likelihood has been renamed to UnbinnedLogLikelihood", PendingDeprecationWarning)
        UnbinnedLogLikelihood.__init__(self, *args, **kwargs)


class BinnedLogLikelihood(LogLikelihoodBase):

    def __init__(self, pdf_base_config, likelihood_config=None, **kwargs):
        LogLikelihoodBase.__init__(self, pdf_base_config, likelihood_config, **kwargs)
        pdf_base_config['pdf_interpolation_method'] = 'piecewise'

        self.model_statistical_uncertainty_handling = self.config.get('model_statistical_uncertainty_handling')

    @inherit_docstring_from(LogLikelihoodBase)
    def prepare(self, *args):
        LogLikelihood.prepare(self, *args)
        self.ps, self.n_model_events = self.base_model.pmf_grids()

        if len(self.shape_parameters):
            self.ps_interpolator = self.morpher.make_interpolator(f=lambda m: m.pmf_grids()[0],
                                                                  extra_dims=list(self.ps.shape),
                                                                  anchor_models=self.anchor_models)

            if self.model_statistical_uncertainty_handling is not None:
                self.n_model_events_interpolator = self.morpher.make_interpolator(f=lambda m: m.pmf_grids()[1],
                                                                                  extra_dims=list(self.ps.shape),
                                                                                  anchor_models=self.anchor_models)

    @inherit_docstring_from(LogLikelihoodBase)
    def set_data(self, d):
        LogLikelihoodBase.set_data(self, d)
        # Bin the data in the analysis space
        dimnames, bins = zip(*self.base_model.config['analysis_space'])
        self.data_events_per_bin = Histdd(bins=bins, axis_names=dimnames)
        self.data_events_per_bin.add(*self.base_model.to_analysis_dimensions(d))

    @inherit_docstring_from(LogLikelihoodBase)
    def _compute_single_pdf(self, **kwargs):
        model = self._compute_single_model(**kwargs)
        mus = model.expected_events()
        ps, n_model_events = model.pmf_grids()
        return mus, ps, n_model_events

    @_needs_data
    @inherit_docstring_from(LogLikelihoodBase)
    def adjust_expectations(self, mus, pmfs, n_model_events):
        if self.model_statistical_uncertainty_handling == 'bb_single':

            source_i = self.config.get('bb_single_source')
            if source_i is None:
                raise ValueError("You need to specify bb_single_source to use bb_single_source expectation adjustment")
            source_i = self.base_model.get_source_i(source_i)

            assert pmfs.shape == n_model_events.shape

            # Get the number of events expected for the sources we will NOT adjust
            counts_per_bin = pmfs.copy()
            for i, (mu, _x) in enumerate(zip(mus, counts_per_bin)):
                if i != source_i:
                    _x *= mu
                else:
                    _x *= 0.
            u_bins = np.sum(counts_per_bin, axis=0)

            p_calibration = mus[source_i] / n_model_events[source_i].sum()

            a_bins = n_model_events[source_i]

            A_bins_1, A_bins_2 = beeston_barlow_roots(a_bins, p_calibration, u_bins, self.data_events_per_bin.histogram)
            assert np.all(A_bins_1 <= 0)  # it seems(?) the 1st root is always negative

            # For U=0, the solution above is singular; we need to use a special case instead
            A_bins_special = (self.data_events_per_bin.histogram + a_bins) / (1. + p_calibration)
            A_bins = np.choose(u_bins == 0, [A_bins_2, A_bins_special])

            assert np.all(0 <= A_bins)
            pmfs[source_i] = A_bins / A_bins.sum()
            mus[source_i] = A_bins.sum() * p_calibration

        return mus, pmfs

    def _compute_likelihood(self, mus, pmfs):
        """Return binned Poisson log likelihood
        :param mus: numpy array with expected rates for each source
        :param pmfs: array (sources, *analysis_space) of PMFs for each source in each bin
        """
        expected_counts = pmfs.copy()
        for mu, _p_bin_source in zip(mus, expected_counts):
            _p_bin_source *= mu         # Works because of numpy view magic...
        expected_total = np.sum(expected_counts, axis=0)

        observed_counts = self.data_events_per_bin.histogram

        ret = observed_counts * np.log(expected_total) - expected_total - gammaln(observed_counts + 1.).real
        return np.sum(ret)


def extended_loglikelihood(mu, ps, outlier_likelihood=0.0):
    """Evaluate an extended unbinned likelihood function
    :param mu: array of n_sources: expected number of events
    :param ps: array of (n_sources, n_events): pdf value for each source and event
    :param outlier_likelihood: if an event has p=0, give it this likelihood (instead of 0, which makes the whole
    loglikelihood infinite)
    :return: ln(likelihood)
    """
    p_events = np.nansum(mu[:, np.newaxis] * ps, axis=0)
    if outlier_likelihood != 0:
        # Replace all likelihoods which are not positive numbers (i.e. 0, negative, or nan) with outlier_likelihood
        p_events[True ^ (p_events > 0)] = outlier_likelihood
    return -mu.sum() + np.sum(np.log(p_events))


def beeston_barlow_root1(a, p, U, d):
    """Solution to the Beeston-Barlow equations for a single finite-statics source and several infinite-statistics
    sources. This is the WRONG root, as far as we can tell -- DO NOT USE IT!!
    We retained it only to keep checking that it is the wrong root. It will be removed soon, when we are more confident.
    """
    return ((-U*p - U + a*p + d*p -
             np.sqrt(U**2*p**2 + 2*U**2*p + U**2 + 2*U*a*p**2 + 2*U*a*p -
                     2*U*d*p**2 - 2*U*d*p + a**2*p**2 + 2*a*d*p**2 + d**2*p**2))/(2*p*(p + 1)))


def beeston_barlow_root2(a, p, U, d):
    """Solution to the Beeston-Barlow equations for a single finite-statics source and several infinite-statistics
    sources. This is the 'right' root, as far as we can tell anyway."""
    return ((-U*p - U + a*p + d*p +
             np.sqrt(U**2*p**2 + 2*U**2*p + U**2 + 2*U*a*p**2 + 2*U*a*p -
                     2*U*d*p**2 - 2*U*d*p + a**2*p**2 + 2*a*d*p**2 + d**2*p**2))/(2*p*(p + 1)))


def beeston_barlow_roots(a, p, U, d):
    return beeston_barlow_root1(a, p, U, d), beeston_barlow_root2(a, p, U, d)


class LogLikelihoodSum(object):
    """
        Class that takes a list of likelihoods to be minimized together, and 
        provides an interface to the inference methods and evaluation similar to likelihoods. 
        Note that the pfd_base_config is a bit of a fudge; only storing guesses from the last likelihood. 
        As different guesses for different likelihoods should be a cause for concern, the safest method is to pass
        manual guesses to the minimization. 
    """
    def __init__(self, likelihood_list):
        self.likelihood_list = []
        self.rate_parameters = dict()
        self.shape_parameters = dict()
        self.source_list = [] # DOES NOT EXIST IN LF!
        #in order to pass to confidence interval
        self.pdf_base_config  ={}#might also have to be fudged

        self.likelihood_parameters=[]

        for ll in likelihood_list:
            self.likelihood_list.append(ll)
            self.rate_parameters.update(ll.rate_parameters)
            self.shape_parameters.update(ll.shape_parameters)
            parameter_names = []
            
            for rate_parameter_name in ll.rate_parameters.keys():
                parameter_names.append(rate_parameter_name + '_rate_multiplier')
                base_value = ll.pdf_base_config.get(rate_parameter_name)
                if base_value is not None:
                    self.pdf_base_config[rate_parameter_name] = base_value
            for shape_parameter_name in ll.shape_parameters.keys():
                parameter_names.append(shape_parameter_name)
                base_value = ll.pdf_base_config.get(shape_parameter_name)
                if base_value is not None:
                    self.pdf_base_config[shape_parameter_name] = base_value
            self.likelihood_parameters.append(parameter_names)
    
    def __call__(self, compute_pdf=False, livetime_days=None, **kwargs):
        ret = 0.
        for i, (ll, parameter_names) in enumerate(zip(self.likelihood_list,
                                                      self.likelihood_parameters)):
            pass_kwargs = {k: v for k, v in kwargs.items() if k in parameter_names}
            livetime = livetime_days
            if isinstance(livetime_days, list):
                livetime = livetime_days[i]
 
            ret += ll(compute_pdf=compute_pdf, livetime_days=livetime, **pass_kwargs)
        return ret

    def split_results(self, result_dict):
        ret = []
        for i,parameter_names in enumerate(self.likelihood_parameters):
            ret.append({k: v for k, v in result_dict.items() if k in parameter_names})
        return ret

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters]
        if parameter_name in self.shape_parameters.keys():
            bounds = []
            for ll in self.likelihood_list:
                if parameter_name in ll.shape_parameters.keys():
                    bounds.append(ll.get_bounds(parameter_name))
            bounds = np.array(bounds)
            ret= np.max(bounds[:,0]), np.min(bounds[:,1])
            if ret[1] <= ret[0]:
                raise InvalidParameterSpecification("lower bound %s higher than upper bound!"%parameter_name)
            return ret
            
        elif parameter_name.endswith('_rate_multiplier'):
            return 0, float('inf')
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)

    # def make_objective(self, guess=None, minus=True, rates_in_log_space=False, **kwargs):
    #     return make_objective(self, guess,minus,rates_in_log_space,**kwargs)


class LogAncillaryLikelihood(object):
    """
        Function to add ancillary (constraint) analytical likelihoods, 
        passed args to initialization: 
        func - python function taking an _OrderedDict_ of (named) input values, plus func_kwargs extra arguments.
        parameter_list - list of names of parameters for which a dict is pulled from the config.  
        func_kwargs - other parameters to pass to function
        config - pdf config containing default values for parameters

        returns: 
            func({parameters:config[parameter]}, **func_kwargs)
    """
    def __init__(self, func, parameter_list, config=None, func_kwargs=None):
        if config is None:
            config = dict()
        if func_kwargs is None:
            func_kwargs = dict()

        self.rate_parameters = dict()
        self.shape_parameters = dict()
        self.source_list = []    # DOES NOT EXIST IN LF!
        # in order to pass to confidence interval
        self.pdf_base_config = config    # might also have to be fudged

        self.func = func
        self.func_kwargs = func_kwargs
        for parameter_name in parameter_list:
            self.shape_parameters.update(OrderedDict([(parameter_name,(None,None,None))]))

    def get_bounds(self, parameter_name=None):
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters]
        if parameter_name in self.shape_parameters.keys():
            return -np.inf, np.inf    # other likelihoods can be more constrictive.
        else:
            raise InvalidParameter("Non-existing parameter %s" % parameter_name)

    def __call__(self, **kwargs):
        pass_kwargs = OrderedDict()   # Use an ordered dict here, so function can rely on order of arguments
        for parameter_name in self.shape_parameters:
            pass_kwargs[parameter_name] = self.pdf_base_config[parameter_name]
        pass_kwargs.update(kwargs)

        #print("pass_kwargs",pass_kwargs)
        #print("func_kwargs",self.func_kwargs)
        return self.func(pass_kwargs, **self.func_kwargs)



# Add the inference methods from .inference
for methodname in inference.__all__:
    for q in (LogLikelihoodBase, LogLikelihoodSum, LogAncillaryLikelihood):
        setattr(q, methodname, getattr(inference, methodname))
