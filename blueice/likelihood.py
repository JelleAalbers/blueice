from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy import stats

from .model import Model
from .parallel import create_models_in_parallel
from .pdf_morphers import MORPHERS


class LogLikelihood(object):
    """Extended log likelihood function with several rate and/or shape parameters

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
        pdf_base_config.update(kwargs)
        if likelihood_config is None:
            likelihood_config = {}
        self.config = likelihood_config
        self.config.setdefault('morpher', 'GridInterpolator')

        self.pdf_base_config = pdf_base_config
        self.base_model = Model(self.pdf_base_config)   # Base model: no variations of any settings
        self.source_list = [s.name for s in self.base_model.sources]

        self.rate_parameters = OrderedDict()     # sourcename_rate -> logprior
        self.shape_parameters = OrderedDict()    # settingname -> (anchors, logprior, base_z).
        # where anchors is dict: representative number -> actual setting
        # From here on representative number will be called 'z-score'.
        # base_value is the default z-score that will be used.
        # We'll take care of sorting the keys in self.prepare()

        self.is_prepared = False
        self.is_data_set = False

        # If there are NO shape parameters:
        self.ps = None                # ps of the data

        # If there are shape parameters:
        self.anchor_models = OrderedDict()  # dictionary mapping z-score -> actual model
        self.mu_interpolator = None     # function mapping z scores -> rates for each source
        self.ps_interpolator = None     # function mapping z scores -> (source, event) p-values

    def prepare(self, ipp_client=None):
        """Prepares a likelihood function with shape parameters for use.
        This will  compute the models for each shape parameters anchor value combination.
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
                configs.append(config)

            # Create the new models
            models = create_models_in_parallel(configs, ipp_client,
                                               block=self.config.get('block_during_paralellization', False))

            # Add the new models to the anchor_models dict
            for zs, model in zip(zs_list, models):
                self.anchor_models[tuple(zs)] = model

            # Build the interpolator for the rates of each source.
            self.mus_interpolator = self.morpher.make_interpolator(f=lambda m: m.expected_events(),
                                                                   extra_dims=[len(self.source_list)],
                                                                   anchor_models=self.anchor_models)
        self._prepare_for_data()

        self.is_data_set = False
        self.is_prepared = True

    def set_data(self, d):
        """Prepare the dataset d for likelihood function evaluation
        :param d: Dataset, must be an indexable object that provides the measurement dimensions
        For example, if your models are on 's1' and 's2', d must be something for which d['s1'] and d['s2'] give
        the s1 and s2 values of your events as numpy arrays.
        """
        if not self.is_prepared and len(self.shape_parameters):
            raise NotPreparedException("You have shape parameters in your model: "
                                       "first do .prepare(), then set the data.")
        self._prepare_data(d)

        self.is_data_set = True

    def _prepare_for_data(self):
        pass

    def _prepare_data(self, d):
        """Called in set_data, specific to type of likelihood"""
        if len(self.shape_parameters):
            self.ps_interpolator = self.morpher.make_interpolator(f=lambda m: m.score_events(d),
                                                                  extra_dims=[len(self.source_list), len(d)],
                                                                  anchor_models=self.anchor_models)
        else:
            self.ps = self.base_model.score_events(d)

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
                raise InvalidShapeParameter("When specifying anchors only by setting values, "
                                            "base setting must have a numerical default.")
            anchors = {z: z for z in anchors}

        if not is_numeric and base_value is None:
            raise InvalidShapeParameter("For non-numeric settings, you must specify what number will represent "
                                        "the default value (the base model setting)")
        if is_numeric and base_value is not None:
            raise InvalidShapeParameter("For numeric settings, base_value is an unnecessary argument.")

        self.shape_parameters[setting_name] = (anchors, log_prior, base_value)

    def __call__(self, livetime_days=None, **kwargs):
        """Evaluate the likelihood function. Pass any values for parameters as keyword arguments.
        For rate uncertainties, pass sourcename_rate_multiplier.
        :param lifetime_days: lifetime in days to use, will affect rates of all sources.
        """
        if not self.is_data_set:
            raise NotPreparedException("First do .set_data(dataset), then start evaluating the likelihood function")
        result = 0

        if len(self.shape_parameters):
            # Get the shape parameter z values
            zs = []
            for setting_name, (_, log_prior, base_value) in self.shape_parameters.items():
                z = kwargs.get(setting_name)
                if z is None:
                    # Get the default value of the number representing this shape parameter
                    base_setting = self.pdf_base_config.get(setting_name)
                    is_numeric = isinstance(base_setting, (float, int))
                    if is_numeric:
                        assert base_value is None
                        z = base_setting
                    else:
                        z = base_value

                if not isinstance(z, (int, float)):
                    raise ValueError("Arguments to likelihood function must be numeric, not %s" % type(z))
                zs.append(z)

                # Test if the anchor value out of range, if so, return -inf (since is impossible)
                minbound, maxbound = self.get_bounds(setting_name)
                if not minbound <= z <= maxbound:
                    return -float('inf')

                if log_prior is not None:
                    result += log_prior(z)

            # The RegularGridInterpolators want numpy arrays: give it to them...
            zs = np.asarray(zs)

            mus = self.mus_interpolator(zs)
            ps = self.ps_interpolator(zs)

        else:
            mus = self.base_model.expected_events()
            ps = self.ps

        # Apply the rate multipliers
        for source_i, source_name in enumerate(self.source_list):
            rate_multiplier = kwargs.get(source_name + '_rate_multiplier', 1)
            mus[source_i] *= rate_multiplier

            log_prior = self.rate_parameters.get(source_name, None)
            if log_prior is not None:
                result += log_prior(rate_multiplier)

        # Apply the lifetime scaling
        if livetime_days is not None:
            mus *= livetime_days / self.pdf_base_config['livetime_days']

        # Handle unphysical rates. Depending on the config, either error or return -float('inf') as loglikelihood
        if not np.all((mus >= 0) & (mus < float('inf'))):
            if self.config.get('unphysical_behaviour') == 'error':
                raise ValueError("Unphysical rates: %s" % str(mus))
            else:
                return -float('inf')

        # Get the loglikelihood. At last!
        result += self._compute_likelihood(mus, ps)
        return result

    def _compute_likelihood(self, mus, pdf_values_at_events):
        return extended_loglikelihood(mus, pdf_values_at_events,
                                      outlier_likelihood=self.config.get('outlier_likelihood', 1e-12))

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters.keys()]
        if parameter_name in self.shape_parameters:
            anchor_settings = list(self.shape_parameters[parameter_name][0].keys())
            return min(anchor_settings), max(anchor_settings)
        elif parameter_name.endswith('_rate_multiplier'):
            return 0, float('inf')
        else:
            raise ValueError("Non-existing parameter %s" % parameter_name)

    # Convenience functions for uncertainties.
    # Adding more general priors is the user's responsibility
    # (either provide prior argument to add_x_parameter, or wrap the loglikelihood function)
    def add_rate_uncertainty(self, source_name, fractional_uncertainty):
        """Adds a rate parameter to the likelihood function with Gaussian prior"""
        self.add_rate_parameter(source_name, log_prior=stats.norm(1, fractional_uncertainty).logpdf)

    def add_shape_uncertainty(self, setting_name, fractional_uncertainty, anchor_zs=(-2, -1, 0, 1, 2)):
        """Adds a shape parameter to the likelihood function, with Gaussian prior around the default value.
        :param anchor_zs: list/tuple/array of z-scores to use as the anchor points
        """
        mu = self.pdf_base_config.get(setting_name)
        if not isinstance(mu, (float, int)):
            raise ValueError("%s does not have a numerical default setting" % setting_name)
        std = mu * fractional_uncertainty
        self.add_shape_parameter(setting_name,
                                 anchors=mu + np.array(anchor_zs) * std,
                                 log_prior=stats.norm(mu, mu * fractional_uncertainty).logpdf)


def extended_loglikelihood(mu, ps, outlier_likelihood=0.0):
    """Evaluate an extended likelihood function
    :param mu: array of n_sources: expected number of events
    :param ps: array of (n_sources, n_events): pdf value for each source and event
    :param outlier_li        if len(self.shape_parameters):
            self.pmf_interpolator = self.morpher.make_interpolator(f=lambda m: m.pmf_grids())

            self.ps_interpolator = self.morpher.make_interpolator(f=lambda m: m.score_events(d),
                                                                  extra_dims=[len(self.source_list), len(d)],
                                                                  anchor_models=self.anchor_models)
        else:
            self.ps = self.base_model.score_events(d)
kelihood: if an event has p=0, give it this likelihood (instead of 0, which makes the whole
    loglikelihood infinite)
    :return: ln(likelihood)
    """
    p_events = np.sum(mu[:, np.newaxis] * ps, axis=0)
    if outlier_likelihood != 0:
        # Replace all likelihoods which are not positive numbers (i.e. 0, negative, or nan) with outlier_likelihood
        p_events[True ^ (p_events > 0)] = outlier_likelihood
    return -mu.sum() + np.sum(np.log(p_events))


class BinnedLogLikelihood(LogLikelihood):
    def __init__(self, pdf_base_config, likelihood_config=None, **kwargs):
        pdf_base_config['pdf_interpolation_method'] = 'piecewise'
        LogLikelihood.__init__(self, pdf_base_config, likelihood_config, **kwargs)

    def _prepare_for_data(self):
        self.ps = self.base_model.pmf_grids()

        if len(self.shape_parameters):
            self.ps_interpolator = self.morpher.make_interpolator(f=lambda m: m.pmf_grids(),
                                                                  extra_dims=[len(self.source_list)] +
                                                                              list(self.ps.shape),
                                                                  anchor_models=self.anchor_models)

    def _prepare_data(self, d):
        """Called in set_data, specific to type of likelihood"""
        # Bin the data in the analysis space
        raise NotImplementedError







class NotPreparedException(Exception):
    pass


class InvalidShapeParameter(Exception):
    pass
