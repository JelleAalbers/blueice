from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from tqdm import tqdm

from .model import Model


class LogLikelihood(object):
    """Extended log likelihood function with several rate and/or shape uncertainties

    Does NOT apply "priors" / penalty terms for subsidiary measurments

    z-scores necessary to ensure interpolation is fair

    """
    def __init__(self, pdf_base_config, config=None, **kwargs):
        if config is None:
            config = {}
        config.update(kwargs)
        self.config = config

        self.pdf_base_config = pdf_base_config
        self.rate_uncertainties = OrderedDict()
        self.shape_uncertainties = OrderedDict()
        self.source_list = []
        self.is_prepared = False
        self.is_data_set = False

        # These are only used in case there are shape uncertainties
        self.mu_interpolator = None     # RegularGridInterpolator mapping z scores -> rates for each source
        self.ps_interpolator = None     # RegularGridInterpolator mapping z scores -> (source, event) p-values
        self.anchor_z_arrays = None     # list of numpy arrays of z-parameters of each anchor model
        self.anchor_z_grid = None       # numpy array: z-parameter combinations grid
        self.anchor_models = dict()

        # These are only used in case there are NO shape uncertainties
        self.lone_model = None        # in case there are no shape uncertainties, the only model used is stored here.
        self.ps = None                # ps of the data

    def prepare(self, *args, **kwargs):
        """Prepares the likelihood function for use,
        computing the models for each shape uncertainty anchor value combination.
        Any arguments are passed through to Model initialization.
        """
        if len(self.shape_uncertainties):
            # Compute the anchor grid
            self.anchor_z_arrays = [np.array(list(sorted(anchors.keys())))
                                    for setting_name, anchors in self.shape_uncertainties.items()]
            self.anchor_z_grid = arrays_to_grid(self.anchor_z_arrays)

            # Compute the anchor models
            for _, zs in tqdm(self.anchor_grid_iterator(),
                              total=np.product(self.anchor_z_grid.shape[:-1]),
                              desc="Computing models for shape uncertainty anchor points"):

                # Construct the config for this model
                config = deepcopy(self.pdf_base_config)
                for i, (setting_name, anchors) in enumerate(self.shape_uncertainties.items()):
                    config[setting_name] = anchors[zs[i]]

                # Build the model
                model = Model(config, *args, **kwargs)
                self.anchor_models[tuple(zs)] = model

                # Get the source list (from any one model would do)
                self.source_list = [s.name for s in model.sources]

            # Build the interpolator for the rates of each source
            self.mus_interpolator = self.make_interpolator(f=lambda m: m.expected_events(),
                                                           extra_dims=[len(self.source_list)])

        else:
            self.lone_model = Model(self.pdf_base_config)
            self.source_list = [s.name for s in self.lone_model.sources]

        self.is_prepared = True

    def set_data(self, d):
        """Prepare the dataset d for likelihood function evaluation
        :param d: Dataset, must be an indexable object that provides the measurement dimensions
        For example, if your models are on 's1' and 's2', d must be something for which d['s1'] and d['s2'] give
        the s1 and s2 values of your events as numpy arrays.
        """
        if not self.is_prepared:
            raise RuntimeError("First do .prepare(), then set the data.")
        if len(self.shape_uncertainties):
            self.ps_interpolator = self.make_interpolator(f=lambda m: m.score_events(d),
                                                          extra_dims=[len(self.source_list), len(d)])
        else:
            self.ps = self.lone_model.score_events(d)

        self.is_data_set = True

    def add_rate_variation(self, source_name, spread):
        """Add a rate uncertainty parameter to the likelihood function
        :param source_name: Name of the source for which rate is uncertain
        :param spread: Fractional uncertainty on the rate
        """
        self.rate_uncertainties[source_name] = spread

    def add_shape_variation(self, setting_name, anchors, spread=0):
        """Add a shape uncertainty parameter to the likelihood function
        :param setting_name: Name of the setting to vary
        :param anchors: a dictionary mapping z scores -> values of the setting to vary
                        OR, if spread is specified, a list of z-scores.
        :param spread: The fractional uncertainty on the setting's value.
                       Use only when the setting is numerical, and you specified anchors as list of z-scores,
        """
        if spread != 0:
            # Convert anchors to a dictionary from the spread and z-points provided
            assert not isinstance(anchors, dict)
            default_setting = self.pdf_base_config.get(setting_name)
            if not isinstance(default_setting, (float, int)):
                raise ValueError("When specifying shape uncertainty as a spread, "
                                 "base setting must have a numerical default.")
            anchors = {z: default_setting * (1 + z * spread) for z in anchors}

        self.shape_uncertainties[setting_name] = anchors

    def __call__(self, **kwargs):
        if not self.is_data_set:
            raise RuntimeError("First do .set_data(dataset), then start evaluating the likelihood function")

        if len(self.shape_uncertainties):
            # Get the shape uncertainty z values
            zs = [kwargs.get(setting_name, 0) for setting_name, _ in self.shape_uncertainties.items()]

            # The RegularGridInterpolators want numpy arrays: give it to them...
            zs = np.asarray([zs])

            # Get mus (rate for each source) and ps (pdf value for each source for each event) at this point
            # The RegularGridInterpolators return numpy arrays with one extra dimension: remove it...
            mus = self.mus_interpolator(np.array(zs)[0])
            ps = self.ps_interpolator(zs)[0]

        else:
            mus = self.lone_model.expected_events()
            ps = self.ps

        # Apply the rate modifiers and uncertainties
        for source_i, source_name in enumerate(self.source_list):
            mus[source_i] *= self.config.get('source_rate_multipliers', {}).get(source_name, 1)
            if source_name + '_rate' in kwargs:
                mus[source_i] *= 1 + kwargs[source_name + '_rate'] * self.rate_uncertainties[source_name]

        # Handle unphysical rates. Depending on the config, either error or return -float('inf') as loglikelihood
        if not np.all((mus > 0) & (mus < float('inf'))):
            if self.config.get('unphysical_behaviour') == 'error':
                raise ValueError("Unphysical rates: %s" % str(mus))
            else:
                return -float('inf')

        # Get the loglikelihood. At last!
        return extended_loglikelihood(mus, ps, outlier_likelihood=self.config.get('outlier_likelihood', 1e-10))

    def anchor_grid_iterator(self):
        """Iterates over the anchor grid, yielding index, z-values"""
        it = np.nditer(np.zeros(list(self.anchor_z_grid.shape)[:-1]), flags=['multi_index'])
        while not it.finished:
            anchor_grid_index = list(it.multi_index)
            yield anchor_grid_index, tuple(self.anchor_z_grid[anchor_grid_index + [slice(None)]])
            it.iternext()

    def make_interpolator(self, f, extra_dims):
        """Return a RegularGridInterpolator which interpolates the extra_dims-valued function f(model)
        between the anchor points.
        :param f: Function which takes a model as argument, and produces an extra_dims shaped array.
        :param extra_dims: tuple of integers, shape of return value of f.
        """
        # Allocate an array which will hold the scores at each anchor model
        anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + extra_dims)

        # Iterate over the anchor grid points
        for anchor_grid_index, zs in self.anchor_grid_iterator():

            # Compute f at this point, and store it in anchor_scores
            anchor_scores[anchor_grid_index + [slice(None)] * len(extra_dims)] = f(self.anchor_models[tuple(zs)])

        return RegularGridInterpolator(self.anchor_z_arrays, anchor_scores)


def extended_loglikelihood(mu, ps, outlier_likelihood=0):
    """Evaluate an extended likelihood function
    :param mu: array of n_sources: expected number of events
    :param ps: array of (n_sources, n_events): pdf value for each source and event
    :param ignore_outliers: if an event has p=0, ignore it (instead of making the whole loglikelihood -inf
    :return: loglikelihood
    """
    p_events = np.sum(mu[:, np.newaxis] * ps, axis=0)
    if outlier_likelihood != 0:
        # Replace all likelihoods which are not positive numbers (i.e. 0, negative, or nan) with outlier_likelihood
        p_events[True ^ (p_events > 0)] = outlier_likelihood
    return -mu.sum() + np.sum(np.log(p_events))


def arrays_to_grid(arrs):
    """Convert a list of n 1-dim arrays to an n+1-dim. array, where last dimension denotes coordinate values at point.
    """
    return np.stack(np.meshgrid(*arrs), axis=-1)

