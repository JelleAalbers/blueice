from collections import OrderedDict
from copy import deepcopy

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy import stats
from tqdm import tqdm

from .model import Model, create_models_in_parallel


class LogLikelihood(object):
    """Extended log likelihood function with several rate and/or shape parameters

    likelihood_config options:
        unphysical_behaviour
        outlier_likelihood
        parallelize_models: True (default) or False
        block_during_paralellization: True or False (default)
    """
    def __init__(self, pdf_base_config, likelihood_config=None, ipp_client=None, **kwargs):
        """
        :param pdf_base_config: dictionary with configuration passed to the Model
        :param likelihood_config: dictionary with options for LogLikelihood itself
        :param ipp_client: ipyparallel client. Use if you want to parallelize computation of the base model.
        :param kwargs: Overrides for pdf_base_config
        :return:
        """
        pdf_base_config.update(kwargs)
        if likelihood_config is None:
            likelihood_config = {}
        self.config = likelihood_config
        self.itp_mode = self.config.get('interpolation_mode', 'regular')
        assert self.itp_mode in ('grid', 'radial')

        self.pdf_base_config = pdf_base_config
        self.base_model = None          # Base model: no variations of any settings
        self.rate_parameters = OrderedDict()     # sourcename_rate -> logprior
        self.shape_parameters = OrderedDict()    # settingname -> (anchors, logprior).
                                                 # where anchors is dict: representative number -> actual setting
                                                 # From here on representative number will be called 'z-score'
                                                 # We'll take care of sorting the keys in self.prepare()
        self.source_list = []
        self.is_prepared = False
        self.is_data_set = False

        # In case there are NO shape parameters:
        self.ps = None                # ps of the data

        # These are only used in case there are shape parameters
        self.mu_interpolator = None     # RegularGridInterpolator mapping z scores -> rates for each source
        self.ps_interpolator = None     # RegularGridInterpolator mapping z scores -> (source, event) p-values
        self.anchor_models = dict()     # dictionary mapping z-score -> actual model
        self.anchor_z_arrays = None     # list of numpy arrays of z-parameters of each anchor model

        # For the 'grid' interpolation mode:
        self.anchor_z_grid = None       # numpy array: z-parameter combinations grid

        # Compute the base model.
        self.base_model = Model(self.pdf_base_config, ipp_client=ipp_client)
        self.source_list = [s.name for s in self.base_model.sources]

    def prepare(self, n_sample_points=100, ipp_client=None):
        """Prepares a likelihood function with shape parameters for use.
        This will  compute the models for each shape parameters anchor value combination.
        """
        if not len(self.shape_parameters):
            return

        if self.itp_mode == 'grid':
            # Compute a regular grid of anchor models at the specified anchor points
            self.anchor_z_arrays = [np.array(list(sorted(anchors.keys())))
                                    for setting_name, (anchors, _) in self.shape_parameters.items()]
            self.anchor_z_grid = arrays_to_grid(self.anchor_z_arrays)

        else:
            # Sample a Latin hypercube of models
            self.anchor_z_arrays = latin(n_sample_points,
                                         len(self.shape_parameters),
                                         box=self.get_bounds())

        # Create the configs for each model
        configs = []
        for zs in self.anchor_z_arrays:
            # Construct the config for this model
            config = deepcopy(self.pdf_base_config)
            config['show_pdf_sampling_progress'] = False
            for i, (setting_name, (anchors, _)) in enumerate(self.shape_parameters.items()):
                config[setting_name] = anchors[zs[i]]
            configs.append(config)

        models = create_models_in_parallel(configs, ipp_client,
                                           block=self.config.get('block_during_paralellization', False))

        for zs, model in tqdm(zip(self.anchor_z_arrays, models),
                             total=len(configs),
                             desc="Computing models for shape parameter anchor points"):
            self.anchor_models[zs] = model

        # Build the interpolator for the rates of each source
        self.mus_interpolator = self.make_interpolator(f=lambda m: m.expected_events(),
                                                       extra_dims=[len(self.source_list)])

        self.is_prepared = True

    def set_data(self, d):
        """Prepare the dataset d for likelihood function evaluation
        :param d: Dataset, must be an indexable object that provides the measurement dimensions
        For example, if your models are on 's1' and 's2', d must be something for which d['s1'] and d['s2'] give
        the s1 and s2 values of your events as numpy arrays.
        """
        if not self.is_prepared and len(self.shape_parameters):
            raise RuntimeError("You have shape parameters in your model: first do .prepare(), then set the data.")
        if len(self.shape_parameters):
            self.ps_interpolator = self.make_interpolator(f=lambda m: m.score_events(d),
                                                          extra_dims=[len(self.source_list), len(d)])
        else:
            self.ps = self.base_model.score_events(d)

        self.is_data_set = True

    def add_rate_parameter(self, source_name, log_prior=None):
        """Add a rate parameters to the likelihood function.
        You don't actually have to use this unless you want to specify the prior.
        "rate" means a rate of events per day in total (not just events that procuce signals in range!)
        :param source_name: Name of the source for which you want to vary the rate
        :param log_prior: prior logpdf function on rate
        """
        self.rate_parameters[source_name] = log_prior

    def add_shape_parameter(self, setting_name, anchors, log_prior=None):
        """Add a shape parameter to the likelihood function
        :param setting_name: Name of the setting to vary
        :param anchors: a list/tuple/array of setting values (if they are numeric)
                        OR a dictionary with some numerical value -> setting values (for non-numeric settings).
        For example, if you have LCE maps with varying reflectivities, use
            add_shape_variation('s1_relative_ly_map', {0.98: 'lce_98%.pklz', 0.99: 'lce_99%.pklz, ...})
        then the argument s1_relative_ly_map of the likelihood function takes values between 0.98 and 0.99.
        """
        if not isinstance(anchors, dict):
            # Convert anchors list to a dictionary
            if not isinstance(self.pdf_base_config.get(setting_name), (float, int)):
                raise ValueError("When specifying anchors only by setting values, "
                                 "base setting must have a numerical default.")
            anchors = {z: z for z in anchors}

        self.shape_parameters[setting_name] = (anchors, log_prior)

    def __call__(self, **kwargs):
        if not self.is_data_set:
            raise RuntimeError("First do .set_data(dataset), then start evaluating the likelihood function")
        result = 0

        if len(self.shape_parameters):
            # Get the shape parameter z values
            zs = []
            for setting_name, (_, log_prior) in self.shape_parameters.items():
                z = kwargs.get(setting_name, self.pdf_base_config.get(setting_name))
                zs.append(z)

                # Test if the anchor value out of range, if so, return -inf (since is impossible)
                minbound, maxbound = self.get_bounds(setting_name)
                if not minbound <= z <= maxbound:
                    return -float('inf')

                if log_prior is not None:
                    result += log_prior(z)

                        # The RegularGridInterpolators want numpy arrays: give it to them...
            zs = np.asarray(zs)

            # Get mus (rate for each source) and ps (pdf value for each source for each event) at this point
                        # The RegularGridInterpolators return numpy arrays with one extra dimension: remove it...
            mus = self.mus_interpolator(zs)
            ps = self.ps_interpolator(zs)

        else:
            mus = self.base_model.expected_events()
            ps = self.ps

        # Apply the rate modifiers
        for source_i, source_name in enumerate(self.source_list):
            if source_name + '_rate' in kwargs:
                # The user gave a rate in total events/day, and this is what goes into the prior.
                new_total_rate = kwargs[source_name + '_rate']
                log_prior = self.rate_parameters.get(source_name, None)
                if log_prior is not None:
                    result += log_prior(new_total_rate)

                # However, the model / mus interpolator provides and the likelihood expects
                # the number of events IN RANGE /day as mu. So we rescale:
                mus[source_i] *= new_total_rate / self.base_model.get_source(source_name).events_per_day

        # Handle unphysical rates. Depending on the config, either error or return -float('inf') as loglikelihood
        if not np.all((mus >= 0) & (mus < float('inf'))):
            if self.config.get('unphysical_behaviour') == 'error':
                raise ValueError("Unphysical rates: %s" % str(mus))
            else:
                return -float('inf')

        # Get the loglikelihood. At last!
        result += extended_loglikelihood(mus, ps, outlier_likelihood=self.config.get('outlier_likelihood', 1e-12))
        return result

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
        if self.itp_mode == 'regular':

            # Allocate an array which will hold the scores at each anchor model
            anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + extra_dims)

            # Iterate over the anchor grid points
            for anchor_grid_index, _zs in self.anchor_grid_iterator():

                # Compute f at this point, and store it in anchor_scores
                anchor_scores[anchor_grid_index + [slice(None)] * len(extra_dims)] = f(self.anchor_models[tuple(_zs)])

            return RegularGridInterpolator(self.anchor_z_arrays, anchor_scores)

        elif self.itp_mode == 'radial':
            anchor_scores = np.array([f(_models_zs) for _models_zs in self.anchor_z_arrays])

            # Rescale the zs to the bounds. These could have been stored as attributes in prepare() instead...
            bounds = np.array(self.get_bounds())
            mins = bounds[:, 0]
            lengths = bounds[:, 1] - bounds[:, 0]
            models_zs = [(_zs - mins)/lengths for _zs in self.anchor_z_arrays]
            r0 = approx_avg_distance(models_zs)

            def interpolator(zs):
                # Closure: Gets mins, lengths, model_zs, r0 from outer scope

                # Compute the distance between the current point and each model
                zs = (zs - mins) / lengths
                rs = np.sqrt([np.dot(zs, _zs) for _zs in models_zs])

                # Compute the weight of each model: inverse multi-quadratic
                weights = 1/np.sqrt(1+(rs/r0)**2)
                weights /= np.sum(weights)

                return np.average(anchor_scores, weights=anchor_scores, axis=0)

            return interpolator

    def get_bounds(self, parameter_name=None):
        """Return bounds on the parameter parameter_name"""
        if parameter_name is None:
            return [self.get_bounds(p) for p in self.shape_parameters.keys()]
        if parameter_name in self.shape_parameters:
            anchor_settings = list(self.shape_parameters[parameter_name][0].keys())
            return min(anchor_settings), max(anchor_settings)
        elif parameter_name.endswith('_rate'):
            return 0, float('inf')
        else:
            raise ValueError("Non-existing parameter %s" % parameter_name)

    # Convenience function for uncertainties.
    # Adding more general priors is the user's responsibility
    # (either provide prior argument to add_x_parameter, or wrap the loglikelihood function)
    def add_rate_uncertainty(self, source_name, fractional_uncertainty):
        """Adds a rate parameter to the likelihood function, with Gaussian prior around the default value"""
        mu = self.base_model.get_source(source_name).events_per_day
        self.add_rate_parameter(source_name, log_prior=stats.norm(mu, mu * fractional_uncertainty).logpdf)

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
    :param outlier_likelihood: if an event has p=0, give it this likelihood (instead of 0, which makes the whole
    loglikelihood infinite)
    :return: ln(likelihood)
    """
    p_events = np.sum(mu[:, np.newaxis] * ps, axis=0)
    if outlier_likelihood != 0:
        # Replace all likelihoods which are not positive numbers (i.e. 0, negative, or nan) with outlier_likelihood
        p_events[True ^ (p_events > 0)] = outlier_likelihood
    return -mu.sum() + np.sum(np.log(p_events))


def arrays_to_grid(arrs):
    """Convert a list of n 1-dim arrays to an n+1-dim. array, where last dimension denotes coordinate values at point.
    """
    return np.stack(np.meshgrid(*arrs, indexing='ij'), axis=-1)


def approx_avg_distance(points):
    """Some approximation to average distance between points (array (n_points, n_dim))
    found in scipy: https://github.com/pv/scipy-work/blob/master/scipy/interpolate/rbf.py#L203
    """
    # default epsilon is the "the average distance between nodes" based
    # on a bounding hypercube
    ximax = np.amax(points, axis=1)
    ximin = np.amin(points, axis=1)
    edges = ximax-ximin
    edges = edges[np.nonzero(edges)]
    return np.power(np.prod(edges)/len(points), 1.0/edges.size)


def latin(n, d, box=None, shuffle_steps=1000):
    """Creates a latin hypercube of n points in d dimensions
    Stolen from https://github.com/paulknysh/blackbox
    """
    # starting with diagonal shape
    pts=np.ones((n,d))

    for i in range(n):
        pts[i]=pts[i]*i/(n-1.)

    # spread function
    def spread(p):
        s=0.
        for i in range(n):
            for j in range(n):
                if i > j:
                    s=s+1./np.linalg.norm(np.subtract(p[i],p[j]))
        return s

    # minimizing spread function by shuffling
    currminspread=spread(pts)

    for m in tqdm(range(shuffle_steps), desc='Shuffling latin hypercube'):

        p1=np.random.randint(n)
        p2=np.random.randint(n)
        k=np.random.randint(d)

        newpts=np.copy(pts)
        newpts[p1,k],newpts[p2,k]=newpts[p2,k],newpts[p1,k]
        newspread=spread(newpts)

        if newspread<currminspread:
            pts=np.copy(newpts)
            currminspread=newspread

    if box is None:
        return pts

    for i in range(len(box)):
        pts[:, i] = box[i][0] + pts[:, i] * (box[i][1] - box[i][0])

    return pts
