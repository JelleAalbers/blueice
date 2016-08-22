import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from tqdm import tqdm

from blueice.utils import arrays_to_grid, inherit_docstring_from, combine_dicts


class NoShapeParameters(Exception):
    pass


class Morpher(object):

    def __init__(self, config, shape_parameters):
        """Initialize the morpher, telling it which shape_parameters we're going to use
        See model for format of shape_parameters
        """
        self.config = config
        self.shape_parameters = shape_parameters
        if not len(self.shape_parameters):
            raise NoShapeParameters("Attempt to initialize a morpher without shape parameters")

    def get_anchor_points(self, bounds, n_models=None):
        """Returns list of anchor z-coordinates at which we should sample n_models between bounds.
        The morpher may choose to ignore your bounds and n_models argument if it doesn't support them.
        """
        raise NotImplementedError

    def make_interpolator(self, f, extra_dims, anchor_models):
        """Return a function which interpolates the extra_dims-valued function f(model)
        between the anchor points.
        :param f: Function which takes a Model as argument, and produces an extra_dims shaped array.
        :param extra_dims: tuple of integers, shape of return value of f.
        :param anchor_models: dictionary {z-score: Model} of anchor models at which to evaluate f.
        """
        raise NotImplementedError


class GridInterpolator(Morpher):

    @inherit_docstring_from(Morpher)
    def __init__(self, config, shape_parameters):
        super().__init__(config, shape_parameters)
        # Compute the regular grid of anchor models at the specified anchor points
        self.anchor_z_arrays = [np.array(list(sorted(anchors.keys())))
                                for setting_name, (anchors, _, _) in shape_parameters.items()]
        self.anchor_z_grid = arrays_to_grid(self.anchor_z_arrays)

    @inherit_docstring_from(Morpher)
    def get_anchor_points(self, bounds, n_models=None):
        return [zs for _, zs in self._anchor_grid_iterator()]

    @inherit_docstring_from(Morpher)
    def make_interpolator(self, f, extra_dims, anchor_models):
        # Allocate an array which will hold the scores at each anchor model
        anchor_scores = np.zeros(list(self.anchor_z_grid.shape)[:-1] + extra_dims)

        # Iterate over the anchor grid points
        for anchor_grid_index, _zs in self._anchor_grid_iterator():
            # Compute f at this point, and store it in anchor_scores
            anchor_scores[anchor_grid_index + [slice(None)] * len(extra_dims)] = f(anchor_models[tuple(_zs)])

        itp = RegularGridInterpolator(self.anchor_z_arrays, anchor_scores)

        # For some reason I'm getting an extra first dimension with everything in the first element, let's remove it...
        return lambda *args: itp(*args)[0]

    def _anchor_grid_iterator(self):
        """Iterates over the anchor grid, yielding index, z-values"""
        fake_grid = np.zeros(list(self.anchor_z_grid.shape)[:-1])
        it = np.nditer(fake_grid, flags=['multi_index'])
        while not it.finished:
            anchor_grid_index = list(it.multi_index)
            yield anchor_grid_index, tuple(self.anchor_z_grid[anchor_grid_index + [slice(None)]])
            it.iternext()


class RadialInterpolator(Morpher):
    """This morpher is highly experimental!!"""

    @inherit_docstring_from(Morpher)
    def __init__(self, config, shape_parameters):
        defaults = dict(r_sample_points=5,
                        hypercube_shuffle_steps=500,
                        decay_response_to_density='constant')
        config = combine_dicts(defaults, config)
        super().__init__(config, shape_parameters)

    @inherit_docstring_from(Morpher)
    def get_anchor_points(self, bounds, n_models=10):
        # Sample a Latin hypercube of models
        zs_list = latin(n_models, len(self.shape_parameters), box=bounds,
                        shuffle_steps=self.config['hypercube_shuffle_steps'])
        zs_list = list(map(tuple, zs_list))

        # Get the bounds needed to scale the zs
        bounds = np.array(bounds)
        self._mins = bounds[:, 0]
        self._lengths = bounds[:, 1] - bounds[:, 0]

        # Rescale the zs to the bounds. It's fine if the zs are outside the bounds, but we need something
        # to scale the different dimensions to similar ranges so norms make sense.
        # Notice zs_list is redefined here to be the list of zs of *all* models, not just the new ones
        self._normed_model_zs = [(np.array(_zs) - self._mins) / self._lengths for _zs in zs_list]

        # Get the average distance to the five closest points
        self._r0s = KDTree(self._normed_model_zs).query(self._normed_model_zs,
                                                        self.config['r_sample_points'])[0].mean(axis=1)
        decay_response = self.config['decay_response_to_density']
        if decay_response == 'constant':
            self._r0s = np.ones_like(self._r0s) * self._r0s.mean()
        elif decay_response == 'proportional':
            pass
        else:
            raise NotImplementedError(decay_response)

        return zs_list

    @inherit_docstring_from(Morpher)
    def make_interpolator(self, f, extra_dims, anchor_models):
        anchor_scores = np.array([f(m) for m in anchor_models.values()])

        def interpolator(zs):
            # Compute the distance between the current point and each model
            normed_zs = (zs - self._mins) / self._lengths
            # print("Normed zs for this point: ", normed_zs)
            rs = np.sqrt([np.dot(normed_zs - _nzs, normed_zs - _nzs)
                          for _nzs in self._normed_model_zs])
            # print("Distances to models: ", rs)

            # Compute the weight of each model: exponential decay
            # Note we use a normalized exponential, so models with small radius of influence (i.e. in dense regions)
            # should have a higher weight when we get close to them than models with a large radius of influence.
            r_of_influence = self._r0s * self.config.get('decay_multiplier', 5)
            weights = np.exp(-rs / r_of_influence) / r_of_influence

            weights /= np.sum(weights)
            # print("Weights of models: ", weights)
            # print("Data scores at models: ", anchor_scores)

            return np.average(anchor_scores, weights=weights, axis=0)

        return interpolator


def latin(n, d, box=None, shuffle_steps=500):
    """Creates a latin hypercube of n points in d dimensions
    Stolen from https://github.com/paulknysh/blackbox
    """
    # starting with diagonal shape
    pts = np.ones((n, d))

    for i in range(n):
        pts[i] = pts[i] * i / (n-1.)

    # spread function
    def spread(p):
        s = 0.
        for i in range(n):
            for j in range(n):
                if i > j:
                    s = s + 1. / np.linalg.norm(np.subtract(p[i], p[j]))
        return s

    # minimizing spread function by shuffling
    currminspread = spread(pts)

    for m in tqdm(range(shuffle_steps), desc='Shuffling latin hypercube'):

        p1 = np.random.randint(n)
        p2 = np.random.randint(n)
        k = np.random.randint(d)

        newpts = np.copy(pts)
        newpts[p1, k], newpts[p2, k] = newpts[p2, k], newpts[p1, k]
        newspread = spread(newpts)

        if newspread < currminspread:
            pts = np.copy(newpts)
            currminspread = newspread

    if box is None:
        return pts

    for i in range(len(box)):
        pts[:, i] = box[i][0] + pts[:, i] * (box[i][1] - box[i][0])

    return pts


MORPHERS = {x.__name__: x for x in [GridInterpolator, RadialInterpolator]}
