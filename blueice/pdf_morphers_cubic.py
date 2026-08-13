"""C1-smooth, shape-preserving morpher using tensor-product PchipInterpolator.

Fixes the C0 kink problem in GridInterpolator (linear RegularGridInterpolator).
Shape preservation:  pchip is monotone between each pair of anchors.  If all
anchor values are non-negative, the interpolated values are guaranteed non-negative.

Activation
----------
In alea, set it in the likelihood term's YAML config:

    likelihood_terms:
      - name: likelihood_name
        ...
        likelihood_config:
          morpher: CubicSplineInterpolator
          morpher_config:
            extrapolate: false   # default

In plain blueice (no alea), pass the equivalent ``likelihood_config`` dict
directly when constructing the likelihood, e.g.:

    likelihood_config = {
        ...
        'morpher': 'CubicSplineInterpolator',
        'morpher_config': {'extrapolate': False},  # default
    }
    ll = UnbinnedLogLikelihood(pdf_base_config, likelihood_config=likelihood_config)

Config keys (morpher_config)
----------------------------
extrapolate : bool, default False
    Whether to extrapolate beyond the outermost anchors.
    Recommended False — extrapolation can produce negative values.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from .pdf_morphers import GridInterpolator
from .utils import inherit_docstring_from


def _make_fast_pchip_eval(x, extrapolate=False):
    """Return a callable eval(xq, y) that evaluates pchip at scalar xq.

    All x-dependent constants are pre-computed and closed over.
    y shape: (n, ...) where axis 0 is the interpolation axis.
    Returns array of shape y.shape[1:].
    """
    x = np.asarray(x, dtype=float)
    h = np.diff(x)
    hinv = 1.0 / h
    n = len(x)

    # Interior derivative weights (Fritsch-Carlson)
    w1 = 2 * h[1:] + h[:-1]  
    w2 = h[1:] + 2 * h[:-1]  

    # Endpoint one-sided coefficients
    ep0a = (2*h[0] + h[1]) / (h[0] + h[1])
    ep0b = -h[0] / (h[0] + h[1])
    epNa = (2*h[-1] + h[-2]) / (h[-1] + h[-2])
    epNb = -h[-1] / (h[-1] + h[-2])

    def eval_at(xq, y):
        xq = float(xq)
        nd = y.ndim
        bcast = (slice(None),) + (np.newaxis,) * (nd - 1)

        delta = np.diff(y, axis=0) * hinv[bcast]  # (n-1, ...)

        d = np.empty_like(y)
        d[0] = ep0a * delta[0] + ep0b * delta[1]
        d[-1] = epNa * delta[-1] + epNb * delta[-2]

        for k in range(1, n - 1):
            ss = delta[k-1] * delta[k] > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                d[k] = np.where(
                    ss,
                    (w1[k-1] + w2[k-1]) / (w1[k-1] / delta[k-1] + w2[k-1] / delta[k]),
                    0.0,
                )

        # Clamp endpoints
        d[0] = np.where(
            np.sign(d[0]) != np.sign(delta[0]), 0.0,
            np.where(np.abs(d[0]) > 3 * np.abs(delta[0]), 3 * delta[0], d[0]),
        )
        d[-1] = np.where(
            np.sign(d[-1]) != np.sign(delta[-1]), 0.0,
            np.where(np.abs(d[-1]) > 3 * np.abs(delta[-1]), 3 * delta[-1], d[-1]),
        )

        # Interval lookup
        if extrapolate:
            ki = np.clip(np.searchsorted(x, xq, side='right') - 1, 0, n - 2)
        else:
            ki = np.clip(np.searchsorted(x, xq, side='right') - 1, 0, n - 2)

        t = (xq - x[ki]) / h[ki]
        t2 = t * t
        t3 = t2 * t
        hk = h[ki]
        return (
            (2*t3 - 3*t2 + 1) * y[ki]
            + (t3 - 2*t2 + t) * hk * d[ki]
            + (-2*t3 + 3*t2) * y[ki+1]
            + (t3 - t2) * hk * d[ki+1]
        )

    return eval_at


class CubicSplineInterpolator(GridInterpolator):
    """C1-smooth, shape-preserving morpher. Drop-in for GridInterpolator.

    Uses pchip (monotone cubic Hermite) interpolation.  Guarantees non-negative
    output when all anchor values are non-negative.  Eliminates the gradient kinks
    that cause MIGRAD to produce invalid minima with linear interpolation.
    """

    @inherit_docstring_from(GridInterpolator)
    def make_interpolator(self, f, extra_dims, anchor_models):
        # Build anchor_scores exactly as GridInterpolator does.
        # Shape: (n_1, n_2, ..., n_N, *extra_dims)
        anchor_scores = np.zeros(
            list(self.anchor_z_grid.shape)[:-1] + list(extra_dims)
        )
        for anchor_grid_index, _zs in self._anchor_grid_iterator():
            anchor_scores[
                tuple(anchor_grid_index + [slice(None)] * len(extra_dims))
            ] = f(anchor_models[tuple(_zs)])

        anchor_z_arrays = self.anchor_z_arrays
        extrapolate = self.config.get('extrapolate', False)
        n_dims = len(anchor_z_arrays)

        # Axis 0: pre-build scipy PchipInterpolator once.
        # Subsequent axes: use the fast numpy eval.
        itp0 = PchipInterpolator(
            anchor_z_arrays[0], anchor_scores, axis=0, extrapolate=extrapolate
        )

        if n_dims == 1:
            return lambda zs: itp0(float(zs[0]))

        fast_evals = [
            _make_fast_pchip_eval(z_arr, extrapolate=extrapolate)
            for z_arr in anchor_z_arrays[1:]
        ]

        def interpolator(zs):
            current = itp0(float(zs[0]))
            for z, fe in zip(zs[1:], fast_evals):
                current = fe(float(z), current)
            return current

        return interpolator
