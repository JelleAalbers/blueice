"""Helper functions for analysis of LogLikelihood functions
If you want to analyze with your own tools, you can just ignore these,
only make_objective is of general use (for wrapping optimizers).
"""
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt


def make_objective(lf, guess=None, minus=True, rates_in_log_space=False, **kwargs):
    """Return convenient stuff for feeding the LogLikelihood lf to an optimizer.
    :param **kwargs: fixed values for certain parameters. These will not be fitted.
    :param guess: dictionary with guesses for the remaining ("floating") parameters
    :param rates_in_log_space: UNTESTED: let the minimizer work on the rates in log space instead
    If you don't supply a guess, the default / base model value will be used for rate parameters,
    and 0 for shape parameters (fixme: use base model value, even if has been turned into monster).

    :param minus: f provides the minimum of the likelihood function instead

    Returns f, guesses, names:
      - f: function which takes a single arraylike argument with only the floating parameters
      - names: list, floating parameter names in correct order
      - guesses: array of guesses in order taken by f
      - bounds: list of tuples of bounds for floating parameters. (None, None) if there are no bounds for a param.
    """
    if guess is None:
        guess = {}
    names = []
    bounds = []
    guesses = []

    # Which rate parameters should we fit?
    for p in lf.rate_parameters.keys():
        if p not in kwargs:
            # Default is the number of events per day total (not just in range).
            # This is what likelihood function expects, it knows how to deal with it.
            g = guess.get('%s_rate' % p, lf.base_model.get_source(p).events_per_day)
            names.append('%s_rate' % p)
            if rates_in_log_space:
                guesses.append(np.log10(g))
                bounds.append((None, None))
            else:
                guesses.append(g)
                bounds.append((0, None))

    # Which shape parameters should we fit?
    for p in list(lf.shape_parameters.keys()):
        if p not in kwargs:
            names.append(p)
            bounds.append(lf.get_bounds(p))
            guesses.append(guess.get(p, 0))

    # Minimize the - log likelihood
    # Uses kwargs, sign, and self from external scope. So don't try to pickle it...
    sign = -1 if minus else 1
    def objective(args):
        # Get the arguments from args, then fill in the ones already fixed in outer kwargs
        call_kwargs = {}
        for i, k in enumerate(names):
            if rates_in_log_space and k.endswith('_rate'):
                # The minimizer provides on the log10 of the rate. Convert it back to a normal rate for the
                # likelihood function
                call_kwargs[k] = 10**args[i]
            else:
                call_kwargs[k] = args[i]
        call_kwargs.update(kwargs)
        return lf(**call_kwargs) * sign

    return objective, names, np.array(guesses), bounds

##
# Wrapper function for scipy minimization. If you want to use another minimizer, you'd write a similar wrapper
##
def bestfit_scipy(lf, minimize_kwargs=None, rates_in_log_space=False, **kwargs):
    """Minimizes the LogLikelihood function lf over the parameters not specified in kwargs.
    Returns {param: best fit}, minimum loglikelihood.

    Optimization is performed with the scipy minimizer
    :param minimize_kwargs: dictionary with optimz to minimize
    Other kwargs are passed to make_objective
    """
    if minimize_kwargs is None:
        minimize_kwargs = {}

    f, names, guess, bounds = make_objective(lf, minus=True, rates_in_log_space=rates_in_log_space, **kwargs)

    optresult = minimize(f, guess,
                         bounds=bounds,
                         **minimize_kwargs)

    if not optresult.success:
        # Try again with a more robust, but slower method
        optresult = minimize(f, guess,
                             bounds=bounds,
                             method='Nelder-Mead',
                             **minimize_kwargs)
        if not optresult.success:
            raise RuntimeError("Optimization failure: ", optresult)

    optimum = optresult.x if len(names) != 1 else [optresult.x.item()]

    results = {}
    for i, name in enumerate(names):
        if rates_in_log_space and name.endswith('_rate'):
            # The minimizer was fooled into seeing the log10 of the rate, convert it back for the user
            results[name] = 10**optimum[i]
        else:
            results[name] = optimum[i]
    return results,  -optresult.fun


def plot_likelihood_ratio(lf, *space, vmax=10, plot_kwargs=None, **kwargs):
    """Plots the loglikelihood ratio derived from LogLikelihood lf in a parameter space
    :param lf: LogLikelihood function with data set.
    :param space: list/tuple of tuples (dimname, points to plot)
    :param vmax: Limit for color bar (2d) or y axis (1d)
    :param plot_kwargs: kwargs passed to plt.plot / plt.pcolormesh
    Further arguments are passed to lf
    :return: Nothing
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    results = []
    label = "Log likelihood ratio"
    if len(space) == 1:
        dim, x = space[0]
        for q in x:
            lf_kwargs = {dim:q}
            lf_kwargs.update(kwargs)
            results.append(lf(**lf_kwargs))
        results = np.array(results)
        results = results.max() - results
        plt.plot(x, results, **plot_kwargs)
        plt.ylim(0, vmax)
        plt.ylabel(label)
        plt.xlabel(dim)
        plt.xlim(x.min(), x.max())

    elif len(space) == 2:
        dims = (space[0][0], space[1][0])
        x, y = (space[0][1], space[1][1])
        for z1 in tqdm(x):
            results.append([])
            for z2 in y:
                lf_kwargs = {dims[0]:z1, dims[1]:z2}
                lf_kwargs.update(kwargs)
                results[-1].append(lf(**lf_kwargs))
        z1, z2 = np.meshgrid(x, y)
        results = np.array(results)
        results = results.max() - results
        plt.pcolormesh(z1, z2, results.T, vmax=vmax, **plot_kwargs)
        plt.colorbar(label=label)
        plt.xlabel(dims[0])
        plt.ylabel(dims[1])
    else:
        raise ValueError("Can't handle %d dimensions" % len(space))
