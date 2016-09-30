"""Helper functions for analysis of LogLikelihood functions
If you want to analyze with your own tools, you can just ignore these,
only make_objective is of general use (for wrapping optimizers).
"""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from iminuit.util import make_func_code
from iminuit import Minuit


class NoOpimizationNecessary(Exception):
    pass


class OptimizationFailed(Exception):
    pass


def make_objective(lf, guess=None, minus=True, rates_in_log_space=False, **kwargs):
    """Return convenient stuff for feeding the LogLikelihood lf to an optimizer.
    :param **kwargs: fixed values for certain parameters. These will not be fitted.
    :param guess: dictionary with guesses for the remaining ("floating") parameters
    If you don't supply a guess for a parameter, the base model setting will be taken as a guess.
    :param minus: if true (default), multiply result of LogLikelihood by -1.
                  Minimizers tend to appreciate this, samplers like MCMC do not.
    :param rates_in_log_space: UNTESTED: let the minimizer work on the rate multipliers in log space instead

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
        if p + '_rate_multiplier' not in kwargs:
            # Default is the number of events per day total (not just in range).
            # This is what likelihood function expects, it knows how to deal with it.
            g = guess.get('%s_rate_multiplier' % p, 1)
            names.append('%s_rate_multiplier' % p)
            if rates_in_log_space:
                guesses.append(np.log10(g))
                bounds.append((None, None))
            else:
                guesses.append(g)
                bounds.append((0, None))

    # Which shape parameters should we fit?
    for p, (_, __, base_value) in lf.shape_parameters.items():
        if p not in kwargs:
            names.append(p)
            bounds.append(lf.get_bounds(p))
            g = guess.get(p)
            if g is None:
                g = lf.pdf_base_config.get(p)
                if not isinstance(g, (int, float)):
                    g = base_value
            guesses.append(g)

    if not len(names):
        raise NoOpimizationNecessary("There are no parameters to fit, no optimization is necessary")

    # Minimize the - log likelihood
    # Uses kwargs, sign, and self from external scope. So don't try to pickle it...
    sign = -1 if minus else 1

    def objective(args):
        # Get the arguments from args, then fill in the ones already fixed in outer kwargs
        call_kwargs = {}
        for i, k in enumerate(names):
            if rates_in_log_space and k.endswith('_rate_multiplier'):
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


def bestfit_scipy(lf, minimize_kwargs=None, rates_in_log_space=False, pass_bounds_to_minimizer=False, **kwargs):
    """Minimizes the LogLikelihood function lf over the parameters not specified in kwargs.
    Returns {param: best fit}, maximum loglikelihood.

    Optimization is performed with the scipy minimizer
    :param minimize_kwargs: dictionary with optimz to minimize
    :param pass_bounds_to_minimizer: if true (default is False), pass bounds to minimizer via the bounds argument.
    This shouldn't be necessary, as the likelihood function returns -inf outside the bounds.
    I've gotten strange results with scipy's L-BFGS-B, scipy's default method with bound problems,
    perhaps it is less well tested?
    If you pass this, I recommend passing a different minimizer method (e.g. TNC or SLSQP).

    Other kwargs are passed to make_objective.
    """
    if minimize_kwargs is None:
        minimize_kwargs = {}

    try:
        f, names, guess, bounds = make_objective(lf, minus=True, rates_in_log_space=rates_in_log_space, **kwargs)
    except NoOpimizationNecessary:
        return {}, lf(**kwargs)

    optresult = minimize(f, guess,
                         bounds=bounds if pass_bounds_to_minimizer else None,
                         **minimize_kwargs)

    if not optresult.success:
        # Try again with a more robust, but slower method
        optresult = minimize(f, guess,
                             bounds=bounds if pass_bounds_to_minimizer else None,
                             method='Nelder-Mead',
                             **minimize_kwargs)
        if not optresult.success:
            raise OptimizationFailed("Optimization failure: ", optresult)

    optimum = optresult.x if len(names) != 1 else [optresult.x.item()]

    results = {}
    for i, name in enumerate(names):
        if rates_in_log_space and name.endswith('_rate_multiplier'):
            # The minimizer was fooled into seeing the log10 of the rate, convert it back for the user
            results[name] = 10**optimum[i]
        else:
            results[name] = optimum[i]
    return results,  -optresult.fun


def bestfit_minuit(lf, minimize_kwargs=None, rates_in_log_space=False, **kwargs):
    """Minimizes the LogLikelihood function lf over the parameters not specified in kwargs.
    Returns {param: best fit}, maximum loglikelihood.

    Optimization is performed with iminuits Minuit
    :param minimize_kwargs: dictionary with optimz to minimize

    Other kwargs are passed to make_objective.
    """
    if minimize_kwargs is None:
        minimize_kwargs = {}

    try:
        f, names, guess, bounds = make_objective(lf, minus=True, rates_in_log_space=rates_in_log_space, **kwargs)
    except NoOpimizationNecessary:
        return {}, lf(**kwargs)

    class MinuitWrap:
        """Wrapper for functions to be called by Minuit"""
        def __init__(self, f, s_args):
            self.func = f
            self.s_args = s_args
            self.func_code = make_func_code(s_args)

        def __call__(self, *args):
            return self.func(args)

    # Make a dict for minuit
    minuit_dict = {}
    for i, name in enumerate(names):
        minuit_dict[name] = guess[i]

    m = Minuit(MinuitWrap(f, names), **minuit_dict)
    m.migrad()

    return m.values, -1*m.fval


def one_parameter_interval(lf, target, bound,
                           confidence_level=0.9, kind='upper',
                           bestfit_routine=bestfit_scipy, **kwargs):
    """Set a confidence_level interval of kind (central, upper, lower) on the parameter target of lf.
    This assumes the likelihood ratio is asymptotically chi2(1) distributed (Wilk's theorem)
    target: parameter of lf to constrain
    bound: bound(s) for the line search. For upper and lower: single value, for central: 2-tuple.
    kwargs: dictionary with arguments to bestfit
    """
    if target is None:
        target = lf.source_list[-1] + '_rate_multiplier'

    # Find the likelihood of the global best fit (denominator of likelihood ratio)
    result, max_loglikelihood = bestfit_routine(lf, **kwargs)
    global_best = result[target]

    def t(hypothesis, critical_quantile):
        """(profile) likelihood ratio test statistic, with critical_value subtracted
        critical_quantile: fraction (percentile/100) of the test statistic distribution you want to find
        """
        # "But I thought I there was a chi2 in Wilk's theorem!" Quite right, but
        # stats.norm.ppf(CL)**2 = stats.chi2(1).ppf(2*CL - 1)
        # So the chi2 formula is often quoted for central CI's, the normal one for bounds...
        # This cost me hours of confusion. Please explain this to your students if you're statistics professor.
        critical_value = stats.norm.ppf(critical_quantile) ** 2

        if kind == 'upper' and hypothesis <= global_best:
            result = 0
        elif kind == 'lower' and hypothesis >= global_best:
            result = 0
        else:
            # Find the best fit assuming the hypothesis (numerator of likelihood ratio)
            lf_kwargs = {target: hypothesis}
            lf_kwargs.update(kwargs)
            fitresult, ll = bestfit_routine(lf, **lf_kwargs)
            result = 2*(max_loglikelihood - ll)

        return result - critical_value

    if kind == 'central':
        a = brentq(t, bound[0], global_best, args=[(1-confidence_level)/2])
        b = brentq(t, global_best, bound[1], args=[1 - (1 - confidence_level) / 2])
        return a, b
    elif kind == 'lower':
        return brentq(t, bound, global_best, args=[1 - confidence_level])
    elif kind == 'upper':
        return brentq(t, global_best, bound, args=[confidence_level])


def plot_likelihood_ratio(lf, *space, vmax=15, plot_kwargs=None, **kwargs):
    """Plots the loglikelihood ratio derived from LogLikelihood lf in a parameter space
    :param lf: LogLikelihood function with data set.
    :param space: list/tuple of tuples (dimname, points to plot)
    :param vmax: Limit for color bar (2d) or y axis (1d)
    :param plot_kwargs: kwargs passed to plt.plot / plt.pcolormesh
    Further arguments are passed to lf, arguments not passed are fitted at each point.
    :return: Nothing
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    results = []
    label = "Log likelihood ratio"
    if len(space) == 1:
        dim, x = space[0]
        for q in x:
            lf_kwargs = {dim: q}
            lf_kwargs.update(kwargs)
            results.append(bestfit_scipy(lf, **lf_kwargs)[1])
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
                lf_kwargs = {dims[0]: z1, dims[1]: z2}
                lf_kwargs.update(kwargs)
                results[-1].append(bestfit_scipy(lf, **lf_kwargs)[1])
        z1, z2 = np.meshgrid(x, y)
        results = np.array(results)
        results = results.max() - results
        plt.pcolormesh(z1, z2, results.T, vmax=vmax, **plot_kwargs)
        plt.colorbar(label=label)
        plt.xlabel(dims[0])
        plt.ylabel(dims[1])
    else:
        raise ValueError("Can't handle %d dimensions" % len(space))
