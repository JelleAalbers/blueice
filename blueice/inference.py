"""Helper functions for analysis using LogLikelihood functions

Blueice's main purpose in life is to provide you with a likelihood function, but some operations
are so common we could not resist adding canned tools for them. Don't worry if your particular analysis is not covered,
just use whatever tools you want with either the LogLikelihood function itself, or the output of make_objective
(if your tools expect a positional-argument only function).

Functions from this file are also made accesible as methods of LogLikelihoodBase.
"""
import warnings
from collections import OrderedDict

import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
from tqdm import tqdm

from copy import deepcopy
from .exceptions import NoOpimizationNecessary, OptimizationFailed

try:
    # Import imunuit here, so blueice works also for people who don't have it installed.
    from iminuit.util import make_func_code     # noqa
    from iminuit import Minuit                  # noqa
except ImportError:
    pass

DEFAULT_BESTFIT_ROUTINE = 'scipy'

__all__ = ['best_anchor', 'make_objective', 'bestfit_scipy', 'bestfit_minuit', 'plot_likelihood_ratio',
           'one_parameter_interval', 'bestfit_emcee']


def best_anchor(lf):
    """Return shape parameter dictionary of anchor model with highest likelihood.
    Useful as a guess for further fitting.
    """
    if not len(lf.shape_parameters):
        return dict()

    shape_par_names = list(lf.shape_parameters.keys())
    anchors = list(lf.anchor_models.keys())
    results = np.zeros(len(anchors))

    def dictzip_shapes(anchor_vals):
        return {shape_par_names[j]: anchor_vals[j]
                for j in range(len(shape_par_names))}

    for i, anchor_vals in enumerate(anchors):
        results[i] = lf(**dictzip_shapes(anchor_vals))

    best_i = np.argmax(results)

    return dictzip_shapes(anchors[best_i])


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
        f, names, guess, bounds = lf.make_objective(minus=True, rates_in_log_space=rates_in_log_space, **kwargs)
    except NoOpimizationNecessary:
        return {}, lf(**kwargs)

    optresult = minimize(f, guess,
                         bounds=bounds if pass_bounds_to_minimizer else None,
                         **minimize_kwargs)

    if not optresult.success:
        # Try again with a more robust, but slower method
        #if method is defined in kwargs, it must be removed
        minimize_kwargs_temp = deepcopy(minimize_kwargs)
        minimize_kwargs_temp.pop('method',None)
        optresult = minimize(f, guess,
                             bounds=bounds if pass_bounds_to_minimizer else None,
                             method='Nelder-Mead',
                             **minimize_kwargs_temp)
        if not optresult.success:
            raise OptimizationFailed("Optimization failure: ", optresult)

    optimum = optresult.x if len(names) != 1 else [optresult.x.item()]

    results = OrderedDict()
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

    # By default, use quiet evaluation, since this is called iteratively in profile likelihoods.
    minimize_kwargs.setdefault('print_level', 0)
    minimize_kwargs.setdefault('pedantic', False)

    try:
        f, names, guess, bounds = lf.make_objective(minus=True, rates_in_log_space=rates_in_log_space, **kwargs)
    except NoOpimizationNecessary:
        return {}, lf(**kwargs)

    # The full iminuit API is documented here:
    # http://iminuit.readthedocs.io/en/latest/api.html

    # Make a dict for minuit with a key for each parameter and the initial
    # guesses as values.
    # Add also the bounds of each parameter as 'limit_'<name> as key and a
    # tuple of the bounds as values
    # TODO add also errors and fixed parameters to this dictionary
    minuit_dict = minimize_kwargs
    for i, name in enumerate(names):
        minuit_dict[name] = guess[i]
        minuit_dict['limit_' + name] = bounds[i]
        
    # Sets up correct magic for meaningful errors for log likelihoods
    minuit_dict['errordef'] = 0.5

    class MinuitWrap:
        """Wrapper for functions to be called by Minuit

        s_args must be a list of argument names of function f
        the names in this list must be the same as the keys of
        the dictionary passed to the Minuit call."""
        def __init__(self, f, s_args):
            self.func = f
            self.s_args = s_args
            self.func_code = make_func_code(s_args)

        def __call__(self, *args):
            return self.func(args)

    # Make the Minuit object
    m = Minuit(MinuitWrap(f, names), **minuit_dict)

    # Call migrad to do the actual minimization
    m.migrad()

    # TODO return more information, such as m.errors

    return dict(m.values), -1*m.fval  # , m.errors


# Must be defined outside bestfit_emcee to avoid pickling error
# TODO: will inevitably create problems, globals are bad...
def _lnprob(x):
    _lnprob.t.update(1)
    return _lnprob.f(x)


def bestfit_emcee(ll, quiet=False, return_errors=False, return_samples=False,
                  n_walkers=40, n_steps=200, n_burn_in=100, n_threads=1,
                  **kwargs):
    """Optimize the loglikelihood function ll using emcee's MCMC.
    The starting position of the walkers is [0.95, 1.05] * the default values / any guess you put in.
    So if your default value is 0 you have to put in a custom guess. (TODO: fix this)

    :param ll: LogLikelihood to optimize
    :param quiet: if False (default), show corner plot and print out passthrough info
    :param return_errors: if True, return a third result, dictionary with 1 sigma errors for each parameter
    :param return_samples: if True, return a third result, flattened numpy array of samples visited (except in burn-in)
    :param n_walkers: Number of walkers to use for the MCMC
    :param n_steps: Number of steps to use for MCMC
    :param n_burn_in: Number of burn-in steps to use. These are added to n_steps but thrown away.
    :param n_threads: Number of concurrent threads to use
    :param kwargs: Passed to ll.make_objective.
    :return: {param: best fit}, maximum loglikelihood.
    """
    import emcee

    f, names, guess, _  = ll.make_objective(minus=False, **kwargs)

    n_dim = len(guess)

    # Hack to show a progress bar during the computation
    _lnprob.f = f
    _lnprob.t = tqdm(desc='Computing likelihoods',
                    total=n_walkers * n_steps / n_threads)

    # Run the MCMC sampler
    p0 = np.array([np.random.uniform(0.95, 1.05, size=n_dim)  * guess for i in range(n_walkers)])
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, _lnprob, threads=n_threads)
    sampler.run_mcmc(p0, n_steps)

    # Remove first n_burn_in samples for each walker (burn-in)
    samples = sampler.chain[:, n_burn_in:, :].reshape((-1, n_dim))

    if not quiet:
        import corner
        import matplotlib.pyplot as plt

        # "This number should be between approximately 0.25 and 0.5 if everything went as planned"
        # http://dan.iel.fm/emcee/current/user/quickstart/
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

        samples = sampler.chain.reshape((-1, n_dim))
        corner.corner(samples, show_titles=True, labels=names,
                      range=[0.99] * len(names),
                      truths=guess,
        )
        plt.show()

    fit_result = np.median(samples, axis=0)
    fit_result_dict = OrderedDict([(names[i], fit_result[i]) for i in range(len(names))])

    best_ll = ll(**fit_result_dict)

    if return_errors:
        l, r = np.percentile(samples, 100 * stats.norm.cdf([-1, 1]), axis=0)
        fit_errors = (r - l)/2
        fit_errors_dict = OrderedDict([(names[i], fit_errors[i]) for i in range(len(names))])

        return fit_result_dict, best_ll, fit_errors_dict

    if return_samples:
        return fit_result_dict, best_ll, samples

    return fit_result_dict, best_ll


def _get_bestfit_routine(key):
    if hasattr(key, '__call__'):
        return key
    if key is None:
        key = DEFAULT_BESTFIT_ROUTINE
    return BESTFIT_ROUTINES[key]


def one_parameter_interval(lf, target, bound,
                           confidence_level=0.9, kind='upper',
                           bestfit_routine=None,
                           t_ppf=None,
                           **kwargs):
    """Set a confidence_level interval of kind (central, upper, lower) on the parameter target of lf.
    This assumes the likelihood ratio is asymptotically chi2(1) distributed (Wilk's theorem)
    target: parameter of lf to constrain
    bound: bound(s) for the line search. For upper and lower: single value, for central: 2-tuple.
    t_ppf: function (hypothesis, level) -> test statistic (-2 Log[ L(test)/L(bestfit) ])
           must return value at which test statistic reaches level'th quantile if hypothesis is true.
           If not specified, Wilks' theorem will be used.
    kwargs: dictionary with arguments to bestfit
    """
    bestfit_routine = _get_bestfit_routine(bestfit_routine)
    if target is None:
        target = lf.source_list[-1] + '_rate_multiplier'

    # Find the likelihood of the global best fit (denominator of likelihood ratio)
    result, max_loglikelihood = bestfit_routine(lf, **kwargs)
    global_best = result[target]

    def t(hypothesis, critical_quantile):
        """(profile) likelihood ratio test statistic, with critical_value subtracted
        critical_quantile: fraction (percentile/100) of the test statistic distribution you want to find
        """
        if t_ppf is None:
            # Use Wilk's theorem
            # "But I thought I there was a chi2 in Wilk's theorem!" Quite right, but
            # stats.norm.ppf(CL)**2 = stats.chi2(1).ppf(2*CL - 1)
            # So the chi2 formula is often quoted for central CI's, the normal one for bounds...
            # This cost me hours of confusion. Please explain this to your students if you're statistics professor.
            critical_value = stats.norm.ppf(critical_quantile) ** 2
        else:
            # Use user-specified function
            critical_value = t_ppf(hypothesis, critical_quantile)

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


def plot_likelihood_ratio(lf, *space, vmax=15,
                          bestfit_routine=None,
                          plot_kwargs=None, **kwargs):
    """Plots the - loglikelihood ratio derived from LogLikelihood lf in a parameter space
    :param lf: LogLikelihood function with data set.
    :param space: list/tuple of tuples (dimname, points to plot)
    :param vmax: Limit for color bar (2d) or y axis (1d)
    :param plot_kwargs: kwargs passed to plt.plot / plt.pcolormesh
    Further arguments are passed to lf, arguments not passed are fitted at each point.
    :return: Nothing
    """
    import matplotlib.pyplot as plt
    bestfit_routine = _get_bestfit_routine(bestfit_routine)
    if plot_kwargs is None:
        plot_kwargs = {}

    results = []
    label = "-Log likelihood ratio"
    if len(space) == 1:
        dim, x = space[0]
        for q in x:
            lf_kwargs = {dim: q}
            lf_kwargs.update(kwargs)
            results.append(bestfit_routine(lf, **lf_kwargs)[1])
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
                results[-1].append(bestfit_routine(lf, **lf_kwargs)[1])
        z1, z2 = np.meshgrid(x, y)
        results = np.array(results)
        best = np.nanmax(results)
        print(best)
        results = best - results
        plt.pcolormesh(z1, z2, results.T, vmax=vmax, **plot_kwargs)
        plt.colorbar(label=label)
        plt.xlabel(dims[0])
        plt.ylabel(dims[1])
    else:
        raise ValueError("Can't handle %d dimensions" % len(space))


BESTFIT_ROUTINES = dict(scipy=bestfit_scipy, minuit=bestfit_minuit, emcee=bestfit_emcee)
