"""Functions which analyze model + data to set limits on model parameters.
These could as well have been methods of Model... but
  - It would become a rather large class!
  - I don't want to have to re-make a large model after a small change in these methods.
  - I want to combine models to get systematic uncertainties in.
"""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy import stats


def extended_loglikelihood(mu, ps):
    """Evaluate an extended likelihood function
    :param mu: array of n_sources: expected number of events
    :param ps: array of (n_sources, n_events): pdf value for each source and event
    :return: loglikelihood
    """
    return -mu.sum() + np.sum(np.log(np.sum(mu[:, np.newaxis] * ps, axis=0)))


def loglikelihood(m, d, wimp_strength, ps=None, rate_modifiers=None):
    """Gives the log-likelihood of the dataset d under the model m
    under the hypothesis that the analysis target source's rate is multiplied by 10**strength.
    rate_modifiers: array of rate adjustments for each source.
                    Rate will be increaed by source.rate * source.rate_uncertainty * rate_modifier events/day,
                    Penalty term in likelihood is normal(0, 1).logpdf(rate_modifier)
    NB: Assumes d is already restricted to analysis space. If not, you will get into trouble!!
    """
    if rate_modifiers is None:
        rate_modifiers = np.zeros(len(m.sources))
    else:
        rate_modifiers = np.asarray(rate_modifiers)
    mu = np.array([m.expected_events(s) for s in m.sources])
    mu *= 1 + rate_modifiers * np.asarray([s.rate_uncertainty for s in m.sources])
    mu[-1] *= 10**wimp_strength
    mu = np.clip(mu, 0, float('inf'))

    if ps is None:
        ps = m.score_events(d)

    result = extended_loglikelihood(mu, ps)
    result += stats.norm.logpdf(rate_modifiers).sum()  # - len(m.sources) * stats.norm.logpdf(0)
    return result



def bestfit(m, d, guess_strength=0, fit_uncertainties=False, fit_strength=True, ps=None):
    """Returns best-fit wimp strength for dataset d, loglikelihood value at that point.
    Throws exception if optimization is unsuccesfull
    if float_uncertainties = True, fits and returns the rate uncertainties for each source
    ps = m.score_events(d), if you've already computed it.
    """
    if ps is None:
        ps = m.score_events(d)

    def add_zeros_if_not_uncertain(_rate_mods_temp):
        # Get rate modifiers from params: fold in zeros for sources without rate uncertainties
        _rate_mods_temp = _rate_mods_temp.tolist()
        return [_rate_mods_temp.pop(0) if s.rate_uncertainty else 0 for s in m.sources]

    n_uncertain_sources = len([s for s in m.sources if s.rate_uncertainty != 0])

    # Determine what kind of minimization we should do.
    if fit_uncertainties and fit_strength:
        guess = [guess_strength] + [0] * n_uncertain_sources

        def objective(params):
            return -loglikelihood(m, d, params[0], ps=ps,
                                  rate_modifiers=add_zeros_if_not_uncertain(params[1:]))


    elif fit_uncertainties:
        guess = [0] * n_uncertain_sources

        def objective(params):
            return -loglikelihood(m, d, guess_strength, ps=ps,
                                  rate_modifiers=add_zeros_if_not_uncertain(params))

    elif fit_strength:
        guess = [guess_strength]

        def objective(wimp_strength):
            return -loglikelihood(m, d, wimp_strength, ps=ps)

    else:
        # Fit nothing: just calculate!
        return guess_strength, loglikelihood(m, d, guess_strength, ps=ps)

    # For some reason the default BGFS minimization fails hard... Powell does much better
    optresult = minimize(objective, guess, method='Powell')
    if not optresult.success:
        raise RuntimeError("Optimization failure: ", optresult)
    return optresult.x, -optresult.fun


def interval(m, d, confidence_level=0.9, kind='limit', profile=False):
    """Return an interval of kind and confidence_level on dataset d
     - d is assumed to be restricted to the analysis range.
     - confidence level is the probability content of the interval (1 - the false positive rate)
     - kind can be 'central' (for a two-sided central CI) or 'limit' (for upper limits)
     - If profile=True, will compute a profile likelihood interval, i.e. will consider the rate uncertainty on each
       source.

    The interval is set using the likelihood ratio method, assuming the asymptotic distribution (Wilk's theorem)
    for the likelihood ratio test statistic.
    """
    ps = m.score_events(d)

    # Find the best-fit wimp strength
    optresult, max_likelihood = bestfit(m, d, ps=ps, fit_uncertainties=profile, fit_strength=True)
    best_strength = optresult[0] if profile else optresult.item()

    def f(wimp_strength, z):
        fit, conditional_max_ll = bestfit(m, d, wimp_strength, fit_strength=False, fit_uncertainties=profile)
        return 2*(max_likelihood - conditional_max_ll) - z**2

    if kind == 'limit':
        return brentq(f,
                      best_strength, m.max_wimp_strength,
                      args=(stats.norm(0, 1).ppf(confidence_level),))

    elif kind == 'central':
        if best_strength <= m.no_wimp_strength:
            a = m.no_wimp_strength
        else:
            a = brentq(f,
                       m.no_wimp_strength, best_strength,
                       args=(stats.norm(0, 1).ppf(1/2 - confidence_level/2),))
        b = brentq(f,
                   best_strength, m.max_wimp_strength,
                   args=(stats.norm(0, 1).ppf(1/2 + confidence_level/2),))
        return a, b

    else:
        raise ValueError("Invalid interval kind %s" % kind)
