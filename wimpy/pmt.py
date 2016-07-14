import numpy as np
from numba import jit


# p_twope_emission is the probability for a photon which is detected by the PMT to emit/have emitted two photoelectrons.
# From Erik's note xenon:xenon1t:analysis:meetings:20151130:single_e_gain_calibration, references Lux paper
# http://arxiv.org/pdf/1506.08748.pdf
p_twope_emission = 0.25
minimum_hit_area = 0
photon_sample_size = int(1e6)

@jit(nopython=True)
def sampling_sum(x, ns, threshold=0):
    """Return sum of ns consecutive elements of x.
    For example if ns = [3, 1, 3], will return:
        np.array([x[0:3].sum(), x[3], x[4:7].sum()])
    """
    n_signals = len(ns)
    n_tot = ns.sum()
    if len(x) < n_tot:
        # print("len(x) = %s, n_tot = %s" % (len(x), n_tot))
        raise ValueError
    signals = np.zeros(n_signals)
    signal_i = 0
    n_to_go = ns[0]

    # Find first signal above threshold
    # TODO: ugly code duplication!
    while n_to_go < threshold:
        signals[signal_i] = -1
        signal_i += 1
        if signal_i == n_signals:
            return signals
        n_to_go = ns[signal_i]

    for i, q in enumerate(x):
        while n_to_go == 0:
            signal_i += 1
            if signal_i == n_signals:
                return signals
            n_to_go = ns[signal_i]
            # Don't even start on signals below threshold
            if n_to_go < threshold:
                n_to_go = 0
                signals[signal_i] = -1
        signals[signal_i] += q
        n_to_go -= 1


def pmt_response(n_photons, gaussian_approximation_from=1e4):
    """Return PMT response (in pe) to arrival of n_photons (which can be array)
    For signals larger than gaussian_approximation_from photons or photon_sample_size, instead draws from Gaussian with
        mu = mean 1ph response
        sigma = 1ph sigma * np.sqrt(n)
    (i.e. uses central limit theorem).
    Maintains an internal cache of single photon areas, which is rolled (not shuffled) each time you call it.
    """
    # Do we need to calculate the photon areas?
    # Otherwise, use cached values (stored as function attribute)
    current_settings = dict(p_twope_emission=p_twope_emission,
                            minimum_hit_area=minimum_hit_area,
                            photon_sample_size=photon_sample_size)
    if hasattr(pmt_response, 'area_ph_settings') and current_settings == pmt_response.area_ph_settings:
        area_ph = np.roll(pmt_response.photon_areas,
                          np.random.randint(0, len(pmt_response.photon_areas) - 1))
    else:
        # Rebuild the cache
        # Single pes
        area_1pe = np.random.normal(1, 0.5, size=photon_sample_size)
        area_1pe = area_1pe[area_1pe >= minimum_hit_area]
        area_1pe = area_1pe[:2 * int(len(area_1pe) / 2)]
        # Double pes
        area_2pe = area_1pe.reshape((-1, 2)).sum(axis=1)
        # Photons
        area_ph = np.concatenate((np.random.choice(area_1pe, size=int(photon_sample_size * (1-p_twope_emission))),
                                  np.random.choice(area_2pe, size=int(photon_sample_size * p_twope_emission))))
        np.random.shuffle(area_ph)
        pmt_response.photon_areas = area_ph
        pmt_response.area_ph_settings = current_settings

    # Support scalar argument
    if isinstance(n_photons, (int, float)):
        n_photons = np.array([n_photons])
    results = np.zeros(len(n_photons))

    # Do the Gaussian approximation for large signals
    gaussian_approximation_from = min(gaussian_approximation_from, len(area_ph))
    use_gauss = n_photons > gaussian_approximation_from
    if len(use_gauss):
        results[use_gauss] = np.random.normal(loc=area_ph.mean() * n_photons[use_gauss],
                                              scale=area_ph.std() * np.sqrt(n_photons[use_gauss]))
    n_photons_todo = n_photons[True ^ use_gauss]

    # Combine individual photons for small signal
    if n_photons_todo.sum() >= len(area_ph):
        # Divide in two, then combine...
        halfway = int(len(n_photons_todo)/2)
        results[True ^ use_gauss] = np.concatenate((pmt_response(n_photons_todo[:halfway]),
                                                    pmt_response(n_photons_todo[halfway:])))
    else:
        results[True ^ use_gauss] = sampling_sum(area_ph, n_photons_todo)
    return results
