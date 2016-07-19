import json
import gzip

import numpy as np
from scipy.interpolate import NearestNDInterpolator

import pax

from . import yields


class Source(object):
    """A source of events in the detector. Takes care of simulating events.
    """
    name = 'unspecified'
    label = 'Catastrophic irreducible noise'
    recoil_type = 'nr'
    color = 'black'
    n_events_for_pdf = 1e6
    rate_uncertainty = 0
    spatial_distribution = 'uniform'

    energy_distribution = None      # Histdd of rate /kg /keV /day.
    events_per_day = 0              # Calculated on init.
    pdf_histogram = None            # Histdd, will be set by Model upon initialization.
    pdf_errors = None               # Histdd, will be set by Model upon initialization.
    fraction_in_range = 0           # Fraction of simulated events that fall in analysis space. Also set by Model.
    analysis_target = False

    def __init__(self, config, spec):
        """
        config: dict with general parameters (e.g. detector geometry, fiducial cut definition.)
        spec: dict with source-specific stuff, e.g. name, energy_distibution, etc. Used to set attributes defined above.
        """
        self.config = c = config
        for k, v in spec.items():       # Store source specs as attributes
            setattr(self, k, v)

        # Compute the integrated event rate (in events / day)
        # This includes all events that produce a recoil; many will probably be out of range of the analysis space.
        h = self.energy_distribution
        if h is None:
            raise ValueError("You need to specify an energy spectrum for the source %s" % self.name)
        self.events_per_day = h.histogram.sum() * self.config['fiducial_mass'] * (h.bin_edges[1] - h.bin_edges[0])

    def pdf(self, *args):
        return self.pdf_histogram.lookup(*args)

    def simulate(self, n_events):
        """Simulate n_events from this source."""
        c = self.config

        # Store everything in a structured array:
        d = np.zeros(n_events, dtype=[('energy', np.float),
                                      ('r2', np.float),
                                      ('theta', np.float),
                                      ('z', np.float),
                                      ('p_photon_detected', np.float),
                                      ('p_electron_detected', np.float),
                                      ('electrons_produced', np.int),
                                      ('photons_produced', np.int),
                                      ('electrons_detected', np.int),
                                      ('s1_photons_detected', np.int),
                                      ('s2_photons_detected', np.int),
                                      ('s1_photoelectrons_produced', np.int),
                                      ('s2_photoelectrons_produced', np.int),
                                      ('s1', np.float),
                                      ('s2', np.float),
                                      ('cs1', np.float),
                                      ('cs2', np.float),
                                      ('source', np.int),  # Not set in this function
                                      ('magic_cs1', np.float)])

        # If we get asked to simulate 0 events, return the empty array immediately
        if not len(d):
            return d

        d['energy'] = self.energy_distribution.get_random(n_events)

        # Sample the positions and relative light yields
        if self.spatial_distribution == 'uniform':
            d['r2'] = np.random.uniform(0, c['fiducial_volume_radius']**2, n_events)
            d['theta'] = np.random.uniform(0, 2*np.pi, n_events)
            d['z'] = np.random.uniform(c['ficudial_volume_zmin'], c['ficudial_volume_zmax'], size=n_events)
            rel_lys = self.config['s1_relative_ly_map'].lookup(d['r2'], d['z'])
        else:
            raise NotImplementedError("Only uniform sources supported for now...")

        # Get the light & charge collection efficiency
        d['p_photon_detected'] = c['ph_detection_efficiency'] * rel_lys
        d['p_electron_detected'] = np.exp(d['z'] / c['v_drift']/ c['e_lifetime'])   # No minus: z is negative

        # Get the mean number of "base quanta" produced
        n_quanta = self.config['base_quanta_yield'] * d['energy']
        n_quanta = np.random.normal(n_quanta,
                                    np.sqrt(self.config['base_quanta_fano_factor'] * n_quanta),
                                    size=n_events)

        # 0 or negative numbers of quanta give trouble with the later formulas.
        # Store which events are bad, set them to 1 quanta for now, then zero them later.
        bad_events = n_quanta < 1
        n_quanta = np.clip(n_quanta, 1, float('inf'))

        p_becomes_photon = \
            d['energy'] * getattr(yields, self.recoil_type + '_photon_yield')(self.config, d['energy']) / n_quanta

        if self.recoil_type == 'er':
            # No quanta get lost as heat:
            p_becomes_electron = 1 - p_becomes_photon

            # Apply extra recombination fluctuation (NEST tritium paper / Atilla Dobii's thesis)
            p_becomes_electron = np.random.normal(p_becomes_electron,
                                                  p_becomes_electron * self.config['recombination_fluctuation'],
                                                  size=n_events)
            p_becomes_electron = np.clip(p_becomes_electron, 0, 1)
            p_becomes_photon = 1 - p_becomes_electron
            n_quanta = np.round(n_quanta).astype(np.int)

        elif self.recoil_type == 'nr':
            # For NR some quanta get lost in heat.
            # Remove them and rescale the p's so we can use the same code as for ERs after this.
            p_becomes_electron = \
                d['energy'] * getattr(yields, self.recoil_type + '_electron_yield')(self.config, d['energy']) / n_quanta
            p_becomes_detectable = p_becomes_photon + p_becomes_electron
            if p_becomes_detectable.max() > 1:
                raise ValueError("p_detected max is %s??!" % p_becomes_detectable.max())
            p_becomes_photon /= p_becomes_detectable
            n_quanta = np.round(n_quanta).astype(np.int)
            n_quanta = np.random.binomial(n_quanta, p_becomes_detectable)

        else:
            raise ValueError('Bad recoil type %s' % self.recoil_type)

        d['photons_produced'] = np.random.binomial(n_quanta, p_becomes_photon)
        d['electrons_produced'] = n_quanta - d['photons_produced']

        # "Remove" bad events (see above); actual removal happens at the very end of the function
        d['photons_produced'][bad_events] = 0
        d['electrons_produced'][bad_events] = 0

        # Detection efficiency
        d['s1_photons_detected'] = np.random.binomial(d['photons_produced'], d['p_photon_detected'])
        d['electrons_detected'] = np.random.binomial(d['electrons_produced'], d['p_electron_detected'])

        # S2 amplification
        d['s2_photons_detected'] = np.random.poisson(d['electrons_detected'] * c['s2_gain'])

        # PMT response
        for si in ('s1', 's2'):
            # Convert photons to photoelectrons, taking double photoelectron emission into account
            d[si + '_photoelectrons_produced'] = d[si + '_photons_detected'] + \
                                                   np.random.binomial(d[si + '_photons_detected'],
                                                                      c['double_pe_emission_probability'])

            # Convert photoelectrons (in reality) to measured pe
            d[si] = np.random.normal(d[si + '_photoelectrons_produced'],
                                     np.clip(c['pmt_gain_width'] * np.sqrt(d[si + '_photoelectrons_produced']),
                                             1e-9,   # Normal freaks out if sigma is 0...
                                             float('inf')))

        # Get the corrected S1 and S2, assuming our posrec + correction map is perfect
        # Note these definitions don't just correct, they also scale back to the number of quanta!
        d['cs1'] = d['s1'] / d['p_photon_detected']
        d['cs2'] = d['s2'] / d['p_electron_detected'] / c['s2_gain']

        # Assuming we know the total number of photons detected (perfect hit counting),
        # give the ML estimate of the number of photons produced.
        d['magic_cs1'] = d['s1_photons_detected'] / d['p_photon_detected']

        # Remove events without an S1 or S1
        if self.config['require_s1']:
            # One photons detected doesn't count as an S1 (since it isn't distinguishable from a dark count)
            d = d[d['s1_photons_detected'] >= 2]
            d = d[d['s1'] > c['s1_area_threshold']]

        if self.config['require_s2']:
            d = d[d['electrons_detected'] >= 1]
            d = d[d['s2'] > c['s2_area_threshold']]

        return d
