import numpy as np

from wimpy.source import MonteCarloSource
from wimpy.utils import InterpolateAndExtrapolate1D

from . import yields

class XENONSource(MonteCarloSource):
    """A Source in a XENON-style experiment"""
    spatial_distribution = 'uniform'
    recoil_type = 'nr'
    energy_distribution = None      # Histdd of rate /kg /keV /day.

    def setup(self):
        # Compute the integrated event rate (in events / day)
        # This includes all events that produce a recoil; many will probably be out of range of the analysis space.
        h = self.energy_distribution
        if h is None:
            raise ValueError("You need to specify an energy spectrum for the source %s" % self.name)
        self.events_per_day = h.histogram.sum() * self.model.config['fiducial_mass'] * (h.bin_edges[1] - h.bin_edges[0])

        # The yield functions are all interpolated in log10(energy) space,
        # Since that's where they are usually plotted in... and curve traced from.
        # The yield points are clipped to 0... a few negative values slipped in while curve tracing, and
        # I'm not going to edit all the leff files to remove them.
        self.yield_functions = {k: InterpolateAndExtrapolate1D(np.log10(self.model.config[k][0]),
                                                               np.clip(self.model.config[k][1], 0, float('inf')))
                                for k in ('leff', 'qy', 'er_photon_yield')}

    def yield_at(self, energies, recoil_type, quantum_type):
        """Return the yield in quanta/kev for the given energies (numpy array, in keV),
        recoil type (string, 'er' or 'nr') and quantum type (string, 'photon' or 'electron')"""
        c = self.model.config
        log10e = np.log10(energies)
        if quantum_type not in ('electron', 'photon'):
            raise ValueError("Invalid quantum type %s" % quantum_type)

        if recoil_type == 'er':
            """
            In NEST the electronic recoil yield is calculated separately for each event,
            based on details of the GEANT4 track structure (in particular the linear energy transfer).
            Here I use an approximation, which is the "old approach" from the MC group, see
                xenon:xenon1t:sim:notes:marco:t2-script-description#generation_of_light_and_charge
            A fixed number of quanta (base_quanta_yield) is assumed to be generated. We get the photon yield in quanta,
            then assume the rest turns into electrons.
            """
            if quantum_type == 'photon':
                return self.yield_functions['er_photon_yield'](log10e)
            else:
                return c['base_quanta_yield'] - self.yield_functions['er_photon_yield'](log10e)

        elif recoil_type == 'nr':
            """
            The NR electron yield is called Qy.
            It is here assumed to be field-independent (but NEST 2013 fig 2 shows this is wrong...).

            The NR photon yield is described by several empirical factors:
                reference_gamma_photon_yield * efield_light_quenching_nr * leff

            The first is just a reference scale, the second contains the electric field dependence,
            the third (leff) the energy dependence.
            In the future we may want to simplify this to just a single function
            """
            if quantum_type == 'photon':
                return self.yield_functions['leff'](log10e) * \
                       c['reference_gamma_photon_yield'] * c['nr_photon_yield_field_quenching']
            else:
                return self.yield_functions['qy'](log10e)

        else:
            raise RuntimeError("invalid recoil type %s" % recoil_type)


    def simulate(self, n_events):
        """Simulate n_events from this source."""
        c = self.model.config

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
            rel_lys = c['s1_relative_ly_map'].lookup(d['r2'], d['z'])
        else:
            raise NotImplementedError("Only uniform sources supported for now...")

        # Get the light & charge collection efficiency
        d['p_photon_detected'] = c['ph_detection_efficiency'] * rel_lys
        d['p_electron_detected'] = np.exp(d['z'] / c['v_drift']/ c['e_lifetime'])   # No minus: z is negative

        # Get the mean number of "base quanta" produced
        n_quanta = c['base_quanta_yield'] * d['energy']
        n_quanta = np.random.normal(n_quanta,
                                    np.sqrt(c['base_quanta_fano_factor'] * n_quanta),
                                    size=n_events)

        # 0 or negative numbers of quanta give trouble with the later formulas.
        # Store which events are bad, set them to 1 quanta for now, then zero them later.
        bad_events = n_quanta < 1
        n_quanta = np.clip(n_quanta, 1, float('inf'))

        p_becomes_photon = d['energy'] * self.yield_at(d['energy'], self.recoil_type, 'photon') / n_quanta
        p_becomes_electron = d['energy'] * self.yield_at(d['energy'], self.recoil_type, 'electron') / n_quanta

        if self.recoil_type == 'er':
            # Apply extra recombination fluctuation (NEST tritium paper / Atilla Dobii's thesis)
            p_becomes_electron = np.random.normal(p_becomes_electron,
                                                  p_becomes_electron * c['recombination_fluctuation'],
                                                  size=n_events)
            p_becomes_electron = np.clip(p_becomes_electron, 0, 1)
            p_becomes_photon = 1 - p_becomes_electron
            n_quanta = np.round(n_quanta).astype(np.int)

        elif self.recoil_type == 'nr':
            # For NR some quanta get lost in heat.
            # Remove them and rescale the p's so we can use the same code as for ERs after this.
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

            # Convert photoelectrons to measured pe
            d[si] = np.random.normal(d[si + '_photoelectrons_produced'],
                                     np.clip(c['pmt_gain_width'] * np.sqrt(d[si + '_photoelectrons_produced']),
                                             1e-9,   # Normal freaks out if sigma is 0...
                                             float('inf')))

        # Get the corrected S1 and S2, assuming our posrec + correction map is perfect
        # Note this does NOT assume the analyst knows the absolute photon detection efficiency:
        # photon detection efficiency / p_photon_detected is just the relative light yield at the position.
        # p_electron_detected is known exactly (since it only depends on the electron lifetime)
        s1_correction = c['ph_detection_efficiency'] / d['p_photon_detected']
        d['cs1'] = d['s1'] * s1_correction
        d['cs2'] = d['s2'] / d['p_electron_detected']

        # Assuming we know the total number of photons detected (perfect hit counting),
        # give the ML estimate of the number of photons produced.
        d['magic_cs1'] = d['s1_photons_detected'] * s1_correction

        # Remove events without an S1 or S1
        if c['require_s1']:
            # One photons detected doesn't count as an S1 (since it isn't distinguishable from a dark count)
            d = d[d['s1_photons_detected'] >= 2]
            d = d[d['s1'] > c['s1_area_threshold']]

        if c['require_s2']:
            d = d[d['electrons_detected'] >= 1]
            d = d[d['s2'] > c['s2_area_threshold']]

        return d

