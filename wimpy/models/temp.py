"""
A XENON1T model

This is a bit off a mess because I don't yet know how to make a nice interface for specifying this.
Maybe INI files or something...
"""
import numpy as np
from multihist import Hist1d

from pax import units
from pax.configuration import load_configuration
pax_config = load_configuration('XENON1T')

energy_bins = np.linspace(1e-7, 100, 10000)      # Bin edges of energies to consider
def make_e_hist(rates, energy_bins=energy_bins):
    assert len(rates) == len(energy_bins) - 1    # Bin centers stuff...
    h = Hist1d(bins=energy_bins)
    h.histogram = rates
    return h
es = Hist1d(bins=energy_bins).bin_centers        # Centers of the energy bins defined above

# Wimp energy spectrum
reference_wimp_cross_section = 1e-45
wimp_mass = 50
from wimpy.wimps import wimp_recoil_spectrum
wimp_sources = [dict(energy_distribution=make_e_hist(wimp_recoil_spectrum(es,
                                                                          mass=wimp_mass,
                                                                          sigma=reference_wimp_cross_section)),
                     color='red',
                     name='wimp_%dgev' % wimp_mass,
                     n_events_for_pdf=5e6,
                     analysis_target=True,
                     recoil_type='nr',
                     label='%d GeV WIMP' % wimp_mass) for wimp_mass in [50, 6, 10, 20, 100, 1000]]

# Uniform ER Background at 2e-4 /day/kg/keV
# From figure 5, right of the MC paper
# The ER background doesn't need to be computed to very high energy, since they generate way more quanta / energy
# so only the low-energy ER background interferes with WIMP searches
n_e_er = int(len(energy_bins) * 0.15)
er_bg = make_e_hist(np.ones(n_e_er - 1) * 2e-4, energy_bins[:n_e_er])

# NR background
# Roughtly curve-traced from fig 8 of the MC paper,
# radiogenic neutrons are not spatially uniform, but for now we'll sample them as such anyway
# TODO: Properly curve trace the NR background!
cnns_bg = make_e_hist(4e-3 * 10**(-es*9/7))
a = 6e-8 * 10**(-es*1/100)
b = 3e-6 * 1/(2+es**2)
neutron_bg = make_e_hist(a + b)

backgrounds = [
    {'energy_distribution': er_bg,
     'color': 'blue',
     'recoil_type': 'er',
     'name': 'er_bg',
     'n_events_for_pdf': 2e7,
     'label': 'ER Background'},
    {'energy_distribution': cnns_bg,
     'color': 'orange',
     'recoil_type': 'nr',
     'name': 'cnns',
     'n_events_for_pdf': 5e6,
     'label': 'CNNS'},
    {'energy_distribution': neutron_bg,
     'color': 'purple',
     'recoil_type': 'nr',
     'name': 'radiogenics',
     'n_events_for_pdf': 5e6,
     'label': 'Radiogenic neutrons'},
]

config=dict(
    # Basic model info
    analysis_space=[('cs1', np.linspace(0, 500, 100)),
                    ('cs2',  np.linspace(0, 300, 100))],
    sources=backgrounds + [wimp_sources[0]],
    dormant_sources=wimp_sources[1:],
    livetime_days=2*365.25,
    require_s1 = True,
    require_s2 = True,
    force_pdf_recalculation = False,
    pdf_sampling_batch_size = int(1e6),

    # Detector parameters
    fiducial_mass = 1000, #kg. np.pi * rmax**2 * (zmax - zmin) * density?
    e_lifetime=1*units.ms,
    v_drift=1.5*units.km/units.s,
    s2_gain=26,
    ph_detection_efficiency=0.118,
    drift_field = 500 * units.V / units.cm,
    pmt_gain_width=0.5,    # Width (in photoelectrons) of the single-photoelectron area spectrum
    double_pe_emission_probability=0.12,   # Probability for a photon detected by a PMT to produce two photoelectrons.

    # For sampling of light and charge yield in space
    n_location_samples = int(1e5),          # Number of samples to take for the source positions (for light yield etc, temporary?)
    fiducial_volume_radius = pax_config['DEFAULT']['tpc_radius'] * 0.9,
    # Note z is negative, so the maximum z is actually the z of the top boundary of the fiducial volume
    ficudial_volume_zmax = - 0.05 * pax_config['DEFAULT']['tpc_length'],
    ficudial_volume_zmin = - 0.95 * pax_config['DEFAULT']['tpc_length'],
    s1_relative_ly_map_filename = pax_config['WaveformSimulator']['s1_light_yield_map'],

    # S1/S2 generation parameters
    nr_electron_yield_cutoff_energy = 1,  # keV.
    nr_electron_yield_behaviour_below_cutoff = 'const',   # 'const' or 'zero'. Be careful with the latter.
    nr_photon_yield_field_quenching = 0.95,      # Monte Carlo note: add ref!
    reference_gamma_photon_yield = 63.4,         # NEST For 122... keV gamma, from MC note (add ref!)
    base_quanta_yield = 73,              # NEST's basic quanta yield, xenon:xenon1t:sim:notes:marco:conversion-ed-to-s1-s2
    # Fano factor for smearing of the base quanta yield
    # xenon:xenon1t:sim:notes:marco:conversion-ed-to-s1-s2 and xenon:xenon1t:sim:notes:marco:t2-script-description,
    # ultimately taken from the NEST code
    base_quanta_fano_factor=0.03,
    # Recombination fluctuation, from LUX tritium paper (p.9) / Atilla Dobii's thesis
    # If I don't misunderstand, they report an extra sigma/mu on the probability of a quantum to end up as an electron.
    recombination_fluctuation=0.067,
)