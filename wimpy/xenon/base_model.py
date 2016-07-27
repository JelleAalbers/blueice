"""
A XENON1T model

This is a bit off a mess because I don't yet know how to make a nice interface for specifying this.
Maybe INI files or something...
"""
import os
import inspect
import numpy as np
from multihist import Hist1d

from pax import units
from pax.configuration import load_configuration
pax_config = load_configuration('XENON1T')

from .XENONSource import XENONSource

# Store the directory of this file
THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


nr_ignore_settings = ['er_photon_yield', 'recombination_fluctuation']   # Ignore these if you're an NR source
er_ignore_settings = ['leff', 'qy', 'nr_photon_yield_field_quenching']  # Ignore these if you're an ER source

config = dict(
    data_dirs = [os.path.join(THIS_DIR, 'data'), '.'],
    default_source_class = XENONSource,
    analysis_space= (('cs1', tuple(np.linspace(0, 70, 70))),
                     ('cs2', tuple(np.linspace(0, 7000, 70)))),
    livetime_days=2*365.25,
    require_s1 = True,
    require_s2 = True,
    force_pdf_recalculation = False,
    pdf_sampling_multiplier = 1,
    pdf_sampling_batch_size = int(1e6),
    # Basic model info
    sources = [
        {'energy_distribution': 'uniform_er_bg.pkl',
         'color': 'blue',
         'recoil_type': 'er',
         'name': 'er_bg',
         'n_events_for_pdf': 2e7,
         'ignore_settings': er_ignore_settings,
         'label': 'ER Background'},
        {'energy_distribution': 'cnns.pkl',
         'color': 'orange',
         'recoil_type': 'nr',
         'name': 'cnns',
         'n_events_for_pdf': 5e6,
         'ignore_settings': nr_ignore_settings,
         'label': 'CNNS'},
        {'energy_distribution': 'radiogenic_neutrons.pkl',
         'color': 'purple',
         'recoil_type': 'nr',
         'name': 'radiogenics',
         'n_events_for_pdf': 5e6,
         'ignore_settings': nr_ignore_settings,
         'label': 'Radiogenic neutrons'},
        {'energy_distribution': 'wimp_50gev_1e-45.pkl',
         'color': 'red',
         'name': 'wimp_50gev',
         'n_events_for_pdf': 5e6,
         'analysis_target': True,
         'recoil_type': 'nr',
         'ignore_settings': nr_ignore_settings,
         'label': '50 GeV WIMP'}
    ],

    # Thresholds on uncorrected S1/S2: for comparison with the Bologna model at low WIMP masses
    s1_area_threshold = 3,
    s2_area_threshold = 150,

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
    s1_relative_ly_map = 's1_rel_ly_pax5.1.pkl',

    # S1/S2 generation parameters
    leff='leff_mcpaper_0.csv',
    qy='qy_bezrukov.csv',
    er_photon_yield='beta_photon_yield_nest_500V.csv',
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

