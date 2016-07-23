"""
This file contains code necessary to create WIMP energy spectra.
Use the wimp_recoil_spectrum function defined at the end, which gives the wimp spectrum (in rate /kg /day /keV)
for Xenon detectors using the standard Halo model.

Almost all of this is "borrowed" from:

  * Andrew's maximum gap limit setting code available here: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon100:andrew:code
    This is the code used for the XENON100 main WIMP analysis maximum-gap cross checks.

  * Chris' WIMPStat repository, available here: https://github.com/tunnell/wimpstat
    This is the code used for XENON100's S2-only analysis.

Ultimately I believe most of this derives from the Lewin&Smith paper.
"""
import numpy as np

# Global variables, needed in several functions below
A = 131.293   # amu, mass of Xenon
mnucl = 0.931494    # Mass of a nucleon, in GeV/c^2
speed_of_light = 2.99792458e8  # m/s, light

##
# Velocity function
##

from scipy.special import erf
def ErfMinusErf(x, y):
    retval = np.sqrt(np.pi)/2
    retval *= (erf(y)-erf(x))
    return retval

def GetI(erec, Mchi, mred, Mnucleus):
    """Gets some integral of some velocity function -- TODO: figure out what this really does
    TODO: Why is Mchi not used???
    erec - recoil energy (keV)
    Mchi - DM mass (GeV)
    mred - reduced mass (GeV)
    Mnucleus - mass of target nucleus (I think; GeV?)
    """
    vsun = 232.0 # Solar velocity in km/s
    vinf = 220.0 # Asymptotic velocity of local system in km/s
    vesc = 544.0 # Galactic escape velocity In km/s

    neta = vsun / vinf # Unitless
    z = vesc / vinf # Unitless

    # sqrt(keV * GeV * (km/s) * (km/s) / GeV**2 / (km/s))
    # = sqrt(keV * (km/s) / GeV) -> so 1e-6
    xmin =  np.sqrt(erec * Mnucleus * speed_of_light * speed_of_light * 1e-12/ (2*mred*mred)) / vinf

    norm = 1.0/(erf(z)-(2/np.sqrt(np.pi))*z*np.exp(-z*z))

    retval = (norm/neta)/(np.sqrt(np.pi) * vinf)

    if xmin < (z-neta):
        retval *= ErfMinusErf(xmin-neta,xmin+neta)-2*neta*np.exp(-z*z)
    if ((z-neta)<=xmin and xmin < (z+neta)):
        retval *= ErfMinusErf(xmin-neta,z)-np.exp(-z*z)*(z+neta-xmin)
    if (xmin>=(z+neta)):
        retval = 0
    return retval


##
# Form factors
##
@np.vectorize
def engel_form_factor(E_recoil, A=A):
    """Return Engel form factor at recoil energy in keV
    Code from a neutrino notebook from Chris; it was a neutrino notebook, so it uses GeV internally.
    # http://arxiv.org/pdf/1202.6073v2.pdf
    # J. Engel, Phys.Lett. B264, 114 (1991).
    """
    E_recoil *= 1e-6   # Convert to GeV

    conversion = 1.97e-14 # GeV * cm, from 197 MeV * femtometer to GeV * cm
    m_nucleus = 122.29865 # GeV, from 131.293 amu to GeV/c^2

    R_0 = 1.14e-13 * pow(A, 1/3) / conversion  # [GeV^-1] Size of nucleus

    k = np.sqrt(2 * E_recoil * m_nucleus)  # GeV
    s = 1 / 0.197  # GeV^-1, from 1 fm

    # R = np.sqrt(1.2 * A / 3)

    r = np.sqrt(R_0**2 - 5 * s**2)
    kr = k * r

    FF = 3 * np.exp(-1 * k**2 * s**2 / 2)
    FF *= (np.sin(kr) - kr * np.cos(kr)) / kr**3

    return FF


def spherical_bessel_j1(x):
    """Spherical Bessel function j1 according to Wolfram Alpha"""
    return np.sin(x)/x**2 + - np.cos(x)/x


@np.vectorize
def helm_form_factor_squared(en, anucl=A):
    """Return Helm form factor squared from Lewin & Smith
    Lifted from Andrew's max gap code with minor edits
    en = nuclear recoild energy in keV (according to in-code comments)
    """
    if anucl <= 0:
        raise ValueError("Invalid value of A!")

    # First we get rn sqared, in fm
    pi = np.pi
    c = 1.23*anucl**(1/3)-0.60
    a = 0.52
    s = 0.9
    rn_sq = c**2+(7.0/3.0)*pi**2*a**2-5*s**2
    rn = np.sqrt(rn_sq)  # units fm
    mass_kev = anucl * mnucl * 1e6
    hbarc_kevfm = 197327  # hbar * c in keV *fm (from Wolfram alpha)

    # E in units keV, rn in units fm, hbarc_kev units keV.fm
    # Formula is spherical bessel fn of Q=sqrt(E*2*Mn_keV)*rn
    q = np.sqrt(en*2.*mass_kev)
    qrn_over_hbarc = q*rn/hbarc_kevfm
    sph_bess = spherical_bessel_j1(qrn_over_hbarc)
    retval = 9. * sph_bess * sph_bess / (qrn_over_hbarc*qrn_over_hbarc)
    qs_over_hbarc = q*s/hbarc_kevfm
    retval *= np.exp(-qs_over_hbarc*qs_over_hbarc)
    return retval


##
# WIMP recoil spectrum
##

def _wimp_recoil_spectrum(energies, mass=100, sigma=2e-47, form_factor='Helm', A=A):
    """Return differential WIMP rate (dR/dE) in events /kg /day /keV
        energies - Recoil energies, i.e. "x-axis" of spectrum (keV)
        mass - Mass of WIMP (GeV)
        sigma - cross section nucleon (cm^-2)
        A - Target nucleus mass in amu
        form_factor - None, 'Helm' (default), or 'Engel'
    """
    erec = energies
    Mchi = mass

    rho = 0.3 # in GeV cm^-3
    avogadro_number = 6.0221415 * 10**26 # 1/kg

    mprot = 0.938272046 # mass of proton, GeV/c^2
     # amu
    Mnucleon = 0.931494    # Mass of a nucleon, in Gev/c^2

    # Helpers
    Mnucleus = A * Mnucleon # GeV

    Nt = avogadro_number / A # #, Number of target nuclei per unit of max

    #Returns the per nucleon scale factor for the know masses
    #Again following Lewin and Smith 1996
    sigma /= ((1 + Mchi/Mnucleus)/(1 + Mchi/mprot))**2
    sigma *= A**2

    mred = (Mnucleus * Mchi) / (Mnucleus + Mchi)  # Reduced mass

    if not form_factor:
        # Ignore form factor squared, true only for low mass analysis!
        F2 = 1
    elif form_factor == 'Helm':
        F2 = helm_form_factor_squared(erec)
    elif form_factor == 'Engel':
        F2 = engel_form_factor(erec)**2
    else:
        raise ValueError("Unknown form factor %s" % form_factor)

    I = 1e-5 * GetI(erec, Mchi, mred, Mnucleus) # Integrate velocity dist, then km -> cm

    # Notice the Mn/med**2 in the returned formula? Convert to 1/keV
    scale = 1e-6  # 1e-6 takes 1/GeV
    scale *= speed_of_light**2 * 1e4 # c^2 in cm/s

    scale *= 60 * 60 * 24 # formula in paper yields days, want seconds

    return Nt * (rho/Mchi) * ((Mnucleus * sigma)/(2 * mred**2)) * F2 * I * scale


wimp_recoil_spectrum = np.vectorize(_wimp_recoil_spectrum, excluded=['mass', 'sigma', 'form_factor', 'A'])

def sample_wimp_energies(size=10, energies=None, **kwargs):
    """Sample energies from wimp spectrum
        energies: controls bins of drde histogram to sample from (default 100 bins from 1 to 100 keV)
                  default 991 bins from 1 to 100 keV (i.e bins of 0.1 keV)
    Other arguments (mass, sigma, form_factor, A) are passed to wimp_recoil_spectrum, see its docstring
    """
    if energies is None:
        energies = np.linspace(1, 100, 99 * 10 + 1)
    drde = wimp_recoil_spectrum(energies)
    drde /= drde.sum()
    return np.random.choice(energies, size=size, replace=True, p=drde)
