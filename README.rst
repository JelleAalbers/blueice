wimpy - A WIMP analysis package for xenon based on python and other cool tools
==============================================================================

Source code: `https://github.com/XENON1T/wimpy`

Documentation: nope.

Does it actually run: if you're lucky


About
=====
This package allows generation of toy XENON1T datasets and computation of (profile) likelihood ratio intervals.
Currently it is just a tool I use to study questions like "is photon counting for S1 a good idea?".

Rather than interfacing directly with NEST and the GEANT4 Monte Carlo, it uses effective descriptions of e.g. the light collection efficiency
or the amount of recombination fluctuations.

Many of the 'physics info' stored in the package is shamelessly borrowed from other sources. In particular:
  * Andrew's maximum-gap limit setting code, used for the XENON100 max-gap cross checks
  * Chris' wimpstat repository, used for the XENON100 S2-only limit setting
  * NEST
  * XEPHYR, one of the main XENON100 profile likelihood packages
  * Several of the Monte Carlo group's excellent notes on this topic.

The statistical model used is very similar to the "Bologna model" (unbinned likelihood). 


Installation
============
Make sure you have pax installed: we only need it for the common unit system and XENON1T.ini.

Then run `python setup.py develop` or `python setup.py install`

