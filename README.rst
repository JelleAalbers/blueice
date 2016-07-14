wimpy - A WIMP analysis package for XENON based on python and other cool tools
==============================================================================
Jelle Aalbers, 2016

Source code: `https://github.com/XENON1T/wimpy`

Documentation: nope. Well, you can check the docstrings and code comments...

Does it actually run: if you're lucky


About
=====
This package allows generation of toy XENON1T datasets and computation of (profile) likelihood ratio intervals and limits. Use of arbitrary analysis dimensions should be supported -- although only 2d spaces have been tested so far.
Rather than interfacing directly with NEST and the GEANT4 Monte Carlo, it uses effective descriptions of e.g. the light collection efficiency
or the amount of recombination fluctuations.

Currently this is just a tool I use to study questions like "is photon counting for S1 a good idea?". I hope it will eventually be useful to others interested in doing statistical analyses. 

The default model includes lots of information I obtained by curve-tracing plots or even just constructing some function that looked like what I saw in an image. Just saying...

Many of the 'physics info' stored in the package is shamelessly "borrowed" from other sources. In particular:
  * Andrew's maximum-gap limit setting code, used for the XENON100 max-gap cross checks
  * Chris' wimpstat repository, used for the XENON100 S2-only limit setting
  * NEST
  * XEPHYR, one of the main XENON100 profile likelihood packages
  * Several of the Monte Carlo group's excellent notes on this topic.

The statistical model used is very similar to the "Bologna model" (unbinned likelihood). 




Installation
============
Make sure you have pax installe -- we need it for the common unit system and XENON1T.ini.

Then run `python setup.py develop` or `python setup.py install`

