blueice: Build Likelihoods Using Efficient Interpolations and monte-Carlo generated Events
==========================================================================================
.. image:: https://travis-ci.org/JelleAalbers/blueice.svg?branch=master
    :target: https://travis-ci.org/JelleAalbers/blueice
.. image:: https://coveralls.io/repos/github/JelleAalbers/blueice/badge.svg?branch=master
    :target: https://coveralls.io/github/JelleAalbers/blueice?branch=master
.. image:: https://readthedocs.org/projects/blueice/badge/?version=latest
         :target: http://blueice.readthedocs.org/en/latest/?badge=latest
         :alt: Documentation Status
.. image:: https://zenodo.org/badge/65375508.svg
   :target: https://zenodo.org/badge/latestdoi/65375508

Source code: `https://github.com/JelleAalbers/blueice`

Documentation: `http://blueice.readthedocs.io/en/latest/index.html`

About
=====
This package allows you to do parametric inference using likelihood functions, in particular likelihoods derived from Monte-Carlo or calibration sources.

Especially when connected to a Monte Carlo, blueice lets you make likelihood functions which measure agreement between data and theory with flexibility: you choose which settings to vary (which parameters the likelihood functions has) and in which space the agreement is measured.

This package contains only generic code: you'll need a few things to make it useful for a particular experiment. Originally this code was developed for XENON1T only; the XENON1T models have since been split off to the `laidbax <https://github.com/XENON1T/laidbax>`_ repository.


Contributors
============
* Jelle Aalbers
* Knut Dundas Moraa
* Bart Pelssers
