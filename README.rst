blueice: Build Likelihoods Using Efficient Interpolations and monte-Carlo generated Events
==========================================================================================
Jelle Aalbers, 2016

.. image:: https://travis-ci.org/JelleAalbers/blueice.svg?branch=master
    :target: https://travis-ci.org/JelleAalbers/blueice

Source code: `https://github.com/JelleAalbers/blueice`


About
=====
This package allows you to do parametric inference using Monte-Carlo derived extended unbinned likelihood functions.

It lets you make likelihood functions which measure agreement between data and Monte Carlos with different settings: you choose which settings to vary (which parameters the likelihood functions has) and in which space the agreement is measured. 

This package contains only generic code: you'll need a few things to make it useful for a particular experiment. Originally this code was developed for XENON1T only; the XENON1T models have since been split off to the `laidbax <https://github.com/XENON1T/laidbax>`_ repository.

