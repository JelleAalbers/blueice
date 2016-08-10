blipflip: Build Likelihoods from Interpolated Pdfs For LImit settting and other Parametric inference
====================================================================================================
Jelle Aalbers, 2016

Source code: `https://github.com/JelleAalbers/blipflip`


About
=====
This package allows you to do parametric inference using Monte-Carlo derived extended unbinned likelihood functions.

It lets you make likelihood functions which measure agreement between data and Monte Carlos with different settings: you choose which settings to vary (which parameters the likelihood functions has) and in which space the agreement is measured. 

This package contains only generic code: you'll need a few things to make it useful for a particular experiment. Originally this code was developed for XENON1T only; the XENON1T models have since been split off to the `laidbax <https://github.com/XENON1T/laidbax>`_ repository.

