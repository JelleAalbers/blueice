Unnamed statistical inference package
=====================================
Jelle Aalbers, 2016

Source code: `https://github.com/XENON1T/wimpy`

Documentation:

- `This note <https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:statspackage_architecture>`_ on the XENON wiki.
- You can also check the docstrings and code comments.


About
=====
This package allows you to do parametric inference using Monte-Carlo derived extended unbinned likelihood functions. 

It lets you make likelihood functions which measure agreement between data and Monte Carlos with different settings: you choose which settings to vary (which parameters the likelihood functions has) and in which space the agreement is measured. For more information, please see `the documentation in this note
<https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:statspackage_architecture>`_ and the `examples in the notebooks folder <https://github.com/XENON1T/wimpy/tree/master/notebooks>`_.

The ``xenon`` folder contains settings and configuration particular to the XENON1T experiment. Much of this is derived or shamelessly "borrowed" from other sources: 

- Andrew's maximum-gap limit setting code, used for the XENON100 max-gap cross checks.
- Chris' wimpstat repository, used for the XENON100 S2-only limit setting (but no longer available?)
- NEST: not directly, but since this is currently the best xenon TPC code out there, the physics model used here 
- Several of the Monte Carlo group's excellent material on this topic, in particular the `XENON1T Monte Carlo paper <http://arxiv.org/abs/1512.07501>`_ and the `notes linked here <https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:org:papers:xe1t_sensitivity>`_. 

The default model includes lots of information I obtained by curve-tracing plots or even just constructing some function that looked like what I saw in an image. Just saying...


Installation
============
Make sure you have pax installed -- we need it for the common unit system and XENON1T.ini.

Then run `python setup.py develop` or `python setup.py install`
