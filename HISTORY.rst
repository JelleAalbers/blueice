------------------
1.2.1 (2024/12/11)
------------------
* Source-wise interpolation by @hammannr in https://github.com/JelleAalbers/blueice/pull/46
* Fix source index mismatch by @hammannr in https://github.com/JelleAalbers/blueice/pull/47

------------------
1.2.0 (2024/01/13)
------------------
* Prevent negative rates being passed to Barlow-Beeston equation, and allow per-event weights (#32)
* Add likelihood that takes coupling as shape parameters (#34)
* Patch for tests (#37)
* Use scipy stats for PoissonLL (#40)
* Do not scale mus when livetime_days is 0 (#41)

------------------
1.1.0 (2021/01/07)
------------------
* Likelihood sum wrapper (#17)
* emcee bestfit and multicore precomputation (#18)
* LogAncillaryLikelihood for constraint terms (#19)
* HistogramPDFSource simulation, order shape parameter dict (#20)
* Efficiency shape parameter, LogLikelihoodSum enhancements (#23)
* Use scipy as default optimizer (#24)
* Minuit support for bounds and errors (#26, #27)
* Per-source efficiencies, weighted LogLikelihoodSum (#28)
* Use atomicwrites for cache to prevent race conditions (#30)

------------------
1.0.0 (2016/10/01)
------------------
* Binned likelihoods (#7)
* Argument validation for LogLikelihood function (#8)
* Automatic handling of statistical uncertainty due to finite MC/calibration statistics (#9):
  * Adjustment of expected counts per bin using Beeston-Barlow method for one source
  * Generalized to multiple sources, but only one with finite statistics.
  * Only for binned likelihoods.
* iminuit integration, use as default minimizer if installed (#10, #13)
* compute_pdf option to do full likelihood model computation on the fly (#11)
* HistogramPDF to provide just histogram lookup/interpolation from DensityEstimatingSource (#12)
* inference functions -> LogLikelihood methods
* Most-used functions/classes available under blueice (blueice.Source, blueice.UnbinnedLogLikelihood, ...)
* compute_pdf auto-called, consistent handling of events_per_day
* Start of documentation, readthedocs integration

------------------
0.4.0 (2016/08/22)
------------------
* Big internal refactor, some API changes (#5)
* DensityEstimatingSource
* Bugfixes, more tests

------------------
0.3.0 (2016/08/21)
------------------

* Renamed to blueice, XENON stuff renamed to laidbax
* Experimental radial template morphing (#4)
* Tests, several bugfixes (e.g. #3)
* Rate parameters are now rate multipliers
* Linear interpolation of density estimator
* Parallel model initialization

------------------
0.2.0 (2016/07/31)
------------------

* Complete makeover centered around LogLikelihood function
* Separation of XENON stuff and general code
* PDF caching
* Example notebooks

------------------
0.1.0 (2016/07/14)
------------------

* First release in separate repository
* Model and Source, pdf sampling.

------------------
0.0.1 (2015/12/18)
------------------

* First release in XeAnalysisScripts
