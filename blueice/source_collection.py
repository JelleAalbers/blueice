import numpy as np
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm

from . import utils
from .pdf_morphers import MORPHERS

__all__ = ['SourceCollection']

def init_source(config,i_source=0,**kwargs):
        """
        :param config: Dictionary specifying detector parameters, source info, etc.
        :param kwargs: Overrides for the config (optional)
        """
        defaults = dict(livetime_days=1,
                        data_dirs=1,
                        nohash_settings=['data_dirs', 'pdf_sampling_batch_size',
                                         'force_recalculation'])
        init_config = utils.combine_dicts(defaults, config, kwargs, deep_copy=True)

        if 'rate_multiplier' in init_config:
            raise ValueError("Don't put a setting named rate_multiplier in the model config please...")

        # Initialize the sources. Each gets passed the entire config (without the 'sources' field)
        # with the settings in their entry in the sources field added to it.
        source_config = config['sources'][i_source]
        if 'class' in source_config:
            source_class = source_config['class']
        else:
            source_class = init_config['default_source_class']
        conf = utils.combine_dicts(init_config,
                                    source_config,
                                    exclude=['sources', 'default_source_class', 'class'])

        # Special handling for the _rate_multiplier settings
        source_name = conf.get('name', 'WHAAAAAA_YOUDIDNOTNAMEYOURSOURCETHIS')
        conf['rate_multiplier'] = conf.get('%s_rate_multiplier' % source_name, 1)
        conf = {k:v for k,v in conf.items() if not k.endswith('_rate_multiplier')}
        s = source_class(conf)
        return s

class SourceCollection(object):
    """
        Object that contains all source instances for a source type (e.g. signal)
        Responsible for initialising and caching the source instances, for initialising and calling the interpolators in the relevant nuisance parameter directions, and simulating events. 
    """
    def __init__(self,ll, source_index=0, **kwargs):

        self.source_index = source_index
        self.ll = ll
        self._kwargs_to_settings = ll._kwargs_to_settings
        self.apply_efficiency = ll.pdf_base_config["sources"][source_index].get("apply_efficiency",False)
        self.sname = ll.pdf_base_config["sources"][source_index].get("name","NAME IS MISSING")
        self.efficiency_name = ll.pdf_base_config["sources"][source_index].get("efficiency_name","")
        #Import shape parameters, only the ones we use:
        ignore_parameters = ll.pdf_base_config["sources"][source_index].get("ignore_parameters",[])
        self.shape_parameter_names = [spn for spn in ll.shape_parameters.keys() if spn not in ignore_parameters]
        self.shape_parameters = OrderedDict()
        for spn in self.shape_parameter_names:
            self.shape_parameters[spn] = ll.shape_parameters[spn]
        if len(self.shape_parameters):
            self.morpher = MORPHERS[ll.config.get("morpher",'GridInterpolator')](ll.config.get("morpher_config",{}),self.shape_parameters)
            zs_list = self.morpher.get_anchor_points(bounds=ll.get_bounds)
            configs = []
            for zs in zs_list:
                config = deepcopy(ll.pdf_base_config)
                for i, (setting_name, (anchors, _, _)) in enumerate(self.shape_parameters.items()):
                    config[setting_name] = anchors[zs[i]]
                configs.append(config)
            self.sources = [init_source(c,i_source=source_index) for c in tqdm(configs,desc="Initialising source "+self.sname)]
            anchor_models = OrderedDict()
            for zs,source in zip(zs_list,self.sources):
                anchor_models[tuple(zs)] = source
            self.anchor_models= anchor_models
            self.mus_interpolator = self.morpher.make_interpolator(f = lambda s: s.events_per_day*s.fraction_in_range, 
                extra_dims=[1],
                anchor_models = anchor_models)
            if self.ll.model_statistical_uncertainty_handling:
                self.n_model_events_interpolator = self.morpher.make_interpolator(f=lambda s:s.pmf_grid(),extra_dims=[1],anchor_models=anchor_models)
            else:
                self.n_model_events_interpolator = None
        else:
            print("initialising source "+self.sname)
            base_source = self.ll.base_model.sources[self.source_index]
            self.mu = base_source.events_per_day*base_source.fraction_in_range
    def to_analysis_dimensions(self, d):
        """Given a dataset, returns list of arrays of coordinates of the events in the analysis dimensions"""
        return utils._events_to_analysis_dimensions(d, self.ll.pdf_base_config['analysis_space'])
    def set_data(self,d):
        """
            Send data array to source to score
        """
        pif = lambda s:np.vstack([s.pdf(*self.to_analysis_dimensions(d))])
        if len(self.shape_parameters):
            self.ps_interpolator = self.morpher.make_interpolator(f=pif,extra_dims=[1,len(d)],anchor_models=self.anchor_models )
        else: self.ps = pif(self.ll.base_model.sources[self.source_index])

    def evaluate(self,livetime_days = None,compute_pdf = False,**kwargs):
        rate_multipliers, shape_parameter_settings = self._kwargs_to_settings(**kwargs)
        rate_multiplier = rate_multipliers[self.source_index]
        ret = 0.
        if len(self.shape_parameters):
            if compute_pdf:
                if self._has_non_numeric:
                    raise NotImplementedError("compute_pdf only works for numerical values")

                mus, ps, n_model_events = self._compute_single_pdf(**kwargs)

            else:
                # We can use the interpolators. They require the settings to come in order:
                zs = []
                for setting_name, (_, log_prior, _) in self.shape_parameters.items():
                    z = shape_parameter_settings[setting_name]
                    zs.append(z)

                    # Test if the z value is out of range; if so, return 0
                    minbound, maxbound = self.ll.get_bounds(setting_name)
                    if not minbound <= z <= maxbound:
                        return 0.
                # The RegularGridInterpolators want numpy arrays: give it to them...
                zs = np.asarray(zs)
                mu = self.mus_interpolator(zs)[0]
                ps = self.ps_interpolator(zs)[0]
        else:
            # No shape parameters
            mu = self.mu
            ps = self.ps
        #rate multiplier + efficiency: 
        mu *= rate_multiplier
        if self.apply_efficiency:
            mu *= shape_parameter_settings.get(self.efficiency_name,1.)
        return mu,ps

