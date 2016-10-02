from tqdm import tqdm
from .model import Model

__all__ = ['create_models_in_parallel']


def create_models_in_parallel(configs, ipp_client=None, block=False):
    """Return Models for each configuration in configs.
    :param ipp_client: ipyparallel client to use for parallelized computation, or None (in which case models will be
                       computed serially. For now only engines running in the same directory as the main code
                       are supported, see #1.
    :param configs: list of Model configuration dictionaries
    :param block: passed to the async map of ipyparallel. Useful for debugging, but disables progress bar.
    :return: list of Models.
    """
    if ipp_client is not None:
        # Fully fledged blueice Models don't pickle, but we can use dill..
        ipp_client[:].use_dill()

        def compute_model(conf):
            return Model(conf)

        asyncresult = ipp_client.load_balanced_view().map(compute_model, configs, ordered=True, block=block)
        result = []
        for m in tqdm(asyncresult,
                      desc="Computing models in parallel",
                      smoothing=0,  # Show average speed, instantaneous speed is extremely variable
                      total=len(configs)):
            result.append(m)

        return result

    else:
        return [Model(conf) for conf in tqdm(configs, 'Computing models on one core')]
