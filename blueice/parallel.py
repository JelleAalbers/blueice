from tqdm import tqdm
from .model import Model


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
        # Fully fledged blueice Models don't pickle, so we have to construct them again later in the main process
        # (but then we can just grab their PDFs from cache, so it's quick)

        def compute_model(conf):
            Model(conf)
            return None

        asyncresult = ipp_client.load_balanced_view().map(compute_model, configs, ordered=False, block=block)
        for _ in tqdm(asyncresult,
                      desc="Computing models in parallel",
                      smoothing=0,   # Show average speed, instantaneous speed is extremely variable
                      total=len(configs)):
            pass

        # (Re)make the models in the main process; hopefully PDFs use the cache...
        return [Model(conf) for conf in configs]

    else:
        return [Model(conf) for conf in tqdm(configs, 'Computing models on one core')]
