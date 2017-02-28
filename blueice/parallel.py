"""Parallel and delayed computation of models/sources for blueice.
"""
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from .utils import read_pickle
from .model import Model

__all__ = ['create_models_ipyparallel', 'compute_single', 'compute_many', 'compute_all']
log = logging.getLogger('blueice.parallel')


def compute_single(hash, task_dir='pdf_tasks', result_dir='pdf_cache'):
    """Computes a single source PDF from a saved task file"""
    # Do we have result with this hash?
    result_filename = os.path.join(result_dir, hash)
    if os.path.exists(result_filename):
        log.debug("Task %s already computed, nothing done." % hash)
        return

    # Do we have a task with this hash?
    task_filename = os.path.join(task_dir, hash)
    if not os.path.exists(task_filename):
        # Abort, not going to compute something random.
        raise ValueError("Hash %s does not correspond to a task or result" % hash)

    source_class, source_config = read_pickle(task_filename)

    # Compute the source pdf
    source_config['pdf_cache_dir'] = result_dir
    source_config['delay_pdf_computation'] = False
    s = source_class(source_config)

    # Remove the task file
    os.remove(task_filename)

    assert os.path.exists(result_filename)

    if s.hash != hash:
        raise ValueError("source hash changed somehow??")


def compute_many(hashes, n_cpus=1, *args, **kwargs):
    if n_cpus != 1:
        pool = ProcessPoolExecutor(max_workers=n_cpus)
        futures = []
        for h in hashes:
            futures.append(pool.submit(compute_single, *args, hash=h, **kwargs))

        # Wait fot the futures to complete; give a progress bar
        with tqdm(total=len(futures), desc='Computing on %d cores' % n_cpus) as pbar:
            while len(futures):
                _done_is = []
                for f_i, f in enumerate(futures):
                    if f.done():
                        _done_is.append(f_i)
                        pbar.update(1)
                futures = [f for f_i, f in enumerate(futures) if not f_i in _done_is]
                time.sleep(0.1)
    else:
        for h in tqdm(hashes, desc='Computing on one core'):
            compute_single(h, *args, **kwargs)


def compute_all(input_dir='./pdf_cache', *args, **kwargs):
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Input directory %s does not exist")
    compute_many(os.listdir(input_dir), *args, **kwargs)


def create_models_ipyparallel(configs, ipp_client=None, block=False):
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
                      desc="Computing models on %d cores" % len(ipp_client.ids),
                      smoothing=0,  # Show average speed, instantaneous speed is extremely variable
                      total=len(configs)):
            result.append(m)

        return result

    else:
        return [Model(conf) for conf in tqdm(configs, 'Computing models on one core')]
