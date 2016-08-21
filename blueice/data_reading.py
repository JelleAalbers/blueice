import os
import pandas as pd
import numpy as np

from .utils import data_file_name
from .utils import read_pickle


def read_csv(filename):
    result = pd.read_csv(filename, delimiter=',', names=['x', 'y'], comment='#')
    result = result.values[1:].astype(np.float).T
    return result


FILE_READERS = {'pkl': read_pickle, 'csv': read_csv}
CACHE = dict()


def read_if_is_filename(x, data_dirs=tuple('.')):
    """If x is a string that ends in a supported extension, return the file contents, else return x.
    A cache is maintained to ensure things are only read once.
    :param data_dirs: directories to look for files. Defaults to '.'.
    """
    global CACHE
    if not isinstance(x, str):
        return x
    if x in CACHE:
        return CACHE[x]
    _, extension = os.path.splitext(x)

    if extension not in FILE_READERS:
        return x

    contents = data_file_name(x, data_dirs)
    CACHE[x] = contents
    return contents
