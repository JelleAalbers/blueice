"""Utilities for reading files specified in configuration dictionaries
"""

from copy import deepcopy
import os
import pandas as pd
import numpy as np

from .utils import data_file_name
from .utils import read_pickle

__all__ = ['read_csv', 'read_files_in']


def read_csv(filename):
    result = pd.read_csv(filename, delimiter=',', names=['x', 'y'], comment='#')
    result = result.values[1:].astype(float).T
    return result


FILE_READERS = {'.pkl': read_pickle, '.csv': read_csv}
CACHE = dict()


def read_files_in(d, data_dirs=tuple('.')):
    """Return a new dictionary in which every value in d that is a string that ends in a supported extension
    is replaced with that file's contents. Leave other keys alone.
    A cache is maintained to ensure things are only read once.
    :param data_dirs: directories to look for files. Defaults to '.'.
    """
    d = deepcopy(d)
    global CACHE

    for k, x in d.items():
        if not isinstance(x, str):
            continue

        _, extension = os.path.splitext(x)

        if extension not in FILE_READERS:
            continue

        # Convert x to the full path
        x = data_file_name(x, data_dirs)

        if x in CACHE:
            d[k] = CACHE[x]
        else:
            d[k] = CACHE[x] = FILE_READERS[extension](x)

    return d
