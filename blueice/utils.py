import os
from hashlib import sha1
import pickle
import re

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def data_file_name(filename, data_dirs=None):
    """Returns filename if a file exists. Also checks data_dirs for the file."""
    if os.path.exists(filename):
        return filename
    if data_dirs is not None:
        return find_file_in_folders(filename, data_dirs)
    return FileNotFoundError(filename)


def find_file_in_folders(filename, folders):
    """Searches for filename in folders, then return full path or raise FileNotFoundError
    Does not recurse into subdirectories
    """
    for folder in folders:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(filename)


def load_pickle(filename, data_dirs=None):
    """Loads a pickle from filename"""
    with open(data_file_name(filename, data_dirs), mode='rb') as infile:
        return pickle.load(infile)


def load_csv(filename, data_dirs=None):
    """Read csv file and return numpy float arrays x, y"""
    return pd.read_csv(data_file_name(filename, data_dirs),
                       delimiter=',', names=['x', 'y'], comment='#').values[1:].astype(np.float).T


def save_pickle(stuff, filename):
    """Saves stuff in a pickle at filename"""
    with open(filename, mode='wb') as outfile:
        pickle.dump(stuff, outfile)


def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


def deterministic_hash(thing):
    """Return a deterministic hash of a container hierarchy using hashablize, pickle and sha1"""
    return sha1(pickle.dumps(hashablize(thing))).hexdigest()


def process_files_in_config(config, data_dirs):
    """Replaces file name values in dictionary config with their files.
    Modifies config in-place
    """
    for k, v in config.items():
        if isinstance(v, str):
            _, ext = os.path.splitext(v)
            if ext == '.pkl':
                config[k] = load_pickle(v, data_dirs)
            elif ext == '.csv':
                config[k] = load_csv(v, data_dirs)
            # Add support for other formats here


class InterpolateAndExtrapolate1D(object):
    """Extends scipy.interpolate.interp1d to do constant extrapolation outside of the data range
    """
    def __init__(self, points, values):
        points = np.asarray(points)
        self.interpolator = interp1d(points, values)
        self.min = points.min()
        self.max = points.max()

    def __call__(self, points):
        try:
            points[0]
        except TypeError:
            points = np.array([points])
        points = np.clip(points, self.min, self.max)
        return self.interpolator(points)


def arrays_to_grid(arrs):
    """Convert a list of n 1-dim arrays to an n+1-dim. array, where last dimension denotes coordinate values at point.
    """
    return np.stack(np.meshgrid(*arrs, indexing='ij'), axis=-1)


def latin(n, d, box=None, shuffle_steps=500):
    """Creates a latin hypercube of n points in d dimensions
    Stolen from https://github.com/paulknysh/blackbox
    """
    # starting with diagonal shape
    pts=np.ones((n,d))

    for i in range(n):
        pts[i]=pts[i]*i/(n-1.)

    # spread function
    def spread(p):
        s=0.
        for i in range(n):
            for j in range(n):
                if i > j:
                    s=s+1./np.linalg.norm(np.subtract(p[i],p[j]))
        return s

    # minimizing spread function by shuffling
    currminspread=spread(pts)

    for m in tqdm(range(shuffle_steps), desc='Shuffling latin hypercube'):

        p1=np.random.randint(n)
        p2=np.random.randint(n)
        k=np.random.randint(d)

        newpts=np.copy(pts)
        newpts[p1,k],newpts[p2,k]=newpts[p2,k],newpts[p1,k]
        newspread=spread(newpts)

        if newspread<currminspread:
            pts=np.copy(newpts)
            currminspread=newspread

    if box is None:
        return pts

    for i in range(len(box)):
        pts[:, i] = box[i][0] + pts[:, i] * (box[i][1] - box[i][0])

    return pts
