import os
import inspect
import pickle


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


def save_pickle(stuff, filename):
    """Saves stuff in a pickle at filename"""
    with open(filename, mode='wb') as outfile:
        pickle.dump(stuff, outfile)