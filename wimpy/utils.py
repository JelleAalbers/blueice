import os
import inspect


# Store the directory of this file
THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def data_file_name(filename):
    """Returns filename if a file exists there, else returns THIS_DIR/data/filename"""
    if os.path.exists(filename):
        return filename
    new_filename = os.path.join(THIS_DIR, 'data', filename)
    if os.path.exists(new_filename):
        return new_filename
    else:
        raise ValueError('File name or path %s not found!' % filename)
