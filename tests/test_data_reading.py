import shutil
import os
import tempfile

import pytest

from blueice import data_reading
from blueice.test_helpers import *
from blueice import utils


@pytest.fixture
def tempdir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_data_reading(tempdir):
    # Save a pickle with an "important message"
    fn = 'important_setting.pkl'
    full_path = os.path.join(tempdir, fn)
    content = 'howdy'

    # Test we could read it back
    utils.save_pickle('howdy', full_path)
    assert utils.read_pickle(full_path) == content

    # Test we can find it
    with pytest.raises(FileNotFoundError):
        utils.find_file_in_folders('gnork', folders=tempdir) == full_path
    assert utils.find_file_in_folders(fn, folders=tempdir) == full_path
    assert utils.data_file_name(fn, data_dirs=tempdir) == full_path

    # Test file-reading leaves ordinary settings alone
    c = dict(bla='nothing_special')
    c_pimped = data_reading.read_files_in(c, data_dirs=tempdir)
    assert isinstance(c_pimped, dict)
    assert c_pimped['bla'] == 'nothing_special'

    # Finally, actually test file-reading from config
    c = dict(bla=fn)
    assert '.pkl' in data_reading.FILE_READERS
    c_pimped = data_reading.read_files_in(c, data_dirs=tempdir)
    assert c['bla'] == fn    # Original setting has been left alone
    assert c_pimped['bla'] == content
    assert full_path in data_reading.CACHE

    # Test if we can read the test config without crashing
    data_reading.read_files_in(conf_for_test(), data_dirs=tempdir)

