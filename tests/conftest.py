"""
pytest configuration for sflow
"""
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from os.path import join, dirname, abspath

import pytest


# --------- Add key paths to pytest namespace


TEST_PATH = abspath(dirname(__file__))
PKG_PATH = dirname(TEST_PATH)
TEST_DATA_PATH = join(TEST_PATH, "test_data")
sys.path.insert(0, PKG_PATH)  # make package importable


def append_func_name(list_like):
    """ decorator to append function to list """

    def _decor(func):
        list_like.append(func.__name__)
        return func

    return _decor


# -------------------------- session fixtures


@pytest.fixture(scope="session")
def process_pool_executor():
    """ return a process pool executor """
    return ProcessPoolExecutor()


@pytest.fixture(scope="session")
def thread_pool_executor():
    """ return a thread pool executor"""
    return ThreadPoolExecutor()


@pytest.fixture
def some_list():
    """ return an empty list for storing state """
    return []


@pytest.fixture
def some_dict():
    """ return an empty list for storing state """
    return {}
