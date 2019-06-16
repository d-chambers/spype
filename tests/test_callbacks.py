"""
Tests for the various built-in callbacks
"""
import os
import tempfile

import pytest

from spype import task, pype_input
from spype.callbacks import log_on_fail
from spype.constants import FAILURE_LOG_COLUMNS
from spype.exceptions import InvalidPype


# --------------- module level fixtures


@pytest.fixture
def report_file():
    with tempfile.NamedTemporaryFile(suffix=".csv") as fi:
        yield fi.name


# ----------------- callback tests


class TestLogOnFailWithPype:
    """ tests for writing failures to file hooked to pype """

    # fixtures
    @pytest.fixture
    def pype(self, report_file):
        """ a simple pype for logging failures """

        @task
        def add_one(num):
            return num + 1

        @task
        def raise_value_error(num):
            raise ValueError(f"I dont like this number {num}")

        pype = pype_input | add_one | raise_value_error
        return pype

    @pytest.fixture
    def pype_raise_on_fail(self, pype, report_file):
        """ add the log_on_fail callback to pype """
        on_fail = log_on_fail(report_file, re_raise=True)
        return pype.add_callback(on_fail, "on_failure")

    @pytest.fixture
    def pype_no_raise_on_fail(self, pype, report_file):
        """ add the log_on_fail callback, don't reraise"""
        on_fail = log_on_fail(report_file, re_raise=False)
        pype.register("test_on_fail")  # register pype
        pype = pype.add_callback(on_fail, "on_failure")
        # trigger pype to cause failure and writting to log
        pype(2)
        return pype

    # tests
    def test_raises_on_unregistered_pype(self, pype_raise_on_fail):
        """ if a pype is not registered it should raise to use callback """
        with pytest.raises(InvalidPype):
            pype_raise_on_fail(2)

    def test_reraise(self, pype_raise_on_fail):
        """ ensure the exception is re-raised if reraise was set to True """
        pype_raise_on_fail.register("bob")

        with pytest.raises(ValueError) as e:
            pype_raise_on_fail(2)

        assert "I dont like" in str(e)

    def test_log_created(self, pype_no_raise_on_fail, report_file):
        """ ensure the report file exists and has the current pype in it """
        pype_no_raise_on_fail(2)
        assert os.path.exists(report_file)
        contents = open(report_file).readlines()
        assert contents
        assert contents[0][:-1] == ", ".join(list(FAILURE_LOG_COLUMNS))
        failure = contents[1].split(", ")
        assert failure[0] == pype_no_raise_on_fail.name
