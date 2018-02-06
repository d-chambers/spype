"""
A collection of useful callbacks.
"""

from collections import namedtuple
from pathlib import Path

import pdb

from spype.constants import FAILURE_LOG_COLUMNS
from spype.exceptions import InvalidPype
from spype.utils import FileLock


def debug_callback(task, args, kwargs, pype):
    """ callback for debugging pype inputs """
    pdb.set_trace()
    return task(*args, **kwargs)


# ----------------------------- Log Failure stuff


Failure = namedtuple('Failure', list(FAILURE_LOG_COLUMNS))


def log_on_fail(log_file: str, re_raise: bool = True):
    """
    Create a callback for logging failures in a csv file.

    Used for retrying failures. In order for this to work inputs to pypes
    (ie the first task inputs) must be str instances (eg file paths).

    Parameters
    ----------
    log_file
        The path to a csv with failure info
    re_raise
        If the exception should be re-raised
        The number of times a file will be re-processed
    """

    def on_failure(task, pype, inputs, meta, e):
        if pype is not None and pype.name is None:
            msg = (f'the following pype does not have a valid name. It must ',
                   f'be registered via the register method')
            raise InvalidPype(msg)
        log_path = Path(log_file)
        _make_file(log_path)
        failure = _make_failure(pype, task, inputs, meta, e)
        with FileLock(log_path):  # write failure to log
            with open(log_file, 'a') as fi:
                fi.write(', '.join(list(failure)) + '\n')
        if re_raise:
            raise e

    return on_failure


def _make_failure(pype, task, inputs, meta, e):
    """ return a Failure named tuple which captures info about failure """
    failure = Failure(
        pype=str(None if pype is None else getattr(pype, 'name')),
        task_inputs=str(inputs),
        pype_inputs=str((meta or {}).get('pype_inputs')),
        number_failures=str(0),
        task=task.get_name(),
        exception=str(e).replace('\n', ' '),
        succeeded=str(False),
    )
    return failure


def _make_file(path):
    """ make the file, ensure the csv headers are inplace """
    path.parent.mkdir(parents=True, exist_ok=True)  # make sure dirs exist
    # if the file doesn't exist or if it is empty, write column headers
    if not path.exists() or path.stat().st_size == 0:
        header = ', '.join(list(FAILURE_LOG_COLUMNS)) + '\n'
        with FileLock(path):
            with open(path, 'w') as fi:
                fi.write(header)
