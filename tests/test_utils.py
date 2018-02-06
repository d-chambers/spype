"""
tests for sflow's utils module
"""
import inspect
import tempfile
from os.path import exists
from typing import Tuple

import pytest

from spype.exceptions import NoReturnAnnotation
from spype.utils import (Context, args_kwargs, de_args_kwargs, apply_partial,
                         sig_to_args_kwargs, get_default_names, FileLock)


# -------------------------- functions for testing

def int2str(a: int) -> str:
    return str(a)


def two_str(a: str, b: str) -> (str, str):
    return int(a), int(b)


def int_str(a: int, b: str) -> Tuple[int, str]:
    """ This function is to test output type of Tuple """
    return int(a), str(b)


# --------------------------- tests


class TestContext:
    """ test the context object """

    # fixtures
    @pytest.fixture
    def basic_dict(self):
        """ return a dict with 'a', 'b' as keys and 0, 1 as values """
        return {'a': 0, 'b': 1}

    @pytest.fixture
    def basic_context(self, basic_dict):
        """ return context instance with a dict that has 'a' and 'b' keys """
        return Context(basic_dict)

    # tests
    def test_bad_key_raises(self, basic_context):
        """ ensure trying to set an option not in dict raises """
        with pytest.raises(Exception):
            basic_context(bob='sucks')

    def test_set_value(self, basic_context, basic_dict):
        """ ensure setting a value updates dict """
        basic_context(a=2)
        assert basic_dict['a'] == 2

    def test_context_manager(self, basic_context, basic_dict):
        """ ensure the context manager works """
        with basic_context(a=2, b=2):
            assert basic_dict['a'] == 2 and basic_dict['b'] == 2
        assert basic_dict['a'] == 0 and basic_dict['b'] == 1

    def test_get_item(self, basic_context, basic_dict):
        """ make sure context get item returns exact content of dict """
        assert basic_context['a'] == basic_dict['a']

    def test_set_item(self, basic_context, basic_dict):
        """ make sure set_item updates dict """
        basic_context['a'] = 10
        assert basic_dict['a'] == 10
        assert basic_context['a'] == 10

    def test_str(self, basic_context):
        """ensure a tring method call doesnt raise """
        assert isinstance(str(basic_context), str)

    def test_repr(self, basic_context):
        """ ensure repr doesn't raise """
        assert isinstance(repr(basic_context), str)


class TestArgsKwargs:
    """ test for putting generic function output into args and kwargs """
    colnames = 'output, adapt, args, kwargs'

    good_pairs = [  # (output, adapter, expected_args, expected_kwargs)
        ((), None, (), {}),
        (None, None, (), {}),
        (1, None, (1,), {}),
        (range(100), None, (range(100),), {}),
        ((1, 2, 3), None, (1, 2, 3), {}),
        ((1, 2, 3), (2, 1, 0), (3, 2, 1), {}),
        ((1, 2, 3), (1, 0, 'bob'), (2, 1), {'bob': 3}),
        (1, ('bob',), (), {'bob': 1}),
    ]

    bad_pairs = [  # these should raise an assertion error
        ((), (1, 2)),
        ((1, 2, 'bob'), (5, 4, 2)),
        ((int, float), ('bob', 10)),
    ]

    @pytest.mark.parametrize(colnames, good_pairs)
    def test_good_pairs(self, output, adapt, args, kwargs):
        """ ensure args_kwargs returns correct output """
        assert args_kwargs(output, adapter=adapt) == (args, kwargs)

    @pytest.mark.parametrize('output, adapt', bad_pairs)
    def test_bad_pairs(self, output, adapt):
        """ test that the bad pairs raise """
        with pytest.raises(AssertionError):
            args_kwargs(output, adapter=adapt)


class TestDeArgsKwargs:
    """ test turning args and kwargs back to a tuple or single object """
    colnames = 'args, kwargs, output'

    good_pairs = [
        ((), {}, None),
        ((1,), {}, 1),
        ((0,), {}, 0),
    ]

    @pytest.mark.parametrize(colnames, good_pairs)
    def test_good_pairs(self, args, kwargs, output):
        """ ensure args_kwargs returns correct output """
        assert de_args_kwargs(args, kwargs) == output


class TestPartial:
    # list of partial arguments, function inputs (args, kwargs), expected outputs

    def func1(self, x, y, z, a=2, b='b', c=None):
        return locals()

    # inputs, partial_dict, expected output
    func1_list = [
        (((1,), {}), dict(x=1, y=2), dict(x=1, y=2, z=1)),
        (((1, 2, 3, 4), {}), dict(a=4), dict(x=1, y=2, z=3, b=4, a=4)),
        (((), {}), dict(x=1, y=2, z=3), dict(x=1, y=2, z=3)),
        (((), {'x': 1, 'y': 1, 'z': 1}), dict(c=1, b=1), dict(c=1, b=1)),
        (((1, 2, 3), {}), {}, dict(x=1, y=2, z=3, a=2, b='b', c=None))
    ]

    @pytest.mark.parametrize('inputs, partial_dict, output', func1_list)
    def test_func1(self, inputs, partial_dict, output):
        """ Ensure all of the variables were bounded as expected for func 1"""
        out = apply_partial(self.func1, *inputs[0], partial_dict=partial_dict,
                            **inputs[1])
        assert output.items() <= out.items()

    def test_partial_no_apply(self):
        """ ensure apply_partial works without a dict """


class TestSigToArgKwargs:
    """ tests for getting args and kwargs of types from signatures """
    # callable, adapter, expected args, expected kwargs
    test_list = (
        (inspect.signature(int2str), (), (str,), {}),
        (int2str, (), (str,), {}),
        (two_str, (), (str, str), {}),
        (two_str, (0, 'b'), (str,), {'b': str}),
        (two_str, (None, 'b'), (), {'b': str}),
        (int_str, (), (int, str), {}),
        (int_str, (None, 0), (str,), {}),
        (int_str, (None, 'bob'), (), {'bob': str}),
        (int_str, ('int', 0), (str,), {'int': int}),
    )

    @pytest.mark.parametrize('func, adapter, args, kwargs', test_list)
    def test_sig_args_kwargs(self, func, adapter, args, kwargs):
        """ ensure the outputs of sig to args kwargs works """
        args_out, kwargs_out = sig_to_args_kwargs(func, adapter=adapter)
        assert args_out == args
        assert kwargs_out == kwargs

    def test_no_annotation_raises(self):
        """ ensure function with no return annotations raises """

        def bob(a):
            pass

        with pytest.raises(NoReturnAnnotation):
            sig_to_args_kwargs(bob)


class TestGetDefaultNames:
    """
    Tests for getting names of arguments with default values from signatures.
    """

    # fixtures
    @pytest.fixture
    def signature(self):
        """ return a signature of a function with default values """

        def func1(a, b, c=None):
            pass

        return inspect.signature(func1)

    # tests
    def test_get_names(self, signature):
        """ ensure get_default_names returns only names of params with
         defaults """
        names = get_default_names(signature)
        assert names == {'c'}


class TestFileLock:
    """ test the FileLock object """

    # fixture
    @pytest.fixture
    def protected_file(self):
        """ create a file that will be protected by the lock """
        with tempfile.NamedTemporaryFile(suffix='.csv') as tf:
            yield tf.name

    @pytest.fixture
    def file_lock(self, protected_file):
        """ return an instance of FileLock """
        return FileLock(protected_file, timeout=2)

    # tests
    def test_file_is_locked(self, file_lock, protected_file):
        """ ensure when the file is lock another filelock cannot obtain the
        lock """
        with file_lock:
            with pytest.raises(IOError):
                with FileLock(protected_file, timeout=.2):
                    pass

    def test_lock_files_created(self, file_lock):
        """ ensure the lock files are created with context manager """
        with file_lock:
            assert exists(file_lock.lockfile)
        assert not exists(file_lock.lockfile)
