"""
A number of utilities for sflow
"""
import copy
import inspect
import os
import time
import types
from collections import Sequence, OrderedDict
from inspect import signature, Signature
from typing import Optional, Callable, Tuple, Set

from spype.constants import adapt_type, args_kwargs_type
from spype.exceptions import NoReturnAnnotation

# ----------------------- context stuff


_SFLOW_CONTEXT = dict(
    check_type=True,
    check_compatibility=True,
    on_failure=None,
    on_success=None,
    on_finish=None,
    on_start=None,
    print_flow=False,
)


class Context:
    """ A class for controlling modifications made to a dictionary,
    used to sensibly store global state """

    def __init__(self, input_dict: dict):
        """
        Parameters
        ----------
        input_dict
            A dictionary for holding modifiable state.
        """
        self._dict = input_dict
        self._previous_state = {}

    def __call__(self, _save_state=True, **kwargs):
        """
        Set global options for how spype behaves.

        If an unsupported value is set a KeyError will be raised.
        """
        if not set(kwargs).issubset(self._dict):
            diff = set(kwargs) - set(self._dict)
            msg = (f'unsupported option(s): {diff} passed to set_options. '
                   f'supported uptions are {set(self._dict)}')
            raise KeyError(msg)

        if _save_state:
            self._previous_state = copy.deepcopy(self._dict)
        self._dict.update(kwargs)
        return self

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, key, value):
        self(**{key: value})

    def items(self):
        return _SFLOW_CONTEXT.items()

    def __enter__(self):
        pass

    def __repr__(self):
        return str(self._dict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self(_save_state=False, **self._previous_state)


context = Context(_SFLOW_CONTEXT)


# ------------------- File lock for thread-safe log


class FileLockException(Exception):
    pass


class FileLock(object):
    """
    File lock based on https://github.com/dmfrey/FileLock
    """

    def __init__(self, file_path, timeout=10, delay=.1):
        """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
        self.file_path = file_path
        self.lockfile = f"{file_path}.lock~"
        self.timeout = timeout
        self.delay = delay

    @property
    def is_locked(self):
        return os.path.exists(self.lockfile)

    def lock(self):
        """ create the lock file """
        with open(self.lockfile, 'w') as fi:
            fi.write('occupied')

    def release(self):
        """ release the lock """
        if self.is_locked:
            os.remove(self.lockfile)

    def acquire(self):
        """
        Try to acquire the lock.
        """
        start_time = time.time()

        while (time.time() - start_time) < self.timeout:
            if self.is_locked:  # try sleeping a bit
                time.sleep(self.delay)
            else:  # create lock file and go about your business
                self.lock()
                break
        else:
            msg = f'{self.lockfile} still exists after {self.timeout} seconds'
            raise IOError(msg)

    def __enter__(self):
        """ Activated when used in the with statement.
            Should automatically acquire a lock to be used in the with block.
        """
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
        self.release()

    def __del__(self):
        """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
        self.release()


# ------------------------- function jiggering


def partial_to_kwargs(func: Callable, *args, partial_dict: Optional[dict] = None,
                      signature: Optional[inspect.Signature] = None, **kwargs
                      ) -> dict:
    """
    Return a kwargs dict compatible with function or siganture.

    Parameters
    ----------
    func
        A callable
    partial_dict
        A dict that may have keys named the same as arguments expected by
        func
    signature
        A signature object, if None then get if from function.
    """
    out = dict(kwargs)
    sig = signature or inspect.signature(func)
    argd = OrderedDict(((item, value) for item, value in sig.parameters.items()
                        if item not in partial_dict))
    # first bind new args taking out any that are also found in partial_dict
    out.update({name: value for name, value in zip(argd, args)})
    # get kwargs to bind
    shared_keys = set(partial_dict) & set(sig.parameters)
    out.update({item: partial_dict[item] for item in shared_keys})
    return out


def apply_partial(func: Callable, *args, partial_dict: Optional[dict] = None,
                  signature: Optional[inspect.Signature] = None, **kwargs
                  ) -> Tuple[tuple, dict]:
    """
    Call func with args and kwargs, supersede with partial_dict.

    Inspects a callable and if any argument names match keys in partial
    those will be applied.

    Parameters
    ----------
    func
        A callable
    partial_dict
        A dict that may have keys named the same as arguments expected by
        func
    signature
        A signature object, if None then get if from function.

    Returns
    -------
    Tuple of args and kwargs which can be input into func
    """
    if not partial_dict:  # bail out if no special binding to perform
        return func(*args, **kwargs)
    out = partial_to_kwargs(func, *args, partial_dict=partial_dict,
                            signature=signature, **kwargs)
    return func(**out)


# --------------------- Args and Kwargs Wrangling


def args_kwargs(output, adapter: Optional[adapt_type] = None
                ) -> args_kwargs_type:
    """
    Take the output of a function and turn it into args and kwargs.

    Parameters
    ----------
    output
        Any output from a function
    adapter
        A sequence of ints/strings for mapping output into args and kwargs

    Returns
    -------
    tuple
        A tuple of args and kwargs
    """
    if output is None:
        return (), {}
    if not isinstance(output, tuple):  #
        output = (output,)
    if adapter is None:
        return tuple(output), {}
    assert len(adapter) == len(output), (
        f'adapter {adapter} and output {output} have different lengths')
    # wrangle output into a tuple and a dict based on adapter
    return _apply_adapter(output, adapter)


def _apply_adapter(output, adapter):
    """ apply an adapter tuple to an output tuple """
    out_list = {}
    out_dict = {}
    for val, item in zip(output, adapter):
        if isinstance(item, int):
            out_list[item] = val
        elif isinstance(item, str):
            out_dict[item] = val
    # change out_list dict into a tuple
    assert set(out_list) == set(range(len(out_list)))
    out_tuple = tuple(out_list[x] for x in range(len(out_list)))
    return out_tuple, out_dict


def de_args_kwargs(args, kwargs):
    """
    Take args and kwargs and turn it into a simple tuple.
    """
    out = tuple([x for x in args] + [val for _, val in kwargs.items()])
    if len(out) == 1:  # unpack if len is 1
        out = out[0]
    return None if out is () else out


def get_default_names(sig: inspect.Signature) -> Set[str]:
    """
    Return a set of parameter names that have default values.
    """
    return {key for key, value in sig.parameters.items()
            if value.default is not inspect._empty}


def sig_to_args_kwargs(sig: inspect.Signature,
                       adapter: Optional[tuple] = None) -> (tuple, dict):
    """
    Return an tuple of args and kwargs of types for signature return type.

    If no return annotation is given raise a NoReturnAnnotation Exception.

    Parameters
    ----------
    sig
        The signature that may have return annotations attached.
    adapter
        A tuple of ints, None, or str to re-arrange the outputs.

    Returns
    -------
    args and kwargs
    """
    sig = sig if isinstance(sig, Signature) else signature(sig)
    # get output args
    if isinstance(sig.return_annotation, tuple):
        args = sig.return_annotation
    elif sig.return_annotation is inspect._empty:
        raise NoReturnAnnotation
    # if this is a Tuple typehint strip out args
    elif sig.return_annotation.__class__ == Tuple.__class__:
        args = sig.return_annotation.__args__
    else:
        args = (sig.return_annotation,)
    # wrangle into args and kwargs
    kwargs = {}
    if adapter:
        args, kwargs = _apply_adapter(args, adapter)
    return args, kwargs


# ------------------------ misc functions


def iterate(obj):
    """ return an iterable object from any sequence or non-sequence. Return
     empty tuple if None """
    if obj is None:
        return ()
    if isinstance(obj, str):
        return (obj,)
    return obj if isinstance(obj, Sequence) else (obj,)


def function_or_class_name(obj):
    """
    Given a callable, try to determine its name. Return 'None' if None.
    """
    try:  # for decorator tasks
        return obj.__name__
    except AttributeError:  # for class Tasks
        return obj.__class__.__name__


def copy_func(f, name=None):
    '''
    return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    '''
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
                            f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn
