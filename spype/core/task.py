import abc
import copy
from inspect import Signature
from collections import ChainMap
from contextlib import suppress
from functools import partial
from types import MappingProxyType as MapProxy
from typing import TypeVar, Optional, Callable, Any, Mapping, Sequence

import spype
from spype.constants import (FIXTURE_NAMES, CALLBACK_NAMES, PYPE_FIXTURES,
                             WRAP_FIXTURES)
from spype.core import wrap
from spype.core.sbase import _SpypeBase
from spype.exceptions import ReturnCallbackValue, UnresolvedDependency, ExitTask
from spype.types import valid_input, compatible_instance
from spype.utils import (iterate, apply_partial, de_args_kwargs, copy_func,
                         get_default_names, function_or_class_name)


_fixtures = {**dict.fromkeys(PYPE_FIXTURES), **dict.fromkeys(WRAP_FIXTURES)}
EMPTY_FIXTURES = MapProxy(_fixtures)


# ----------------------------- Task class


class _TaskMeta(abc.ABCMeta):
    """ meta class for Tasks """

    def __instancecheck__(self, instance):
        """ special check to see if instance is task-like """
        return hasattr(instance, 'run') and callable(instance)


def _get_return_type(bound):
    """ based on bound signature get the expected return type, mainly
    this function is to account for typevars """
    # check if there are any typevars, swap these out for types bound
    out = list(iterate(bound.signature.return_annotation))

    if any([isinstance(x, TypeVar) for x in out]):
        # get the expected value for typevars by transversing signature

        params = bound.signature.parameters
        vals = bound.arguments
        new_types = {val.annotation: type(vals[item])
                     for item, val in params.items()
                     if isinstance(val.annotation, TypeVar)}
        # swap out return annotations
        for num, value in enumerate(out):
            if value in new_types:
                out[num] = new_types[value]
        return tuple(out) if len(out) > 1 else out[0]

    return bound.signature.return_annotation


class Task(_SpypeBase, metaclass=_TaskMeta):
    """
    An abstract class whose subclasses encapsulate a unit of work.
    """

    # spype attributes
    signature: Optional[Signature] = None
    pype: Optional['Pype'] = None
    _decorator_task: bool = False
    supported_fixtures = FIXTURE_NAMES
    _is_task_wrap = False  # if this is a function wraped with task
    _option_getter = _SpypeBase()

    def __init_subclass__(cls):
        # ensure a run method is defined
        if '__call__' not in cls.__dict__:
            # if not hasattr(cls, '__call__') or not callable(cls.run):
            msg = f'{cls} must have a __call__ method defined'
            raise TypeError(msg)
        # ensure run method is not defined (this would get overwritten)
        if 'run' in cls.__dict__:
            msg = (f'{cls} defines a run method, this is not permitted')
            raise TypeError(msg)

    def get_signature(self) -> Signature:
        """ return signature of bound run method """
        if self.signature is None:
            self.signature = spype.signature(self.__call__)
        return self.signature

    def get_option(self, option: str) -> Any:
        """
        Returns an option defined in self or defer to Task MRO.

        Parameters
        ----------
        option
            A supported Spype option. See spype.options.
        """
        try:
            return getattr(self, option)
        except AttributeError:  # this happens when self is a function
            return getattr(Task, option)

    def get_name(self) -> str:
        """
        Return the name of task.
        """
        return function_or_class_name(self)

    def run(self, *args, _fixtures: Optional[Mapping[str, Any]] = None,
            _callbacks: Optional[Sequence[Callable]]=None, **kwargs):
        """
        Call the task's __call__ and handle spype magic in the background.

        Run essentially performs the following steps:
            1. Try to bind args and kwawrgs to the task signature
            2. If bind raises, look for missing arguments in _fixtures
            3. Rebind args and kwargs to signature with new args if needed
            4. Run on_start callback, if defined
            5. Run task call method (or function)
            6. Run on_failure callback if defined and an exception was raised
            7. Run on_success callback if defined and no exception was raised
            8. Run on_finsih callback, if defined
            9. Return output of call method, or output of any callback if
               any non-None values were returned.

        Parameters
        ----------
        _fixtures
            A dict of fixtures. Keys are paramters that might be used by
            callbacks and values are the values to substitute.
        _callbacks
            A
        """
        # get fixtures passed in from wraps/pypes or use empty dicts
        _fixtures = _fixtures or EMPTY_FIXTURES
        meta = _fixtures.get('meta', {}) or {}  # meta dict from pype or {}
        # a proxy of outputs from previous tasks
        task_outputs = meta.get('outputs', {})
        # get a signature and determine if type checking should happen
        sig = self.get_signature()
        check_type = self.get_option('check_type')
        # get bound arguments raise Appropriate Exceptions if bad imputs
        bound = self._bind(sig, args, kwargs, _fixtures, task_outputs)
        out_type = self._check_inputs(bound, *args, **kwargs,
                                      check_type=check_type)
        # create a dictionary of possible fixtures callbacks can ask for
        control = dict(task=self, self=self, signature=sig, e=None,
                       outputs=None, inputs=(args, kwargs), args=args,
                       kwargs=kwargs, )
        fixtures = ChainMap(control, _fixtures)
        try:
            self._run_callbacks('on_start', fixtures, _callbacks)
            try:
                fixtures['outputs'] = self(*bound.args, **bound.kwargs)
            except Exception as e:
                fixtures['e'] = e
                self._run_callbacks('on_failure', fixtures, _callbacks)
            else:
                self._run_callbacks('on_success', fixtures, _callbacks)
            finally:
                self._run_callbacks('on_finish', fixtures, _callbacks)
        # if a callback returned a value or raised a
        except ReturnCallbackValue:  # if a callback returns a value
            pass
        except ExitTask:
            return
        if meta.get('print_flow'):
            print(f'{self.get_name()} got {(bound.args, bound.kwargs)} and '
                  f'returned {fixtures["outputs"]}')
        return self._check_outputs(fixtures['outputs'], out_type,
                                   check_type=check_type)

    # --- validators

    def _bind(self, signature, args, kwargs, fixtures, outputs):
        """
        Bind args and kwargs to signature. If it fails, look for fixture that
        may satisfy binding. If it does not have a value yet, raise an
        UnresolvedDependency Exception.
        """
        defaults = get_default_names(signature)
        _kwargs = {key: fixtures[key] for key in (defaults & set(fixtures))}

        try:
            bind = signature.bind(*args, **{**kwargs, **_kwargs})
        except TypeError:  # need to look for fixtures
            params = signature.parameters
            # determine if any unresolved dependencies exist and raise if so
            overlap_keys = set(params) & set(fixtures)
            # get values that should be given to parameters.
            values = {}
            for key in overlap_keys:
                if fixtures[key] in outputs:
                    args_, kwargs_ = outputs[fixtures[key]]
                    values[key] = de_args_kwargs(args_, kwargs_)
                else:
                    if isinstance(fixtures[key], Task):
                        raise UnresolvedDependency
                    values[key] = fixtures[key]
            try:  # try binding with new inputs
                bind = apply_partial(signature.bind, *args, signature=signature,
                                     partial_dict=values, **kwargs)
            except TypeError:
                msg = (f'{args} and {kwargs} are not valid inputs for {self} '
                       f'which expects a signature of {signature}')
                raise TypeError(msg)
        return bind

    def _check_inputs(self, bound, *args, check_type=True, **kwargs):
        """
        Ensure the inputs are of compatible types with the signature and get
        return type.

        Also take into account possible fixture values.
        """
        sig = bound.signature
        valid = valid_input(sig, *args, bound=bound, check_type=check_type,
                            **kwargs)
        if not valid:
            msg = (f'{args} and {kwargs} are not valid inputs for {self} '
                   f'which expects a signature of {sig}')
            raise TypeError(msg)

        return _get_return_type(bound)

    def _check_outputs(self, out, out_type, check_type=True):
        """ if out is not None, check compatibility """
        if out is None:  # bail early on None (none always should work)
            return None
        if check_type and not compatible_instance(out, out_type):
            msg = (f'task: {self} returned: {out} which is not consistent with '
                   f'expected output type of: {out_type}')
            raise TypeError(msg)
        return out

    # --- callback stuff

    def _run_callbacks(self, name: str, control: dict, _callbacks=None):
        """ call the callbacks of type name. If a single callback is
        defined just call it, else iterate sequence and call each """
        ex_callbacks = [] if _callbacks is None else _callbacks.get(name, [])
        func = ex_callbacks + list(iterate(self.get_option(name)) or [])

        # set default raise if not on_failures are set
        if not func and name == 'on_failure':
            func = raise_exception

        for callback in iterate(func):
            # if this is a bound method, revert to unbound
            # this enables using functions that havent defined self
            with suppress(AttributeError):
                callback = callback.__func__
            # determine needed values from original task parameters
            cb_sig = spype.signature(callback)  # callback signature
            expected = set(cb_sig.parameters) & set(control['signature'].parameters)
            if expected:
                signature = control['signature']
                args, kwargs = control['args'], control['kwargs']
                bound_args = signature.bind(*args, *kwargs).arguments
                kwargs = {item: bound_args[item] for item in expected}
            else:
                kwargs = {}
            # run callback
            out = apply_partial(callback, partial_dict=control, **kwargs)
            # if any output raise so task returns that
            if out is not None:
                control['outputs'] = out
                raise ReturnCallbackValue

    def validate_callback(self, callback: Callable) -> None:
        """
        Raise TypeError if callback is not a valid callback for this task.

        Parameters
        ----------
        callback
            Any callable
        """
        assert callable(callback)
        sig = spype.signature(callback)
        call_params = set(self.get_signature().parameters)
        supported = set(FIXTURE_NAMES) | call_params
        if not set(sig.parameters).issubset(supported):
            unsupported_parrams = set(sig.parameters) - FIXTURE_NAMES
            msg = (f'{unsupported_parrams} are not a valid parameter names '
                   f'for a task callback fixtures. Supported fixtures are: '
                   f'{FIXTURE_NAMES} and call params are {call_params}')
            raise TypeError(msg)

    def validate_callbacks(self) -> None:
        """
        Iterate over all attached callbacks and raise TypeError if
        any problems are detected.
        """
        for name in CALLBACK_NAMES:
            for cb in iterate(getattr(self, name, None)):
                self.validate_callback(cb)

    # --- dunders

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """ Task subclasses must implement their own call methods. """

    def __getattr__(self, item):
        # if the item is supported by _Task class return wrapped task
        if item in wrap.Wrap._wrap_funcs:
            return getattr(wrap.Wrap(self), item)
        else:
            raise AttributeError

    def __str__(self):
        return f'{self.__class__.__name__} instance'

    def __repr__(self):
        return str(self)

    # --- supported operators

    def __or__(self, other):
        return wrap.Wrap(self) | other

    def __lshift__(self, other):
        return wrap.Wrap(self).__lshift__(other)

    def __rshift__(self, other):
        return wrap.Wrap(self).__rshift__(other)

    # --- misc

    def wrap(self, *args, **kwargs) -> 'wrap.Wrap':
        """
        Instantiate a Wrap instance from this task.

        Args and kwargs are passed to Wrap constructor.

        Returns
        -------
        Wrap
        """
        return wrap.Wrap(self, *args, **kwargs)

    def copy(self) -> 'Task':
        """
        Return a deep copy of task.
        """
        try:  # if this is a function task
            return copy_func(self)
        except AttributeError:  # else an class task
            return copy.deepcopy(self)


# --------------------------- Task decorator


def task(func: Optional[Callable] = None, *,
         on_start: Optional[Callable] = None,
         on_failure: Optional[Callable] = None,
         on_success: Optional[Callable] = None,
         on_finish: Optional[Callable] = None,
         **kwargs) -> Task:
    """
    Decorator for registering a callable as a tasks.

    This essentially adds the Task class attributes to a function and returns
    the function. This means the function will behave as before, but will
    have the Task class attributes attached. This approach is needed so that
    the tasks are pickable, else returning Task instances would work.

    Parameters
    ----------
    func:
        A callable to use as a task
    on_start:
        Callable which is called before running task
    on_failure:
        Callable which will be called when a task fails
    on_success:
        Callable that gets called when a task succeeds
    on_finish:
        Callable that gets called whenever a task finishes

    Returns
    -------
    Task
        An instance of Task

    """
    # handle called decorator (ie no function passed yet)
    if func is None:
        return partial(task, on_failure=on_failure, on_success=on_success,
                       on_finish=on_finish, on_start=on_start, kwargs=kwargs)

    for name, item in Task.__dict__.items():
        if callable(item):
            setattr(func, name, item.__get__(func))  # binds functions to func
    # set all task class attributes of Task using func as self
    update_dict = {it: val for it, val in locals().items()
                   if it != 'func' and val is not None}
    func.__call__ = func
    func.signature = spype.signature(func)
    func.__dict__.update(update_dict)
    # add all attributes that return a wrapped task
    for item in wrap.Wrap._wrap_funcs:
        def _func(*args, item=item, **kwargs):
            wrap_ = wrap.Wrap(func)
            wrap_func = getattr(wrap_, item)
            return wrap_func(*args, **kwargs)

        setattr(func, item, _func)
    return func


# ---------------------------- special tasks


class PypeInput(Task):
    """
    A singleton Task that is used, explicitly or implicitly, to begin
    each pype.
    """
    singleton_instance = None

    def __new__(cls, *args, **kwargs):  # ensure singleton is instantiated
        if cls.singleton_instance is None:
            new = super().__new__(cls)
            cls.singleton_instance = new
        return cls.singleton_instance

    def __call__(self, *args):
        return args


pype_input = PypeInput()


class Forward(Task):
    """
    A task for simply forwarding inputs to the next task.
    """

    def __call__(self, *args):
        return args


forward = Forward()


# --------------------------- misc functions


def raise_exception(e):
    raise e
