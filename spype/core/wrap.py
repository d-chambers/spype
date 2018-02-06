"""
Wrap class. Used to wrap tasks defined in pype processing lines.
"""
import inspect
from copy import deepcopy
from functools import partial
from types import MappingProxyType as MapProxy
from typing import Optional, Union, Mapping

from spype.constants import conditional_type, CALLBACK_NAMES
from spype.core import pype
from spype.core import task
from spype.core.sbase import _SpypeBase
from spype.exceptions import TaskReturnedNone, NoReturnAnnotation
from spype.types import compatible_callables, valid_input
from spype.utils import (args_kwargs, iterate, de_args_kwargs,
                         partial_to_kwargs, sig_to_args_kwargs,
                         function_or_class_name)


# --- Callback Descriptor


class _CallbackDescriptor:
    """ A simple descriptor """

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        assert callable(value)
        instance._callbacks[self.name].append(value)

    def __get__(self, instance, owner):
        return instance._callbacks[self.name]

    def __delete__(self, instance):
        instance._callbacks[self.name] = []


# ------------------------- Wrap class


class Wrap(_SpypeBase):
    """
    Class to encapsulate a task.
    """

    # wrap attributes that can be used to alter flow of data
    _wrap_funcs = ('iff', 'fan', 'agg', 'partial', 'fit', 'compatible')

    # supported features
    _in_edge_labels = {'is_conditional', 'is_aggregate'}
    _out_edge_labels = {'is_fan'}
    _fnames = {'predicate', 'adapter', 'aggregate_on', 'dependencies'}
    _feature_names = frozenset(_in_edge_labels | _fnames | _out_edge_labels)

    # functions for modifying how wrap instances take or puts items on queue
    _after_task_funcs = None  # functions for putting work on queue
    _before_task_funcs = None  # functions for taking work from queue

    # wrap level callbacks
    on_success = _CallbackDescriptor()
    on_failure = _CallbackDescriptor()
    on_finish = _CallbackDescriptor()
    on_start = _CallbackDescriptor()

    pype = None

    def __init__(self, task: 'task.Task', **kwargs):
        # if a wrap is fed to init just strip out task
        if isinstance(task, Wrap):
            self.__dict__.update(task.__dict__)
        else:
            kwargs = dict(kwargs)
            self.task = task
            # a dict that stores features of wrap
            self.features = dict.fromkeys(self._feature_names)
            self.features['dependencies'] = []
            # set callbacks on the wrap level
            self._callbacks = {cb_name: list() for cb_name in CALLBACK_NAMES}
            for callback in (set(kwargs) & set(CALLBACK_NAMES)):
                setattr(self, callback, kwargs.pop(callback))
            # setup dependencies
            self.session_fixtures = {'wrap': self}  # fixtures that dont change
            # dependencies that change for each object {param_name: task}
            self._partials = {}
            # functions that modify how objects are queued
            self._get_queue_func = None
            self._put_queue_func = None
            self.adapter = None  # todo get rid of this
            assert not len(kwargs), f'{kwargs} are not supported arguments'

    def __repr__(self):
        return f'task wrap of {self.task}'

    def __call__(self, *args, _pype_fixtures=None, **kwargs):
        fixtures = MapProxy({**(_pype_fixtures or {}), **self._wrap_fixtures,
                             **self._partials})
        out = self.task.run(*args, **kwargs, _fixtures=fixtures,
                            _callbacks=self._callbacks)
        if out is None:
            raise TaskReturnedNone
        return args_kwargs(out, adapter=self.adapter)

    # # ---- supported operators

    def __or__(self, other):
        return pype.Pype(self) | other

    def __rshift__(self, other):
        return pype.Pype(self).__rshift__(other)

    def __lshift__(self, other):
        return pype.Pype(self).__lshift__(other)

    # --- methods for adding features to wraps

    def partial(self, **kwargs) -> 'Wrap':
        """
        Set values for paramters.

        If this task does not receive all the required arguments the ones set
        with this function will be used.

        """
        for item, val in kwargs.items():
            if item not in self.signature.parameters:
                msg = (f'{item} is not a valid paramter of {self.task_name}, '
                       f'valid parameters are {set(self.signature.parameters)}')
                raise TypeError(msg)
            if isinstance(val, task.Task):
                # add task to list of dependencies
                self.features['dependencies'].append(val)
            self._partials[item] = val
        return self

    par = partial  # alias for lazy people like myself

    def iff(self, predicate: Optional[conditional_type] = None) -> 'Wrap':
        """
        Register a condition that must be true for data to continue in pype.

        Parameters
        ----------
        predicate
            A function that takes the same inputs as the task and returns a
            boolean.

        Returns
        -------
        Wrap
        """
        predicate_list = list(iterate(predicate))
        if not predicate_list:
            return self  # do do anything for None
        for func in iterate(predicate):  # ensure compatible signatures
            self._check_condtion(func)
        self.features['predicate'] = predicate
        self.features['is_conditional'] = True
        self._before_task_funcs = _iff
        return self

    def _check_condtion(self, condition: conditional_type):
        """ format and check conditional inputs to be bound to instance """
        # ensure we are dealing with a list of callables
        if self.task.get_option('check_compatibility'):
            compat = compatible_callables(self.task, condition,
                                          func1_type='input')
            if not compat:
                msg = (f'run method incompatible with {condition} '
                       f'for {self.task}')
                raise TypeError(msg)
        return condition

    def fan(self) -> 'Wrap':
        """
        Mark Wrap as fanning out.

        This will cause it to iterate output and queue one item at a time.
        """
        self._before_task_funcs = _fan
        self.features['is_fan'] = True
        # TODO make function to strip sequence rather than disable type_check
        self.check_type = False
        return self

    def agg(self, scope='object') -> 'Wrap':
        """
        Mark wrap as aggregating output from input tasks.

        This will store all outputs of previous task in a list then feed to
        this task when it is done.
        """
        self._before_task_funcs = partial(_aggregate, scope=scope)
        self.features['is_aggregate'] = True
        return self

    def fit(self, *args):
        """
        Method to adapt order/name of the outputs.

        This is useful if the output order/name needs to be adjusted to work
        with the next Wrap in the Pype.

        Parameters
        ----------
        args
            A sequence of ints/strings for mapping output into args and kwargs

        Returns
        -------
        Wrap instance
        """
        if len(args):
            self.features['adapter'] = args
        return self

    # --- validation machinery

    def compatible(self, other: Union['task.Task', 'Wrap'],
                   extra_params: Optional[Mapping] = None) -> bool:
        """
        Return True if self (current wrap) provides valid inputs for other.

        Parameters
        ----------
        other
            Another task or wrap
        extra_params
            A mapping of extra parameters
        Returns
        -------
        bool
        """
        if isinstance(other, task.Task):
            other = other.wrap()  # ensure we are working with a wrap
        if not isinstance(other, Wrap):
            return False
        adapter = self.features.get('adapter', None)
        try:
            args, kwargs = sig_to_args_kwargs(self.signature, adapter)
        except NoReturnAnnotation:  # if there is no return annotation
            return True
        sig = other.signature
        check_type = self.check_type and other.check_type
        # check if raw inputs work, if not look in extra_params for args
        if valid_input(sig, *args, check_type=check_type, **kwargs):
            return True
        extra_params = extra_params or other._partials
        # else determine if any params should be provided by extra_params
        return _compatible_extra(sig, args, kwargs, extra_params, check_type)

    @property
    def signature(self) -> inspect.Signature:
        """ return signature which indicates the arguments expected as
        input, excluding partials  """
        return self.task.get_signature()

    @property
    def _wrap_fixtures(self):
        """ return potential fixtures """
        return {**self._partials, **self.session_fixtures}

    def _validate_callbacks(self):
        """
        Raise TypeError if not all wrap and task callbacks are valid.
        """
        for name in CALLBACK_NAMES:
            wrap_cbs = list(iterate(getattr(self, name, None)))
            task_cbs = list(iterate(getattr(self.task, name, None)))
            for cb in wrap_cbs + task_cbs:
                self.task.validate_callback(cb)

    # --- methods for controlling how data flow through wrap

    def _queue_up(self, inputs, _meta, que, sending_wrap=None,
                  used_functions=None):
        """
        Add this task onto que with given inputs.

        Normally this wrap and inputs are simply appended to the que, unless
        a special queue function is defined to allow custom behavior.
        """
        # bail out early if nothing special needs to happen
        if not (sending_wrap._after_task_funcs or self._before_task_funcs):
            que.append((self, inputs))
            return
        # get the functions that should be executed. If None do normal queue
        after_funcs = set(iterate(sending_wrap._after_task_funcs))
        before_funcs = set(iterate(self._before_task_funcs))
        used_funcs = set(used_functions) if used_functions else set()
        # if there are funcs to call after operating on data
        # current no after funcs should be un-used
        assert not (after_funcs - used_funcs)
        # if there are funcs to call before allowing task to operate on data
        if (before_funcs - used_funcs):
            for func in (before_funcs - used_funcs):
                func(self, inputs, _meta, que, sending_wrap, used_funcs)
            return
        que.append((self, inputs))  # else just do the normal thing

    def copy(self) -> 'Wrap':
        return deepcopy(self)

    @property
    def task_name(self):
        """ return the short name of the wrapped task """
        return function_or_class_name(self.task)

    @property
    def conditional_name(self):
        """ return the name of the predicate, else None """
        return function_or_class_name(self.features['predicate'])

    @property
    def _in_edge_lab(self) -> set:
        """ return a set of str that classify the type of edge leading to
        this task """
        return {x for x in self._in_edge_labels if self.features[x]}

    @property
    def _out_edge_lab(self) -> set:
        """ return a set of labels for the type of edge leaving this task """
        return {x for x in self._out_edge_labels if self.features[x]}

    def __str__(self):
        return str(self.signature)


# ----------- functions to control how data is put on the queue

def _iff(wrap: Wrap, inputs, _meta, que, sending_wrap=None,
         used_functions=None):
    """
    Function to ensure some condition(s) are true else dont put data on queue.
    """
    for func in iterate(wrap.features['predicate']):
        if not func(*inputs[0], **inputs[1]):
            return  # if a condition fails bail out
    wrap._queue_up(inputs, _meta, que, sending_wrap, used_functions={_iff})


def _fan(wrap: Wrap, inputs, _meta, que, sending_wrap=None,
         used_functions=None):
    """ fan out the output of sending task """
    for val in reversed(de_args_kwargs(*inputs)):
        wrap._queue_up(args_kwargs(val), _meta, que, sending_wrap,
                       used_functions={_fan})


def _aggregate(wrap: Wrap, inputs, _meta, que, sending_wrap=None,
               used_functions=None, scope='object'):
    """ aggregate outputs coming from sending wrap to call on warp """
    # determine if aggregation has taken place yet
    if not _meta[scope + '_scope_finished']:
        # if not aggregate and exit
        _meta[scope + '_scope_map'][wrap] = sending_wrap
        puts = de_args_kwargs(*inputs)
        _meta[scope + "_aggregates"][sending_wrap].append(puts)
        return


# -------------------- misc wrap functions

def _compatible_extra(sig, args, kwargs, extra_params, check_type):
    """
    Determine if args, kwargs are valid inputs to sig if supplemented with
    extra_params
    """
    extra_params = extra_params or {}
    commons = set(sig.parameters) & set(extra_params)
    if not commons:
        return False
    # iterate over commons and pull out values or parameter annotations
    partials = {}
    for common in commons:
        new = extra_params[common]
        # need to extact type from task signature return value
        if isinstance(new, (task.Task, Wrap)):
            partials[common] = Wrap(new).signature.return_annotation
        else:  # else this is the desired value, just get the type
            partials[common] = type(new)
    new_kwargs = partial_to_kwargs(None, *args, partial_dict=partials,
                                   signature=sig, **kwargs)
    return valid_input(sig, check_type=check_type, **new_kwargs)
