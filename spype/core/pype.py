"""
Pype class and supporting functions.
"""

import itertools
import time
from collections import defaultdict, deque
from contextlib import suppress
from copy import deepcopy
from types import MappingProxyType as MapProxy
from typing import Union, Dict, Optional, Callable, Any, List, Sequence, Hashable

from spype.callbacks import debug_callback
from spype.constants import CALLBACK_NAMES, HOW_ARGS
from spype.core import task
from spype.core import wrap
from spype.core.digraph import _WrapDiGraph
from spype.core.sbase import _SpypeBase
from spype.exceptions import UnresolvedDependency, TaskReturnedNone
from spype.utils import iterate, args_kwargs, de_args_kwargs

pype_input_type = Optional[Union["task.Task", "wrap.Wrap", "Pype", str]]


class Pype(_SpypeBase):
    """
    Class to control the data flow between tasks.

    Parameters
    ----------
    arg
        A task, pype, or any hashable that has been registered as a pype.
        If hashable, a copy of the registered pype will be returned.
    """

    _registered_pypes = {}  # a store for optionally registering pypes
    name = None

    def __init__(self, arg: pype_input_type = None, name: Optional[Hashable] = None):
        """

        """
        # graphs for data flow and data dependency
        self.flow = _WrapDiGraph()
        self.flow.add_wrap(task.pype_input.wrap())
        self.dependencies = _WrapDiGraph()
        # validation state and counts
        self.validated = False
        # dict for outputs of last call
        self.outputs = {}
        self._partials = {}
        # add first task to pype
        if arg is not None:  # add task to pype
            with suppress(TypeError):
                arg = self.__class__._registered_pypes.get(arg, arg)
            _connect_to_pype(self, arg, inplace=True)
        self.register(name or getattr(arg, "name", None))

    # --- task hookup

    def __or__(self, other):
        return _connect_to_pype(self, other)

    def __ior__(self, other):
        return _connect_to_pype(self, other, inplace=True)

    def __and__(self, other):
        return _connect_to_pype(self, other, how="first")

    def __iand__(self, other):
        return _connect_to_pype(self, other, how="first", inplace=True)

    def __lshift__(self, other):
        return _connect_to_pype(self, other, wrap_func=lambda x: x.fan())

    def __rshift__(self, other):
        def task_func(x):
            return x.agg(scope="object")

        return _connect_to_pype(self, other, wrap_func=task_func)

    # --- call stuff

    def __call__(self, *args, **kwargs):
        if not self.validated:  # validate pype if it is not already
            self.validate()
        _meta = self._create_run_dict()  # create main control structure
        # iterate over iterable, extract args and kwargs and run queue
        que = deque()
        que.append((self.flow.get_input_wrap(), (args, kwargs)))
        self._run_queue(_meta, que)
        # set results
        self.outputs = _meta["outputs"]
        return _meta["output"][0]

    def _run_queue(self, _meta, que):
        """ run the queue until complete """
        # run que until complete or all tasks are waiting agg results
        assert self.flow.get_input_wrap().task is task.pype_input
        fixtures = MapProxy({"meta": _meta, "pype": self, **self._partials})

        while len(que):
            wrap_, (args, kwargs) = que.pop()
            wrap_: wrap.Wrap
            try:
                output = wrap_(*args, **kwargs, _pype_fixtures=fixtures)
            except UnresolvedDependency:  # task needs to be put back
                _meta["defer_count"][wrap_] += 1  # up task deferment counter
                que.appendleft((wrap_, (args, kwargs)))
                continue
            except TaskReturnedNone:  # task returned None
                continue
            else:  # everything went fine
                _meta["outputs"][wrap_.task] = output
                for neighbor in self.flow.neighbors(wrap_):  # queue neighbors
                    neighbor._queue_up(output, _meta, que, sending_wrap=wrap_)
        # run tasks that waited for object scoped aggregations
        self._run_aggregations(_meta, que)
        _meta["output"].append(de_args_kwargs(*output))

    def _run_aggregations(self, _meta, que):
        """ run aggregated values """
        # collect aggregated things and run queue again
        for wrap_ in _meta["object_scope_map"]:
            # dont run again if object scope is finished
            if wrap_ in _meta["object_scope_finished"]:
                continue
            _meta["object_scope_finished"].add(wrap_)  # mark as complete
            needed_task = _meta["object_scope_map"][wrap_]
            inputs = _meta["object_aggregates"][needed_task]
            if len(inputs):
                que.append((wrap_, args_kwargs(inputs)))
        if len(que):  # if there is anything on the queue run it again
            self._run_queue(_meta, que)

    def _create_run_dict(self) -> Dict:
        """
        Creates the main control structure for running Spype
        """
        out = dict(
            time=time.time(),  # start time of the call
            outputs={},  # store for most recent output of each task
            # a set of tasks that have completed aggregations on needed scope
            object_scope_finished=set(),
            # map of tasks that need outputs of other tasks for aggregating
            object_scope_map={},
            # map of tasks to their outputs for both applicable scopes
            object_aggregates=defaultdict(list),
            # a counter of how many times a task has been deferred
            defer_count=defaultdict(lambda: 0),
            output=[],  # a list for storing results
            pype_inputs=None,  # store inputs for first pype
            print_flow=getattr(self, "print_flow", False),
        )
        return out

    # --- validation

    def validate(self):
        """
        Run checks on the pype to detect potential problems.

        Will raise an InvalidPype exception if compatibility issues are found,
        or a TypeError if any invalid callbacks are found.
        """
        # validate task compatibility
        self.flow.validate(extra_params=MapProxy(self._partials))
        # validate callbacks
        for wrap_ in self.flow.wraps:
            wrap_._validate_callbacks()

    # --- flow control attributes

    def iff(self, predicate: Callable[[Any], bool], inplace=False) -> "Pype":
        """
        Run data through the pype only if predicate evaluates to True.

        Parameters
        ----------
        predicate
            A callable that returns a boolean and takes the same inputs as
            the first task in the pype (excluding pype_input)
        inplace
            If True modify the pype in place, else return a pype with iff
            applied.

        Returns
        -------
        Pype
        """
        pype = self if inplace else self.copy()
        for wrap_ in pype._first_tasks:  # iterate first tasks and set predicate
            wrap_.iff(predicate)
        return pype

    # --- misc dunders

    def __getitem__(self, item):
        if isinstance(item, task.Task):
            return _TaskView(item, self)

    def __setitem__(self, key, value):
        if isinstance(value, task.Task):
            if value not in self.flow.tasks:
                _connect_to_pype(self, value, how="first", inplace=True)
        self._partials[key] = value

    def __len__(self):
        """ len should be equal to the number of nodes """
        return len(self.flow.wraps)

    def __str__(self):
        nodes = [x.task for x in self.flow.wraps]
        edges = [(x.task, y.task) for x, y in self.flow.edges]
        deps = [(x.task, y.task) for x, y in self.dependencies.edges]
        msg = (
            f"Pype isntance\n\nNODES: \n\n{nodes} \n\n EDGES: \n\n"
            f"{edges} \n\nDEPENDENCIES:\n\n {deps}\n"
        )
        return msg

    def __repr__(self):
        return str(self)

    def __enter__(self):
        return self.copy()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        # ensue tasks are not copied
        for task_ in self.flow.tasks:
            memo[id(task_)] = task_
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    # --- visualization

    def plot(self, file_name: Optional[str] = None, view: bool = True):
        """
        Plot the graph
        Parameters
        ----------
        file_name
            The name of the graph viz file output.
        view
            If True display the graph network.

        Returns
        -------
        Instance of graphviz.Digraph
        """
        return self.flow.plot(file_name, view)

    # --- task stuff

    @property
    def _first_tasks(self) -> List["wrap.Wrap"]:
        """ return all nodes (wraps) attached to pype input"""
        input_wrap = self.flow.get_input_wrap()
        return list(self.flow.neighbors(input_wrap))

    @property
    def _last_tasks(self):
        """
        Return all the tasks that do not connect to other tasks.
        """
        connected = {x for x in self.flow.node_map if self.flow.node_map[x]}
        return set(self.flow.wraps) - connected

    def add_callback(
        self,
        callback: callable,
        callback_type: str,
        tasks: Optional[Sequence["task.Task"]] = None,
    ) -> "Pype":
        """
        Add a callback to all, or some, tasks in the pype. Return new Pype.

        Parameters
        ----------
        callback
            The callable to attach to the tasks in the pype
        callback_type
            The type of callback: supported types are:
                on_start, on_failure, on_success, and on_exception
        tasks
            A sequence of tasks to apply callback to, else apply to all tasks.
        Returns
        -------
        A copy of Pype
        """
        assert (
            callback_type in CALLBACK_NAMES
        ), f"unsported callback type {callback_type}"
        pype = self.copy()
        # get a list of wraps to apply callbacks to
        if tasks is None:
            wraps = pype.flow.wraps
        else:
            wraps_ = [
                iterate(selected_wraps)
                for task in iterate(tasks)
                for selected_wraps in pype.flow.tasks[task]
            ]
            wraps = itertools.chain.from_iterable(wraps_)
        # apply callbacks
        for wrap_ in wraps:
            setattr(wrap_, callback_type, callback)
        return pype

    def debug(
        self, tasks: Optional[Sequence["task.Task"]] = None, callback_type="on_start"
    ) -> "Pype":
        """
        Return of copy of Pype with debugging callbacks set.

        Optionally, a list of tasks to set debug on can be defined to limit
        the breakpoints to only include those tasks. The callback where to
        debugger is called is also configurable.

        Parameters
        ----------
        tasks
            If not None a task or sequence of tasks to debug. If None
            debug all tasks.

        callback_type
            The callback to set debug function. Controls where in the
            execution cycle debugging tasks place

        Returns
        -------
        A copy of this pype.
        """
        return self.add_callback(debug_callback, callback_type, tasks=tasks)

    def register(self, name: Hashable) -> None:
        """
        Register a pype under name.

        Allows accessing the pype, or copies of it, later.

        Parameters
        ----------
        name
            Any Non-None hashable


        """
        if name is not None:
            self.__class__._registered_pypes[name] = self
            self.name = name


class _TaskView(_SpypeBase):
    """ A class for selecting a task from a pype """

    def __init__(self, task_: "task.Task", pype: Pype):
        if len(pype.flow.tasks[task_]) != 1:
            msg = (
                f"pype has {len(pype.flow.tasks[task])} instances of {task}"
                f"it must have exactly one to use this feature."
            )
            raise TypeError(msg)
        self.task = task_
        self.pype = pype
        self.wrap = pype.flow.tasks[task_][0]

    def __call__(self, *args, **kwargs):
        return self.wrap(*args, **kwargs)

    # --- overloaded operators

    def __or__(self, other):
        return _connect_to_pype(self.pype, other, how=self.task)

    def __ior__(self, other):
        return _connect_to_pype(self.pype, other, inplace=True, how=self.task)

    def __lshift__(self, other):
        def task_func(x):
            return x.fan()

        return _connect_to_pype(self.pype, other, wrap_func=task_func, how=self.task)

    def __rshift__(self, other):
        def task_func(x):
            return x.agg(scope="object")

        return _connect_to_pype(self.pype, other, wrap_func=task_func, how=self.task)


# ------------------- pype functions


pype_or_wrap = Union[Pype, "wrap.Wrap"]
pype_wrap_or_task = Union[Pype, "wrap.Wrap", "task.Task"]


def _get_attach_wraps(pype, how):
    """ return a list of Wraps which should be connected based on how arg """
    out = []  # list of tasks to be connected to other
    for arg in iterate(how):
        # make sure input is valid
        assert how in HOW_ARGS or isinstance(arg, task.Task)
        if isinstance(arg, task.Task):
            assert arg in pype.flow.tasks and len(pype.flow.tasks[arg]) == 1
            out.append(pype.flow.tasks[arg][0])
        elif how == "last":
            out += pype._last_tasks
        elif how == "first":
            out.append(pype.flow.tasks[task.pype_input][0])
    return out


def _pype_to_pype(pype1, attach_tasks, pype2):
    """ attach pype1 to pype2 on attach_tasks (list of tasks in pype1)
     modifies pype1 in place """
    pype1.flow = pype1.flow | pype2.flow
    for wrap_ in pype2._first_tasks:
        assert wrap_ in pype1.flow.wraps and wrap_ in pype1.flow.wraps
        _wrap_to_pype(pype1, attach_tasks, wrap_)


def _wrap_to_pype(pype1: Pype, attach_wraps, wrap):
    """ attach attach warps to wrap, modifies pype1 in place """
    for wrap1 in attach_wraps:
        if wrap.task is not task.pype_input:  # pype_input must always be first
            pype1.flow.add_edge(wrap1, wrap)


def _yield_first_wraps(obj: pype_or_wrap):
    """ Yield the first wrap in obj, or obj if it is a Wrap instance """
    if isinstance(obj, wrap.Wrap):
        yield obj
    else:
        for first_wrap in obj._first_tasks:
            yield first_wrap


def _apply_wrap_func(obj: pype_or_wrap, func: Callable[["wrap.Wrap"], None]):
    """ apply func to the first wrap in obj if pype or to obj itself
    if it is a wrap """
    assert isinstance(obj, (Pype, wrap.Wrap)), f"{obj} is not a wrap or pype"
    for wrap_ in _yield_first_wraps(obj):
        func(wrap_)


def _route_to_pype(route):
    """ recursively convert dict objects to pypes """
    if isinstance(route, dict):
        out = []
        for predicate, task_like in route.items():
            # make sure inputs are compatible
            assert callable(predicate) or isinstance(predicate, bool)
            out.append(_route_to_pype(task_like).iff(predicate))
        return task.pype_input | tuple(out)
    else:
        assert hasattr(route, "iff")
        return route


def _connect_to_pype(
    pype: Pype,
    other,
    how: Union[str, "task.Task"] = "last",
    inplace: bool = False,
    wrap_func: Optional[Callable] = None,
):
    """
    Add task or pype to the pype structure.

    Parameters
    ----------
    pype
        Pype to join
    other
        Pype, Task, or Wrap instance to connect to pype.
    how
        How the connection should be done. Supported options are:
        "first" : connect other to input_task of pype
        "last" : connect other to last tasks in pype
        Task instance : connect other to a specific task in pype
    inplace
        If False deepcopy pype before modfiying, else modify in place.
    wrap_func
        A function to call on the first wrap of other.

    Returns
    -------
    Pype connectect
    """
    pype1 = pype if inplace else deepcopy(pype)
    # get attach points (where the other should be hooked)
    attach_wraps = _get_attach_wraps(pype1, how)
    # iterate items to be attached to pype
    for oth in reversed(iterate(other)):
        # handle route objects by converting them to pypes
        if isinstance(oth, dict):
            oth = _route_to_pype(oth)
        # wrap or deepcopy to ensure data is ready for next step
        oth = deepcopy(oth) if isinstance(oth, Pype) else _wrap_task(oth)
        # apply task_func to other
        if wrap_func is not None:
            _apply_wrap_func(oth, wrap_func)
        if isinstance(oth, Pype):  # handle hooking up pypes
            _pype_to_pype(pype1, attach_wraps, oth)
        elif isinstance(oth, (task.Task, wrap.Wrap)):  # hook up everything else
            _wrap_to_pype(pype1, attach_wraps, oth)
    # ensure input task was handled correctly
    assert len(pype1.flow.tasks[task.pype_input]) == 1
    assert task.pype_input in pype1.flow.tasks
    assert pype1.flow.get_input_wrap().task is task.pype_input
    return pype1


def _wrap_task(obj):
    """ wrap up an object or sequence of objects"""
    if isinstance(obj, task.Task):
        return _wrap_task(wrap.Wrap(obj))
    elif isinstance(obj, (list, tuple)):
        return [_wrap_task(x) for x in obj]
    assert isinstance(obj, wrap.Wrap), "non_wrap returned by _wrap_task"
    return obj
