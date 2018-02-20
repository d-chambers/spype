"""
Constants, and their explanations, used by sflow
"""
from types import MappingProxyType as MapProxy
from typing import (Callable, Sequence, Union, Tuple, Dict, Any,
                    Hashable)

# ---------------------- Fixtures/callbacks

# supported generic task fixtures
TASK_FIXTURES = MapProxy(dict(
    signature="The signature of the task's run method",
    task='A reference to the current task object',
    self='A reference to the current task object',
    e='The exception object if one was raised, else None',
    inputs='A tuple of (args, kwargs) passed as input to current task',
    args='A tuple of arguments passed to current task',
    kwargs='A dict of keywork arguments passed to current task',
    outputs='The outputs of calling a task, or None',
))

# supported wrap fixtures
WRAP_FIXTURES = MapProxy(dict(
    wrap='A refence to the wrap object around the task',
))

# supported pype fixtures
PYPE_FIXTURES = MapProxy(dict(
    pype='A reference to the pype object running the task, or None',
    meta='The controld dict running the task que (advanced fixture)',
    print_flow='If True print the inputs/outputs of each task to screen',
))

# names of supoprted fixtures
FIXTURE_NAMES = frozenset({**TASK_FIXTURES, **PYPE_FIXTURES, **WRAP_FIXTURES})

# names of supported callbacks
CALLBACK_NAMES = ('on_start', 'on_failure', 'on_success', 'on_finish')

# ------------------------- type hints

# any callable that returns a boolean
bool_func = Callable[..., bool]

# the type expected by spype.core._wraps for conditional flow control
predicate_type = Union[bool_func, Sequence[bool_func]]

# The type for adapt argument of slflow.utils.kwargs_args
adapt_type = Sequence[Union[str, int]]

# The type of output from args and kwargs
args_kwargs_type = Tuple[Tuple[Any, ...], Dict[str, Any]]

# node types for digraph
nodes_type = Sequence[Hashable]

# edge types for digraph
edge_type = Tuple[Hashable, Hashable]
edges_type = Sequence[edge_type]

# callback types
callback_type = Union[Callable, Sequence[Callable]]

# -------------------- pype constannts

# supported str arguments for how of hooking pypes together
HOW_ARGS = ('last', 'first')

# ----------------- utility constants

# columns for the failure log csv
FAILURE_LOG_COLUMNS = ('pype', 'task_inputs', 'pype_inputs', 'number_failures',
                       'task', 'exception', 'succeeded')
