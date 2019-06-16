"""
module for handling type checking stuff.
Note: much of this was inspired by the mypy and enforce

"""

import collections
import functools
import inspect
from inspect import Signature, _empty, Parameter, BoundArguments
from typing import (
    Union,
    Callable,
    Any,
    get_type_hints,
    Optional,
    Sequence,
    List,
    Dict,
    Mapping,
    Tuple,
    Set,
    TypeVar,
)

# Create a dictionary that contains functions for handling typing types
TYPE_FUNCTIONS = {}


# --------------------- helper functions


def _hash_type(type_: type):
    """
    Get a hash for a type.

    Use either __qualname__ attr or call type
    """
    try:
        return f"{type_.__module__}.{type_.__qualname__}"
    except AttributeError:
        if isinstance(type_, object):
            return _hash_type(type(type_))


# Create a dict that maps types from typing to collections or base types
TYPE_MAP = {
    _hash_type(List): list,
    _hash_type(Tuple): tuple,
    _hash_type(Sequence): collections.Sequence,
    _hash_type(Set): collections.Set,
    _hash_type(Mapping): collections.Mapping,
    _hash_type(Dict): dict,
    _hash_type(Callable): collections.Callable,
}

# a hash of types that should not be turned into tuples in the callable check
NO_TUPPLIZE_HASH = {
    _hash_type(Sequence),
    _hash_type(Tuple),
    _hash_type(List),
    _hash_type(Any),
    _hash_type(_empty),
}


def type_func(type_: type, check_instance_and_attrs: bool = True):
    """ register a function for handling a certain type """

    def deco(func):

        if hasattr(func, "_original_func"):
            func = func._original_func

        @functools.wraps(func)
        def wraper(obj, type_, strict):
            if check_instance_and_attrs:
                check = _instance_type_check(obj, type_)
                if isinstance(check, bool):
                    return check
            return func(obj, type_, strict)

        wraper._original_func = func

        TYPE_FUNCTIONS[_hash_type(type_)] = wraper

        return wraper

    return deco


def _instance_type_check(obj, type_) -> Union[bool, tuple]:
    """ check if obj is instance or subclass of type_, if not return False,
    Then check if type_ has __args__, if not return True.
    Examples
    --------
    >>> _instance_type_check([1], List)
    True
    >>> _instance_type_check([1], List[int])
    True
    >>> _instance_type_check([1]. List[str])
    False
    """
    func = get_isinstance_issubclass(obj, type_)
    expected_type = TYPE_MAP[_hash_type(type_)]
    if not func(obj, expected_type):
        return False

    if type_.__args__ is None:
        return True


def get_isinstance_issubclass(obj: Union[object, type], type_: Union[object, type]):
    """
    Determine whether two objects should be compared with isinstance or
    issubclass, return the appropriate function.
    Parameters
    ----------
    obj
    type_

    Returns
    -------

    """
    if isinstance(obj, type) and type_ is not type:
        return issubclass
    else:
        return isinstance


def iter_sequence(seq_like):
    """ iterate a sequence or Sequence type """
    try:
        for item in seq_like:
            yield item
    except TypeError:
        for arg in seq_like.__args__:
            yield arg


def iter_mapping(mapping_like):
    """ Iterate a mapping or Mapping like type """
    try:
        for item, val in mapping_like.items():
            yield item, val
    except TypeError:
        yield mapping_like.__args__


def _get_func_output(func):
    """ given a function return the 'return' type annotation. If None
    exists return Any """
    # check if func is a typing.Callable class
    if _hash_type(func) == _hash_type(Callable):
        try:
            return func.__args__[-1]
        except (IndexError, TypeError):
            return Any
    if not isinstance(func, Signature):
        return get_type_hints(func).get("return", Any)
    else:
        return func.return_annotation


def _get_func_input(func):
    """" given a function return the annotations attached to the inputs """
    if _hash_type(func) == _hash_type(Callable):
        try:
            return func.__args__[:-1]
        except (IndexError, TypeError):
            return (Any,)
    if not isinstance(func, Signature):
        # hints = get_type_hints(func)
        sig = signature(func)
        # return tuple([hints.get(x, Any) for x in list(sig.parameters)])
    else:
        sig = func
    return tuple(sig.parameters[x].annotation for x in sig.parameters)


def _prepare_type(obj):
    """
    Prepare the type or object for further processing
    """
    # try to convert sequences to Tuple types
    if isinstance(obj, (list, tuple)):
        try:
            return Tuple[tuple(obj)]
        except (TypeError, SyntaxError):  # if items are not classes
            pass
    return obj


def _get_len(obj):
    """ try to get the length of an object, if TypeError return len of
    __args__"""
    try:
        return len(obj)
    except TypeError:
        return len(obj.__args__)


# ----------------------- functions to handle various types


@type_func(Union, check_instance_and_attrs=False)
@type_func(Optional, check_instance_and_attrs=False)
def nested_args_typing(obj, type_, strict):
    """ handle nested Types, Union, Optional"""
    func = all if strict else any
    if _hash_type(obj) in TYPE_FUNCTIONS:
        return func([compatible_type(x, type_, strict) for x in obj.__args__])
    else:
        for arg in type_.__args__:
            if compatible_type(obj, arg, strict):
                return True
    return False


@type_func(List)
@type_func(Sequence)
def handle_collections(obj, type_, strict):
    """ handle collections """

    # check the types of the sequence
    for item in iter_sequence(obj):
        if not all([compatible_type(item, x, strict) for x in type_.__args__]):
            return False
    return True


@type_func(Mapping)
@type_func(Dict)
def handle_mapping(obj, type_, strict):
    """ handle dict like types """
    assert len(type_.__args__) == 2, "mapping must have exactly 2 values"
    ktype, vtype = type_.__args__
    for key, val in iter_mapping(obj):
        con1 = compatible_type(key, ktype, strict)
        con2 = compatible_type(val, vtype, strict)
        if not (con1 and con2):
            return False
    return True


@type_func(Tuple)
def handle_tuple(obj, type_, strict):
    args = type_.__args__

    if not _get_len(obj) == len(args):
        return False

    for item, itype in zip(iter_sequence(obj), args):
        if not compatible_type(item, itype, strict):
            return False
    return True


@type_func(Callable, check_instance_and_attrs=False)
def handle_callable(obj, type_, strict):
    if not type_.__args__ and callable(obj):
        return True

    f1_input = _get_func_input(obj)
    f1_output = _get_func_output(obj)

    f2_input = _get_func_input(type_)
    f2_output = _get_func_output(type_)

    # check params
    for p1, p2 in zip(f1_input, f2_input):
        if p2 is Ellipsis or p1 is Ellipsis:
            break
        if not compatible_type(p1, p2, strict):
            return False
    else:  # if no ellipsis is used and args are not the same length
        if len(f1_input) != len(f2_input) and len(f1_input):
            return False

    # check return types
    if not compatible_type(f1_output, f2_output, strict):
        return False

    return True


# @type_func(_empty, check_instance_and_attrs=False)
# def handle_empty_from_signature(obj, type_, strict):
#     """ _empty should be treated as Any """
#     return True


@type_func(TypeVar, check_instance_and_attrs=False)
def handle_type_var(obj, type_, strict):
    if isinstance(obj, TypeVar):  # if obj is a typevar compare constraints
        return compatible_type(Tuple[obj.__constraints__], Tuple[type_.__constraints__])
    else:
        if type_.__constraints__ == ():  # no types attached to TypeVar
            return True
        return isinstance(obj, type_.__constraints__ or Any)


def compatible_type(
    type1_: Union[type, str], type2_: type, strict: bool = True
) -> bool:
    """
    Determine if a type

    Parameters
    ----------
    obj
        Some object or type
    type_
        Some type
    strict
        If True, obj must always be compatible with type_, if False, return
        True if obj can be compatible with type_. This mainly effects Union
        comparisons.

    Returns
    -------
    bool

    Examples
    ----------
    >>> from typing import List, Union
    >>> compatible_type(str, Union[str, float, int])
    True
    >>> compatible_type(float, Union[str, Tuple[bytes]])
    False
    >>> compatible_type(Union[int, float], Union[str, float])
    False
    """
    type1_, type2_ = _prepare_type(type1_), _prepare_type(type2_)
    # Any is compatible with everything
    if type1_ is Any or type1_ is _empty or type2_ is Any or type2_ is _empty:
        return True
    # get a hash for the obj and type_, see if a special func is needed
    hash1, hash2 = _hash_type(type1_), _hash_type(type2_)
    if hash2 in TYPE_FUNCTIONS:
        return TYPE_FUNCTIONS[hash2](type1_, type2_, strict)
    elif not strict and hash1 in TYPE_FUNCTIONS:
        # switch the type_/obj, this should only happen when not strict
        return TYPE_FUNCTIONS[hash1](type2_, type1_, strict)
    else:  # isinstance/issublcass should work; determine which and call
        func = get_isinstance_issubclass(type1_, type2_)
        return func(type1_, type2_)


def compatible_instance(obj: object, type_: type, strict: bool = True) -> bool:
    """
    Determine if an instance is compatible with a type.

    Parameters
    ----------
    obj
        Some object or type
    type_
        Some type
    strict
        If True, obj must always be compatible with type_, if False, return
        True if obj can be compatible with type_. This mainly effects Union
        comparisons.

    Returns
    -------
    bool

    Examples
    ----------
    >>> from typing import List, Union
    >>> compatible_instance(1, int)
    True
    >>> compatible_instance('bob', Union[float, str, int])
    True
    >>> compatible_instance([1, 2., str], List[Union[int, float, str]])
    True
    >>> compatible_instance(9.3, int)
    False
    >>> compatible_instance(['mark'], List[Union[int, float]])
    False

    """
    type_ = _prepare_type(type_)
    # Any is compatible with everything
    if obj is Any or obj is _empty or type_ is Any or type_ is _empty:
        return True
    # get a hash for the obj and type_, see if a special func is needed
    hash2 = _hash_type(type_)
    if hash2 in TYPE_FUNCTIONS:
        return TYPE_FUNCTIONS[hash2](obj, type_, strict)
    else:  # isinstance/issublcass should work; determine which and call
        func = get_isinstance_issubclass(obj, type_)
        return func(obj, type_)


def _get_tupled_output(func):
    out = _get_func_output(func)
    is_type = _hash_type(out) in NO_TUPPLIZE_HASH
    is_collection = isinstance(out, collections.Sequence)
    if not is_collection and not is_type:
        out = (out,)  # make iterable if needed
    return out


FUNC_TYPE_DICT: Dict[str, Callable] = {
    "output": _get_tupled_output,
    "input": _get_func_input,
}


def compatible_callables(
    func1: Callable,
    func2: Callable,
    func1_type="output",
    func2_type="input",
    strict=True,
) -> bool:
    """
    Function for determining if callable input/outputs are compatible.

    func1_type and func2_type parameters control if the input or outputs
    are checked for each function.

    Parameters
    ----------
    func1
        The function whose output will be checked.
    func2
        The function whose input will be checked.
    func1_type: str ('input' or 'output')
        Control of func1 input or output is examined
    func2_type: str ('input' or 'output')
        Control of func2 input or output is examined
    strict
        If True return True only if the types must be compatible, not if
        they can be compatible.

    Returns
    -------
    bool

    Examples
    --------
    >>> from typing import Union
    >>> def func1(a: int) -> int:
    ...    return a + 1
    >>> def func2(a: int) -> str:
    ...     return str(a)
    >>> def func3(a: Union[int, float]) -> float:
    ...     return float(a) / 10.
    >>> def func4(a: Union[float, str]) -> Union[float, str]:
    ...     return a
    >>> compatible_callables(func1, func2)
    True
    >>> compatible_callables(func2, func1)
    False
    >>> compatible_callables(func1, func3)
    True
    >>> compatible_callables(func4, func3)
    False
    >>> compatible_callables(func4, func3, strict=False)
    True
    """
    func1_comp = FUNC_TYPE_DICT[func1_type](func1)
    func2_comp = FUNC_TYPE_DICT[func2_type](func2)
    return compatible_type(func1_comp, func2_comp, strict=strict)


def valid_input(
    func: Union[Signature, Callable],
    *args,
    check_type: bool = True,
    bound: Optional[BoundArguments] = None,
    **kwargs,
):
    """
    Return True if inputs are valid for a callable or signature.

    Parameters
    ----------
    func
        A signature to bind to
    check_type
        If True also perform type-check
    bound
        The bound arguments, optional

    Returns
    -------
    bool
        True if inputs are compatible else False
    """
    sig = func if isinstance(func, Signature) else signature(func)
    if bound is None:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:  # incompatible inputs for signature
            return False
    if check_type:
        return _valid_signature_types(bound)
    return True


def _valid_signature_types(bind):
    """ return True if the types are consistent with the inputs bound
    to signature """
    args = bind.arguments
    arg_types = bind.signature._parameters
    for arg in args:
        if not compatible_instance(args[arg], arg_types[arg].annotation):
            return False
    return True


def signature(func: Callable) -> Signature:
    """
    Function for getting signature and handling types indicated by strings.

    Since some type hints can be passed as strings, this is a function to
    return a signature of a callable that has the string types evaluated.

    Parameters
    ----------
    func
        Any callable.

    Returns
    -------
    A signature object.
    """
    # get signature of original functions
    sig = inspect.signature(func)
    params = collections.OrderedDict(sig.parameters)
    # get hints attached to functions. This evals string hints
    hints = get_type_hints(func)
    return_annotation = hints.pop("return", Parameter.empty)
    # swap out unevaluated annotations for evaluated ones
    for name, hint in hints.items():
        params[name] = params[name].replace(annotation=hint)
    # get new signature with evaluated annotations
    out = sig.replace(
        parameters=list(params.values()), return_annotation=return_annotation
    )
    # ensure the same argument names are there and return
    assert set(out.parameters) == set(sig.parameters)
    return out
