"""
Tests for the type checking in sflow
"""
# import inspect
from typing import (
    Union,
    Any,
    Text,
    Optional,
    Sequence,
    Mapping,
    Dict,
    List,
    Tuple,
    Callable,
    TypeVar,
)

import pytest

from spype.types import (
    compatible_type,
    compatible_callables,
    valid_input,
    compatible_instance,
    signature,
)


# --------------- helper functions/classes


class Car:
    pass


class Honda(Car):
    pass


def int2str(a: int) -> str:
    return str(a)


def str2float(a: str) -> float:
    return bool(a)


def bunch_of_args(a, b, c, d) -> str:
    pass


def func_with_kwargs(a: int, b: float = 2.0) -> str:
    pass


def func_with_req_kwargs(a: int, *, b: float = 2.0, c: str = "hey"):
    pass


def multiple_out() -> (int, str):
    pass


def union_func1(a: Union[int, float]) -> float:
    pass


def union_output_1(a: int) -> Union[float, Car]:
    pass


def func_3in_2out(a: int, b: float, c: str) -> Tuple[int, float]:
    pass


def func_2in_3out(a: int, b: float) -> (int, float, str):
    pass


def func_2in_no_hints(a, b):
    pass


def one_callable(a: Callable) -> bool:
    return callable(a)


def higher_order_func1(a: int) -> Callable[..., int]:
    pass


def higher_order_func2(a: Callable[[int, float, str], int]) -> Callable:
    pass


def raise_two(num):
    return num ** 2


def return_true(x):
    return True


def args_func(*args):
    """ function that takes *args """
    return args


def kwargs_func(**kwargs):
    """ function that just takes kwargs """
    return kwargs


def args_kwargs_func(*args, **kwargs):
    pass


def range_func(num: int) -> List[int]:
    return range_func(num)


def int_mas_one(a: int) -> int:
    return a + 1


def str_honda(a: "Honda") -> "Honda":
    pass


def str_car(a: Car) -> "Car":
    pass


blank_tv = TypeVar("blank_tv")

# ----------------------- type variables


int_float_str = TypeVar("int_float_str", int, float, str)
car_str = TypeVar("car_str", Car, str)


# ---------------------- object_is_type tests


class TestSignature:
    """ tests for the custom signature to eval string types """

    funcs_with_str_types = [str_car, str_honda]
    normal_funcs = [int_mas_one, range_func, kwargs_func, args_func]

    @pytest.mark.parametrize("func", funcs_with_str_types + normal_funcs)
    def test_no_string_annotations(self, func):
        """ ensure string types are not """
        sig = signature(func)
        for name, sig_type in sig.parameters.items():
            assert not isinstance(sig_type.annotation, str)


class TestObjectIsType:
    """ tests for using unions, Optional, Generics, etc. """

    is_type_tuple = (
        (1, int),
        (2.0, float),
        ("hey", Text),
        ("arg", Any),
        (Car(), Car),
        (Honda(), Car),
        (1, Union[int, float, Car]),
        (2.0, Union[float, Car]),
        (Honda(), Union[float, Car]),
        (None, Optional[Union[int, float, Car]]),
        (None, Optional[int]),
        (Honda(), Optional[Union[float, Car]]),
        ([1, 2, 3], Sequence[int]),
        ([1, 2, 3], Union[Sequence[int], Sequence[float]]),
        ([1.0, 2, "bob"], Sequence[Union[int, float, str]]),
        ([1, 2, 3], Sequence),
        ({float: 1}, Mapping[type, int]),
        ({1: 1, "bob": 2}, Dict[Union[int, str], int]),
        ({1: 2, 2: 3}, Any),
        ([Car(), Honda()], List[Car]),
        ((1, 2.0, Honda()), Tuple[int, float, Car]),
        ((1, [2]), Tuple[int, List[Union[int, float]]]),
        (int2str, Callable),
        (int2str, Callable[[int], str]),
        (lambda x: x, Callable),
        (lambda x: x, Callable[[int], float]),
        (bunch_of_args, Callable[..., str]),
        (func_with_kwargs, Callable[[int, float], str]),
        (func_with_req_kwargs, Callable[[int, float, str], "str"]),
        (multiple_out, Callable[..., Tuple[int, str]]),
        (multiple_out, Callable[..., Tuple[Union[int, str], Union[int, str]]]),
        ((1, 2.0, Car()), (int, float, Car)),
        ("str", car_str),
        (Car(), car_str),
        (Honda(), car_str),
        (1, int_float_str),
        (2.3, int_float_str),
        (["szechuan", "sauce", "snafu"], list),
        (1, blank_tv),
        (1.1, blank_tv),
        (Car(), blank_tv),
        (["bob"], blank_tv),
        ((1, 2), blank_tv),
        ("marco polo".split(" "), List[str]),
    )

    is_not_type_tuple = (
        (1, str),
        ("bob", float),
        (2.0, int),
        (1, Union[float, str]),
        ("bob", Union[int, float]),
        (1, Optional[str]),
        ("bob", Optional[int]),
        ([1.0, "bob", 3], Sequence[int]),
        ([1, 2, 4], Sequence[Union[float, str]]),
        ([1, 2.0, 3], Sequence[int]),
        (1, Sequence[int]),
        ({1: 2, "b": 3}, Mapping[int, Any]),
        ((1, 2, "b"), Tuple[int, int]),
        ((1, "2"), Tuple[int, List[Union[int, float]]]),
        (int2str, Callable[[int, int], str]),
        (bunch_of_args, Callable[..., int]),
        (multiple_out, Callable[..., List[int]]),
        (Car(), int_float_str),
        (int, int_float_str),
        (1, car_str),
    )

    # tests
    @pytest.mark.parametrize("obj, type_", is_type_tuple)
    def test_object_is_type(self, obj, type_):
        if not compatible_instance(obj, type_):
            breakpoint()
            compatible_instance(obj, type_)
        assert compatible_instance(obj, type_)

    @pytest.mark.parametrize("obj, type_", is_not_type_tuple)
    def test_object_is_not_type(self, obj, type_):
        assert not compatible_instance(obj, type_)


class TestCompatibleObjects:
    compat_tuple = (
        (int, int),
        (int, Union[int, float]),
        (Union[int, float], Union[int, float, str]),
        (Union[int, float], Optional[Union[int, float, str, Car, Callable]]),
        (int, Any),
        (List, Any),
        (List, List),
        (list, List),
        (List[Union[str, float]], List[Union[str, float, int]]),
        (Dict[int, float], Dict[int, Union[float, int]]),
        (Callable[..., int], Callable[..., Union[int, float]]),
        (Tuple[int, int, int], Sequence[int]),
        (Tuple[str], Tuple[str]),
        (Callable, Callable[..., Any]),
        (Callable[[int, float, str], Any], Callable[[int, float, str], None]),
        (Any, (int, float)),
        (int_float_str, int_float_str),
        (TypeVar("new_type", int, float, str), int_float_str),
        (int, Callable),
        (str, Callable),
        (Honda, type),
    )

    incompat_tuple = (
        (int, Optional[float]),
        (int, Union[float, str, Car]),
        (Union[int, float], Union[float, str]),
        (tuple, List),
        (Mapping, dict),
        (Dict[int, Union[float, int]], Dict[int, float]),
        (Callable[..., Union[int, str]], Callable[..., Union[int, float]]),
        (Callable[..., str], Callable[..., Car]),
        (Callable[[int, float, int], Any], Callable[[int, float, str], None]),
        (car_str, int_float_str),
        (Honda(), type),
    )

    compat_if_not_strict = (
        (Union[int, float], Union[str, float]),
        (Union[float, int, Car, Honda], Car),
    )

    not_compat_strict = list(incompat_tuple) + list(compat_if_not_strict)

    # tests
    @pytest.mark.parametrize("obj, type_", compat_tuple)
    def test_object_is_type(self, obj, type_):
        assert compatible_type(obj, type_)

    @pytest.mark.parametrize("obj, type_", not_compat_strict)
    def test_object_is_not_type(self, obj, type_):
        assert not compatible_type(obj, type_)

    @pytest.mark.parametrize("obj, type_", compat_if_not_strict)
    def test_object_compat_not_strict(self, obj, type_):
        assert compatible_type(obj, type_, strict=False)


class TestCompatibleFunctions:
    """ tests checking """

    compat_functions = (
        (int2str, str2float),
        (func_3in_2out, func_2in_3out),
        (func_2in_3out, func_3in_2out),
        (func_2in_no_hints, func_2in_3out),
        (func_3in_2out, func_2in_no_hints),
        (higher_order_func1, higher_order_func2),
        (int_mas_one, int2str),
        (str_honda, str_car),
    )
    incompat_functions = (
        (str2float, int2str),
        (func_2in_3out, func_2in_3out),
        (func_3in_2out, func_3in_2out),
        (func_2in_3out, func_2in_no_hints),
        (higher_order_func2, higher_order_func1),
    )
    compat_with_no_strict = ((union_output_1, union_func1),)

    compat_both_inputs = (
        (raise_two, return_true),
        (lambda x, y, z: None, lambda a, b, c: None),
    )

    incompat_strict = list(incompat_functions) + list(compat_with_no_strict)

    # tests
    @pytest.mark.parametrize("func1, func2", compat_functions)
    def test_compat_functions(self, func1, func2):
        assert compatible_callables(func1, func2)

    @pytest.mark.parametrize("func1, func2", compat_functions)
    def test_signatures(self, func1, func2):
        """ ensure signatures can aslo be passed to compatible callables """
        sig1 = signature(func1)
        sig2 = signature(func2)
        assert compatible_callables(sig1, sig2)

    @pytest.mark.parametrize("func1, func2", incompat_strict)
    def test_incompat_functions(self, func1, func2):
        assert not compatible_callables(func1, func2)

    @pytest.mark.parametrize("func1, func2", compat_with_no_strict)
    def test_compat_no_strict(self, func1, func2):
        assert compatible_callables(func1, func2, strict=False)

    @pytest.mark.parametrize("func1, func2", compat_both_inputs)
    def test_compat_both_inputs(self, func1, func2):
        """ test that function inputs can be compared """
        assert compatible_callables(func1, func2, func1_type="input")


class TestIsValidInput:
    compat_inputs = (
        (int2str, (1,), {}),
        (int2str, (), {"a": 2}),
        (func_2in_no_hints, (1, 2), {}),
        (args_func, (1, 3, 4), {}),
        (args_func, (), {}),
        (kwargs_func, (), dict(bob="no good", bill="alright")),
        (kwargs_func, (), {}),
        (args_kwargs_func, (), {}),
        (args_kwargs_func, (1,), {}),
        (args_kwargs_func, (1, 2), {"hey": "there"}),
        (args_kwargs_func, (), {"hey": "there"}),
        (int2str, (int,), {}),
        (one_callable, (int,), {}),
    )

    non_compat_inputs = (
        (int2str, (1,), {"bad": "to the bone"}),
        (int2str, ("str",), {}),
        (int2str, (), {"a": "bob"}),
        (args_func, (1, 3), {"hey": "Jude"}),
        (kwargs_func, (1, 3), dict(ok=1)),
        (kwargs_func, (1,), dict()),
        (int2str, (str,), {}),
    )

    # tests
    @pytest.mark.parametrize("func, args, kwargs", compat_inputs)
    def test_compat_inputs(self, func, args, kwargs):
        assert valid_input(func, *args, **kwargs)

    @pytest.mark.parametrize("func, args, kwargs", non_compat_inputs)
    def test_non_compat_inputs(self, func, args, kwargs):
        assert not valid_input(func, *args, **kwargs)
