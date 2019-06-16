"""
Tests for Pype object.
"""
import itertools
import os
import pickle
from typing import List, TypeVar

import pytest

import spype
from spype import pype_input, Pype, Task, forward, task
from spype.exceptions import IncompatibleTasks
from spype.utils import de_args_kwargs


# --------------------------- helper functions


def append_func_to_list(a_list: list):
    """ decorator to append func to a_list, return func """

    def register(func):
        a_list.append(func)
        return func

    return register


def append_func_name(list_like):
    """ decorator to append function to list """

    def _decor(func):
        list_like.append(func.__name__)
        return func

    return _decor


# --- some conditionals. Must define on module level for pickle to work


def bigger_than_0(x):
    return x > 0


def gt2(x):
    return x > 2


def lt2(x):
    return x < 2


def eq2(x):
    return x == 2


# ---------------------------- tasks setup for testing


CLASS_TASKS = []
class_task = append_func_to_list(CLASS_TASKS)

FUNCTION_TASK = []
function_task = append_func_to_list(FUNCTION_TASK)


@class_task
class PlusOne(Task):
    def __call__(self, obj: int) -> int:
        return obj + 1


@function_task
@task
def add1(obj: int) -> int:
    return obj + 1


@function_task
@task
def add2(num: int) -> int:
    """ add two to a real number """
    return num + 2


@function_task
@spype.core.task.task
def mult2(num: int) -> int:
    """ multiply a number by 2 """
    return num * 2


@function_task
@spype.core.task.task
def raise2(num: int) -> int:
    """ square a number """
    return num ** 2


@function_task
@spype.core.task.task
def divide2(num):
    """ divide a number by 2 """
    return num / 2


@function_task
@spype.core.task.task
def divide_numbers(num1: int, num2: int) -> float:
    """ divide two numbers """
    return num1 / num2


@function_task
@spype.core.task.task
def return_range(num1: int) -> List[int]:
    """ return a range cast to a list """
    return list(range(num1))


@function_task
@spype.core.task.task
def sum_range(num_list: List[int]) -> int:
    """ sum a range """
    return sum(num_list)


@function_task
@spype.task
def join_str(obj: List[str]) -> str:
    return "".join(obj)


TV = TypeVar("tv")


@function_task
@spype.task
def pass_through(x: TV) -> TV:
    return x


@function_task
@spype.task
def int_to_str(obj: int) -> str:
    return str(obj)


@function_task
@spype.task
def split_on_space(obj: str) -> list:
    """ yield the string split on spaces """
    return obj.split(" ")


@function_task
@spype.task
def join_on_str(obj: List[str], join_on="") -> str:
    """ yield the string split on spaces """
    return join_on.join(obj)


@function_task
@spype.task
def many_args(a, b, c, d, e):
    return locals()


@function_task
@spype.task
def multiply2(num: int) -> int:
    return num * 2


INSTANCE_TASKS = [x() for x in CLASS_TASKS]


@pytest.fixture(params=INSTANCE_TASKS + FUNCTION_TASK)
def generic_task(request):
    """ fixture to collect and parametrize simple tasks """
    return request.param


# ------------------------ pypes to use for testing

VALID_PYPES = []


@pytest.fixture
@append_func_name(VALID_PYPES)
def add2_mult2():
    """ hook three tasks together """
    return pype_input | add2 | mult2


@pytest.fixture
@append_func_name(VALID_PYPES)
def raise2_add1():
    """ hook three tasks together """
    return pype_input | raise2 | add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def add2_mult2_add2():
    """ chain three tasks together """
    return pype_input | add2 | mult2 | add2


@pytest.fixture
@append_func_name(VALID_PYPES)
def add2_fork_mult2_raise2():
    return pype_input | add2 | (mult2, raise2)


@pytest.fixture
@append_func_name(VALID_PYPES)
def forked_aggregated_pype():
    return pype_input | add2 | (mult2, raise2) | divide2


@pytest.fixture
@append_func_name(VALID_PYPES)
def joined_pypes(add2_mult2, raise2_add1):
    """ connect two pypes together """
    return pype_input | add2_mult2 | raise2_add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def joined_pypes_via_and(add2_mult2, raise2_add1):
    """ connect two pypes together """
    return pype_input | add2_mult2 & raise2_add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def iff_pype(add2_mult2):
    """ return a pype with an attached if statement. """
    return add2_mult2.iff(bigger_than_0)


@pytest.fixture
@append_func_name(VALID_PYPES)
def route_pype(add2_mult2):
    """ setup a pype with a simple route object """
    pype = pype_input | {
        gt2: mult2,
        lt2: {lt2: {lt2: add2}},  # this is just to test nested dicts
        eq2: add2_mult2,
    }
    return pype


@pytest.fixture
@append_func_name(VALID_PYPES)
def fan_aggregate_pype():
    """ a pype that fans out and aggregates """
    return pype_input | return_range.fan() | sum_range.agg()


@pytest.fixture
@append_func_name(VALID_PYPES)
def fan_forward_aggregate():
    """ return a pype that has a fan and an aggregate """
    return (pype_input | return_range) << add1 | forward >> add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def nested_tuple_pype():
    """ return a pype that has a fan and an aggregate """
    return pype_input | ((add2, add1), (mult2, divide2)) | add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def aggregate_to_multiple_tasks():
    """ return a pype that aggregates to multple tasks """
    p1 = pype_input | add1 | add2
    return pype_input << (add2, add1, p1) | add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def pype_with_nested_wrap():
    """ return a pype that aggregates to multple tasks """
    return pype_input | (add1.wrap(), add2) | add1


@pytest.fixture
@append_func_name(VALID_PYPES)
def pype_agg_fan_on_task_view():
    """ return a pype created with task views that has aggregate and fan """
    p1 = pype_input | return_range
    p2 = p1[return_range] << int_to_str
    p3 = p2[int_to_str] >> join_on_str
    return p3


@pytest.fixture
@append_func_name(VALID_PYPES)
def empty_pype():
    """ return an empty pype """
    return Pype()


@pytest.fixture
@append_func_name(VALID_PYPES)
def pype2_from_quickstart():
    """ The second pype in the quickstart example """
    return spype.pype_input | (add2, raise2) | (divide2, multiply2) | add2


@pytest.fixture(params=VALID_PYPES)
def pype_object(request):
    """ return pype objects """
    return request.getfixturevalue(request.param)


# ------------------------ Pype tests


class TestPypeConstruction:
    """ hopefully we dont get protesters... """

    # helper functions
    def get_task_name_set(self, pype: Pype):
        """ return a set of the names of tasks in a pype """
        out = set()
        for task_ in pype.flow.tasks:
            try:  # function based tasks
                out.add(task_.__name__)
            except AttributeError:  # class based tasks
                out.add(task_.__class__.__name__)
        return out

    # --- general pype tests
    def test_pipe_constructor(self, pype_object):
        """ ensure tasks can be hooked up with | to return pypes """
        assert isinstance(pype_object, Pype)

    def test_tasks(self, pype_object):
        """ both the tasks and wraps objects should contain Task and Wrap
        instances respectively. """
        assert pype_object.flow.wraps is not None
        assert pype_object.flow.tasks is not None
        # get wraps from the task dict and make sure all are in wraps
        wraps = itertools.chain.from_iterable(pype_object.flow.tasks.values())
        assert set(pype_object.flow.wraps) == set(wraps)
        # do the opposite, get tasks from wraps and ensure they are in tasks
        tasks = list(pype_object.flow.wraps.values())
        assert set(tasks) == set(pype_object.flow.tasks)

    def test_pype_can_be_copied(self, pype_object):
        """ make sure pypes can be copied """
        pype2 = pype_object.copy()
        assert pype2 is not pype_object

    def test_len(self, pype_object):
        """ the len of a pype should be equal to the number of nodes """
        assert len(pype_object) == len(pype_object.flow.wraps)

    def test_str(self, pype_object):
        """ All pypes should be able to produce a str rep """
        assert isinstance(str(pype_object), str)

    def test_repr(self, pype_object):
        """ All pypes should return valid str from repr method """
        assert isinstance(repr(pype_object), str)

    def test_pype_input_in_network_once(self, pype_object):
        """"  The special task pype_input should be in the network """
        network = pype_object.flow
        wraps = list(network.wraps)
        tasks = [x.task for x in wraps]
        assert pype_input in tasks
        assert len(pype_object.flow.tasks[pype_input]) == 1

    def test_connected_tasks_have_first_and_last_tasks(self, pype_object):
        """ ensure that any pype with more than one task has a non-zero
        first and last task attribute """
        if len(pype_object.flow.tasks) > 1:
            assert pype_object._last_tasks

    def test_can_pickle_pype(self, pype_object):
        """ test that pypes can be pickled, pype_input should not change """
        p1 = pype_object
        pickle_str = pickle.dumps(p1)
        p2 = pickle.loads(pickle_str)
        assert pype_input in p2.flow.tasks and pype_input in p1.flow.tasks
        assert p2.flow.get_input_wrap() is not None
        assert p1.flow.get_input_wrap() is not None
        # ensure the same named tasks exist in both
        task_names1 = self.get_task_name_set(p1)
        task_names2 = self.get_task_name_set(p2)
        assert task_names1 == task_names2

    def test_pype_constructor_returns_pype(self, pype_object):
        """ ensure calling the pype constructor returns and equal pype """
        pype1 = Pype(pype_object)
        assert isinstance(pype1, Pype)
        assert len(pype1) == len(pype_object)

    # specific tests

    def test_pipe_no_args_doesnt_raise(self):
        """ ensure a pype can be instantiated with no input args """
        Pype()  # passes if this doesn't raise

        try:
            Pype()  # passes if this doesn't raise
        except Exception:
            pytest.fail("should not raise no empty pype creation")

    def test_pipe_no_args_last_task(self):
        """ ensure the last tasks of an empty pype is pype_input """
        pype = Pype()
        assert pype.flow.get_input_wrap() in pype._last_tasks

    def test_print(self, capsys, joined_pypes):
        """ ensure the outputs are printed when told to do so """
        joined_pypes(2)

        with spype.options(print_flow=True):
            joined_pypes(2)
            out, err = capsys.readouterr()
            assert "got" in out
            assert "and returned" in out

    def test_tasks_in_pype(self):
        """ create a pype from tasks, ensure tasks are in wraps and tasks """
        tasks = [add2, add1, mult2, raise2, pype_input]
        pype = pype_input | add2 | add1 | mult2 | raise2
        for task_ in tasks:
            assert task_ in pype.flow.tasks
        # get a list of tasks in the _task dict
        used_tasks = {val for _, val in pype.flow.wraps.items()}
        assert set(tasks) == used_tasks

    def test_copy_on_pype(self):
        """ ensure each pipe operation returns an new object """
        p1 = pype_input | add2 | add1
        p2 = pype_input | mult2 | raise2
        p3 = pype_input | p1 | p2
        # make sure p1 was not modified
        assert p1 is not p3
        assert p2 is not p3
        # ensure wraps are different
        s1 = set(p1.flow.wraps) | set(p2.flow.wraps)
        s2 = set(p3.flow.wraps)
        assert len(s1 & s2) == 0  # no wraps should overlap
        # but tasks should be the same
        assert set(p1.flow.tasks) | set(p2.flow.tasks) == set(p3.flow.tasks)

    def test_broadcast_to_pypes(self):
        """ test that pypes can be broadcast to otther pypes """
        p1 = pype_input | add2 | raise2 | raise2 | add2
        p2 = pype_input | add2 | add2
        # hook pypes to the end of add_two and feed outputs into divide_two
        pype = pype_input | add2 | (p1, p2, add2) | divide2

        # get a list of expected tasks
        expected_tasks = set.union(
            set(p1.flow.tasks), set(p2.flow.tasks), {add2}, {divide2}
        )
        # ensure all the tasks are in the tasks dict
        assert expected_tasks == set(pype.flow.tasks)

    def test_tasks_not_copied(self, add2_mult2):
        """ ensure the constituent tasks are still in pypes tasks """
        assert add2 in add2_mult2.flow.tasks
        assert mult2 in add2_mult2.flow.tasks

    def test_register_pype(self, add2_mult2):
        """ ensure pype can be registered """
        name = "add2_mult2"
        pype = add2_mult2
        pype.register(name)
        assert pype.name == name
        out = Pype(name)
        assert set(out.flow.wraps) != set(pype.flow.wraps)
        assert set(out.flow.tasks) == set(pype.flow.tasks)
        assert out.name == pype.name == name

    def test_instance_tasks_in_network(self):
        """ instances should be in the network """
        instances = [PlusOne() for _ in range(3)]
        pype = instances[0] | instances[1] | instances[2]

        for instance in instances:
            assert instance in pype.flow.tasks


class TestValidatePype:
    """ tests for detecting problems in pypes before running """

    def test_valid(self, pype_object):
        """ the validate_pype method should not raise """
        try:
            pype_object.validate()
        except Exception:
            pytest.fail("valid pype should not raise")


class TestCallPypes:
    """ tests for calling pypes """

    pype_input_output = [  # tuple of pype fixture, input and output
        ("add2_mult2", 2, 8),
        ("raise2_add1", 2, 5),
        ("add2_mult2_add2", 5, 16),
        ("joined_pypes", 2, 65),
    ]

    # tests
    @pytest.mark.parametrize("pype_name, in_arg, expected", pype_input_output)
    def test_output(self, pype_name, in_arg, expected, request):
        pype = request.getfixturevalue(pype_name)
        assert pype(in_arg) == expected

    def test_call_empty_pipe(self):
        """ calling an empty pype should return values input  """
        pype = Pype()
        assert pype() is None
        assert pype(1) == 1


class TestJoinPypes:
    """ tests for joining pype objects together """

    # fixtures
    @pytest.fixture
    def pype1(self, add2_mult2):
        return add2_mult2

    @pytest.fixture
    def pype2(self, raise2_add1):
        return raise2_add1

    @pytest.fixture
    def or_joined_pypes(self, pype1, pype2):
        return pype1 | pype2

    @pytest.fixture
    def ior_joined_pypes(self, pype1, pype2):
        pype1 |= pype2
        return pype1

    @pytest.fixture
    def and_joined_pypes(self, pype1, pype2):
        return pype1 & pype2

    # tests
    def test_pype1_not_modified(self, add2_mult2, pype1, or_joined_pypes):
        """ ensure pype 1 was not modified by join operation """
        assert add2_mult2 is pype1
        assert add2_mult2 == pype1
        assert pype1 is not or_joined_pypes

    def test_task_order(self, pype1, pype2, or_joined_pypes):
        """ tests for ensuring the task graphs are right """
        p1 = or_joined_pypes
        p1_last = list(pype1._last_tasks)[0]
        p2_first = list(pype2._first_tasks)[0]
        # get wraps from task instances
        _task1 = p1.flow.tasks[p1_last.task][0]
        _task2 = p1.flow.tasks[p2_first.task][0]
        # ensure we have the right wraps
        assert _task1 in p1.flow.wraps
        assert _task2 in p1.flow.wraps
        # make sure the first was connected to the last
        assert or_joined_pypes(0) == 17

    def test_tasks_in_graph(self, pype1, pype2, or_joined_pypes):
        """ ensure the combine pypes has tasks from both in its graph """
        task_set1 = set(pype1.flow.tasks)
        task_set2 = set(pype2.flow.tasks)
        assert set(or_joined_pypes.flow.tasks) == (task_set1 | task_set2)

    def test_wrapped_tasks_unique(self, pype1, pype2, or_joined_pypes):
        """ ensure wrapped tasks are not in joined_pype (wraps should be
        unique, except pype_input's, tasks should not be unique """
        wraps = set(pype1.flow.wraps) | set(pype2.flow.wraps)
        assert len(set(or_joined_pypes.flow.wraps) & wraps) == 0

    def test_ior_joined_pype(self, pype1, ior_joined_pypes):
        """ __ior__ |= should join pypes in place """
        assert ior_joined_pypes is pype1

    def test_and_joined_pypes(self, pype1, pype2, and_joined_pypes):
        """ test that the first task of pype2 was hooked to the first
        task of pype 1"""
        last_task1 = list(pype1._last_tasks)[0].task
        last_task2 = list(pype2._last_tasks)[0].task
        # call joined pypes, make sure independent
        and_joined_pypes(1)
        t1 = de_args_kwargs(*and_joined_pypes.outputs[last_task1])
        t2 = de_args_kwargs(*and_joined_pypes.outputs[last_task2])
        assert t1 == pype1(1)
        assert t2 == pype2(1)

    def test_and_joined_tasks(self, pype1):
        """ that a single task can be joined to the start of a pype """
        pype = pype1.copy()
        # join a single task
        new_pype = pype & divide2
        input_wrap = new_pype.flow.tasks[pype_input][0]
        divide2_wrap = new_pype.flow.tasks[divide2][0]
        assert (input_wrap, divide2_wrap) in new_pype.flow.edges

    def test_hook_to_and_joined_task(self):
        """ ensure joining tasks with & then with | joins on all ends """
        p1 = pype_input | add1 | add2
        p2 = pype_input | mult2 | raise2
        pype = (p1 & p2) | divide2
        # get last wrapped tasks
        last_wrap1 = pype.flow.tasks[add2][0]
        last_wrap2 = pype.flow.tasks[raise2][0]
        divide2_wrap = pype.flow.tasks[divide2][0]
        # ensure these are connected in edges
        assert (last_wrap1, divide2_wrap) in pype.flow.edges
        assert (last_wrap2, divide2_wrap) in pype.flow.edges


class TestJoinPypeWithGetItem:
    """ test that pypes can be joined together with get_item interface """

    def test_raise_on_non_unique_task(self):
        """ trying to access a non-unique """
        p1 = pype_input | add1 | add1
        with pytest.raises(TypeError) as e:
            p1[add1]
        assert "must have exactly one" in str(e)

    def test_call_wrap(self, add2_mult2):
        """ calling the object returned from getitem should call wrap """
        assert add2_mult2[add2](1) == add2.wrap()(1)

    def test_join_task_to_pype(self, add2_mult2):
        """ test that get_item can join pypes """
        p1 = add2_mult2[add2] | add1
        assert p1 is not add2_mult2
        assert add1 in p1.flow.tasks
        add1_wrap = p1.flow.tasks[add1][0]
        add2_wrap = p1.flow.tasks[add2][0]
        mult2_wrap = p1.flow.tasks[mult2][0]
        assert (add2_wrap, add1_wrap) in p1.flow.edges
        assert {add1_wrap, mult2_wrap} == set(p1._last_tasks)

    def test_equal_join_task_to_pype(self, add2_mult2):
        """ test that get_item can join pypes in place """
        p1 = add2_mult2
        add2_mult2[add2] |= add1
        assert p1 is add2_mult2
        assert add1 in p1.flow.tasks
        add1_wrap = p1.flow.tasks[add1][0]
        add2_wrap = p1.flow.tasks[add2][0]
        mult2_wrap = p1.flow.tasks[mult2][0]
        assert (add2_wrap, add1_wrap) in p1.flow.edges
        assert {add1_wrap, mult2_wrap} == set(p1._last_tasks)

    def test_join_pypes(self, add2_mult2: Pype, raise2_add1: Pype):
        """ test that a pype can be joined to a task using the get_item
        interface """
        pype = add2_mult2[add2] | raise2_add1
        # get wraps for testing conditions
        mult2_wrap = pype.flow.tasks[mult2][0]
        add1_wrap = pype.flow.tasks[add1][0]
        add2_wrap = pype.flow.tasks[add2][0]
        raise2_wrap = pype.flow.tasks[raise2][0]
        # ensure proper edge connections have been made
        assert (add2_wrap, mult2_wrap) in pype.flow.edges
        assert (add2_wrap, raise2_wrap) in pype.flow.edges
        # ensure last task set correctly
        assert mult2_wrap in pype._last_tasks
        assert add1_wrap in pype._last_tasks


class TestFixtureViaPartial:
    """ test that dependencies get injected using partial method """

    # fixtures
    @pytest.fixture
    def pype_simple_dep(self):
        """ create a pype that has a partial dependency that is constant """
        return pype_input | add2 | mult2 | divide_numbers.partial(num1=8)

    @pytest.fixture
    def pype_task_dep(self):
        """ create a pype that has a partial dependency that is a pype """
        p1 = pype_input | add2 | divide_numbers.partial(num1=add1)
        p1 &= add1
        return p1

    @pytest.fixture
    def pype_input_dep(self):
        """ getting pype_input should also work in partial """
        return pype_input | add2 | divide_numbers.partial(num1=pype_input)

    @pytest.fixture
    def pype_delayed_dep(self):
        """ create a pype that has a task with a fixture that will cause the
        task to be kicked to the start of the queue """
        p1 = pype_input | add2 | mult2
        p1 &= pype_input | add1 | divide_numbers.partial(num1=mult2)
        return p1

    # tests
    def test_task_dependencies_dont_get_copied(self):
        """ ensure that when tasks are used to specify dependencies those
        tasks do not get copied """
        p1 = pype_input | add2 | mult2
        p1 &= pype_input | add1 | divide_numbers.partial(num1=mult2)
        # get tasks from task and wraps
        tasks1 = p1.flow.tasks
        tasks2 = {x for x in p1.flow.wraps.values()}
        assert mult2 in tasks1 and mult2 in tasks2
        # ensure the fixture value (task) has not been copied
        cfs = set(p1.flow.tasks[divide_numbers][0]._partials.values())
        assert mult2 in cfs

    def test_simple_pype(self, pype_simple_dep):
        """ ensure the dependency gets resolved correctly """
        assert pype_simple_dep(2) == 1

    def test_pype_dependency(self, pype_task_dep):
        """ ensure the pype dependencies get correctly resolved """
        assert pype_task_dep(2) == 3 / 4.0

    def test_pype_input_dependency(self, pype_input_dep):
        """ ensure using the kwarg 'input' returns the input to the pype """
        assert pype_input_dep(1) == 1 / 3.0

    def test_deplayed_dep(self, pype_delayed_dep):
        """ ensure a task that needs a dependency that is not yet calculated
        can get it. """
        pype_delayed_dep(1)
        assert de_args_kwargs(*pype_delayed_dep.outputs[divide_numbers]) == 3.0

    def test_bad_value_input_raises(self):
        """ ensure a partial with a bad input type raises if check_type """
        pype = pype_input | add2 | divide_numbers.partial(num1="hey you")
        with pytest.raises(IncompatibleTasks):
            pype.validate()

    def test_bad_value_input_doesnt_raises(self):
        """ ensure a partial with a bad input type raises if check_type """
        pype = pype_input | divide_numbers.partial(num1="hey you")
        with spype.set_options(check_type=False):
            try:
                pype.validate()
            except Exception:
                pytest.fail("should not raise with type check off")

    def test_bad_task_dependency_raises(self):
        """ if a dependency is defiend as a task, but the task returns the
        incorrect type it should raise on validation """
        p1 = pype_input | int_to_str
        p2 = p1 & (pype_input | add2 | divide_numbers.partial(num1=int_to_str))
        with pytest.raises(IncompatibleTasks):
            p2.validate()


class TestFixtureViaSetItem:
    """ test that dependencies get injected using set item on pypes """

    # fixtures
    @pytest.fixture
    def pype_simple_dep(self):
        """ create a pype that has a partial dependency that is constant """
        p1 = pype_input | add2 | mult2 | divide_numbers
        p1["num1"] = 3
        return p1

    @pytest.fixture
    def pype_task_dep(self):
        """ create a pype that has a partial dependency that is a pype """
        p1 = pype_input | add2 | divide_numbers
        p1 &= add1
        p1["num1"] = add1
        return p1

    @pytest.fixture
    def pype_input_dep(self):
        """ getting the input to a function should also work with the
        pype_input """
        p1 = pype_input | add2 | divide_numbers
        p1["num1"] = pype_input
        return p1

    @pytest.fixture
    def pype_many_deps(self):
        """ test defining many dependencies """
        p1 = pype_input | many_args
        p1["a"], p1["b"], p1["c"] = add1, add2, mult2
        p1["d"], p1["e"] = raise2, divide2
        return p1

    # tests
    def test_simple_pype(self, pype_simple_dep):
        """ ensure the dependency gets resolved correctly """
        assert pype_simple_dep(1) == 0.5

    def test_pype_dependency(self, pype_task_dep):
        """ ensure the pype dependencies get correctly resolved """
        assert pype_task_dep(0) == 0.5

    def test_pype_input_dependency(self, pype_input_dep):
        """ ensure using the kwarg 'input' returns the input to the pype """
        assert pype_input_dep(1) == 1 / 3.0

    def test_many_deps(self, pype_many_deps):
        """ ensure a pype with many deps still works """
        pype_many_deps(2)

    def test_set_item_implicit_and(self):
        """ test that setting an item dep of a task not in the network
        implicitly connects it in parallel """
        p1 = pype_input | add2 | divide_numbers
        p1["num1"] = add1
        assert add1 in p1.flow.tasks

    def test_task_doc_example(self):
        pype = spype.pype_input | add2 | raise2 | divide_numbers
        pype["num2"] = add2
        assert pype(2) == 4

    def test_bad_value_input_raises(self):
        """ ensure a partial with a bad input type raises if check_type """
        pype = pype_input | add2 | divide_numbers
        pype["num1"] = "hey you"
        with pytest.raises(IncompatibleTasks):
            pype.validate()

    def test_bad_value_input_doesnt_raises(self):
        """ ensure a partial with a bad input type raises if check_type """
        pype = pype_input | divide_numbers
        pype["num1"] = "hey you"
        with spype.set_options(check_type=False):
            try:
                pype.validate()
            except Exception:
                pytest.fail("should not raise with type check off")

    def test_bad_task_dependency_raises(self):
        """ if a dependency is defiend as a task, but the task returns the
        incorrect type it should raise on validation """
        p1 = pype_input | int_to_str
        p2 = p1 & (pype_input | add2 | divide_numbers)
        p2["num1"] = int_to_str
        with pytest.raises(IncompatibleTasks):
            p2.validate()


class TestFixturesOnPypeLevel:
    """ test that tasks can get information about applicable pypes,
    tasks, wraps """

    def test_pype_level_fixture_on_callback(self, some_list):
        """ ensure a callback on a task can ask for, and receive, a reference
        to the task, pype, and wrap objects driving it. """

        out = set()

        def on_start(pype, wrap, task):
            out.add(pype)
            out.add(wrap)
            out.add(task)

        @task(on_start=on_start)
        def add_another(num):
            return num * 2

        p1 = pype_input | add2 | add1 | add_another

        p1(2)
        assert len(out) == 3
        assert p1 in out
        assert add_another in out
        assert p1.flow.tasks[add_another][0] in out


class TestConditionals:
    """ test that conditionals can be used to silently drop data """

    def test_basic_iff(self):
        def number_is_even(num):
            return (num % 2) == 0

        pype = pype_input | add2 | mult2.iff(number_is_even) | raise2

        # if number is not even only add_two gets run
        pype(3)

        assert add2 in pype.outputs
        assert mult2 not in pype.outputs
        assert raise2 not in pype.outputs


class TestIff:
    """ test that iff works on the pype level """

    def test_inplace(self, add2_mult2):
        """ ensure inplace=False returns a copy, inplace=True returns same
        pype """
        p1 = add2_mult2.iff(lambda x: True)
        assert p1 is not add2_mult2
        p2 = add2_mult2.iff(lambda x: True, inplace=True)
        assert p2 is add2_mult2

    def test_iff(self, iff_pype):
        """ ensure the data only flows through the pype if the predicate evals
        to True """
        assert iff_pype(1) == 6
        assert iff_pype(0) == 0

    def test_single_fixture(self, some_dict):
        """ ensure a predicate with a single """

        def log_pype(pype):
            some_dict["pype"] = pype
            return True

        pype = pype_input | divide_numbers.iff(log_pype)
        pype(1, 2)

        assert some_dict["pype"] is pype


class TestFan:
    """ tests for fanning """

    @pytest.fixture
    def append_to_list(self, some_list):
        """ return a task that will just append to some_list """

        @spype.core.task.task
        def append_to_list(item):
            some_list.append(item)

        return append_to_list

    @pytest.fixture
    def fan_pype(self, append_to_list):
        """ create a pype that will fan out """
        return pype_input << forward | append_to_list

    def test_fan_append_to_list(self, fan_pype, some_list):
        """ ensure that each item was individually append to list """
        fan_pype(range(10))
        assert len(some_list) == 10
        assert list(range(10)) == some_list

    def test_doc_example(self, some_list):
        """ test the fan example from the documentation """

        @spype.task
        def print_input(obj: str):
            some_list.append(obj)

        p1 = spype.pype_input | split_on_space
        pype = p1 << print_input
        pype("szechuan sauce snafu")

        assert some_list == "szechuan sauce snafu".split(" ")


class TestAggregate:
    """ tests for aggregating on the object level """

    @pytest.fixture
    def agg_pype(self, some_list):
        @task
        def range_it(num: int):
            return range(num)

        @task
        def pass_it_on(num):
            return num

        @task
        def funnel(num_list):
            for item in num_list:
                some_list.append(item)
            return num_list

        pype = (pype_input | range_it) << pass_it_on >> funnel
        # ensure the tasks are in the pype
        assert range_it in pype.flow.tasks
        assert pass_it_on in pype.flow.tasks
        assert funnel in pype.flow.tasks
        return pype

    # tests
    def test_basic_object_aggregate(self, agg_pype, some_list):
        """ ensure aggregating works on the object level """
        agg_pype(5)
        assert set((range(5))) == set(some_list)

    def test_empty_list_aggregate(self, agg_pype, some_list):
        """ ensure aggregating on an empty list returns and empty list """
        agg_pype(0)
        assert len(some_list) == 0

    def test_doc_example(self, some_list):
        """ test the example from the docs """

        @spype.task
        def print_input(obj: str):
            some_list.append(obj)

        p1 = (spype.pype_input | split_on_space) << pass_through.agg()
        pype = p1 >> join_on_str | print_input
        pype("a full string")


class TestRoute:
    """ Tests for routing data using dictionaries """

    def test_outputs(self, route_pype):
        """ test the outputs of the route pype """
        assert route_pype(3) == 6  # goes to mult2
        assert route_pype(1) == 3  # goes to add2
        assert route_pype(2) == 8  # goes to add2_mult2


class TestPlot:
    """ test that the graph structures are plotted when graphviz """

    # skip these test cases if graphviz is not installed
    pytest.importorskip("graphviz", "you gotta install graphviz bro")
    import graphviz  # checks for python graphviz

    try:
        graphviz.version()  # checks for C library
    except graphviz.ExecutableNotFound:
        pytest.skip("graphviz is not installed")

    file_name = ".deleteme"

    # fixtures
    @pytest.fixture(autouse=True)
    def cleanup_pdf(self):
        """ delete pdf file """
        yield
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

    def test_plot_all_pypes(self, pype_object):
        """ test that all the valid pypes can be plotted """
        pype_object.plot(view=False)

    def test_basic_plot(self, forked_aggregated_pype):
        """ensure calling plot creates the expected pdf from graphviz """
        forked_aggregated_pype.plot(file_name=self.file_name, view=False)

    def test_if_plot(self, iff_pype):
        """ ensure the conditional pype get plotted """
        iff_pype.plot(file_name=self.file_name, view=False)

    def test_fan_plot(self):
        """ plots a simple fan """
        pype = pype_input << add1
        pype.plot(file_name=self.file_name, view=False)

    def test_aggregate_plot(self):
        """ plot an aggregate pype """
        pype = pype_input << add1 | forward >> add1
        pype.plot(file_name=self.file_name, view=False)

    def test_fan_aggregate_plot(self, fan_aggregate_pype):
        """ plots a pype that usses aggregations """
        fan_aggregate_pype.plot(file_name=self.file_name, view=False)


class TestMap:
    """ Basic tests for mapping iterable onto pype """

    # tests
    def test_map_single_thread(self, add2_mult2):
        """ test maping without any concurrency options. some_list and
         output of map should be the same """
        iterable = range(15)
        out = map(add2_mult2, iterable)
        assert [(x + 2) * 2 for x in range(0, 15)] == list(out)

    def test_process_pool(self, process_pool_executor, add2_mult2):
        """ ensure the process pool works the same as single thread """
        ppout = process_pool_executor.map(add2_mult2, range(10))
        # ppout = map(add2_mult2, (range(10), client=process_pool_executor))
        stout = map(add2_mult2, range(10))
        assert set(ppout) == set(stout)

    def test_thread_pool(self, thread_pool_executor, add2_mult2):
        """ ensure the thread pool works the same as single thread """
        # ppout = add2_mult2.map(range(10), client=thread_pool_executor)
        ppout = thread_pool_executor.map(add2_mult2, range(10))
        stout = map(add2_mult2, range(10))
        assert set(ppout) == set(stout)

    def test_none_removed(self):
        """ ensure None has been removed from map output """

        @task
        def gt_two(num):
            if num > 2:
                return num

        pype = pype_input | gt_two
        out = map(pype, range(10))
        assert all([x is not None for x in out])


class TestTemporaryCallbacks:
    """ tests for adding callbacks to all, or only some, tasks """

    def test_add_callback_copies_pype(self, add2_mult2):
        """ the add_callback method on the pype class creates a copy """

        def on_start():
            pass

        p2 = add2_mult2.add_callback(on_start, "on_start")
        assert p2 is not add2_mult2
        # tasks should not have been copied
        assert set(p2.flow.tasks) == set(add2_mult2.flow.tasks)
        # but wraps should all be unique
        assert len(set(p2.flow.wraps) & set(add2_mult2.flow.wraps)) == 0

    def test_callbacks_get_called(self, add2_mult2):
        """ ensure the added callbacks get called """
        out = {}

        def on_start(task, inputs):
            out[task] = inputs

        p2 = add2_mult2.add_callback(on_start, "on_start")
        p2(2)
        assert set(out) == set(p2.flow.tasks)

    def test_select_tasks(self, add2_mult2):
        """ ensure that tasks can limit the callback application """
        out = {}

        def on_start(task, inputs):
            out[task] = inputs

        p2 = add2_mult2.add_callback(on_start, "on_start", tasks=[add2])
        p2(2)
        assert set(out) == {add2}

    def test_debug(self, add2_mult2, monkeypatch):
        """ test the debug function """

        out = {}

        def new_set_trace():
            out["trace_set"] = True

        monkeypatch.setattr("pdb.set_trace", new_set_trace)

        with add2_mult2.debug() as p:
            p(2)

        assert out["trace_set"]
