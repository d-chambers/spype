"""
Tests for Task object usage
"""
import copy
import inspect
import itertools
import pickle
from contextlib import suppress
from typing import TypeVar, Union, Tuple, Any

import pytest

import spype
from spype import ExitTask
from spype import Task, task, Wrap, pype_input
from spype.constants import CALLBACK_NAMES, FIXTURE_NAMES


# --------------------------- helper functions


def append_func_to_list(a_list: list):
    """ decorator to append func to a_list, return func """

    def register(func):
        a_list.append(func)
        return func

    return register


# a super ugly hack, look away!
def _dynamic_func():
    None


_dhack = {}

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
def add2(num: Union[int, float]):
    """ add two to a real number """
    return num + 2


@function_task
@spype.core.task.task
def mult2(num):
    """ multiply a number by 2 """
    return num * 2


@function_task
@spype.core.task.task
def raise2(num):
    """ square a number """
    return num ** 2


@function_task
@spype.core.task.task
def divide2(num):
    """ divide a number by 2 """
    return num / 2


@function_task
@spype.core.task.task
def divide_numbers(num1, num2):
    """ divide two numbers """
    return num1 / num2


# --- simple tasks mainly for compatibility checks


@function_task
@spype.core.task.task
def plus_one_plus_two(num: int) -> Tuple[int, int]:
    return num + 1, num + 2


@function_task
@spype.core.task.task
def divide_ints(num1: int, num2: int) -> int:
    """ divide two numbers """
    return num1 // num2


@function_task
@spype.core.task.task
def int2str(num1: int) -> str:
    return str(num1)


@function_task
@spype.core.task.task
def str2int(str1: str) -> int:
    return int(str1)


def _task_from_closure():
    @spype.task
    def any_to_str(arg1: Any) -> str:
        return str(arg1)

    return any_to_str


_type = TypeVar('int_float_str', list, int, float, str)


@function_task
@spype.core.task.task
def add_things(num1: _type, num2: _type) -> _type:
    return num1 + num2


# --- gather up tasks


INSTANCE_TASKS = [x() for x in CLASS_TASKS]


@pytest.fixture(params=INSTANCE_TASKS + FUNCTION_TASK)
def generic_task(request):
    """ fixture to collect and parametrize simple tasks """
    return request.param


# ---------------------- task tests


class TestClassTasks:
    """ test the tasks defined using classes """

    # fixtures
    @pytest.fixture
    def p_one(self):
        return PlusOne()

    @pytest.fixture
    def callback_class(self, some_list):
        """ return a class with callbacks """

        def on_success_one(outputs):
            some_list.append('on_success_one called')

        def on_success_two(outputs):
            some_list.append('on_success_two called')

        class TimeTwo(spype.core.task.Task):
            def on_failure(self):
                some_list.append('on_fail called')

            def on_finish(self):
                some_list.append('on_finish called')

            on_success = [on_success_one, on_success_two]

            def __call__(self, num):
                if num is None:
                    raise ValueError

        return TimeTwo

    # tests
    def test_task_no_run_raises(self):
        """ insure defining a task without a __call__ raises a type error """

        with pytest.raises(TypeError):
            class BadTask(Task):
                pass

    def test_task_with_call_raises(self):
        """ tasks cannot define the run method, raise if they do """

        with pytest.raises(TypeError):
            class BadTask(Task):
                def run(self):
                    pass

                def __call__(self):
                    pass

    def test_call_returns_output(self, p_one):
        """ ensure calling the task returns output """
        assert p_one(1) == 2

    def test_callbacks(self, some_list, callback_class):
        """ test that the fail and finish callbacks go off """
        callback_class().run(None)
        assert 'on_fail called' == some_list[0]
        assert 'on_finish called' == some_list[1]

    def test_list_of_callbacks(self, some_list, callback_class):
        """ test that multiple callbacks get called when in list """
        callback_class().run(1)
        assert len(some_list) == 3
        assert 'on_success_one called' == some_list[0]
        assert 'on_success_two called' == some_list[1]


class TestDecoratorTask:
    """ tests to ensure the task decorator correctly constructs tasks """

    # fixtures
    @pytest.fixture
    def task_fail_and_finish(self, some_list):
        """ test a decorator task with an on_fail and on_finish method """

        def on_failure(task, e, *args, **kwargs):
            some_list.append('on_fail called')

        def on_finish(task, outputs, *args, **kwargs):
            some_list.append('on_finish called')

        @task(on_finish=on_finish, on_failure=on_failure)
        def raise_value_error(obj):
            """ just raise a value error """
            raise ValueError

        # assert on_failure is raise_value_error.on_failure

        return raise_value_error

    @pytest.fixture
    def task_multiple_callbacks(self, some_list):
        """ test a decorator task with an on_fail and on_finish method """

        def on_fail1(task, e, *args, **kwargs):
            some_list.append('on_fail called first')

        def on_fail2(task, e, *args, **kwargs):
            some_list.append('on_fail called second')

        @task(on_failure=[on_fail1, on_fail2])
        def raise_value_error(obj):
            """ just raise a value error """
            raise ValueError

        return raise_value_error

    # tests
    def test_is_task(self):
        """ assert that plus_one is a task """
        assert isinstance(add1, Task)

    def test_callbacks(self, some_list, task_fail_and_finish):
        """ test that the fail and finish callbacks go off """
        task_fail_and_finish.run(1)
        assert 'on_fail called' == some_list[0]
        assert 'on_finish called' == some_list[1]

    def test_list_of_callbacks(self, task_multiple_callbacks, some_list):
        """ test that multiple callbacks get called when in list """
        task_multiple_callbacks.run(1)
        assert len(some_list) == 2
        assert 'on_fail called first' == some_list[0]
        assert 'on_fail called second' == some_list[1]

    def test_signatures(self):
        """ ensure signatures of run methods are the same for class and
        decorator based tasks """
        p1_class = PlusOne()
        sig1 = inspect.signature(p1_class.run)
        sig2 = inspect.signature(add1.run)
        assert sig1 == sig2

    def test_func_with_self_doesnt_raise(self):
        """ ensure a function that uses self doesn't raise """
        try:
            @task
            def func(self, a, b):
                pass
        except Exception:
            pytest.fail('valid function failed')


class TestTasks:
    """ tests for both function and class based tasks """

    # tests
    def test_can_pickle(self, generic_task):
        """ test that all tasks can be pickled. This is, unfortunately,
        important to have simple multiprocessing functionality """

        pickle_string = pickle.dumps(generic_task)
        loaded_func = pickle.loads(pickle_string)

        assert generic_task.__dict__ == loaded_func.__dict__

    def test_is_virtual_instance_of_task(self, generic_task):
        """ ensure each task is a virtual instance of Task """
        assert isinstance(generic_task, Task)

    def test_wrap_functions(self, generic_task):
        """ certain attributes should wrap the task and return it """
        attrs_to_test = list(Wrap._wrap_funcs)
        attrs_to_test.remove('compatible')
        for attr in attrs_to_test:
            try:
                wrap = getattr(generic_task, attr)()
            except Exception:
                pytest.fail()
            assert isinstance(wrap, Wrap)

    def test_copy(self, generic_task):
        """ ensure copy returns a new object """
        assert generic_task.copy() is not generic_task


class TestTaskCompatibility:
    """ tests for checking if tasks are compatible """
    # fixtures

    # lists of compatible and incompatible tasks
    compatible_tasks = (
        (plus_one_plus_two, divide_ints),
        (divide_ints, plus_one_plus_two),
        (int2str, str2int),
        (str2int, int2str),
    )

    incompatible_tasks = (
        (int2str, divide_ints),
        (int2str, plus_one_plus_two),
    )

    # general tests
    @pytest.mark.parametrize('task1, task2', compatible_tasks)
    def test_compatible(self, task1, task2):
        """ ensure task1 and task2 are compatible """
        assert task1.compatible(task2)

    @pytest.mark.parametrize('task1, task2', incompatible_tasks)
    def test_not_compatible(self, task1, task2):
        """ ensure task1 outputs are valid inputs to task2 """
        assert not task1.compatible(task2)

        # specific tests


class TestTypeEnforcement:
    # fixtures
    @pytest.fixture(scope='class', autouse=True)
    def ensure_type_check_is_on(self):
        """ ensure type checking is on for task input/ouput for this suite of
         tests """
        with spype.options(check_type=True):
            yield

    @pytest.fixture(params=[add1, PlusOne()])
    def plus_one(self, request):
        """ plus one parametrization """
        return request.param

    @pytest.fixture
    def task_bad_output(self):
        """ return a task that outputs a type different than stated """

        @spype.core.task.task
        def bad_boy(obj: int) -> int:
            return str(obj)

        return bad_boy

    @pytest.fixture
    def task_with_typevar(self):
        """ return task with typevar """
        TV = TypeVar('TV', int, float, str)

        @spype.core.task.task
        def add_to_self(obj: TV) -> TV:
            """ add an object to itself, return """
            if obj == 13:  # unlucky number, here is another bug
                return float(13 + 13)
            return obj + obj

        return add_to_self

    # test
    def test_bad_type_raises(self, plus_one):
        """ make sure feeding a task a bad type will raise TypeError """

        with pytest.raises(TypeError):
            plus_one.run(1.1)

    def test_bad_type_doesnt_raise(self, plus_one):
        """ make sure the bad type doesnt raise when type-checking is
        disabled """
        with spype.options(check_type=False):
            plus_one.run(1.1)

    def test_bad_ouputs(self, task_bad_output):
        """ a task that gives a bad output should raise type error """
        # feeding it a str should raise (it expects int)
        with pytest.raises(TypeError):
            task_bad_output.run('hey')

        # when it tries to return a str it should raise (stated as int)
        with pytest.raises(TypeError):
            task_bad_output.run(1)

    def test_typevar(self, task_with_typevar):
        """ make sure the typevar is enforced """
        # if a type the same as input is returned it should not fail
        try:
            task_with_typevar.run(10)
        except Exception:
            pytest.fail('this should not fail')

        with pytest.raises(TypeError):
            task_with_typevar.run(13)

    def test_bad_number_of_args(self):
        """ ensure passing a bad number of arguments raises """
        with pytest.raises(TypeError):
            add1.run(1, 2, 3)

    def test_compatibility_check_before_callbacks(self, some_list):
        """ compatibility checks should be performed before any callbacks """

        def on_start():
            some_list.append(1)

        @task(on_start=on_start)
        def int_in_int_out(a: int) -> int:
            return int(a)

        with suppress(TypeError):
            int_in_int_out.run('a')

        assert len(some_list) == 0


class TestValidateCallbacks:
    """ tests for validating Task Callbacks """

    # tests
    def test_defining_bad_callback_on_task_subclass_rasies(self):
        """ ensure defining a bad callback in a class definintion raises """
        with pytest.raises(TypeError) as e:
            class BadCallback(spype.core.task.Task):
                def __call__(self):
                    pass

                def on_failure(self, not_a_valid_fixture):
                    pass

            BadCallback().validate_callbacks()
        assert 'not a valid parameter names' in str(e)

    def test_call_params_are_ok(self):
        """ ensure any parameter names in call method are not flagged as
        invalid """

        def on_fail(num1):
            pass

        @task(on_failure=on_fail)
        def some_task(num1):
            pass

        try:
            some_task.validate_callbacks()
        except TypeError:
            pytest.fail('should not raise on valid call parameters')


class TestPartials:
    """ test that the partial method works (Note this is slightly different
    from the functools partial)"""

    @pytest.fixture
    def some_task(self):
        """ return a a task for partiallizing """

        @task
        def some_func(arg1, arg2, arg3=None, arg4=42):
            """ return local dict """
            return dict(arg1=arg1, arg2=arg2, arg3=arg3, arg4=arg4)

        return some_func

    @pytest.fixture
    def partial1(self, some_task):
        """ return a partial of some task"""
        return some_task.partial(arg3=2)

    # tests
    def test_basic_partial(self, partial1):
        """ ensure calling partial task still works """
        out = partial1(1, 2)
        assert dict(arg3=2, arg1=1, arg2=2).items() <= out[0][0].items()

    def test_bad_partial_arguments_raises(self):
        """ passing a key that is not a parameter in task should raise
        a TypeError"""
        with pytest.raises(TypeError) as e:
            divide_numbers.partial(bob=2)
        assert 'is not a valid paramter' in str(e)


class TestIff:
    """ tests for using if statements on a task check compatibility between
     the tasks run method and the callable(s) fed to the iff statement """
    _should_raise = []
    _should_not_raise = []

    def test_iff_fixtures(self, some_dict):
        """ iff should use fixtures just like callbacks """

        def some_iff(e, num1, pype):
            some_dict['num1'] = num1
            return True

        wrap = divide_numbers.iff(some_iff)
        wrap(1, 2)

        assert some_dict['num1'] == 1


class TestPypeInputWrapAndTask:
    """ tests for the special task PypeInput """

    def test_wrap_is_not_singleton(self):
        """ ensure passing pype_input to _Task returns different objects  """
        assert pype_input.wrap() is not pype_input.wrap()

    def test_deep_copy_returns_same(self):
        """ ensure a deep copy still returns the same objects in instance """
        wrap = pype_input.wrap()
        wrap2 = copy.deepcopy(wrap)

        assert wrap2 is not wrap
        assert wrap2.task is wrap.task


class TestCallbacksInDepth:
    """ more in-depth tests for callbacks on tasks """

    def test_on_failure_from_docs(self):
        """ on failure example wasn't working in docs, this test
        replicates it """

        out = []

        # attach callbacks in task class
        class StrToInt(spype.core.task.Task):
            def __call__(self, obj: str) -> int:
                if obj == '13':
                    raise ValueError('unlucky numbers not accepted')
                return None

            def on_failure(self, inputs):
                out.append(inputs)

        str_to_int = StrToInt()
        str_to_int.run('13')
        assert (('13',), {}) in out

    def test_class_tasks_raise_on_default(self):
        """ make sure when no callbacks are defined the task still raises on
        exception """

        class DumbTask(spype.core.task.Task):
            def __call__(self, obj):
                return float(obj)

        dumbtask = DumbTask()
        with pytest.raises(ValueError):
            dumbtask.run('bob')

    def test_decorator_task_raise_on_default(self):
        """ ensure the decorator task rasies by default when to on_failure
        provided """

        @spype.core.task.task
        def dumbtask(obj):
            return float(obj)

        with pytest.raises(ValueError):
            dumbtask.run('bob')

    def test_decorator_task_on_can_be_skipped(self):
        """ ensure defining a custom on_fail overrides the default """

        def on_fail():
            pass

        @spype.core.task.task(on_failure=on_fail)
        def dumbtask(obj):
            return float(obj)

        try:
            dumbtask.run('hey')
        except ValueError:
            pytest.fail('should not raise')

    def test_on_start(self):
        """ ensure on_start can be used to get input args """
        some_list = []

        def on_start(inputs):
            some_list.append(inputs)

        @spype.core.task.task(on_start=on_start)
        def dumbtask(obj):
            return obj

        dumbtask.run(10)
        assert ((10,), {}) in some_list

    def test_task_class_definition_can_be_overwritten(self, monkeypatch):
        """ ensure when an instance callbacks are not defined it defaults
        to the class's """

        # define an on_finish callback that will raise
        def raise_some_error():
            raise ValueError('some error')

        # some simple task
        @spype.core.task.task
        def passit():
            pass

        # make sure calling the task doesn't raise
        try:
            passit.run()
        except ValueError:
            pytest.fail('should not raise')

        # monkey patch the on_finish callback to the Task class
        monkeypatch.setattr(spype.core.task.Task, 'on_finish', raise_some_error)

        # ensure it raises
        with pytest.raises(ValueError):
            passit.run()

    def test_values_from_input_can_be_used_as_fixtures(self):
        """ ensure a callback can get an input value from a fixture """
        out = {}

        def on_failure(a, c):
            out['on_failure'] = (a, c)

        def on_start(task, a):
            out['on_start'] = (task, a)

        def on_success(c):
            out['on_success'] = c

        def on_finish(outputs):
            out['on_finish'] = outputs

        @task(on_failure=on_failure, on_success=on_success, on_start=on_start,
              on_finish=on_finish)
        def some_task(a, b, c=None):
            return a + b

        some_task.run(1, 2, 'bob')

        assert out['on_start'] == (some_task, 1)
        assert out['on_success'] == 'bob'
        assert out['on_finish'] == 3

        some_task.run('a', 1, 'hey')

        assert out['on_failure'] == ('a', 'hey')

    combos = itertools.product(CALLBACK_NAMES, tuple(FIXTURE_NAMES))

    @pytest.mark.parametrize('callback, fixture', combos)
    def test_task_fixtures(self, callback, fixture):
        """ ensure each of the supported task fixtures works """

        # define dynamic function requesting particular callback
        str_func = (f'def _dynamic_func({fixture}):\n'
                    f'    _dhack["{fixture}"] = {fixture}')
        exec(str_func, globals())

        # define normal and failing tasks
        @spype.task(**{callback: _dynamic_func})
        def normal_task():
            pass

        @spype.task(**{callback: _dynamic_func})
        def fail_task():
            raise ValueError('I fail at everything... sad!')

        # figure out which tasks to run
        if callback == 'on_failure':
            fail_task.run()
        else:
            normal_task.run()

        # make sure global has fixture value set
        assert fixture in _dhack
        _dhack.clear()  # empty dict for next run

    def test_stop_task_execution(self):
        """ raising a ExitTask exception in a callback should result in """

        def stop_task():
            raise ExitTask('get outta here')

        @spype.task(on_finish=stop_task)
        def some_task():
            return True

        assert some_task.run() is None


class TestCallbackReturnValues:
    """ tests for callbacks returning values to interupt task execution """

    def test_return_value(self):
        """ ensure if a callback returns a None None value it """

        @task
        def return_int(a):
            return int(a)

        return_int.on_failure = lambda: 3

        assert return_int.run(2) == 2
        assert return_int.run('2') == 2
        assert return_int.run('bb') == 3

        return_int.on_success = lambda: 'two'

        assert return_int.run(1) == 'two'
        assert return_int.run('2') == 'two'
        assert return_int.run('bob') == 3

        return_int.on_start = lambda: 'java sucks'
        assert return_int.run(1) == 'java sucks'
        assert return_int.run(22) == 'java sucks'
        assert return_int.run('bob') == 'java sucks'

    def test_callback_execution(self):
        """ as soon as a callback returns a value it should be returned right
         away and no other callbacks should get called """
        out = {}

        def on_start():
            out['on_start'] = True
            return 1

        def on_finish():
            out['on_finish'] = True
            return 2

        @task(on_start=on_start, on_finish=on_finish)
        def some_task():
            pass

        assert some_task.run() == 1
        assert 'on_finish' not in out
