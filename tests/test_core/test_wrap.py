"""
Tests for the wrap object
"""
import pytest

from spype import task, Pype, Wrap, Task


# --------------------------- tasks for wrapping


@pytest.fixture
def add_two():
    @task
    def add_two(a: int, b: int) -> int:
        return a + b

    return add_two


@pytest.fixture
def return_two():
    @task
    def return_two(a: int) -> (int, int):
        return a, a

    return return_two


@pytest.fixture
def add_two_strs():
    @task
    def add_two_strs(a: str, b: str) -> str:
        return a + b

    return add_two_strs


@pytest.fixture
def add_three():
    @task
    def add_three(a: int, b: int, c: int) -> int:
        return a + b + c

    return add_three


# --------------------------- tests


class TestWrapBasics:
    """ test basic functionality of wraps """

    # fixtures
    @pytest.fixture
    def wrap(self, add_two):
        return add_two.wrap()

    def test_copy(self, add_two):
        """ ensure copying a wrap returns a new object """
        wrap = add_two.wrap()
        wrap2 = wrap.copy()
        assert wrap is not wrap2
        assert wrap.task is wrap2.task

    def test_join_to_pypes(self, add_two, return_two):
        """ test that wraps can join to become pypes """
        wrap1, wrap2 = add_two.wrap(), return_two.wrap()
        p1 = wrap1 | wrap2
        assert isinstance(p1, Pype)

    def test_str(self, wrap):
        """ ensure str work """
        assert isinstance(str(wrap), str)

    def test_repr(self, wrap):
        """ ensure repr returns str """
        assert isinstance(repr(wrap), str)

    def test_wrap_wrap_copies(self, wrap):
        """ ensure wrapping a wrap function copies the task """
        wrap2 = Wrap(wrap)
        assert isinstance(wrap2, Wrap)
        assert isinstance(wrap2.task, Task)
        assert wrap2.task is wrap.task


class TestCompatibility:
    """ tests for wrap compatibility """

    def test_compatible(self, add_two, return_two, add_three, add_two_strs):
        wrap1, wrap2 = return_two.wrap(), add_two.wrap()
        wrap3, wrap4 = add_three.wrap(), add_two_strs.wrap()
        assert wrap1.compatible(wrap2)
        assert wrap2.compatible(wrap1)
        assert not wrap1.compatible(wrap3)
        assert not wrap2.compatible(wrap3)
        assert not wrap4.compatible(wrap1)
        assert not wrap1.compatible(1)


class TestWrapCallbacks:
    """ Test that callbacks that are added to wrap get executed, not just the
     task callbacks """

    # fixtures
    @pytest.fixture
    def callback_wrap(self, add_two, some_dict):
        """ return a wrap with two callbacks that write to some dict """

        def on_start(a, b):
            some_dict["on_start"] = (a, b)

        def on_finish(b):
            some_dict["on_finish"] = b

        return add_two.wrap(on_start=on_start, on_finish=on_finish)

    def test_wrap_callbacks_get_executed(self, callback_wrap, some_dict):
        """ ensure on fail gets executed """

        callback_wrap(1, 2)

        assert some_dict["on_start"] == (1, 2)
        assert some_dict["on_finish"] == 2

    def test_get_callback(self, callback_wrap):
        """ tests for the descriptor getters """
        assert callback_wrap.on_start
        assert len(callback_wrap.on_start)
        for cb in callback_wrap.on_start:
            assert callable(cb)

    def test_del_callback(self, callback_wrap):
        """ ensure callbacks can be deleted, which clears the list """
        del callback_wrap.on_finish
        assert len(callback_wrap.on_finish) == 0
