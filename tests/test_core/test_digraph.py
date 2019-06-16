"""
Tests for the network digraph
"""
import sys
from typing import Tuple, TypeVar

import pytest

import spype
from spype import task, pype_input
from spype.core.digraph import _WrapDiGraph
from spype.exceptions import InvalidPype
from spype.utils import iterate


def append_func_name(list_like):
    """ decorator to append function to list """

    def _decor(func):
        list_like.append(func.__name__)
        return func

    return _decor


# --------------------- tasks/wraps for digraph testing

# ts = {num: task(lambda obj: obj) for num in range(10)}

ts = {}
# rename tasks
for num in range(10):
    t = task(lambda obj: obj)
    t.__name__ = str(num)
    ts[num] = t

ws = {num: task.wrap() for num, task in ts.items()}


class TaskOne(spype.Task):
    def __call__(self, obj):
        return obj


@task
def two_int(a: int, b: int) -> Tuple[int, int]:
    return a, b


two_arg_task2 = two_int.copy()


@task
def return_int(x: int) -> int:
    return int(x)


@task
def return_str(x: str) -> str:
    return str(x)


int_float_or_list = TypeVar("int_or_float", int, float, list)


@task
def add_two(x: int_float_or_list, y: int_float_or_list) -> int_float_or_list:
    return x + y


# ------------------------ module fixtures

VALID_DIGRAPHS = []
INVALID_DIGRAPHS = []


# --- valid graphs


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def digraph():
    """ init an empty digraph """
    return _WrapDiGraph()


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def filled_digraph():
    """ create a digraph that is populated with some wraps """
    digraph = _WrapDiGraph()
    digraph.add_wrap([ws[1], ws[2], ws[3], ws[4]])
    digraph.add_edge([(ws[2], ws[3]), (ws[1], ws[2]), (ws[3], ws[4])])
    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def pype_net1():
    """ a simple network that uses the input pype """
    digraph = _WrapDiGraph()
    digraph.add_edge([(pype_input.wrap(), ws[1]), (ws[1], ws[2]), (ws[2], ws[3])])
    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def pype_net2(digraph):
    """ another network that has the input_pype """
    digraph.add_edge([(pype_input.wrap(), ws[4]), (ws[4], ws[5])])
    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def graph_with_deps(digraph):
    """ return a digraph that has wraps with dependencies """

    w1 = ts[0].wrap()
    w2 = two_int.partial(a=ts[0]).fit(0)
    w3 = ts[1].wrap()

    digraph.add_edge(w1, w2)
    digraph.add_edge(w2, w3)

    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def simple_dependencies_graph(digraph):
    """ create a graph with dependencies that are not cyclic """
    digraph.add_edge(ws[0], ws[1])
    digraph.add_edge(ws[1], two_int.partial(a=ws[2].task))

    digraph.add_edge(ws[0], ws[2])
    digraph.add_edge(ws[2], two_arg_task2.partial(a=ws[1].task))
    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def graph_with_valid_task_partial(digraph):
    """ create a graph with tasks that connect and are only compatible due
    to partial """
    digraph.add_edge(return_int.wrap(), two_int.partial(b=2))
    return digraph


@pytest.fixture
@append_func_name(VALID_DIGRAPHS)
def graph_with_valid_task_fit(digraph):
    """ create a graph with tasks that connect and are only compatible due
    to fit (which reshapes outputs) """
    digraph.add_edge(two_int.fit(0), return_int.wrap())
    return digraph


# --- invalid graphs


@pytest.fixture
@append_func_name(INVALID_DIGRAPHS)
def cyclic_graph(digraph):
    """ create a graph with a single cycle """
    digraph.add_edge(ws[0], ws[1])
    digraph.add_edge(ws[1], ws[2])
    digraph.add_edge(ws[2], ws[0])
    return digraph


@pytest.fixture
@append_func_name(INVALID_DIGRAPHS)
def cyclic_dependencies_graph(digraph):
    """ create a graph with cyclic dependencies """
    two_to_one1 = task(lambda a, b: a + b).wrap()
    two_to_one2 = task(lambda a, b: a + b).wrap()

    tt1 = two_to_one1.partial(a=ws[2].task)
    tt1.task.__name__ = "tt1"
    tt2 = two_to_one2.partial(a=ws[1].task)
    tt2.task.__name__ = "tt2"

    digraph.add_edge(ws[0], tt1)
    digraph.add_edge(tt1, ws[1])

    digraph.add_edge(ws[0], tt2)
    digraph.add_edge(tt2, ws[2])
    return digraph


@pytest.fixture
@append_func_name(INVALID_DIGRAPHS)
def graph_with_incompatible_arg_numbers(digraph):
    """ create a graph with tasks that connect but do not have compatible
    input and outputs (different numbers of arguments) """
    digraph.add_edge(return_int.wrap(), add_two.wrap())
    return digraph


@pytest.fixture
@append_func_name(INVALID_DIGRAPHS)
def graph_with_incompatible_types(digraph):
    """ A digraph with wraps that have incompatible types. """
    digraph.add_edge(return_int.wrap(), return_str.wrap())
    return digraph


# --- meta fixtures


@pytest.fixture(params=VALID_DIGRAPHS)
def valid_digraph(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=INVALID_DIGRAPHS)
def invalid_digraph(request):
    return request.getfixturevalue(request.param)


# ------------------------ tests


class TestNodeWraps:
    """ tests for adding/removing wrap nodes to network """

    # a list of objects and expected values in nodes attr
    nodes_to_add = [(ws[1], {ws[1]}), ([ws[1], ws[3], ws[3]], {ws[1], ws[3]})]

    # tests
    @pytest.mark.parametrize("nodes, expected", nodes_to_add)
    def test_add_single_node(self, digraph, nodes, expected):
        """ ensure the values put into the digraph node pool show up as
        expected """
        digraph.add_wrap(nodes)
        assert set(expected) == set(digraph.wraps)
        tasks = {x.task for x in iterate(nodes)}
        assert set(digraph.tasks) == tasks

    def test_add_node(self, digraph):
        """ test that adding node does just that """
        digraph.add_wrap(ws[1])
        assert ws[1] in digraph.wraps

    def test_remove_node(self, filled_digraph):
        """ ensure a node can be removed from the digraph """
        remove_node = ws[2]
        remove_task = ws[2].task
        assert remove_task in filled_digraph.tasks
        filled_digraph.remove_wrap(remove_node)
        assert set(filled_digraph.wraps) == {ws[1], ws[3], ws[4]}
        assert set(filled_digraph.edges) == {(ws[3], ws[4])}
        assert remove_task not in filled_digraph.tasks

    def test_pype_input(self, pype_net1):
        """ A test for checking pype_input is in """
        assert pype_net1.get_input_wrap().task is pype_input

    def test_bad_neighbors_rasie(self, pype_net1):
        """ asking for the neighbors of a non-existant node should raise """
        with pytest.raises(KeyError):
            pype_net1.neighbors(1)

    def test_init_graph_with_wraps(self, digraph):
        wraps = list(ws.values())
        graph = _WrapDiGraph(wraps=wraps)
        assert set(wraps) == set(graph.wraps)


class TestReplaceNode:
    """ tests for replacing one node with another """

    # tests
    def test_replace(self, filled_digraph):
        """ ensure replace_wrap works for all edges and such """
        filled_digraph.replace_wrap(ws[2], ws[5])
        # ensure wraps dict was updated
        assert ws[5] in filled_digraph.wraps
        assert ws[2] not in filled_digraph.wraps
        # ensure edges were updated
        assert (ws[5], ws[3]) in filled_digraph.edges
        assert (ws[1], ws[5]) in filled_digraph.edges
        assert (ws[2], ws[3]) not in filled_digraph.edges
        assert (ws[1], ws[2]) not in filled_digraph.edges


class TestEdge:
    """ tests for adding/removing edges to network """

    def test_edges_get_added_to_nodes(self, digraph):
        """ ensure adding edges not in nodes adds the nodes """
        digraph.add_edge(ws[1], ws[2])
        assert {(ws[1], ws[2])} == set(digraph.edges)

    def test_add_edges(self, digraph):
        """ ensure a sequence gets added """
        edge_list = [(ws[1], ws[2]), (ws[3], ws[4]), (ws[5], ws[6])]
        digraph.add_edge(edge_list)
        assert set([ws[num] for num in range(1, 7)]) == set(digraph.wraps)
        assert set(edge_list) == set(digraph.edges)


class TestCombineDigraphs:
    """ tests for adding (combining) two digraphs """

    def test_combine_two_digraphs(self):
        """ ensure adding two networks combines all sets and dicts but does
        not modify the original networks """
        # create 2 networks
        edges1 = [(ws[1], ws[2]), (ws[3], ws[4])]
        edges2 = [(ws[5], ws[6]), (ws[7], ws[8]), (ws[1], ws[4])]
        d1 = _WrapDiGraph(edges=edges1)
        d2 = _WrapDiGraph(edges=edges2)
        # create new network
        d3 = d1 | d2
        # ensure nodes updated
        assert set(ws[num] for num in range(1, 9)) == set(d3.wraps)
        # ensure edges updated
        assert set(edges1 + edges2) == set(d3.edges)
        # make sure old graphs didn't change
        assert d3 is not d1 and d3 is not d2
        assert set(d1.edges) == set(edges1)
        assert set(d2.edges) == set(edges2)
        assert set(d2.neighbors(ws[1])) == {ws[4]}
        assert {**d1.tasks, **d2.tasks} == d3.tasks
        assert {**d1.wraps, **d2.wraps} == d3.wraps

    def test_combine_digraph_with_inputs(self, pype_net1, pype_net2):
        """ ensure two digraphs can be combined with pype_input objects """
        net3 = pype_net1 | pype_net2
        assert net3.get_input_wrap().task is pype_input


class TestCopy:
    """ ensure copying Digraph behaves as expected """

    def test_copy_doesnt_modify_original(self, digraph: _WrapDiGraph):
        """ ensure removing a node from a copy doesn't do the same on
        original"""
        ws5 = TaskOne().wrap()
        nodes = {ws[1], ws[2], ws[3], ws[4], ws5}
        edges = [(ws[1], ws[2]), (ws[3], ws[4]), (ws[4], ws5)]

        digraph.add_edge(edges)
        d2 = digraph.copy()
        assert set(edges) == set(d2.edges)

        d2.remove_wrap(ws[2])

        assert set(d2.wraps) == {ws[1], ws[3], ws[4], ws5}
        assert set(digraph.wraps) == nodes

        assert (set(edges) - set(d2.edges)) == {(ws[1], ws[2])}
        assert set(edges) == set(digraph.edges)

    def test_copy_leaves_objects(self, digraph):
        """ ensure copying digraph does not deep copy nodes """
        edges = [(ws[1], ws[2]), (ws[3], ws[4])]
        digraph.add_edge(edges)

        d1 = digraph.copy()
        for edge in edges:
            for node in edge:
                assert node in d1.wraps

    def test_nested_lists_copied(self, filled_digraph):
        """ ensure the nested list structures are also copied, eg on tasks
        attribute """
        n1 = filled_digraph
        n2 = filled_digraph.copy()
        # iterate tasks and ensure each list was copied
        for task_ in set(n1.tasks) & set(n2.tasks):
            assert n1.tasks[task_] is not n2.tasks[task_]


class TestValidateDigraph:
    """ ensure validate function catches unresolvable networks """

    def test_doesnt_raise_on_valid_digraphs(self, valid_digraph):
        """ ensure calling validate doesn't raise on valid digraphs """
        try:
            valid_digraph.validate()
        except Exception:
            pytest.fail("should not raise")

    def test_raies_on_invalid_digraph(self, invalid_digraph):
        """ ensure digraphs do raise on invalid digraphs """
        with pytest.raises(InvalidPype):
            invalid_digraph.validate()

    def test_disable_type_check(self, graph_with_incompatible_types):
        """ ensure digraph that have incompatible types are valid when type
        checking is disabled. """
        with spype.options(check_type=False):
            try:
                graph_with_incompatible_types.validate()
            except InvalidPype:
                pytest.fail("should not fail when type checking is turned off")

    def test_disable_compat_check(self, graph_with_incompatible_arg_numbers):
        """ ensure compat. checks can be disabled """
        # ensure global options can disable compat check
        graph: _WrapDiGraph = graph_with_incompatible_arg_numbers
        with spype.options(check_compatibility=False):
            try:
                graph.validate()
            except InvalidPype:
                pytest.fail("should not fail when type checking is turned off")
        # ensure the validate call can disable compat check
        try:
            graph.validate(check_task_compatibility=False)
        except InvalidPype:
            pytest.fail("should not raise")


class TestPlotting:
    """ tests for plotting the digraph """

    @pytest.fixture
    def digraph_with_dep(self, digraph):
        """ return a digraph with a dependency """
        wrap1 = add_two.wrap()
        wrap2 = two_int.partial(b=add_two)
        digraph.add_edge(wrap1, wrap2)
        return digraph

    def test_plot_digraph_with_dep(self, digraph_with_dep: _WrapDiGraph):
        """ ensure plotting a digraph with dependencies doesnt raise """
        digraph_with_dep.plot(view=False)

    def test_plotting_no_graphviz(self, digraph_with_dep, monkeypatch):
        """ patch graphviz to not be importable, ensure str returned """
        monkeypatch.setitem(sys.modules, "graphviz", None)
        with pytest.warns(UserWarning) as uw:
            digraph_with_dep.plot()
        msg = str(uw.list[0].message)
        assert "graphviz is not installed" in msg
