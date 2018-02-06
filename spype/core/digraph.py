"""
Digraph implementation for wraps. Used by spype.core.pype.Pype to direct data
"""

import copy
import warnings
from collections import defaultdict
from contextlib import suppress
from typing import Optional, Union, Sequence, Tuple, Set

from spype.core import task
from spype.core import wrap
from spype.exceptions import UnresolvableNetwork, IncompatibleTasks
from spype.utils import iterate

wrap_node_type = Union['wrap.Wrap', Sequence['wrap.Wrap']]
edge_type = Sequence[Tuple['wrap.Wrap', 'wrap.Wrap']]


def _get_edge_plot_attrs(wrap1, wrap2):
    """ get the kwargs to pass to graphviz plot edges based on edge labels """
    labels = wrap1._out_edge_lab | wrap2._in_edge_lab

    out = {}
    if 'is_conditional' in labels:
        out['label'] = wrap2.conditional_name
        out['style'] = 'dashed'
        out['fontsize'] = '10'
    if 'is_fan' in labels or 'is_aggregate' in labels:
        out['style'] = 'tapered'
        out['arrowtail'] = 'none'
        out['arrowhead'] = 'none'
        out['penwidth'] = '7'
        if 'is_fan' in labels and 'is_aggregate' in labels:
            out['dir'] = 'both'
        elif 'is_aggregate' in labels:
            out['dir'] = 'forward'
        elif 'is_fan' in labels:
            out['dir'] = 'back'
    return out


class _WrapDiGraph:
    """
    Class to encapsulate a wrap network with directional connections.

    Note: This class is only ment to handle networks of Wrap instances,
    if you need something more general purpose checkout networkx.
    """

    def __init__(self, wraps: Optional[wrap_node_type] = None,
                 edges: Optional[edge_type] = None):
        """
        Parameters
        ----------
        wraps
            Wrap objects that serve as nodes in the network.
        edges
            Directional edges to connect the nodes. Will add any nodes
            included in the edge tuple that are not already in the network.

        Attributes
        ----------
        tasks
            A mapping of each unique tasks to the wraps that cover it.
            Eg task:[wrap, ...]
        wraps
            A mapping of wraps to the tasks they cover. Eg wrap: task
        """
        # init data structures for representing graphs and dependencies
        self.wraps = {}
        self.edges = defaultdict(set)
        self.tasks = defaultdict(list)
        self.dependencies = defaultdict(set)
        self.node_map = defaultdict(set)
        # add wraps and edges
        for wrap_ in iterate(wraps):
            self.add_wrap(wrap_)
        for edge in iterate(edges):
            self.add_edge(edge)

    # --- functions for handling wraps (nodes)

    def add_wrap(self, wrap_: wrap_node_type) -> None:
        """
        Add a single wrap or a sequence of wraps to the network.

        Parameters
        ----------
        wrap_
            A Wrap object or sequence of wrap objects to add to network.
        """
        for wrap_ in iterate(wrap_):
            assert isinstance(wrap_, wrap.Wrap)
            if wrap_ not in self.wraps:
                self.wraps[wrap_] = wrap_.task
                self.tasks[wrap_.task].append(wrap_)
                # add dependencies
                for dep in iterate(wrap_.features['dependencies']):
                    self.dependencies[dep].add(wrap_)

    def remove_wrap(self, wrap: wrap_node_type, edges=True):
        """
        remove a wrap or sequence of wraps from the network.

        Also removes all edges that use the removed wraps(s) if edges.
        """
        for wr in iterate(wrap):
            # pop out of wraps dict
            self.wraps.pop(wr, None)
            # pop out of map
            self.node_map.pop(wr, None)
            # remove from edges
            if edges:
                for edge in set(self.edges):
                    if wr in edge:
                        self.edges.pop(edge, None)
            # remove tasks that are empty from task list
            if wrap.task in set(self.tasks):
                with suppress(ValueError):
                    self.tasks[wr.task].remove(wr)
                if not len(self.tasks[wr.task]):
                    self.tasks.pop(wr.task, None)

    def replace_wrap(self, old_wrap, new_wrap):
        """ replace instances of old_wrap with new_wrap """
        self.remove_wrap(old_wrap, edges=False)
        self.add_wrap(new_wrap)
        old2new = {old_wrap: new_wrap}
        # replace edges
        for edge in set(self.edges):
            if old_wrap in edge:
                self.edges.pop(edge)
                self.add_edge(tuple(old2new.get(x, x) for x in edge))

    # --- functions for handling edges

    def add_edge(self, *args, edge_type=None):
        """ add a either a single edge (wrap, wrap) or a sequences of edges to
        network """
        # args is two wraps, should be connected as an edge
        if len(args) == 2 and isinstance(args[0], wrap.Wrap):
            wrap1, wrap2 = args
            # set edge types (eg conditional, fan, aggregate, etc.)
            # used for determining how to plot edge
            self.edges[(wrap1, wrap2)] = None
            # add to nodes
            self.add_wrap(wrap1)
            self.add_wrap(wrap2)
            # add to map
            self.node_map[wrap1].add(wrap2)
        # args contains a sequece of edges
        else:
            for item in args:
                self.add_edge(*item, edge_type=edge_type)

    # --- functions for finding out about networks structure

    def neighbors(self, wrap_: 'wrap.Wrap', include_deps=False
                  ) -> Set['wrap.Wrap']:
        """
        Return a set of all neighbors to node.
        Parameters
        ----------
        wrap_
            Any hashable (Wrap instance) in network.
        include_deps
            If True also include nodes that depend on this node.
        Returns
        -------
        A list of Wrap_ objects next to wrap_.
        """
        if wrap_ not in self.wraps:
            msg = f'{wrap_} is not in the network'
            raise KeyError(msg)
        if include_deps:
            touching = self.node_map[wrap_] | self.dependencies[wrap_.task]
        else:
            touching = self.node_map[wrap_]
        # need to return union of node map and wraps because node_map wont
        # pop removed wraps out of value set when wraps are removed
        return touching & set(self.wraps)

    def yield_connected(self, wrap_, have_yielded=None, **kwargs):
        """ yield all wraps (nodes) connected to wrap_ """
        have_yielded = have_yielded or set()
        for neighbor in (self.neighbors(wrap_, **kwargs) - have_yielded):
            if neighbor not in have_yielded:
                have_yielded.add(neighbor)
                yield neighbor
            yield from self.yield_connected(neighbor, have_yielded, **kwargs)

    def connected(self, wrap1, wrap2, include_deps=False) -> bool:
        """
        Return True if wrap1 is connected to wrap2 else False.

        Parameters
        ----------
        wrap1
            First wrap
        wrap2
            Second wrap
        include_deps
            If True include the dependency paths as well in figure out
            which nodes are connected.
        """
        return wrap2 in self.yield_connected(wrap1, include_deps=include_deps)

    def get_input_wrap(self):
        input_wrap_list = self.tasks[task.pype_input]
        assert len(input_wrap_list) == 1
        return input_wrap_list[0]

    # --- plotting functions

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
        try:
            import graphviz
        except (ImportError, ModuleNotFoundError):
            warnings.warn('graphviz is not installed, displaying str repr')
            print(str(self))
            return
        dot = graphviz.Digraph(graph_attr=dict(rankdir="LR"))
        # plot all nodes in flow graph
        for nx_node in self.wraps:
            dot.node(str(id(nx_node)), nx_node.task_name)
        # plot connections
        for edge in self.edges:
            dot.edge(str(id(edge[0])), str(id(edge[1])),
                     **_get_edge_plot_attrs(edge[0], edge[1]))
        # plot dependencies
        for task_, wraps in self.dependencies.items():
            if wraps:
                assert len(self.tasks[task_]) == 1  # dep task should be unique
                twrap = self.tasks[task_][0]
                for wrap_ in wraps:
                    dot.edge(str(id(twrap)), str(id(wrap_)), color='blue',
                             arrowsize='.5')
        # save/view
        if file_name or view:
            file_name = file_name or '.network.pdf'
            dot.render(filename=file_name, view=view)
        return dot

    # --- validation methods

    def validate(self, check_task_compatibility=True, extra_params=None):
        """
        Test for several digraph issues and raise an exception if needed.
        """
        # ensure there are no cycles in the network
        for wrap_ in self.wraps:
            # test for cycles
            if self.connected(wrap_, wrap_, include_deps=True):
                msg = f'{wrap_} is connected to itself without a condition'
                raise UnresolvableNetwork(msg)
        # ensure all edges are compatible or continue
        if not check_task_compatibility:
            return
        for wrap1, wrap2 in self.edges:
            # if either wrap or parent have compat checks turned off skip
            cw1 = wrap1.task.get_option('check_compatibility')
            cw2 = wrap2.task.get_option('check_compatibility')
            if not (cw1 and cw2):
                continue
            # perform compatibility check and raise if needed
            if not wrap1.compatible(wrap2, extra_params=extra_params):
                msg = f'output of {wrap1} is not valid input to {wrap2}'
                raise IncompatibleTasks(msg)

    # --- merging and copying

    def __or__(self, other: '_WrapDiGraph') -> '_WrapDiGraph':
        """ used to merge two networks together. Roughly equivilent to
        set1 | set2. Returns a digraph for which self and other are subsets"""
        # ensure we don't change any objects in network with subsequent calls
        assert isinstance(other, _WrapDiGraph), 'other must be a _WrapDigraph'
        new = _WrapDiGraph()
        other = other.copy()
        # try to pop out wrap_input from other it it has one
        try:
            other.remove_wrap(other.get_input_wrap())
        except AssertionError:
            pass
        # add wraps to node map and add wraps
        for item in set(self.wraps) | set(other.wraps):
            new.node_map[item].update(self.node_map.get(item, set()))
            new.node_map[item].update(other.node_map.get(item, set()))
            new.add_wrap(item)
        new.edges = {**self.edges, **other.edges}
        return new

    def copy(self):
        new = _WrapDiGraph()
        for item, val in self.__dict__.items():
            setattr(new, item, copy.copy(val))
            if isinstance(val, dict):
                # copy nested sequences
                for item1, val1 in val.items():
                    if isinstance(val1, Sequence):
                        getattr(new, item)[item1] = copy.copy(val1)
        return new
