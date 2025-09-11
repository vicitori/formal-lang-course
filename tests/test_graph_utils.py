import pytest
import pydot
import os
import networkx as nx
from project.graph_utils import *
import shutil

############# GraphAnalyzer.nodes_cnt #############


def test_nodes_cnt__empty_graph():
    analyzer = GraphAnalyzer("empty")
    assert analyzer.get_nodes_cnt() == 0


def test_nodes_cnt__graph_one_node_several_edges():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_nodes_cnt() == 1


def test_nodes_cnt__graph_from_dataset():
    analyzer = GraphAnalyzer("wine")
    # https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/wine.html#wine
    assert analyzer.get_nodes_cnt() == 733


############# GraphAnalyzer.edges_cnt #############


def test_edges_cnt__empty_graph():
    analyzer = GraphAnalyzer("empty")
    assert analyzer.get_edges_cnt() == 0


def test_edges_cnt__graph_with_loops():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_edges_cnt() == 3


def test_edges_cnt__multigraph():
    analyzer = GraphAnalyzer("two_nodes_several_edges")
    assert analyzer.get_edges_cnt() == 4


def test_edges_cnt__graph_from_dataset():
    analyzer = GraphAnalyzer("wine")
    # https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/wine.html#wine
    assert analyzer.get_edges_cnt() == 1839


######### GraphAnalyzer.get_all_attributes ########


def test_get_all_attributes__no_attributes():
    analyzer = GraphAnalyzer("without_attributes")
    assert analyzer.get_all_attributes() == {}


def test_get_all_attributes__two_attributes():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_all_attributes() == {
        "animal": {"cat", "dog"},
        "flower": {"rose"},
    }


############# GraphAnalyzer.get_labels ############


def test_get_labels__has_attributes_no_labels():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_labels() == {}


def test_get_labels__has_not_only_labels():
    analyzer = GraphAnalyzer("has_attributes_including_labels")
    assert analyzer.get_labels() == {"1", "2", "3", "4"}


################## get_graph_data #################


def test_get_graph_data__empty_graph():
    assert get_graph_data("empty") == GraphData(0, 0, {})


def test_get_graph_data__normal_graph_with_different_attributes():
    assert get_graph_data("has_attributes_including_labels") == GraphData(
        9, 9, {"1", "2", "3", "4"}
    )


############### get_two_cycles_graph ##############

TEST_CYCLIC_GRAPHS_PATH = CYCLIC_GRAPHS_PATH.parent.parent / "tests" / "tmp"

# auxiliary for dot files reading


def read_dot_graph(path) -> nx.MultiDiGraph:
    graphs = pydot.graph_from_dot_file(str(path))
    return nx.drawing.nx_pydot.from_pydot(graphs[0])


def test_get_two_cycles_graph__file_creating():
    try:
        path = TEST_CYCLIC_GRAPHS_PATH / "empty.dot"
        TEST_CYCLIC_GRAPHS_PATH.mkdir(exist_ok=True)
        get_two_cyclic_graph(3, 3, ("a", "b"), path)
        assert os.path.isfile(path)
    finally:
        if TEST_CYCLIC_GRAPHS_PATH.exists():
            shutil.rmtree(TEST_CYCLIC_GRAPHS_PATH)


def test_get_two_cycles_graph__invalid_nodes():
    path = TEST_CYCLIC_GRAPHS_PATH / "empty.dot"
    TEST_CYCLIC_GRAPHS_PATH.mkdir(exist_ok=True)

    try:
        with pytest.raises(ValueError):
            get_two_cyclic_graph(0, 0, ("a", "b"), path)

        with pytest.raises(ValueError):
            get_two_cyclic_graph(5, 0, ("a", "b"), path)

        with pytest.raises(ValueError):
            get_two_cyclic_graph(-5, 0, ("a", "b"), path)

    finally:
        if TEST_CYCLIC_GRAPHS_PATH.exists():
            shutil.rmtree(TEST_CYCLIC_GRAPHS_PATH)
