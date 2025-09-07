import pytest
import networkx as nx
from pathlib import Path
from project.graph_utils import GraphAnalyzer

############# GraphAnalyzer.nodes_cnt ############# 

def test_nodes_cnt__empty_graph():
    analyzer = GraphAnalyzer("empty")
    assert analyzer.get_nodes_cnt() == 0

def test_nodes_cnt__graph_one_node_several_edges():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_nodes_cnt() == 1

def test_nodes_cnt__graph_from_dataset():
    analyzer = GraphAnalyzer("wine")
    assert analyzer.get_nodes_cnt() == 733 # https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/wine.html#wine


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
    assert analyzer.get_edges_cnt() == 1839 # https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/data/wine.html#wine


######### GraphAnalyzer.get_all_attributes ########

def test_get_all_attributes__no_attributes():
    analyzer = GraphAnalyzer("without_attributes")
    assert analyzer.get_all_attributes() == {}

def test_get_all_attributes__two_attributes():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_all_attributes() == {"animal": {"cat", "dog"}, "flower": {"rose"}}


############# GraphAnalyzer.get_labels ############

def test_get_labels__has_attributes_no_labels():
    analyzer = GraphAnalyzer("one_node_three_loops_has_attributes")
    assert analyzer.get_labels() == {}

def test_get_labels__has_not_only_labels():
    analyzer = GraphAnalyzer("has_attributes_including_labels")
    assert analyzer.get_labels() == {"1", "2", "3", "4"}
