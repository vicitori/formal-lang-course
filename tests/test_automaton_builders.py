import pytest
from project.graph_utils import CYCLIC_GRAPHS_PATH
from project.automaton_builders import regex_to_dfa, graph_to_nfa
from pyformlang.regular_expression.regex_objects import MisformedRegexError
import networkx as nx

testdata_rtd_accepts_and_rejects = [
    pytest.param(
        ["a b", "a.b"],
        {"accepts": ["ab"], "rejects": ["a", "b", "ba", ""]},
        id="concatenation",
    ),
    pytest.param(
        ["a|b", "a+b"], {"accepts": ["a", "b"], "rejects": ["ab", "ba", ""]}, id="union"
    ),
    pytest.param(
        ["epsilon", "$"],
        {"accepts": [""], "rejects": ["a", "b", "ab"]},
        id="epsilon_symbols",
    ),
    pytest.param(
        ["a*"],
        {"accepts": ["", "a", "aa", "aaa"], "rejects": ["b", "ab"]},
        id="kleene_star",
    ),
]


@pytest.mark.parametrize(
    "regexs, expected_properties", testdata_rtd_accepts_and_rejects
)
def test_regex_to_dfa__accepts_and_rejects(regexs, expected_properties):
    for regex in regexs:
        dfa = regex_to_dfa(regex)

        for test_str in expected_properties["accepts"]:
            assert dfa.accepts(test_str)

        for test_str in expected_properties["rejects"]:
            assert not dfa.accepts(test_str)


testdata_rtd_invalid_regex = ["(a|b", "a|b)", "|", "*", "a||b"]


@pytest.mark.parametrize("invalid_regex", testdata_rtd_invalid_regex)
def test_regex_to_dfa__invalid_regex(invalid_regex):
    with pytest.raises(MisformedRegexError):
        assert regex_to_dfa(invalid_regex)


def create_test_graph_three_states():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")
    graph.add_edge(0, 2, label="c")
    return graph


testdata_gtn_accepts_and_rejects = [
    pytest.param(
        create_test_graph_three_states(),
        {0},
        {2},
        {"accepts": ["ab", "c"], "rejects": ["", "a", "b"]},
        id="specific_start_final",
    ),
    pytest.param(
        create_test_graph_three_states(),
        set(),
        set(),
        {"accepts": ["", "a", "ab", "c"], "rejects": ["d"]},
        id="empty_start_final",
    ),
    pytest.param(
        create_test_graph_three_states(),
        set(),
        {2},
        {"accepts": ["", "ab", "c", "b"], "rejects": ["a"]},
        id="empty_start_specific_final",
    ),
    pytest.param(
        create_test_graph_three_states(),
        {0},
        set(),
        {"accepts": ["", "a", "ab", "c"], "rejects": ["d"]},
        id="specific_start_empty_final",
    ),
]


@pytest.mark.parametrize(
    "graph, start_states, final_states, expected_properties",
    testdata_gtn_accepts_and_rejects,
)
def test_graph_to_nfa__accepts_and_rejects(
    graph, start_states, final_states, expected_properties
):
    nfa = graph_to_nfa(graph, start_states, final_states)

    for test_str in expected_properties["accepts"]:
        assert nfa.accepts(test_str)

    for test_str in expected_properties["rejects"]:
        assert not nfa.accepts(test_str)


@pytest.mark.parametrize(
    "start_states, final_states, expected_properties",
    [
        ({1}, {8}, {"start_states": {1}, "final_states": {8}}),
        (
            set(),
            set(),
            {
                "start_states": {0, 1, 2, 3, 4, 5, 6, 7, 8},
                "final_states": {0, 1, 2, 3, 4, 5, 6, 7, 8},
            },
        ),
    ],
)
def test_graph_to_nfa__dot_two_cycle_graph(
    start_states, final_states, expected_properties
):
    graph = nx.nx_pydot.read_dot(CYCLIC_GRAPHS_PATH / "example_graph.dot")
    nfa = graph_to_nfa(graph, start_states, final_states)

    assert nfa.start_states == expected_properties["start_states"]
    assert nfa.final_states == expected_properties["final_states"]
