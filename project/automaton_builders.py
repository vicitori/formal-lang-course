from typing import Set
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    FiniteAutomaton,
    State,
)

import networkx as nx


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    enfa = Regex(regex).to_epsilon_nfa()
    dfa = enfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    enfa = FiniteAutomaton.from_networkx(graph)

    graph_nodes = set()
    if not start_states or not final_states:
        graph_nodes = set(graph.nodes)

    start_set = start_states if len(start_states) != 0 else graph_nodes
    final_set = final_states if len(final_states) != 0 else graph_nodes

    for state in start_set:
        enfa.add_start_state(State(state))
    for state in final_set:
        enfa.add_final_state(State(state))

    return enfa.remove_epsilon_transitions()
