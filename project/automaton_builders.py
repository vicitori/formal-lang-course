from typing import Set
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    FiniteAutomaton,
    State,
)

import networkx as nx
from pyformlang.rsa import RecursiveAutomaton


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


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    result_nfa = NondeterministicFiniteAutomaton()
    for automaton_name in rsm.labels:
        automaton = rsm.boxes[automaton_name].dfa
        transition_map = automaton.to_dict()

        initial_nfa_state = (automaton.start_state, automaton_name)
        result_nfa.add_start_state(initial_nfa_state)

        for accepting_state in automaton.final_states:
            final_nfa_state = (accepting_state, automaton_name)
            result_nfa.add_final_state(final_nfa_state)

        for origin_state in transition_map:
            for transition_symbol, destination_state in transition_map[
                origin_state
            ].items():
                nfa_origin = (origin_state, automaton_name)
                nfa_destination = (destination_state, automaton_name)
                result_nfa.add_transition(
                    nfa_origin, transition_symbol, nfa_destination
                )

    return result_nfa
