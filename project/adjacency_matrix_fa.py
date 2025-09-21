import numpy as np
import networkx as nx
from typing import Iterable
from scipy import sparse
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)

from automaton_builders import (regex_to_dfa, graph_to_nfa)


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton = None):
        self.boolean_decomp_by_symbol = {}
        self.start_vector = np.zeros(0, dtype=bool)
        self.final_vector = np.zeros(0, dtype=bool)
        self.id_of_state = {}
        self.state_of_id = {}
        self.state_cnt = 0

        if automaton is None:
            return

        self.state_cnt = len(automaton.states)

        self.id_of_state = {state: id for id,
                            state in enumerate(automaton.states)}
        self.state_of_id = {id: state for id,
                            state in enumerate(automaton.states)}

        start_ids = {
            self.id_of_state[state] for state in automaton.start_states}
        self.start_vector = np.zeros(self.state_cnt, dtype=bool)
        for id in start_ids:
            self.start_vector[id] = True

        final_ids = {
            self.id_of_state[state] for state in automaton.final_states}
        self.final_vector = np.zeros(self.state_cnt, dtype=bool)
        for id in final_ids:
            self.final_vector[id] = True

        transitions_by_symbol = {}
        graph = automaton.to_networkx()
        edges = graph.edges(data="label")
        for edge in edges:
            from_state = edge[0]
            to_state = edge[1]
            symbol = edge[2]

            if symbol is not None:
                transitions_by_symbol.setdefault(
                    symbol, []).append((from_state, to_state))

        for symbol in transitions_by_symbol.keys:
            transitions = transitions_by_symbol[symbol]

            rows = np.fromiter(
                (self.id_of_state[from_state] for from_state, _ in transitions), dtype=int)
            cols = np.fromiter(
                (self.id_of_state[to_state] for _, to_state in transitions), dtype=int)
            data = np.ones(len(transitions), dtype=bool)

            matrix = sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(self.state_cnt, self.state_cnt),
            )

            self.boolean_decomp_by_symbol[symbol] = matrix

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_vector = self.start_vector
        for symbol in word:
            # all reachable states from the starting ones
            current_vector = (
                current_vector @ self.boolean_decomp_by_symbol[symbol]
            )
        return np.any(current_vector & self.final_vector)

    def is_empty(self) -> bool:
        transitive_closure = self.get_transitive_closure()

        # all reachable states from the starting ones
        current_vector = self.start_vector @ transitive_closure
        return not np.any(current_vector & self.final_vector)

    def get_transitive_closure(self):
        # initialized identity matrix
        transitive_closure = sparse.csr_matrix(
            (np.ones(self.state_cnt, dtype=bool),
             (range(self.state_cnt), range(self.state_cnt)),),
            shape=(self.state_cnt, self.state_cnt),
        )

        for matrix in self.boolean_decomp_by_symbol.values():
            transitive_closure = matrix + transitive_closure

        # all possible paths of length self.state_cnt - 1 are taken into account
        transitive_closure = transitive_closure ** (self.state_cnt - 1)
        return transitive_closure.tocsr()


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    size1 = automaton1.state_cnt
    size2 = automaton2.state_cnt
    kron_matrix_size = size1 * size2

    kron_boolean_decomp_by_symbol = {}
    kron_symbols = (
        automaton1.boolean_decomp_by_symbol.keys()
        & automaton2.boolean_decomp_by_symbol.keys()
    )

    for symbol in kron_symbols:
        kron_boolean_decomp_by_symbol[symbol] = sparse.kron(
            automaton1.boolean_decomp_by_symbol[symbol],
            automaton2.boolean_decomp_by_symbol[symbol],
        )

    start_ids = [id1 * size2 + id2
                 for id1, data1 in enumerate(automaton1.start_vector)
                 if data1
                 for id2, data2 in enumerate(automaton2.start_vector)
                 if data2
                 ]
    start_vector = np.zeros(kron_matrix_size, dtype=bool)
    start_vector[start_ids] = True

    final_ids = [id1 * size2 + id2
                 for id1, data1 in enumerate(automaton1.final_vector)
                 if data1
                 for id2, data2 in enumerate(automaton2.final_vector)
                 if data2
                 ]
    final_vector = np.zeros(kron_matrix_size, dtype=bool)
    final_vector[final_ids] = True

    intersected_fa = AdjacencyMatrixFA()

    intersected_fa.state_cnt = kron_matrix_size
    intersected_fa.boolean_decomp_by_symbol = kron_boolean_decomp_by_symbol
    intersected_fa.start_vector = start_vector
    intersected_fa.final_vector = final_vector
    intersected_fa.id_of_state = {id: id for id in range(kron_matrix_size)}
    intersected_fa.state_of_id = {id: id for id in range(kron_matrix_size)}

    return intersected_fa


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_fa = AdjacencyMatrixFA(regex_dfa)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_fa = AdjacencyMatrixFA(graph_nfa)

    intersected_fa = intersect_automata(graph_fa, regex_fa)
    transitive_closure = intersected_fa.get_transitive_closure()

    graph_start_ids = np.where(graph_fa.start_vector)[0]
    graph_final_ids = np.where(graph_fa.final_vector)[0]

    regex_start_ids = np.where(regex_fa.start_vector)[0]
    regex_final_ids = np.where(regex_fa.final_vector)[0]

    result = set()
    for g_start in graph_start_ids:
        for r_start in regex_start_ids:
            intersect_start = g_start * regex_fa.state_cnt + r_start

            for g_final in graph_final_ids:
                for r_final in regex_final_ids:
                    intersect_final = g_final * regex_fa.state_cnt + r_final

                    if transitive_closure[intersect_start, intersect_final]:
                        start_state = graph_fa.state_of_id[g_start]
                        final_state = graph_fa.state_of_id[g_final]
                        result.add((start_state, final_state))

    return result
