import numpy as np
from typing import Iterable
from scipy import sparse
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)


class AdjacencyMatrixFA:
    def __init__(
        self,
        automaton: NondeterministicFiniteAutomaton = None,
        matrix_class=sparse.csr_matrix,
    ):
        self.boolean_decomp_by_symbol = {}
        self.start_vector = np.zeros(0, dtype=bool)
        self.final_vector = np.zeros(0, dtype=bool)
        self.id_of_state = {}
        self.state_of_id = {}
        self.state_cnt = 0
        self.matrix_class = matrix_class

        if automaton is None:
            return

        self.state_cnt = len(automaton.states)

        self.id_of_state = {state: id for id, state in enumerate(automaton.states)}
        self.state_of_id = {id: state for id, state in enumerate(automaton.states)}

        start_ids = {self.id_of_state[state] for state in automaton.start_states}
        self.start_vector = np.zeros(self.state_cnt, dtype=bool)
        for id in start_ids:
            self.start_vector[id] = True

        final_ids = {self.id_of_state[state] for state in automaton.final_states}
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
                transitions_by_symbol.setdefault(symbol, []).append(
                    (from_state, to_state)
                )

        for symbol in transitions_by_symbol.keys():
            transitions = transitions_by_symbol[symbol]

            rows = np.fromiter(
                (self.id_of_state[from_state] for from_state, _ in transitions),
                dtype=int,
            )
            cols = np.fromiter(
                (self.id_of_state[to_state] for _, to_state in transitions), dtype=int
            )
            data = np.ones(len(transitions), dtype=bool)

            matrix = self.create_matrix(data, rows, cols)

            self.boolean_decomp_by_symbol[symbol] = matrix

    def create_matrix(self, data, rows, cols):
        if self.matrix_class in [sparse.lil_matrix, sparse.dok_matrix]:
            lil_mat = self.matrix_class((self.state_cnt, self.state_cnt), dtype=bool)
            for i, j, val in zip(rows, cols, data):
                lil_mat[i, j] = val
            return lil_mat
        else:
            return self.matrix_class(
                (data, (rows, cols)), shape=(self.state_cnt, self.state_cnt)
            )

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_vector = self.start_vector
        for symbol in word:
            # all reachable states from the starting ones
            current_vector = current_vector @ self.boolean_decomp_by_symbol[symbol]
        return np.any(current_vector & self.final_vector)

    def is_empty(self) -> bool:
        transitive_closure = self.get_transitive_closure()

        # all reachable states from the starting ones
        current_vector = self.start_vector @ transitive_closure
        return not np.any(current_vector & self.final_vector)

    def get_transitive_closure(self):
        # initialized identity matrix
        matrix_format = self.matrix_class.format.fget(self.matrix_class)
        transitive_closure = sparse.identity(
            self.state_cnt, dtype=bool, format=matrix_format
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

    id_of_state = {}
    state_of_id = {}
    composite_states = []
    for id_state1 in range(size1):
        state1 = automaton1.state_of_id[id_state1]
        for id_state2 in range(size2):
            state2 = automaton2.state_of_id[id_state2]
            composite_states.append((state1, state2))

    for global_id, composite_state in enumerate(composite_states):
        id_of_state[composite_state] = global_id
        state_of_id[global_id] = composite_state

    start_vector = np.zeros(kron_matrix_size, dtype=bool)
    final_vector = np.zeros(kron_matrix_size, dtype=bool)

    for global_id, composite_state in state_of_id.items():
        state1, state2 = composite_state
        id_state1 = automaton1.id_of_state[state1]
        id_state2 = automaton2.id_of_state[state2]

        if automaton1.start_vector[id_state1] and automaton2.start_vector[id_state2]:
            start_vector[global_id] = True

        if automaton1.final_vector[id_state1] and automaton2.final_vector[id_state2]:
            final_vector[global_id] = True

    intersected_fa = AdjacencyMatrixFA()
    if automaton1.matrix_class == automaton2.matrix_class:
        intersected_fa.matrix_class = automaton1.matrix_class

    intersected_fa.state_cnt = kron_matrix_size
    intersected_fa.boolean_decomp_by_symbol = kron_boolean_decomp_by_symbol
    intersected_fa.start_vector = start_vector
    intersected_fa.final_vector = final_vector
    intersected_fa.id_of_state = id_of_state
    intersected_fa.state_of_id = state_of_id

    return intersected_fa
