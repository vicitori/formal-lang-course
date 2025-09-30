import numpy as np
import networkx as nx
from scipy import sparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automaton_builders import regex_to_dfa, graph_to_nfa


def tensor_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_fa = AdjacencyMatrixFA(regex_dfa)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_fa = AdjacencyMatrixFA(graph_nfa)

    intersected_fa = intersect_automata(regex_fa, graph_fa)
    transitive_closure = intersected_fa.get_transitive_closure()

    start_ids = np.where(intersected_fa.start_vector)[0]
    final_ids = np.where(intersected_fa.final_vector)[0]

    result = set()
    for start_id in start_ids:
        for final_id in final_ids:
            if transitive_closure[start_id, final_id]:
                start_state = intersected_fa.state_of_id[start_id]
                final_state = intersected_fa.state_of_id[final_id]
                result.add((start_state[1], final_state[1]))

    return result


def ms_bfs_based_rpq(
    regex: str, graph: nx.MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    regex_fa = AdjacencyMatrixFA(regex_dfa)

    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_fa = AdjacencyMatrixFA(graph_nfa)

    dfa_size = regex_fa.state_cnt
    nfa_size = graph_fa.state_cnt

    dfa_start_id = np.where(regex_fa.start_vector)[0][0]
    nfa_starts_ids = np.where(graph_fa.start_vector)[0]
    block_count = len(nfa_starts_ids)

    symbols = (
        regex_fa.boolean_decomp_by_symbol.keys()
        & graph_fa.boolean_decomp_by_symbol.keys()
    )

    dfa_matrices_transposed = {
        symbol: regex_fa.boolean_decomp_by_symbol[symbol].transpose()
        for symbol in symbols
    }

    nfa_matrices = graph_fa.boolean_decomp_by_symbol

    start_front = sparse.csr_matrix((dfa_size * block_count, nfa_size), dtype=bool)
    for block_num, nfa_start in enumerate(nfa_starts_ids):
        start_front[dfa_start_id + dfa_size * block_num, nfa_start] = True

    front = start_front
    visited = sparse.csr_matrix((dfa_size * block_count, nfa_size), dtype=bool)

    while front.count_nonzero() > 0:
        visited += front
        front_by_symbol = {}
        for symbol in symbols:
            front_by_symbol[symbol] = front @ nfa_matrices[symbol]

            for block_num in range(block_count):
                start_id = block_num * dfa_size
                end_id = (block_num + 1) * dfa_size
                block = front_by_symbol[symbol][start_id:end_id]
                front_by_symbol[symbol][start_id:end_id] = (
                    dfa_matrices_transposed[symbol] @ block
                )

        new_front = sum(
            front_by_symbol.values(),
            sparse.csr_matrix((dfa_size * block_count, nfa_size), dtype=bool),
        )
        front = new_front > visited

    result = set()
    dfa_finals_ids = np.where(regex_fa.final_vector)[0]
    for block_num, start_node_id in enumerate(nfa_starts_ids):
        for dfa_final_id in dfa_finals_ids:
            row_id = block_num * dfa_size + dfa_final_id
            reachable = visited.getrow(row_id)

            reachable_vector = np.zeros(nfa_size, dtype=bool)
            reachable_vector[reachable.indices] = True

            final_vector = reachable_vector & graph_fa.final_vector
            nfa_finals_ids = np.where(final_vector)[0]

            for nfa_final_id in nfa_finals_ids:
                start = graph_fa.state_of_id[start_node_id]
                end = graph_fa.state_of_id[nfa_final_id]
                result.add((start, end))

    return result
