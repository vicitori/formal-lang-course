from pyformlang.cfg import CFG, Terminal
import networkx as nx
from project.cfg_utils import cfg_to_weak_normal_form
from scipy import sparse


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)

    known_triples = set()

    known_triples.update(
        (production.head, u, v)
        for u, v, symbol in graph.edges(data="label")
        for production in wcnf.productions
        if [Terminal(symbol)] == production.body
    )

    known_triples.update(
        (nonterminal, v, v)
        for v in graph.nodes
        for nonterminal in wcnf.get_nullable_symbols()
    )

    triples_to_process = set(known_triples)
    binary_productions = [prod for prod in wcnf.productions if len(prod.body) == 2]

    while triples_to_process:
        curr_nonterm, curr_start, curr_end = triples_to_process.pop()
        merged_triples = set()

        for known_nonterm, known_start, known_end in known_triples:
            if curr_end == known_start:
                for prod in binary_productions:
                    if prod.body == [curr_nonterm, known_nonterm]:
                        new_triple = (prod.head, curr_start, known_end)
                        if new_triple not in known_triples:
                            merged_triples.add(new_triple)
            if curr_start == known_end:
                for prod in binary_productions:
                    if prod.body == [known_nonterm, curr_nonterm]:
                        new_triple = (prod.head, known_start, curr_end)
                        if new_triple not in known_triples:
                            merged_triples.add(new_triple)

        for triple in merged_triples:
            if triple not in known_triples:
                triples_to_process.add(triple)
                known_triples.add(triple)

    result = set()
    for nonterm, start, end in known_triples:
        is_correct_path = (
            nonterm == wcnf.start_symbol
            and (start_nodes is None or start in start_nodes)
            and (final_nodes is None or end in final_nodes)
        )
        if is_correct_path:
            result.add((start, end))
    return result


def matrix_based_cfpq(
    grammar: CFG,
    input_graph: nx.DiGraph,
    starting_vertices: set[int] = None,
    ending_vertices: set[int] = None,
) -> set[tuple[int, int]]:
    vertex_index_map = {}
    index_vertex_map = {}

    for idx, vertex in enumerate(input_graph.nodes):
        vertex_index_map[vertex] = idx
        index_vertex_map[idx] = vertex

    normalized_grammar = cfg_to_weak_normal_form(grammar)
    grammar_rules = normalized_grammar.productions
    nullable_symbols = normalized_grammar.get_nullable_symbols()

    graph_size = input_graph.number_of_nodes()

    terminal_matrices = {
        var: sparse.csr_matrix((graph_size, graph_size), dtype=bool)
        for var in normalized_grammar.variables
    }

    graph_edges = input_graph.edges(data="label")

    for source, target, edge_label in graph_edges:
        for rule in grammar_rules:
            if rule.body == [Terminal(edge_label)]:
                src_idx = vertex_index_map[source]
                tgt_idx = vertex_index_map[target]
                terminal_matrices[rule.head][src_idx, tgt_idx] = True

    for null_sym in nullable_symbols:
        for diagonal_idx in range(graph_size):
            terminal_matrices[null_sym][diagonal_idx, diagonal_idx] = True

    processing_queue = set(normalized_grammar.variables)

    while processing_queue:
        current_var = processing_queue.pop()

        for rule in grammar_rules:
            if current_var in rule.body:
                product_matrix = (
                    terminal_matrices[rule.body[0]] @ terminal_matrices[rule.body[1]]
                )

                if (product_matrix > terminal_matrices[rule.head]).count_nonzero() > 0:
                    terminal_matrices[rule.head] += product_matrix
                    processing_queue.add(rule.head)

    path_pairs = set()

    for row_idx in range(graph_size):
        for col_idx in range(graph_size):
            start_vertex = index_vertex_map[row_idx]
            end_vertex = index_vertex_map[col_idx]

            if (
                terminal_matrices[normalized_grammar.start_symbol][row_idx, col_idx]
                and (not starting_vertices or start_vertex in starting_vertices)
                and (not ending_vertices or end_vertex in ending_vertices)
            ):
                path_pairs.add((start_vertex, end_vertex))

    return path_pairs
