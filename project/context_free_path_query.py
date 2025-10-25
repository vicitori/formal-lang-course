from pyformlang.cfg import CFG, Terminal
import networkx as nx
from project.cfg_utils import cfg_to_weak_normal_form


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
