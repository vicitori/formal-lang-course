from pyformlang.cfg import CFG, Terminal
import networkx as nx
from pyformlang.rsa import RecursiveAutomaton

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automaton_builders import graph_to_nfa, rsm_to_nfa
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


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_fa = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    g_size = graph_fa.state_cnt
    changed = True

    while changed:
        changed = False
        combined = intersect_automata(rsm_fa, graph_fa)
        closure = combined.get_transitive_closure()
        rows, cols = closure.nonzero()

        for pos_r, pos_c in zip(rows, cols):
            left_rsm = pos_r // g_size
            left_graph = pos_r % g_size

            right_rsm = pos_c // g_size
            right_graph = pos_c % g_size

            _, s_lbl = rsm_fa.state_of_id[left_rsm].value
            _, t_lbl = rsm_fa.state_of_id[right_rsm].value

            if s_lbl != t_lbl:
                continue

            if (
                left_rsm not in rsm_fa.start_vector.nonzero()[0]
                or right_rsm not in rsm_fa.final_vector.nonzero()[0]
            ):
                continue

            mats = graph_fa.boolean_decomp_by_symbol
            if s_lbl not in mats:
                mats[s_lbl] = sparse.csr_matrix((g_size, g_size), dtype=bool)

            if not mats[s_lbl][left_graph, right_graph]:
                mats[s_lbl][left_graph, right_graph] = True
                changed = True

    results = set()

    if rsm.initial_label not in graph_fa.boolean_decomp_by_symbol:
        return results

    mtx = graph_fa.boolean_decomp_by_symbol[rsm.initial_label]

    for i in range(g_size):
        for j in range(g_size):
            if not mtx[i, j]:
                continue

            start_ok = (not start_nodes) or (graph_fa.state_of_id[i] in start_nodes)
            final_ok = (not final_nodes) or (graph_fa.state_of_id[j] in final_nodes)

            if start_ok and final_ok:
                results.add((graph_fa.state_of_id[i], graph_fa.state_of_id[j]))

    return results


RsmState = tuple[object, object]  # (Symbol, State)
GraphState = int
GllConfig = tuple[RsmState, GraphState, int]


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    def collect_rsm_transitions(_rsm) -> dict[RsmState, dict[object, set[RsmState]]]:
        res: dict[RsmState, dict[object, set[RsmState]]] = {}
        for box_label, box in _rsm.boxes.items():
            dfa = box.dfa
            net = dfa.to_networkx()
            for u, v, data in net.edges(data=True):
                lbl = data.get("label")
                key = (box_label, u)
                res.setdefault(key, {})
                res[key].setdefault(lbl, set()).add((box_label, v))
            for s in dfa.states:
                res.setdefault((box_label, s), {})
        return res

    def collect_graph_transitions(
        graph: nx.DiGraph,
    ) -> dict[GraphState, dict[object, set[GraphState]]]:
        res: dict[GraphState, dict[object, set[GraphState]]] = {}
        for u, v, data in graph.edges(data=True):
            lbl = data.get("label")
            res.setdefault(u, {})
            res[u].setdefault(lbl, set()).add(v)
        for n in graph.nodes:
            res.setdefault(n, {})
        return res

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    rsm_trans = collect_rsm_transitions(rsm)
    graph_trans = collect_graph_transitions(graph)

    rsm_labels = set(rsm.boxes.keys())

    gss_nodes: dict[int, dict] = {}
    gss_edges: dict[int, list] = {}
    next_gss_id = 0

    def make_gss_node(rsm_state: RsmState, graph_state: GraphState) -> int:
        nonlocal next_gss_id
        for nid, info in gss_nodes.items():
            if info["node"] == (rsm_state, graph_state):
                return nid
        nid = next_gss_id
        next_gss_id += 1
        gss_nodes[nid] = {"node": (rsm_state, graph_state), "returns": set()}
        gss_edges[nid] = []
        return nid

    start_label = rsm.initial_label
    start_dfa = rsm.boxes[start_label].dfa
    start_state = (start_label, start_dfa.start_state)

    init_configs: set[GllConfig] = set()
    for sn in start_nodes:
        gss_id = make_gss_node(start_state, sn)
        cfg: GllConfig = (start_state, sn, gss_id)
        init_configs.add(cfg)

    work: set[GllConfig] = set(init_configs)
    seen: set[GllConfig] = set(init_configs)

    def step_term(cfg: GllConfig) -> set[GllConfig]:
        rsm_state, gstate, gss_id = cfg
        out = set()
        rsm_map = rsm_trans.get(rsm_state, {})
        graph_map = graph_trans.get(gstate, {})
        common_labels = set(rsm_map.keys()) & set(graph_map.keys())
        for lbl in common_labels:
            for nr in rsm_map[lbl]:
                for ng in graph_map[lbl]:
                    out.add((nr, ng, gss_id))
        return out

    def step_nonterm(cfg: GllConfig) -> set[GllConfig]:
        rsm_state, gstate, gss_id = cfg
        out = set()
        candidate_labels = set(rsm_trans.get(rsm_state, {}).keys()) & rsm_labels
        for label in candidate_labels:
            callee_dfa = rsm.boxes[label].dfa
            callee_start = (label, callee_dfa.start_state)
            callee_gss_id = make_gss_node(callee_start, gstate)

            for return_rsm_state in rsm_trans[rsm_state][label]:
                if gss_nodes[callee_gss_id]["returns"]:
                    for ret_graph_state in gss_nodes[callee_gss_id]["returns"]:
                        out.add((return_rsm_state, ret_graph_state, gss_id))
                gss_edges[callee_gss_id].append((gss_id, return_rsm_state))
            out.add((callee_start, gstate, callee_gss_id))
        return out

    def step_return(cfg: GllConfig) -> set[GllConfig]:
        rsm_state, gstate, gss_id = cfg
        out = set()
        box_label, dfa_state = rsm_state
        dfa = rsm.boxes[box_label].dfa
        if dfa_state not in dfa.final_states:
            return out
        gss_nodes[gss_id]["returns"].add(gstate)
        for parent_id, return_rsm_state in gss_edges[gss_id]:
            out.add((return_rsm_state, gstate, parent_id))
        return out

    while work:
        cfg = work.pop()
        new_cfgs = set()
        new_cfgs |= step_term(cfg)
        new_cfgs |= step_nonterm(cfg)
        new_cfgs |= step_return(cfg)

        for nc in new_cfgs:
            if nc not in seen:
                seen.add(nc)
                work.add(nc)

    result: set[tuple[int, int]] = set()
    for init in init_configs:
        _, init_graph_state, init_gss_id = init
        for ret in gss_nodes[init_gss_id]["returns"]:
            if ret in final_nodes:
                result.add((init_graph_state, ret))
    return result
