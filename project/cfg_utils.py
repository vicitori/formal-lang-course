import pyformlang
from pyformlang.cfg import CFG, Production, Epsilon


def cfg_to_weak_normal_form(cfg: CFG) -> pyformlang.cfg.CFG:
    cnf = cfg.to_normal_form()
    weak_cnf_productions = set(cnf.productions)

    for nonterminal in cfg.get_nullable_symbols():
        weak_cnf_productions.add(Production(nonterminal, [Epsilon()]))

    return CFG(
        variables=cnf.variables,
        terminals=cnf.terminals,
        start_symbol=cnf.start_symbol,
        productions=weak_cnf_productions,
    )
