import pyformlang
from pyformlang.rsa import RecursiveAutomaton


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
