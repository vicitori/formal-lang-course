from cfpq_data import *
from typing import Set, Dict, Any
import networkx as nx

class GraphAnalyzer:
    def __init__(self, name):
        path = download(name)
        self.graph = graph_from_csv(path)

    def nodes_cnt(self) -> int:
        return self.graph.number_of_nodes()

    def edges_cnt(self) -> int:
        return self.graph.number_of_edges()

    def get_all_attributes(self) -> Dict[str, Set[Any]]:
        attributeMap = {}
        for _, _, data in self.graph.edges(data=True):
            for attrName, attrValue in data.items():
                if attrName not in self.attributeMap:
                    attributeMap[attrName] = set()
                attributeMap[attrName].add(attrValue)
        return attributeMap
    
    def build_cyclic_graph(self, n: int, m: int, data: tuple[str, str], file_path: str):
        networkx_graph = labeled_two_cycles_graph(n, m, labels=data)
        pydot_graph = nx.drawing.nx_pydot.to_pydot(networkx_graph)
        pydot_graph.write_raw(file_path)
        