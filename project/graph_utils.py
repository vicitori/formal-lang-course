from cfpq_data import *
from typing import Set, Dict, Any
import networkx as nx
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GraphData:
    nodes_count:int
    edges_count:int
    attributes:Dict[str, Set[Any]]

def get_graph_data(name:str):
    analyzer = GraphAnalyzer(name)
    return GraphData(analyzer.get_nodes_count(), analyzer.get_edges_count(), analyzer.get_labels())

def build_two_cycles_graph(cycle_size1: int, cycle_size2: int, data: tuple[str, str], file_path: str):
        networkx_graph = labeled_two_cycles_graph(cycle_size1, cycle_size2, labels=data)
        pydot_graph = nx.drawing.nx_pydot.to_pydot(networkx_graph)
        pydot_graph.write_raw(file_path)

class GraphAnalyzer:
    def __init__(self, name):
        if name in cfpq_data.DATASET:
            path = download(name)
            self.graph = graph_from_csv(path)

        # for custom graphs written like edgelist (https://networkx.org/documentation/latest/reference/readwrite/edgelist.html)
        else:
            path = Path(__file__) / "custom_graphs" / name
            self.graph = nx.read_edgelist(path, create_using=nx.MultiDiGraph)


    def get_nodes_cnt(self) -> int:
        return self.graph.number_of_nodes()

    def get_edges_cnt(self) -> int:
        return self.graph.number_of_edges()

    def get_all_attributes(self) -> Dict[str, Set[Any]]:
        attributeMap = {}
        for _, _, data in self.graph.edges(data=True):
            for attrName, attrValue in data.items():
                if attrName not in self.attributeMap:
                    attributeMap[attrName] = set()
                attributeMap[attrName].add(attrValue)
        return attributeMap
    
    def get_labels(self):
        return self.get_all_attributes["labels"]