import cfpq_data
from typing import Set, Dict, Any
import networkx as nx
from dataclasses import dataclass
from pathlib import Path

CYCLIC_GRAPHS_PATH = Path(__file__).parent / "two_cycles_graphs"


@dataclass
class GraphData:
    nodes_count: int
    edges_count: int
    attributes: Dict[str, Set[Any]]


def get_graph_data(name: str):
    analyzer = GraphAnalyzer(name)
    return GraphData(
        analyzer.get_nodes_cnt(), analyzer.get_edges_cnt(), analyzer.get_labels()
    )


def write_dot_two_cycles_graph(
    fst_cycle_nodes: int, snd_nodes_cnt: int, labels: tuple[str, str], file_path: str
):
    if fst_cycle_nodes <= 0 or snd_nodes_cnt <= 0:
        raise ValueError(
            "Error: write_dot_two_cycles_graph: Count of nodes should be positive number. Try to input other values."
        )

    networkx_graph = cfpq_data.labeled_two_cycles_graph(
        n=fst_cycle_nodes, m=snd_nodes_cnt, labels=labels
    )
    pydot_graph = nx.drawing.nx_pydot.to_pydot(networkx_graph)
    CYCLIC_GRAPHS_PATH.mkdir(exist_ok=True)
    pydot_graph.write_raw(CYCLIC_GRAPHS_PATH / file_path)


class GraphAnalyzer:
    def __init__(self, name):
        if name in cfpq_data.DATASET:
            path = cfpq_data.download(name)
            self.graph = cfpq_data.graph_from_csv(path)

        # for custom graphs written like edgelist (https://networkx.org/documentation/latest/reference/readwrite/edgelist.html)
        else:
            path = Path(__file__).parent / "custom_graphs" / name
            self.graph = nx.read_edgelist(path, create_using=nx.MultiDiGraph)

    def get_nodes_cnt(self) -> int:
        return self.graph.number_of_nodes()

    def get_edges_cnt(self) -> int:
        return self.graph.number_of_edges()

    def get_all_attributes(self) -> Dict[str, Set[Any]]:
        attributeMap = {}
        for _, _, data in self.graph.edges(data=True):
            for attrName, attrValue in data.items():
                if attrName not in attributeMap:
                    attributeMap[attrName] = set()
                attributeMap[attrName].add(attrValue)
        return attributeMap

    def get_labels(self):
        try:
            return (self.get_all_attributes())["label"]
        except KeyError:
            return {}
