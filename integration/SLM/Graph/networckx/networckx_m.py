import networkx as nx


def get_connected_components(graph,min_size=1):
    res = nx.connected_components(graph)
    return [x for x in res if len(x) >= min_size]

