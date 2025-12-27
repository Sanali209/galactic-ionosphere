# Detect communities using Louvain algorithm
# partition = community.best_partition(G)
import os

import networkx as nx
from neo4j import GraphDatabase
from tqdm import tqdm

# Output the communities
# communities = {}
# for node, community_id in partition.items():
# if community_id not in communities:
# communities[community_id] = []
# communities[community_id].append(node)

# for community_id, nodes in communities.items():
# print(f"Community {community_id}: {nodes}"

URI = "bolt://localhost"
Auth = ("neo4j", "12345678")

def_db = "neo4j"


def read_env():
    global def_db
    env_db = os.environ.get('NEO4J_DB')
    if env_db is not None:
        def_db = env_db


def get_neo4j_driver():
    read_env()
    return GraphDatabase.driver(URI, auth=Auth, database=def_db)


def add_node(session, node_label, **kwargs):
    """
    Add a node to the graph
    @param session: Neo4j session object
    @param node_label: label of the node
    @param kwargs: set of properties for the node
    @return: None
    """
    properties = ', '.join([f"{key}: ${key}" for key in kwargs.keys()])
    query = f"CREATE (n:{node_label} {{{properties}}})"
    session.run(query, **kwargs)


def add_node_bulk(session, label, edges_params_list):
    """
    Add nodes to the graph in bulk.

    Returns:
        None
        @param edges_params_list:
        @param session:
        @param label:
    """
    if len(edges_params_list) == 0:
        return
    node_label = label
    # Construct Cypher query with parameters to add nodes in bulk
    query = (
        "UNWIND $params AS params "
        f"CREATE (n:{node_label}) SET n = params"
    )

    # Execute Cypher query with parameters
    session.run(query, params=edges_params_list)


def get_node(session, node_label, **kwargs):
    """
    Get a node from the graph
    @param session: Neo4j session object
    @param node_label: label of the node
    @param kwargs: set of properties for the node
    @return: Node
    """
    properties = ' AND '.join([f"a.{key} = ${key}" for key in kwargs.keys()])
    query = f"MATCH (a:{node_label}) WHERE {properties} RETURN a"
    result = session.run(query, **kwargs)
    return result.single()


def get_nodes_by_prop(session, node_label, prop_name, value_list: list):
    """
    Get bach of nodes from the graph where the source_property value is in the list
    @param session: Neo4j session object
    @param node_label: label of the node
    @param kwargs: list of values for the source_property
    @return: list of nodes
    """
    query = f"MATCH (a:{node_label}) WHERE a.{prop_name} IN $value_list RETURN a"
    result = session.run(query, value_list=value_list)
    return list(result)


def get_edge(session, node_1_label, node_1_prop_name, node_1_prop_val,
             node_2_label, node_2_prop_name, node_2_prop_val,
             edge_label):
    """
    Retrieve an edge from the graph.

    Args:
        session: Neo4j session object.
        node_1_label: Label of the first node.
        node_1_prop_name: Name of the source_property for the first node.
        node_1_prop_val: Value of the source_property for the first node.
        node_2_label: Label of the second node.
        node_2_prop_name: Name of the source_property for the second node.
        node_2_prop_val: Value of the source_property for the second node.
        edge_label: Label of the edge.

    Returns:
        Record: Record containing the edge.
    """
    # Construct Cypher query with parameters to retrieve the edge
    query = (
        f"MATCH (a:{node_1_label} {{{node_1_prop_name}: ${node_1_prop_name}}}) "
        f"MATCH (b:{node_2_label} {{{node_2_prop_name}: ${node_2_prop_name}}}) "
        f"MATCH (a)-[r:{edge_label}]->(b) "
        "RETURN r"
    )

    # Execute Cypher query with parameters
    result = session.run(query,
                         **{node_1_prop_name: node_1_prop_val, node_2_prop_name: node_2_prop_val})

    # Return the record containing the edge
    return result.single()


def add_edge(session, node_1_label, node_1_prop_name, node_1_prop_val,
             node_2_label, node_2_prop_name, node_2_prop_val,
             edge_label, **kwargs):
    """
    Add an edge to the graph.

    Args:
        session: Neo4j session object.
        node_1_label: Label of the first node.
        node_1_prop_name: Name of the source_property for the first node.
        node_1_prop_val: Value of the source_property for the first node.
        node_2_label: Label of the second node.
        node_2_prop_name: Name of the source_property for the second node.
        node_2_prop_val: Value of the source_property for the second node.
        edge_label: Label of the edge.
        **kwargs: Additional properties for the edge.

    Returns:
        None
    """
    # Construct Cypher query with parameters
    properties = ', '.join([f"{key}: ${key}" for key in kwargs.keys()])
    query = (
        f"MATCH (a:{node_1_label} {{{node_1_prop_name}: ${node_1_prop_name}}}) "
        f"MATCH (b:{node_2_label} {{{node_2_prop_name}: ${node_2_prop_name}}}) "
        f"CREATE (a)-[r:{edge_label} {{{properties}}}]->(b)"
    )

    # Execute Cypher query with parameters
    session.run(query, **kwargs,
                **{node_1_prop_name: node_1_prop_val, node_2_prop_name: node_2_prop_val})


def add_edge_bulk(session, node_type, node_value_name, edge_type, edges):
    query = "UNWIND $rels AS rel " \
            f"MATCH (a:{node_type} {{{node_value_name}: rel.val1}}) " \
            f"MATCH (b:{node_type} {{{node_value_name}: rel.val2}}) " \
            f"CREATE (a)-[r:{edge_type} ]->(b)" \
            "SET r = rel.params"

    session.run(query, rels=edges)


def add_edge_bulk_by_node_id(session,  edge_type, edges):
    if len(edges) == 0:
        return

    query = "UNWIND $rels AS rel " \
            f"MATCH (a) WHERE ID(a) = rel.val1 " \
            f"MATCH (b) WHERE ID(b) = rel.val2 " \
            f"CREATE (a)-[r:{edge_type}]->(b)" \
            "SET r = rel.params"

    session.run(query, rels=edges)


def is_node_exist(session, node_label, **kwargs):
    """
    Check if a node exists in the graph.

    Args:
        session: Neo4j session object.
        node_label: Label of the node.
        **kwargs: Properties of the node.

    Returns:
        bool: True if the node exists, False otherwise.
    """
    # Construct Cypher query with parameters
    properties = ' AND '.join([f"a.{key} = ${key}" for key in kwargs.keys()])
    query = f"MATCH (a:{node_label}) WHERE {properties} RETURN a"

    # Execute Cypher query with parameters
    result = session.run(query, **kwargs)

    # Return True if the node exists, False otherwise
    return result.single() is not None


def is_edge_exist(session, node_1_label, node_1_prop_name, node_1_prop_val,
                  node_2_label, node_2_prop_name, node_2_prop_val,
                  edge_label):
    """
    Check if an edge exists in the graph.

    Args:
        session: Neo4j session object.
        node_1_label: Label of the first node.
        node_1_prop_name: Name of the source_property for the first node.
        node_1_prop_val: Value of the source_property for the first node.
        node_2_label: Label of the second node.
        node_2_prop_name: Name of the source_property for the second node.
        node_2_prop_val: Value of the source_property for the second node.
        edge_label: Label of the edge.

    Returns:
        bool: True if the edge exists, False otherwise.
    """
    # Construct Cypher query with parameters
    query = (
        f"MATCH (a:{node_1_label} {{{node_1_prop_name}: ${node_1_prop_name}}}) "
        f"MATCH (b:{node_2_label} {{{node_2_prop_name}: ${node_2_prop_name}}}) "
        f"RETURN EXISTS((a)-[:{edge_label}]->(b))"
    )

    # Execute Cypher query with parameters
    result = session.run(query,
                         **{node_1_prop_name: node_1_prop_val, node_2_prop_name: node_2_prop_val})

    # Return True if the edge exists, False otherwise
    return result.single()[0]


def clear_db(session):
    deleted = True
    while deleted:
        result = session.run("MATCH (n) "
                             "WITH n LIMIT 100 "
                             "DETACH DELETE n "
                             "RETURN count(n) as deletedNodes")
        deleted = result.single()["deletedNodes"] > 0


def edit_edge_property(session, edge_id, prop_name, prop_val):
    result = session.run("MATCH ()-[r]-() WHERE ID(r) = $edge_id SET r." + prop_name + " = $prop_val"
                         , edge_id=edge_id, prop_val=prop_val)
    return result.single()


def load_graph_from_neo4j(session, node_label, edge_type):
    shift = 0
    items_per_query = 1000
    graph = nx.Graph()
    count_edges = session.run(f"MATCH (n:{node_label})-[r:{edge_type}]->(m:{node_label}) RETURN count(r) as count")
    progress = tqdm(total=count_edges.single()['count'])
    while True:
        result = session.run(f"MATCH (n:{node_label})-[r:{edge_type}]->(m:{node_label}) RETURN n,m,r"
                                f" SKIP {shift} LIMIT {items_per_query}")

        reslist = list(result)
        progress.update(len(reslist))
        if len(reslist) == 0:
            break
        for record in tqdm(reslist):
            node1 = record['n']
            node2 = record['m']
            r = record['r']
            id_1 = node1.id
            id_2 = node2.id
            node_1_prop = dict(node1.items())
            node_2_prop = dict(node2.items())

            graph.add_node(id_1, **node_1_prop)
            graph.add_node(id_2, **node_2_prop)
            edge_prop = dict(r.items())
            graph.add_edge(id_1, id_2, **edge_prop)
        shift += items_per_query
    return graph
