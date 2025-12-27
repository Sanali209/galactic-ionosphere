import unittest
from SLM.Graph.neo4j.neo4j_m import clear_db, add_node_bulk, add_edge_bulk, get_neo4j_driver


class TestNeo4j_m_test(unittest.TestCase):
    def test_add_node_bulk_test(self):
        with get_neo4j_driver().session() as session:
            clear_db(session)
            nodes = [{ "path": "path1"}, {"path": "path2"}]
            session.execute_write(add_node_bulk,"Image", nodes)
            result = session.run("MATCH (a:Image) RETURN a")
            res_list = list(result)
            self.assertEqual(2, len(res_list))
            print(res_list)
            edges = [{'val1': 'path1', 'val2': 'path2','params': {'metric': 'mobilenetv3', 'distance': 0.5, 'is_wrong': False, 'type': 'SIMILAR_EMBBEDING'}}]
            add_edge_bulk(session, 'Image','path','DUPLICATE', edges)
            result = session.run("MATCH (a:Image)-[r:DUPLICATE]->(b:Image) RETURN r")
            res = result.single()
            self.assertEqual("path1", res["path1"])



if __name__ == '__main__':
    unittest.main()
