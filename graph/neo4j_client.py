"""
graph/neo4j_client.py
Neo4j Knowledge Graph client with NetworkX in-memory fallback.

Neo4j stores entities as nodes and relations as edges.
When Neo4j is unavailable, NetworkX provides the same API.
"""

from typing import List, Dict, Optional, Tuple
from utils.logger import logger
from config.settings import settings


class NetworkXGraph:
    """
    In-memory graph using NetworkX.
    Used as fallback when Neo4j is unavailable.
    Same interface as Neo4jGraph.
    """

    def __init__(self):
        import networkx as nx
        self.G = nx.MultiDiGraph()
        logger.info("NetworkX fallback graph initialized.")

    def add_entity(self, name: str, entity_type: str, properties: Dict = None) -> bool:
        props = properties or {}
        self.G.add_node(
            name.lower(),
            display_name=name,
            entity_type=entity_type,
            **props
        )
        return True

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: str = "Entity",
        object_type: str = "Entity",
        properties: Dict = None,
    ) -> bool:
        # Ensure nodes exist
        self.add_entity(subject, subject_type)
        self.add_entity(obj, object_type)

        self.G.add_edge(
            subject.lower(),
            obj.lower(),
            predicate=predicate,
            **(properties or {})
        )
        return True

    def query_relations(self, entity: str) -> List[Dict]:
        """Get all relations for an entity."""
        entity_lower = entity.lower()
        results = []

        if entity_lower not in self.G:
            return results

        # Outgoing
        for neighbor in self.G.successors(entity_lower):
            for _, edge_data in self.G[entity_lower][neighbor].items():
                results.append({
                    "subject": self.G.nodes[entity_lower].get("display_name", entity),
                    "predicate": edge_data.get("predicate", "RELATED_TO"),
                    "object": self.G.nodes[neighbor].get("display_name", neighbor),
                })

        # Incoming
        for predecessor in self.G.predecessors(entity_lower):
            for _, edge_data in self.G[predecessor][entity_lower].items():
                results.append({
                    "subject": self.G.nodes[predecessor].get("display_name", predecessor),
                    "predicate": edge_data.get("predicate", "RELATED_TO"),
                    "object": self.G.nodes[entity_lower].get("display_name", entity),
                })

        return results

    def find_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two nodes."""
        import networkx as nx
        try:
            path = nx.shortest_path(self.G, source.lower(), target.lower())
            return [self.G.nodes[n].get("display_name", n) for n in path]
        except Exception:
            return []

    def get_all_triples(self) -> List[Dict]:
        """Return all triples in the graph."""
        triples = []
        for u, v, data in self.G.edges(data=True):
            triples.append({
                "subject": self.G.nodes[u].get("display_name", u),
                "predicate": data.get("predicate", "RELATED_TO"),
                "object": self.G.nodes[v].get("display_name", v),
            })
        return triples

    def get_stats(self) -> Dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "backend": "NetworkX",
        }

    def get_graph_data_for_viz(self) -> Tuple[List[Dict], List[Dict]]:
        """Return nodes and edges for PyVis visualization."""
        nodes = []
        for node_id, data in self.G.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("display_name", node_id),
                "type": data.get("entity_type", "Entity"),
            })
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "label": data.get("predicate", ""),
            })
        return nodes, edges

    def clear(self):
        import networkx as nx
        self.G = nx.MultiDiGraph()

    def close(self):
        pass  # No-op for NetworkX


class Neo4jGraph:
    """
    Neo4j-backed Knowledge Graph.
    Creates nodes for entities and directed edges for relations.
    """

    def __init__(self):
        self.driver = None
        self.database = settings.neo4j.database
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.username, settings.neo4j.password),
            )
            # Verify connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j connected: {settings.neo4j.uri}")
            self._create_constraints()
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            self.driver = None

    def _create_constraints(self):
        """Create uniqueness constraints for efficient node lookup."""
        if not self.driver:
            return
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
        except Exception as e:
            logger.debug(f"Constraint creation (may already exist): {e}")

    def add_entity(self, name: str, entity_type: str, properties: Dict = None) -> bool:
        if not self.driver:
            return False
        props = {"name": name.lower(), "display_name": name, **(properties or {})}
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    f"MERGE (e:{entity_type} {{name: $name}}) "
                    "SET e.display_name = $display_name",
                    name=name.lower(),
                    display_name=name,
                )
            return True
        except Exception as e:
            logger.error(f"add_entity failed: {e}")
            return False

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: str = "Entity",
        object_type: str = "Entity",
        properties: Dict = None,
    ) -> bool:
        if not self.driver:
            return False
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    f"""
                    MERGE (s:{subject_type} {{name: $s_name}})
                    SET s.display_name = $s_display
                    MERGE (o:{object_type} {{name: $o_name}})
                    SET o.display_name = $o_display
                    MERGE (s)-[r:{predicate}]->(o)
                    """,
                    s_name=subject.lower(),
                    s_display=subject,
                    o_name=obj.lower(),
                    o_display=obj,
                )
            return True
        except Exception as e:
            logger.error(f"add_relation failed: {e}")
            return False

    def run_cypher(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a raw Cypher query."""
        if not self.driver:
            return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **(params or {}))
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}\nQuery: {query}")
            return []

    def query_relations(self, entity: str) -> List[Dict]:
        """Get all relations connected to an entity."""
        cypher = """
        MATCH (s)-[r]->(o)
        WHERE toLower(s.name) = toLower($entity)
           OR toLower(o.name) = toLower($entity)
        RETURN s.display_name AS subject, type(r) AS predicate, o.display_name AS object
        LIMIT 20
        """
        return self.run_cypher(cypher, {"entity": entity})

    def find_path(self, source: str, target: str) -> List[Dict]:
        """Find shortest path between two entities."""
        cypher = """
        MATCH path = shortestPath(
            (s {name: toLower($source)})-[*..5]-(t {name: toLower($target)})
        )
        RETURN [n IN nodes(path) | n.display_name] AS path_nodes,
               [r IN relationships(path) | type(r)] AS relations
        LIMIT 1
        """
        return self.run_cypher(cypher, {"source": source, "target": target})

    def get_all_triples(self) -> List[Dict]:
        """Return all triples (for visualization, limited)."""
        cypher = """
        MATCH (s)-[r]->(o)
        RETURN s.display_name AS subject, type(r) AS predicate, o.display_name AS object
        LIMIT 100
        """
        return self.run_cypher(cypher)

    def get_stats(self) -> Dict:
        """Return graph statistics."""
        node_count = self.run_cypher("MATCH (n) RETURN count(n) AS count")
        edge_count = self.run_cypher("MATCH ()-[r]->() RETURN count(r) AS count")
        return {
            "nodes": node_count[0]["count"] if node_count else 0,
            "edges": edge_count[0]["count"] if edge_count else 0,
            "backend": "Neo4j",
        }

    def get_graph_data_for_viz(self) -> Tuple[List[Dict], List[Dict]]:
        """Return nodes and edges for visualization."""
        nodes_q = self.run_cypher(
            "MATCH (n) RETURN n.name AS id, n.display_name AS label, labels(n)[0] AS type LIMIT 100"
        )
        edges_q = self.run_cypher(
            "MATCH (s)-[r]->(o) RETURN s.name AS source, o.name AS target, type(r) AS label LIMIT 100"
        )
        return nodes_q, edges_q

    def get_schema(self) -> str:
        """Get graph schema for Cypher generation prompts."""
        try:
            labels = self.run_cypher("CALL db.labels() YIELD label RETURN label")
            rel_types = self.run_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            label_str = ", ".join(r["label"] for r in labels)
            rel_str = ", ".join(r["relationshipType"] for r in rel_types)
            return f"Node labels: {label_str}\nRelationship types: {rel_str}"
        except Exception:
            return ""

    def clear(self):
        """Delete all nodes and edges."""
        self.run_cypher("MATCH (n) DETACH DELETE n")

    def close(self):
        if self.driver:
            self.driver.close()


def get_graph_client():
    """
    Factory: return Neo4j or NetworkX based on settings.
    """
    if settings.neo4j.use_networkx_fallback:
        logger.info("Using NetworkX fallback graph.")
        return NetworkXGraph()

    client = Neo4jGraph()
    if client.driver is None:
        logger.warning("Neo4j unavailable. Falling back to NetworkX.")
        return NetworkXGraph()
    return client


# Singleton
graph_client = get_graph_client()

__all__ = ["graph_client", "Neo4jGraph", "NetworkXGraph", "get_graph_client"]
