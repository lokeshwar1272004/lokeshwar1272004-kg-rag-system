"""
graph/visualizer.py
Knowledge Graph visualization using PyVis.
Generates interactive HTML graphs for Streamlit embedding.
"""

import tempfile
import os
from typing import List, Dict, Optional, Tuple

from utils.logger import logger


# Color palette for different entity types
ENTITY_COLORS = {
    "Person":       "#4ECDC4",
    "Organization": "#FF6B6B",
    "Location":     "#45B7D1",
    "Product":      "#96CEB4",
    "Event":        "#FFEAA7",
    "Date":         "#DDA0DD",
    "Group":        "#98D8C8",
    "WorkOfArt":    "#F7DC6F",
    "Law":          "#AED6F1",
    "Entity":       "#BDC3C7",
    "Facility":     "#F8C471",
    "Money":        "#82E0AA",
}

EDGE_COLORS = {
    "WORKS_FOR":    "#FF6B6B",
    "LOCATED_IN":   "#45B7D1",
    "LEADS":        "#4ECDC4",
    "OWNS":         "#96CEB4",
    "PARTNERS_WITH":"#FFEAA7",
    "CREATED":      "#DDA0DD",
    "STUDIED_AT":   "#98D8C8",
    "RELATED_TO":   "#BDC3C7",
}


class GraphVisualizer:
    """
    Creates interactive PyVis HTML visualizations of Knowledge Graphs.
    """

    def __init__(self):
        self._pyvis_available = self._check_pyvis()

    def _check_pyvis(self) -> bool:
        try:
            import pyvis
            return True
        except ImportError:
            logger.warning("pyvis not installed. Graph visualization unavailable.")
            return False

    def create_graph_html(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        height: str = "500px",
        width: str = "100%",
        title: str = "Knowledge Graph",
    ) -> Optional[str]:
        """
        Generate interactive HTML visualization.
        
        Args:
            nodes: [{"id": ..., "label": ..., "type": ...}]
            edges: [{"source": ..., "target": ..., "label": ...}]
            height: Canvas height
            width: Canvas width
            title: Graph title
            
        Returns:
            HTML string, or None if pyvis unavailable
        """
        if not self._pyvis_available:
            return None

        if not nodes:
            return None

        try:
            from pyvis.network import Network

            net = Network(
                height=height,
                width=width,
                bgcolor="#0E1117",
                font_color="#FAFAFA",
                directed=True,
            )

            # Configure physics for better layout
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.3,
                  "springLength": 95,
                  "springConstant": 0.04
                },
                "stabilization": {"iterations": 150}
              },
              "interaction": {
                "hover": true,
                "navigationButtons": true,
                "tooltipDelay": 100
              },
              "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "smooth": {"type": "curvedCW", "roundness": 0.2},
                "font": {"size": 11, "color": "#CCCCCC"}
              },
              "nodes": {
                "font": {"size": 13, "face": "monospace"}
              }
            }
            """)

            # Add nodes
            added_nodes = set()
            for node in nodes:
                node_id = str(node.get("id", node.get("label", "")))
                label = str(node.get("label", node_id))
                entity_type = str(node.get("type", "Entity"))
                color = ENTITY_COLORS.get(entity_type, "#BDC3C7")

                if node_id in added_nodes:
                    continue
                added_nodes.add(node_id)

                net.add_node(
                    node_id,
                    label=label[:25],  # Truncate long labels
                    color=color,
                    title=f"Type: {entity_type}\nID: {node_id}",
                    size=25,
                    borderWidth=2,
                    borderWidthSelected=4,
                )

            # Add edges
            for edge in edges:
                source = str(edge.get("source", ""))
                target = str(edge.get("target", ""))
                rel_label = str(edge.get("label", ""))

                if source not in added_nodes or target not in added_nodes:
                    continue

                edge_color = EDGE_COLORS.get(rel_label, "#7F8C8D")

                net.add_edge(
                    source,
                    target,
                    label=rel_label,
                    color=edge_color,
                    title=rel_label,
                    width=2,
                )

            # Save to temp file and return HTML
            tmp_path = tempfile.mktemp(suffix=".html")
            net.save_graph(tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                html_content = f.read()
            os.unlink(tmp_path)

            return html_content

        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return None

    def create_from_triples(
        self,
        triples: List[Dict],
        max_nodes: int = 50,
        **kwargs,
    ) -> Optional[str]:
        """
        Build visualization directly from triple dicts.
        
        triples: [{"subject": ..., "predicate": ..., "object": ...}]
        """
        nodes_map = {}
        edges = []

        for triple in triples[:max_nodes * 2]:
            subj = triple.get("subject", "")
            pred = triple.get("predicate", triple.get("relation", ""))
            obj = triple.get("object", "")

            if not (subj and pred and obj):
                continue

            subj_id = subj.lower().replace(" ", "_")
            obj_id = obj.lower().replace(" ", "_")

            if subj_id not in nodes_map:
                nodes_map[subj_id] = {
                    "id": subj_id,
                    "label": subj,
                    "type": triple.get("subject_type", "Entity"),
                }

            if obj_id not in nodes_map:
                nodes_map[obj_id] = {
                    "id": obj_id,
                    "label": obj,
                    "type": triple.get("object_type", "Entity"),
                }

            edges.append({
                "source": subj_id,
                "target": obj_id,
                "label": pred,
            })

            if len(nodes_map) >= max_nodes:
                break

        return self.create_graph_html(
            nodes=list(nodes_map.values()),
            edges=edges,
            **kwargs,
        )

    def create_ego_graph(
        self,
        center_entity: str,
        relations: List[Dict],
        **kwargs,
    ) -> Optional[str]:
        """
        Create a graph centered on one entity.
        Useful for 'show me everything about X' queries.
        """
        nodes_map = {}
        edges = []

        center_id = center_entity.lower().replace(" ", "_")
        nodes_map[center_id] = {
            "id": center_id,
            "label": center_entity,
            "type": "Focus",
        }

        for rel in relations:
            subj = rel.get("subject", "")
            pred = rel.get("predicate", rel.get("relation", ""))
            obj = rel.get("object", "")

            subj_id = subj.lower().replace(" ", "_")
            obj_id = obj.lower().replace(" ", "_")

            if subj_id not in nodes_map:
                nodes_map[subj_id] = {"id": subj_id, "label": subj, "type": "Entity"}
            if obj_id not in nodes_map:
                nodes_map[obj_id] = {"id": obj_id, "label": obj, "type": "Entity"}

            edges.append({"source": subj_id, "target": obj_id, "label": pred})

        return self.create_graph_html(
            nodes=list(nodes_map.values()),
            edges=edges,
            **kwargs,
        )


# Singleton
graph_visualizer = GraphVisualizer()

__all__ = ["graph_visualizer", "GraphVisualizer"]
