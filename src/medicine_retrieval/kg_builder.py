"""Build a small medicine safety knowledge graph from processed CSV tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .schemas import ProjectTables, load_processed_tables


@dataclass
class SimpleMultiDiGraph:
    """Tiny fallback graph used when NetworkX is not installed locally."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = {**self.nodes.get(node_id, {}), **attrs}

    def add_edge(self, source: str, target: str, key: str | None = None, **attrs: Any) -> None:
        self.edges.append((source, target, key or attrs.get("relation", ""), attrs))

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return len(self.edges)

    def successors(self, node_id: str) -> list[str]:
        return [target for source, target, _, _ in self.edges if source == node_id]


def _new_graph():
    try:
        import networkx as nx

        return nx.MultiDiGraph()
    except ModuleNotFoundError:
        return SimpleMultiDiGraph()


def node_id(kind: str, raw_id: str) -> str:
    return f"{kind}:{raw_id}"


def add_relation(graph: Any, source: str, target: str, relation: str, **attrs: Any) -> None:
    graph.add_edge(source, target, key=relation, relation=relation, **attrs)


def build_kg(tables: ProjectTables | None = None):
    tables = tables or load_processed_tables()
    graph = _new_graph()

    for row in tables["drug_items"]:
        graph.add_node(node_id("drug_item", row["item_code"]), kind="drug_item", **row)

    for row in tables["ingredients"]:
        graph.add_node(node_id("ingredient", row["ingredient_id"]), kind="ingredient", **row)

    for row in tables["evidence_sources"]:
        graph.add_node(node_id("source", row["source_id"]), kind="source", **row)

    for row in tables["product_images"]:
        image_node = node_id("product_image", row["image_id"])
        drug_node = node_id("drug_item", row["item_code"])
        graph.add_node(image_node, kind="product_image", **row)
        add_relation(graph, image_node, drug_node, "RECOGNIZED_AS")

        ocr_node = node_id("ocr_text", row["image_id"])
        graph.add_node(
            ocr_node,
            kind="ocr_text",
            image_id=row["image_id"],
            text=row["ocr_text"],
            confidence=row["ocr_confidence"],
        )
        add_relation(graph, image_node, ocr_node, "HAS_OCR_TEXT")

    for row in tables["drug_item_ingredients"]:
        add_relation(
            graph,
            node_id("drug_item", row["item_code"]),
            node_id("ingredient", row["ingredient_id"]),
            "HAS_INGREDIENT",
            amount=row["amount"],
            unit=row["unit"],
        )

    for row in tables["dur_rules"]:
        rule_node = node_id("dur_rule", row["rule_id"])
        graph.add_node(rule_node, kind="dur_rule", **row)
        target_kind = "drug_item" if row["ref_type"] == "item" else "ingredient"
        add_relation(graph, node_id(target_kind, row["ref_id"]), rule_node, "HAS_DUR_RULE")
        add_relation(graph, rule_node, node_id("source", row["source_id"]), "FROM_SOURCE")

    for row in tables["food_rules"]:
        rule_node = node_id("food_rule", row["food_rule_id"])
        graph.add_node(rule_node, kind="food_rule", **row)
        add_relation(graph, node_id("drug_item", row["item_code"]), rule_node, "HAS_FOOD_RULE")
        if row["ingredient_id"]:
            add_relation(graph, node_id("ingredient", row["ingredient_id"]), rule_node, "HAS_FOOD_RULE")
        add_relation(graph, rule_node, node_id("source", row["source_id"]), "FROM_SOURCE")

    return graph


def graph_summary(graph: Any) -> dict[str, Any]:
    kinds: dict[str, int] = defaultdict(int)
    node_values = graph.nodes.values() if isinstance(graph.nodes, dict) else graph.nodes.values()
    for attrs in node_values:
        kinds[attrs.get("kind", "unknown")] += 1

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_kinds": dict(sorted(kinds.items())),
    }


if __name__ == "__main__":
    kg = build_kg()
    print(graph_summary(kg))
