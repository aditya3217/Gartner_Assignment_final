# ontology_pipeline.py
from py2neo import Graph, Node, Relationship
from triples_extraction import extract_triples

graph = Graph("bolt://localhost:7687", auth=("neo4j", "ontology123"))  # 🔁 Replace with your password

def create_graph(triples):
    for subj, pred, obj in triples:
        subj_node = Node("Entity", name=subj)
        obj_node = Node("Entity", name=obj)
        rel = Relationship(subj_node, pred.upper().replace(" ", "_"), obj_node)
        graph.merge(subj_node, "Entity", "name")
        graph.merge(obj_node, "Entity", "name")
        graph.merge(rel)

if __name__ == "__main__":
    with open("default.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    sample_text = full_text[:2000]  # Adjust size if needed
    triples = extract_triples(sample_text)

    print("\nExtracted Triples:")
    for t in triples:
        print(t)

    create_graph(triples)
    print("\n✅ Triples successfully added to Neo4j.")
