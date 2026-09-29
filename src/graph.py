import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict
from google import genai
from src.extraction import Extraction

class KnowledgeGraph:
    def __init__(self, chunks, API_KEY, MODEL_NAME, SEED=42):
        self.client=genai.Client(api_key=API_KEY)
        self.API_KEY=API_KEY
        self.MODEL_NAME=MODEL_NAME
        self.graph=nx.MultiDiGraph()
        self.seed=SEED
        self.chunks=chunks
        self.node2cid=dict()
        self.cid2node=defaultdict(list)
        self.chunk_lookup={
            f"{chunk.metadata["source"]} : {chunk.metadata["id"]}" : chunk.page_content
            for chunk in chunks
        }

        self.extractor = Extraction(self.API_KEY, self.MODEL_NAME)
        
    def _add_object_to_graph(self, chunk):
        chunk_id = f"{chunk.metadata["source"]} : {chunk.metadata["id"]}"
        self.graph.add_node(chunk_id, type="Chunk")

        extraction = self.extractor.extract(chunk)
        entities = extraction.entities
        relationships = extraction.relationships
        
        for entity in entities:
            name = entity.entity_name.lower().strip()
            if not name:
                continue
            if not self.graph.has_node(name):
                self.graph.add_node(name, type=entity.entity_type, count=0)
            self.graph.nodes[name]["count"] = self.graph.nodes[name]["count"] + 1 
            self.graph.add_edge(chunk_id, name, key='CONTAINS', relation_type='CONTAINS', weight=1.0)

        entity_names = set(e.entity_name.lower().strip() for e in entities if e.entity_name.strip())
        for rel in relationships:
            source, target = rel.source.lower().strip(), rel.target.lower().strip()
            if source not in entity_names or target not in entity_names:
                continue
            key = rel.relationshipType.lower().strip()
            prev = self.graph.get_edge_data(source, target, key, default={})
            self.graph.add_edge(source, target, key=key, relation_type=key,
                               weight=prev.get("weight", 0.0) + rel.weight)

    def build_graph(self):
        for index, chunk in enumerate(self.chunks):
            self._add_object_to_graph(chunk)
        return self.graph

    def _detect_communities(self):
        undirected = nx.Graph()
        undirected.add_nodes_from(self.graph.nodes())

        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if undirected.has_edge(u, v):
                undirected[u][v]["weight"] += weight
            else: 
                undirected.add_edge(u, v, weight=1.0)

        communities = louvain_communities(undirected, weight='weight', seed=42)
        for cid, nodes in enumerate(communities):
            for node in nodes:
                self.node2cid[node] = cid
                self.cid2node[cid].append(node)

    def get_node2cid(self):
        return self.node2cid

    def generate_summary_communities(self):
        self._detect_communities()
        results = {}
        for cid, nodes in self.cid2node.items():
            entities_in_community = []
            chunks_in_community = set()

            for node in nodes:
                node_type = self.graph.nodes[node].get("type")
                if node_type == "Chunk":
                    chunks_in_community.add(node)
                else:
                    entities_in_community.append(node)

            for node in entities_in_community:
                chunks_in_community.update(n for n in self.graph.predecessors(node) if self.graph.nodes[n]["type"] == "Chunk")

            subgraph = self.graph.subgraph(entities_in_community)
            edges_info = []
            for source, target, data in subgraph.edges(data=True):
                rel_type = data.get("relationshipType", "RELATED_TO")
                edges_info.append(f"- {source} [{rel_type}] -> {target}")

            chunks_text = [
                f"[Chunk {cid}] : {self.chunk_lookup[cid]}"
                for cid in chunks_in_community
                if cid in self.chunk_lookup
            ]

            entities_str = (
                ", ".join(e for e in entities_in_community[:15] or None)
            )
            edges_str = "\n".join(edges_info[:10]) or "None"
            chunks_str = "\n\n".join(chunks_text[:5]) or "None"

            prompt = (
                "You are an expert data analyst specializing in Knowledge Graphs and GraphRAG systems.\n"
                "Your task is to analyze a cluster of entities and their corresponding context extracted from text data to generate a high-quality community summary.\n\n"
                f"Key entities in this cluster: {entities_str}\n\n"
                f"Key relationships between entities in this cluster:\n{edges_str}\n\n"
                f"Relevant transcript excerpt:\n{chunks_str}\n\n"
                "--- INSTRUCTIONS ---\n"
                "Based strictly on the provided data, perform the following tasks:\n"
                "Executive Summary: Write a 2-3 sentence paragraph summarizing the core theme, the main entities involved, "
                "and their relationships or significance within this context. Do not invent facts; be deeply grounded in the transcript text."
            )

            try: 
                response = self.client.models.generate_content(
                    model=self.MODEL_NAME, 
                    contents=prompt, 
                    config={
                        "temperature": 0.3,
                    }
                )
                summary_text = response.text.strip()
            except Exception as e: 
                raise RuntimeError("Không tạo được tóm tắt cộng đồng.") from e
                
            results[cid] = summary_text

        return results 
