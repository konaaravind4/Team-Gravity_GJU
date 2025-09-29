import numpy as np
from typing import Dict, List, Any
import uuid

class VectorDatabase:
    """
    A simulated vector database to store organism embeddings and metadata.
    In a real application, this would interact with Faiss, Pinecone, etc.
    """
    def __init__(self):
        self._embeddings: Dict[str, np.ndarray] = {} # organism_id -> embedding
        self._metadata: Dict[str, Dict[str, Any]] = {} # organism_id -> metadata
        self._relationships: Dict[str, List[str]] = {} # organism_id -> list of related_organism_ids

    def add_organism(self, embedding: np.ndarray, metadata: Dict[str, Any], relationships: List[str] = None) -> str:
        """Adds a new organism to the database."""
        organism_id = str(uuid.uuid4())
        self._embeddings[organism_id] = embedding
        self._metadata[organism_id] = metadata
        self._relationships[organism_id] = relationships if relationships is not None else []
        print(f"Added organism: {organism_id} - {metadata.get('name', 'Unnamed')}")
        return organism_id

    def get_organism_data(self, organism_id: str) -> Dict[str, Any]:
        """Retrieves an organism's metadata and embedding."""
        if organism_id not in self._metadata:
            return None
        return {
            "id": organism_id,
            "embedding": self._embeddings.get(organism_id),
            "metadata": self._metadata[organism_id],
            "relationships": self._relationships.get(organism_id, [])
        }

    def update_organism_metadata(self, organism_id: str, new_metadata: Dict[str, Any]):
        """Updates metadata for an existing organism."""
        if organism_id in self._metadata:
            self._metadata[organism_id].update(new_metadata)
            print(f"Updated metadata for {organism_id}")
        else:
            print(f"Organism {organism_id} not found for update.")

    def find_similar_organisms(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Finds the most similar organisms based on embedding similarity (e.g., cosine similarity).
        In a real system, this would be highly optimized by the vector DB.
        """
        similarities = []
        for org_id, db_embedding in self._embeddings.items():
            if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(db_embedding) == 0:
                similarity = 0 
            else:
                similarity = np.dot(query_embedding, db_embedding) / \
                             (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            similarities.append((similarity, org_id))

        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for sim, org_id in similarities[:top_k]:
            metadata = self._metadata.get(org_id, {})
            results.append({
                "organism_id": org_id,
                "similarity": sim,
                "name": metadata.get('name', 'Unnamed'),
                "taxonomy": metadata.get('taxonomy', 'Unknown'),
                "characteristics": metadata.get('characteristics', [])
            })
        return results

    def get_all_embeddings_and_ids(self) -> List[tuple[str, np.ndarray]]:
        """Returns all organism IDs and their embeddings."""
        return [(org_id, emb) for org_id, emb in self._embeddings.items()]

    def get_all_relationships(self) -> Dict[str, List[str]]:
        """Returns all stored relationships."""
        return self._relationships
