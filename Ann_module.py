import numpy as np
from typing import Dict, List, Any
from vector_database_module import VectorDatabase

class SimulatedGNN:
    """
    A highly simplified simulation of a Graph Neural Network (GNN)
    for estimating characteristics and finding close relatives.
    In a real scenario, this would be a complex model trained on
    your organism relationship graph.
    """
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        

    def _get_avg_characteristics(self, organism_ids: List[str]) -> List[str]:
        """Helper to average/combine characteristics from a list of organisms."""
        all_characteristics = set()
        for org_id in organism_ids:
            data = self.vector_db.get_organism_data(org_id)
            if data and 'characteristics' in data['metadata']:
                all_characteristics.update(data['metadata']['characteristics'])
        return sorted(list(all_characteristics))

    def estimate_characteristics_and_relatives(self, query_embedding: np.ndarray, top_k_similar: int = 5) -> Dict[str, Any]:
        """
        Simulates GNN output. Finds similar organisms, infers characteristics
        based on them, and suggests close relatives.
        """
        print("\n--- GNN Simulation: Estimating Characteristics and Relatives ---")
        similar_organisms = self.vector_db.find_similar_organisms(query_embedding, top_k=top_k_similar)
        
        if not similar_organisms:
            print("No similar organisms found for GNN estimation.")
            return {
                "estimated_characteristics": ["unknown_trait"],
                "suggested_relatives": []
            }

        top_similar_ids = [s['organism_id'] for s in similar_organisms if s['similarity'] > 0.7] 
        
        estimated_characteristics = self._get_avg_characteristics(top_similar_ids)

      
        suggested_relatives = []
        for i, sim_org in enumerate(similar_organisms):
            if i >= 2: break # Only consider top 2 for relatives
            data = self.vector_db.get_organism_data(sim_org['organism_id'])
            if data and data['relationships']:
                for rel_id in data['relationships']:
                    rel_data = self.vector_db.get_organism_data(rel_id)
                    if rel_data:
                        suggested_relatives.append({
                            "id": rel_id,
                            "name": rel_data['metadata'].get('name', 'Unnamed'),
                            "taxonomy": rel_data['metadata'].get('taxonomy', 'Unknown')
                        })
        
        print(f"Top similar organisms: {[s['name'] for s in similar_organisms]}")
        print(f"Estimated Characteristics: {estimated_characteristics}")
        print(f"Suggested Relatives: {[r['name'] for r in suggested_relatives]}")

        return {
            "estimated_characteristics": estimated_characteristics,
            "suggested_relatives": suggested_relatives,
            "top_similar_organisms_details": similar_organisms # for further processing
        }
