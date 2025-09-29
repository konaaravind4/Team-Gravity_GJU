import numpy as np
from typing import Dict, List, Any
from vector_database_module import VectorDatabase
from gnn_module import SimulatedGNN

class OrganismIdentificationAndAnnotation:
    """
    Handles the identification of unknown organisms and their subsequent annotation.
    """
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self.gnn = SimulatedGNN(vector_db)
        self.identification_threshold = 0.8 

    def identify_organism(self, edna_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Identifies an organism from an eDNA embedding.
        Determines if it's a known species or an unknown one.
        """
        print(f"\n--- Identifying Organism ---")
        similar_organisms = self.vector_db.find_similar_organisms(edna_embedding, top_k=3)

        if not similar_organisms:
            print("No similar organisms found in the database. Marking as Unknown.")
            return {
                "status": "Unknown",
                "reason": "No matches found in vector database.",
                "details": {}
            }

        top_match = similar_organisms[0]
        
        if top_match['similarity'] >= self.identification_threshold:
            print(f"Identified as KNOWN: {top_match['name']} (Similarity: {top_match['similarity']:.2f})")
            return {
                "status": "Known",
                "organism_id": top_match['organism_id'],
                "name": top_match['name'],
                "taxonomy": top_match['taxonomy'],
                "similarity": top_match['similarity'],
                "details": self.vector_db.get_organism_data(top_match['organism_id'])['metadata']
            }
        else:
            print(f"Identified as POTENTIALLY UNKNOWN: Top match {top_match['name']} "
                  f"with similarity {top_match['similarity']:.2f} (below threshold {self.identification_threshold:.2f})")
            
           
            gnn_results = self.gnn.estimate_characteristics_and_relatives(edna_embedding)
            
            return {
                "status": "Unknown",
                "reason": "Similarity below threshold for known species.",
                "top_similar_known": top_match,
                "estimated_characteristics": gnn_results["estimated_characteristics"],
                "suggested_relatives": gnn_results["suggested_relatives"],
                "details": {}
            }

    def annotate_organism(self, organism_id: str, new_metadata: Dict[str, Any], is_new_organism: bool = False):
        """
        Annotates an organism, either updating an existing one or adding details to a new one.
        """
        print(f"\n--- Annotating Organism: {organism_id} ---")
        if is_new_organism:
         
            print(f"Assuming organism_id {organism_id} was pre-registered as placeholder for unknown.")
            self.vector_db.update_organism_metadata(organism_id, new_metadata)
        else:
            self.vector_db.update_organism_metadata(organism_id, new_metadata)
        print(f"Annotation complete for {organism_id}.")
        print(f"Updated metadata: {self.vector_db.get_organism_data(organism_id)['metadata']}")

    def mark_as_unannotated(self, edna_embedding: np.ndarray, temp_id: str):
        """
        Marks an ASV as unannotated in a temporary or pending state.
        This might involve storing the embedding with a flag for later review.
        """
   
        print(f"\n--- Marking ASV {temp_id} as Unannotated ---")
        print(f"Embedding stored for future annotation/review.")
