import numpy as np
from vector_database_module import VectorDatabase
from gnn_module import SimulatedGNN
from identification_annotation_module import OrganismIdentificationAndAnnotation

def generate_mock_embedding(size=128):
    """Generates a random mock embedding for demonstration."""
    return np.random.rand(size).astype(np.float32)

def main():
    # 1. Initialize Vector Database
    vector_db = VectorDatabase()

    # 2. Populate with some mock known organisms
    print("--- Populating Mock Vector Database ---")
    org1_embedding = generate_mock_embedding()
    org1_metadata = {
        "name": "Escherichia coli",
        "taxonomy": "Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; Escherichia",
        "characteristics": ["rod-shaped", "gram-negative", "facultative anaerobe", "commensal"],
        "habitat": ["intestines"]
    }
    org1_id = vector_db.add_organism(org1_embedding, org1_metadata)

    org2_embedding = generate_mock_embedding() + np.random.rand(128) * 0.1 # Slightly different
    org2_metadata = {
        "name": "Salmonella enterica",
        "taxonomy": "Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; Salmonella",
        "characteristics": ["rod-shaped", "gram-negative", "facultative anaerobe", "pathogen"],
        "habitat": ["intestines", "environment"]
    }
    org2_id = vector_db.add_organism(org2_embedding, org2_metadata, relationships=[org1_id])

    org3_embedding = generate_mock_embedding() * 0.5 # Very different
    org3_metadata = {
        "name": "Saccharomyces cerevisiae",
        "taxonomy": "Eukaryota; Fungi; Ascomycota; Saccharomycetes; Saccharomycetales; Saccharomycetaceae; Saccharomyces",
        "characteristics": ["yeast", "eukaryotic", "fermenting"],
        "habitat": ["fermented foods", "soil"]
    }
    org3_id = vector_db.add_organism(org3_embedding, org3_metadata)
    
    # 3. Initialize Identification and Annotation Module
    identifier = OrganismIdentificationAndAnnotation(vector_db)

    print("\n" + "="*50)
    print("Scenario 1: Identifying a KNOWN organism")
    print("="*50)
    # Simulate an eDNA embedding very similar to E. coli
    known_edna_embedding = org1_embedding + np.random.rand(128) * 0.01 
    identification_result = identifier.identify_organism(known_edna_embedding)
    print("\nIdentification Result:")
    print(identification_result)

    if identification_result['status'] == "Known":
        # Example of updating annotation for a known organism (e.g., adding a new characteristic)
        current_metadata = vector_db.get_organism_data(identification_result['organism_id'])['metadata']
        new_characteristics = current_metadata['characteristics'] + ["motile"]
        identifier.annotate_organism(
            identification_result['organism_id'],
            {"characteristics": new_characteristics}
        )
        print("Updated E.coli characteristics: ", vector_db.get_organism_data(identification_result['organism_id'])['metadata']['characteristics'])

    print("\n" + "="*50)
    print("Scenario 2: Identifying a POTENTIALLY UNKNOWN organism")
    print("="*50)
    # Simulate an eDNA embedding somewhat similar to Salmonella but not identical to any known
    unknown_edna_embedding = (org2_embedding + generate_mock_embedding()) / 2 + np.random.rand(128) * 0.05
    identification_result_unknown = identifier.identify_organism(unknown_edna_embedding)
    print("\nIdentification Result (Unknown):")
    print(identification_result_unknown)

    if identification_result_unknown['status'] == "Unknown":
        print("\n--- User Interaction for Unknown Organism ---")
      
        proposed_name = "New Enterobacter Species X"
        proposed_taxonomy = "Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; UnknownGenus; NewSpeciesX"
        
       
        new_org_placeholder_id = vector_db.add_organism(
            unknown_edna_embedding, 
            {"name": proposed_name + " (Placeholder)", "status": "pending_annotation"}
        )

        user_provided_metadata = {
            "name": proposed_name,
            "taxonomy": proposed_taxonomy,
            "characteristics": identification_result_unknown['estimated_characteristics'] + ["user_observed_trait"],
            "suggested_relatives_from_gnn": identification_result_unknown['suggested_relatives'],
            "status": "annotated"
        }
        identifier.annotate_organism(
            new_org_placeholder_id,
            user_provided_metadata,
            is_new_organism=True 
        )
        print("Finalized new organism details: ", vector_db.get_organism_data(new_org_placeholder_id)['metadata'])

    print("\n" + "="*50)
    print("Scenario 3: Marking as Unannotated (e.g., very low confidence or noise)")
    print("="*50)
    noisy_edna_embedding = generate_mock_embedding() * 0.05 
    

    
    temp_asv_id = "ASV_001_NOISE"
    identifier.mark_as_unannotated(noisy_edna_embedding, temp_asv_id)


if __name__ == "__main__":
    main()
