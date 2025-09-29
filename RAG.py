import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

# --- 1. Simulate eDNABERT-S_16S Embedding ---
# In a real scenario, you would load your eDNABERT-S_16S model
# and use its 'encode' or 'embed' function here.
# For this example, we'll just generate random embeddings for sequences.
# A real eDNABERT embedding would be much more sophisticated and biologically meaningful.

EMBEDDING_DIM = 768 # eDNABERT-S_16S often outputs 768-dim embeddings

def get_ednabert_embedding(sequence: str) -> np.ndarray:
    """
    Simulates getting an embedding from the eDNABERT-S_16S model.
    In a real system, this would involve:
    1. Loading the eDNABERT-S_16S tokenizer.
    2. Tokenizing the sequence.
    3. Passing tokens through the eDNABERT-S_16S model.
    4. Extracting the desired embedding (e.g., CLS token output or mean pooling).

    For this demo, we'll return a random normalized vector.
    """
    # print(f"Simulating embedding for sequence: {sequence[:20]}...")
    embedding = np.random.rand(EMBEDDING_DIM)
    return embedding / np.linalg.norm(embedding) # Normalize for cosine similarity

# --- 2. Simulate Knowledge Base (Retrieval Corpus) ---
# This is our database of known species 16S sequences and their metadata.

known_species_data = [
    {"name": "E. coli", "sequence": "AGCTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCT", "family": "Enterobacteriaceae"},
    {"name": "Bacillus subtilis", "sequence": "CCGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCT", "family": "Bacillaceae"},
    {"name": "Vibrio parahaemolyticus", "sequence": "GCTAGCTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA", "family": "Vibrionaceae"},
    {"name": "Prorocentrum micans", "sequence": "TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC", "family": "Prorocentraceae"},
    {"name": "Pseudo-nitzschia multiseries", "sequence": "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT", "family": "Bacillariophyceae"},
    {"name": "Saccharomyces cerevisiae", "sequence": "GATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC", "family": "Saccharomycetaceae"},
    {"name": "Streptococcus pneumoniae", "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG", "family": "Streptococcaceae"},
    {"name": "Anabaena flos-aquae", "sequence": "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA", "family": "Nostocaceae"},
    # Add more species data here...
]

# Pre-compute embeddings for the knowledge base
# In a real system, these would be stored in a vector database
knowledge_base_embeddings = []
knowledge_base_metadata = [] # To store original data associated with embeddings

print("Pre-computing embeddings for known species...")
for i, species in enumerate(known_species_data):
    if i % 100 == 0 and i > 0:
        print(f"  Processed {i} species...")
    embedding = get_ednabert_embedding(species["sequence"])
    knowledge_base_embeddings.append(embedding)
    knowledge_base_metadata.append(species) # Store the full dict

knowledge_base_embeddings_array = np.array(knowledge_base_embeddings)
print(f"Knowledge base loaded with {len(knowledge_base_embeddings)} species embeddings.")

# --- RAG System Configuration ---
SIMILARITY_THRESHOLD_KNOWN_SPECIES = 0.95 # Above this, consider it a known species
SIMILARITY_THRESHOLD_KNOWN_FAMILY = 0.85 # Above this, consider it a new species in known family
TOP_K_RETRIEVAL = 5 # How many top similar species to retrieve

# --- RAG System Core Logic ---

def identify_species_rag(new_edna_sequence: str):
    """
    Identifies a species based on a new eDNA sequence using the RAG system.
    """
    print(f"\n--- Identifying New Sequence ---")
    print(f"New sequence (first 20 chars): {new_edna_sequence[:20]}...")

    # 1. Embedding the new sequence
    new_seq_embedding = get_ednabert_embedding(new_edna_sequence)
    new_seq_embedding = new_seq_embedding.reshape(1, -1) # Reshape for cosine_similarity

    # 2. Retrieval: Find similar species in the knowledge base
    if not knowledge_base_embeddings_array.shape[0]:
        print("Error: Knowledge base is empty. Cannot perform retrieval.")
        return

    similarities = cosine_similarity(new_seq_embedding, knowledge_base_embeddings_array)[0]
    
    # Get top K indices and their scores
    top_k_indices = np.argsort(similarities)[::-1][:TOP_K_RETRIEVAL]
    
    retrieved_results = []
    for idx in top_k_indices:
        retrieved_results.append({
            "species_data": knowledge_base_metadata[idx],
            "similarity_score": similarities[idx]
        })

    print("\n--- Retrieval Results (Top K) ---")
    for res in retrieved_results:
        print(f"  Species: {res['species_data']['name']}, Family: {res['species_data']['family']}, Similarity: {res['similarity_score']:.4f}")

    # 3. Generative Component (Decision Logic)
    # This interprets the retrieval results and forms a hypothesis.
    best_match = retrieved_results[0]
    
    if best_match["similarity_score"] >= SIMILARITY_THRESHOLD_KNOWN_SPECIES:
        print("\n--- RAG System Output: Known Species ---")
        print(f"Result: Identified as '{best_match['species_data']['name']}' (Family: {best_match['species_data']['family']})")
        print(f"Confidence: Very High (Similarity: {best_match['similarity_score']:.4f})")
        return {
            "status": "known_species",
            "identified_species": best_match['species_data']['name'],
            "confidence": best_match['similarity_score'],
            "details": best_match['species_data']
        }
    elif best_match["similarity_score"] >= SIMILARITY_THRESHOLD_KNOWN_FAMILY:
        print("\n--- RAG System Output: Candidate New Species (Known Family) ---")
        print(f"Result: Candidate for a NEW species, most closely related to '{best_match['species_data']['name']}'")
        print(f"  (within the family '{best_match['species_data']['family']}').")
        print(f"Confidence: Moderate (Highest Similarity: {best_match['similarity_score']:.4f})")
        print("Recommendation: Further phylogenetic analysis and morphological studies recommended for validation.")
        return {
            "status": "candidate_new_species_known_family",
            "closest_known_species": best_match['species_data']['name'],
            "closest_family": best_match['species_data']['family'],
            "confidence": best_match['similarity_score']
        }
    else:
        print("\n--- RAG System Output: Candidate Completely New Species ---")
        print(f"Result: Strong candidate for a COMPLETELY NEW species!")
        print(f"  No sufficiently similar known species found in the database.")
        print(f"Confidence: Low-Moderate (Highest Similarity: {best_match['similarity_score']:.4f} with {best_match['species_data']['name']})")
        print("Recommendation: Urgent follow-up, potentially full genome sequencing, culturing, and detailed microscopic examination required.")
        return {
            "status": "candidate_completely_new_species",
            "closest_known_species": best_match['species_data']['name'], # Still provide closest for context
            "confidence": best_match['similarity_score']
        }

# --- 4. Simulate Knowledge Base Update (Add New Species) ---
def add_new_species_to_kb(name: str, sequence: str, family: str):
    """
    Adds a newly validated species to the RAG system's knowledge base.
    """
    global knowledge_base_embeddings_array # We need to update the global array
    
    print(f"\n--- Adding New Species to Knowledge Base: {name} ---")
    new_species_data = {"name": name, "sequence": sequence, "family": family}
    
    # 1. Generate embedding for the new species
    new_species_embedding = get_ednabert_embedding(sequence)
    
    # 2. Add to in-memory lists (in a real system, update vector DB and metadata DB)
    knowledge_base_embeddings.append(new_species_embedding)
    knowledge_base_metadata.append(new_species_data)
    
    # Recreate the numpy array for efficient similarity search
    knowledge_base_embeddings_array = np.array(knowledge_base_embeddings)
    
    print(f"'{name}' successfully added. Knowledge base now has {len(knowledge_base_embeddings)} entries.")


# --- Example Usage ---

if __name__ == "__main__":
    # Example 1: Sequence very similar to a known species (E. coli)
    # (In a real scenario, embeddings would be consistent, so small changes lead to high sim)
    known_ecoli_sequence = "AGCTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCT"
    # Create a slightly modified version
    very_similar_ecoli = known_ecoli_sequence[:20] + "T" + known_ecoli_sequence[21:] 
    
    print("\n##### TEST CASE 1: Very Similar to Known E. coli #####")
    identify_species_rag(very_similar_ecoli)

    # Example 2: Sequence somewhat similar to a known family but likely a new species
    # (Simulate by making it somewhat distinct from any known, but with some overlap)
    # Here, we'll just use a random sequence for distinctness
    new_species_candidate_seq_1 = "GCTAGCTAGCTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTACGTAGCTAC"
    print("\n##### TEST CASE 2: Candidate for New Species (within a family) #####")
    identify_species_rag(new_species_candidate_seq_1)

    # Example 3: Sequence completely novel
    completely_new_species_seq = "TTTTTTTTTTTTTTAAAAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    print("\n##### TEST CASE 3: Candidate for Completely New Species #####")
    identify_species_rag(completely_new_species_seq)

    # Example 4: Adding a new species and then identifying it
    print("\n##### TEST CASE 4: Add New Species and Re-identify #####")
    new_validated_species_name = "Oceanicus novus"
    new_validated_species_sequence = "GGGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCA"
