import torch
from sentence_transformers import SentenceTransformer, util
import pinecone
import os
import random


EMBEDDING_MODEL_NAME = 'OceanOmics/eDNABERT-S_16S'
EMBEDDING_DIM = 384 '


PINECONE_API_KEY = os.environ.get("dvndfjd3423ffd0", "4dercgh4se6frw400")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "transformer-similarity-search"


if "YOUR_PINECONE_API_KEY" in PINECONE_API_KEY:
    print("WARNING: Please set your Pinecone API key in environment variable PINECONE_API_KEY or replace 'YOUR_PINECONE_API_KEY' in the script.")
   


class TransformerEmbedder:
    def __init__(self, model_name):
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

    def embed_texts(self, texts):
        """
        Generates embeddings for a list of texts using the loaded Transformer model.
        """
       
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().tolist() 

# --- 2. Generating and Storing Embeddings---
def generate_sample_data(num_sequences=100):
    """
    Generates dummy DNA-like sequences and descriptions for demonstration.
    In a real scenario, this would be your actual SILVA eDNA data.
    """
    corpus = []
    bases = ['A', 'T', 'G', 'C']
    for i in range(num_sequences):
        seq_len = random.randint(100, 300) # Random sequence length
        dna_sequence = "".join(random.choice(bases) for _ in range(seq_len))
        description = f"Marine eDNA sequence {i+1} from sample X, habitat Y. Dominant base: {random.choice(bases)}"
        
        corpus.append({
            "id": f"seq_{i+1}",
            "text_to_embed": description,
            "original_dna_sequence": dna_sequence,
            "metadata": {
                "sample_id": f"sample_{random.randint(1, 10)}",
                "habitat_type": random.choice(["pelagic", "benthic", "hydrothermal_vent"]),
                "length": seq_len
            }
        })
    return corpus

# --- 3. Vector Database (Pinecone Example) ---
def initialize_pinecone_index(api_key, environment, index_name, dimension):
    """Initializes Pinecone and creates an index if it doesn't exist."""
    try:
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            print(f"Creating Pinecone index: {index_name} with dimension {dimension}...")
            # 'cosine' is often a good choice for transformer embeddings
            pinecone.create_index(index_name, dimension=dimension, metric='cosine', pod_type="s1")
            print("Index created. Waiting for index to be ready...")
            while not pinecone.describe_index(index_name).status['ready']:
                import time
                time.sleep(1)
            print("Index is ready.")
        else:
            print(f"Index '{index_name}' already exists.")
        return pinecone.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None

def upload_embeddings_to_pinecone(pinecone_index, embedded_data, batch_size=32):
    """
    Uploads embeddings and metadata to Pinecone in batches.
    `embedded_data` is a list of dicts: `{"id": id, "values": embedding, "metadata": {...}}`
    """
    if not pinecone_index:
        print("Pinecone index not initialized. Skipping upload.")
        return

    print(f"Uploading {len(embedded_data)} records to Pinecone in batches...")
    for i in range(0, len(embedded_data), batch_size):
        batch = embedded_data[i:i + batch_size]
        try:
            # Ensure 'id', 'values', and 'metadata' are correctly formatted
            pinecone_index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1}/{len(embedded_data) // batch_size + 1} to Pinecone.")
        except Exception as e:
            print(f"Error upserting batch starting at {i}: {e}")

# --- 4. Performing Similarity Search ---
def perform_similarity_search(pinecone_index, embedder, query_text, top_k=5, filter_metadata=None):
    """
    Embeds a query text and searches for similar vectors in the Pinecone index.
    `filter_metadata`: Optional dictionary for metadata filtering (e.g., {"habitat_type": "benthic"})
    """
    if not pinecone_index:
        print("Pinecone index not initialized. Cannot perform search.")
        return None

    print(f"\nQuery: '{query_text}'")
    query_embedding = embedder.embed_texts([query_text])[0] # Embed the query text

    print(f"Searching Pinecone for top {top_k} similar results...")
    try:
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_metadata
        )
        return query_results
    except Exception as e:
        print(f"Error during Pinecone query: {e}")
        return None

# --- Main Execution ---
def main():
    # 1. Initialize Transformer Embedder
    embedder = TransformerEmbedder(EMBEDDING_MODEL_NAME)

    # 2. Generate Sample Data and Embed
    sample_corpus = generate_sample_data(num_sequences=200)
    texts_to_embed = [item["text_to_embed"] for item in sample_corpus]
    embeddings = embedder.embed_texts(texts_to_embed)

    # Prepare data for Pinecone upload
    vectors_to_upload = []
    for i, item in enumerate(sample_corpus):
        vectors_to_upload.append({
            "id": item["id"],
            "values": embeddings[i],
            "metadata": item["metadata"]
        })

    # 3. Initialize and Upload to Pinecone
    pinecone_index = initialize_pinecone_index(
        PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_DIM
    )
    upload_embeddings_to_pinecone(pinecone_index, vectors_to_upload)

   
  
    query_text_1 = "sequences from a deep sea hydrothermal vent sample"
    results_1 = perform_similarity_search(pinecone_index, embedder, query_text_1, top_k=3)
    if results_1:
        print("\n--- Results for Query 1 ---")
        for match in results_1.matches:
            original_item = next(item for item in sample_corpus if item["id"] == match.id)
            print(f"  Match ID: {match.id}, Score: {match.score:.4f}")
            print(f"    Description: {original_item['text_to_embed']}")
            print(f"    DNA Sequence (first 50): {original_item['original_dna_sequence'][:50]}...")
            print(f"    Metadata: {match.metadata}")

    # Example 2: Similarity search with metadata filtering
    query_text_2 = "sequences related to algae or plant-like organisms"
    filter_by_habitat = {"habitat_type": "pelagic"} # Filter for only pelagic samples
    results_2 = perform_similarity_search(pinecone_index, embedder, query_text_2, top_k=2, filter_metadata=filter_by_habitat)
    if results_2:
        print("\n--- Results for Query 2 (filtered by habitat_type=pelagic) ---")
        for match in results_2.matches:
            original_item = next(item for item in sample_corpus if item["id"] == match.id)
            print(f"  Match ID: {match.id}, Score: {match.score:.4f}")
            print(f"    Description: {original_item['text_to_embed']}")
            print(f"    DNA Sequence (first 50): {original_item['original_dna_sequence'][:50]}...")
            print(f"    Metadata: {match.metadata}")

    
     print(f"\nDeleting Pinecone index: {PINECONE_INDEX_NAME}...")
     pinecone.delete_index(PINECONE_INDEX_NAME)

if __name__ == "__main__":
    main()
