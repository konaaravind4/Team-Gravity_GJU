import os
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch
import pinecone

# --- Configuration ---
FASTA_DIR = "fasta_files_directory" # Directory where  downloaded SILVA FASTA files are
EMBEDDING_MODEL_NAME = "OceanOmics/eDNABERT-S_16Sl" 
# suitable pre-trained DNA sequence embedding model 

# Pinecone configuration 
PINECONE_API_KEY = "i4f452riw4559grtgf/41i"
PINECONE_ENVIRONMENT = "PINECONE_ENVIRONMENT"
PINECONE_INDEX_NAME = "silva-marine-edna"
EMBEDDING_DIM = 768 

# --- 1. & 2. Mass Data Retrieval and Parsing ---
def parse_fasta_files(fasta_directory):
    """
    Parses all FASTA files in a given directory and yields sequence records.
    """
    for filename in os.listdir(fasta_directory):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            filepath = os.path.join(fasta_directory, filename)
            print(f"Parsing {filepath}...")
            try:
                for record in SeqIO.parse(filepath, "fasta"):
                    yield record
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")

# --- 3. Sequence Embedding (Conceptual) ---
class DNASequenceEmbedder:
    def __init__(self, model_name):
        
        print(f"Initializing embedding model: {model_name}")
        try:
            
            
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
            self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
            self.model.eval() 
        except Exception as e:
            print(f"Failed to load embedding model {model_name}. Please ensure you have a suitable DNA model. Error: {e}")
            self.tokenizer = None
            self.model = None

    def embed_sequence(self, dna_sequence):
        """
        Converts a DNA sequence into a numerical vector (embedding).
        This method needs to be implemented based on your chosen embedding approach.
        """
        if self.model is None or self.tokenizer is None:
            print("Embedding model not loaded. Returning dummy embedding.")
            return [0.0] * EMBEDDING_DIM # Return dummy vector

      
         from collections import Counter
         k = 3 # Example k-mer size
         k_mers = [dna_sequence[i:i+k] for i in range(len(dna_sequence) - k + 1)]
         k_mer_counts = Counter(k_mers)
        # # Convert counts to a dense vector (e.g., using a fixed vocabulary of k-mers)
        # # This can get very large, very fast.

        # --- Placeholder Transformer-based Embedding ---
        # For actual DNA models,     often need to tokenize differently (e.g., k-mer tokenization)
        # or using models designed to handle raw sequences.
        try:
            # Simple tokenization for illustrative purposes. Real DNA models are more complex.
            inputs = self.tokenizer(
                dna_sequence,
                return_tensors="pt",
                padding="max_length", 
                truncation=True,
                max_length=512 # Max sequence length for the model
            )
            with torch.no_grad():

                # For classification models, often the CLS token embedding is used.
                # For dedicated embedding models, there might be a direct 'encode' method.
                outputs = self.model(**inputs, output_hidden_states=True)
                # Take the embedding of the [CLS] token as the sequence embedding
                
                embedding = outputs.hidden_states[-1][:, 0, :].squeeze().tolist()
            return embedding
        except Exception as e:
            print(f"Error embedding sequence: {e}. Returning dummy embedding.")
            return [0.0] * EMBEDDING_DIM

# --- 4. Vector Database Storage ---
def initialize_pinecone_index(api_key, environment, index_name, dimension):
    """Initializes Pinecone and creates an index if it doesn't exist."""
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        print(f"Creating Pinecone index: {index_name} with dimension {dimension}")
        pinecone.create_index(index_name, dimension=dimension, metric='cosine') # or 'euclidean', 'dotproduct'
    return pinecone.Index(index_name)

def upload_embeddings_to_pinecone(pinecone_index, records, batch_size=100):
    """
    Uploads embeddings and metadata to Pinecone in batches.
    `records` should be a list of dictionaries like:
    `{"id": seq_id, "values": embedding_vector, "metadata": {...}}`
    """
    print(f"Uploading {len(records)} records to Pinecone in batches...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            pinecone_index.upsert(vectors=batch)
            print(f"Upserted batch {i // batch_size + 1} to Pinecone.")
        except Exception as e:
            print(f"Error upserting batch starting at {i}: {e}")


# --- Main Execution Flow ---
def main():
    if not os.path.exists(FASTA_DIR):
        print(f"Error: FASTA directory '{FASTA_DIR}' not found. Please create it and place your .fasta files inside.")
        
         os.makedirs(FASTA_DIR, exist_ok=True) 
        with open(os.path.join(FASTA_DIR, "dummy.fasta"), "w") as f:
            f.write(">Seq1 Description1\nATGCATGCATGC\n>Seq2 Description2\nGGGCCCATGC\n")
        print("Created dummy FASTA directory and file for demonstration.")
        return




    # implement a k-mer based embedding or a simple CNN from scratch.
    dna_embedder = DNASequenceEmbedder(EMBEDDING_MODEL_NAME)
    if dna_embedder.model is None:
        print("Embedding model could not be loaded. Please check your EMBEDDING_MODEL_NAME and model availability.")
        print("This script will continue with dummy embeddings if the model is unavailable.")
   

    # Initialize Pinecone
    try:
        pinecone_index = initialize_pinecone_index(
            PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_DIM
        )
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}. Please check your API key and environment.")
        return

    all_vectors_to_upload = []
    for record in parse_fasta_files(FASTA_DIR):
        sequence_id = record.id
        dna_sequence = str(record.seq)

        if not dna_sequence:
            print(f"Skipping empty sequence: {sequence_id}")
            continue

        print(f"Processing sequence: {sequence_id[:50]}...") # Print first 50 chars of ID
        embedding = dna_embedder.embed_sequence(dna_sequence)

        # Ensure embedding has the correct dimension
        if len(embedding) != EMBEDDING_DIM:
            print(f"Warning: Embedding for {sequence_id} has dimension {len(embedding)}, expected {EMBEDDING_DIM}. Using dummy.")
            embedding = [0.0] * EMBEDDING_DIM # Fallback to dummy

        metadata = {
            "description": record.description,
            "sequence_length": len(dna_sequence),
            "original_id": record.id, # Store original ID if different from `id` for vector DB
   
        }

        all_vectors_to_upload.append({
            "id": sequence_id, # Pinecone ID for the vector
            "values": embedding,
            "metadata": metadata
        })

    if all_vectors_to_upload:
        upload_embeddings_to_pinecone(pinecone_index, all_vectors_to_upload)
        print(f"\nSuccessfully processed and uploaded {len(all_vectors_to_upload)} sequences to Pinecone.")
    else:
        print("No sequences processed or uploaded.")

    
    print("\n---  Query ---")
    if pinecone_index and all_vectors_to_upload:
        # Take the first uploaded sequence's embedding as a query
        query_vector = all_vectors_to_upload[0]["values"]
        query_id = all_vectors_to_upload[0]["id"]
        print(f"Querying for sequences similar to {query_id}")
        try:
            query_results = pinecone_index.query(
                vector=query_vector,
                top_k=5, # Retrieve top 5 similar sequences
                include_metadata=True
            )
            for match in query_results.matches:
                print(f"  Match: {match.id}, Score: {match.score:.4f}, Description: {match.metadata.get('description', 'N/A')}")
        except Exception as e:
            print(f"Error during query: {e}")
    else:
        print("Cannot perform query: no data uploaded or Pinecone not initialized.")

