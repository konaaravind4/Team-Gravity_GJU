'''Go to the SILVA rRNA Database Project website: https://www.arb-silva.de/
   Navigate to "Download": https://www.arb-silva.de/download/
Choose the appropriate database type:
For prokaryotes: SILVA_138.1_SSURef_NR99 (Small Subunit - 16S, non-redundant 99% identity) or LSURef (Large Subunit - 23S).
For eukaryotes: SILVA_138.1_SSURef_NR99_Euk (Small Subunit - 18S, non-redundant 99% identity).
Note: Always check the latest release number.

Select Files: You'll want:
SILVA_<version>_SSURef_NR99.fasta (the sequences)
SILVA_<version>_SSURef_NR99.tax (the taxonomy)
You might also consider the .arb files if you use the ARB software, or alignment files if you need pre-aligned data.'''



fasta_file = 'path/to/downloaded_files/SILVA_138.1_SSURef_NR99_Euk.fasta'
tax_file = 'path/to/downloaded_files/SILVA_138.1_SSURef_NR99_Euk.tax'
output_fasta = 'marine_18S_eukaryotes.fasta'
output_tax = 'marine_18S_eukaryotes.tax'

marine_keywords = ['marine', 'ocean', 'sea', 'fish', 'algae', 'crustacean', 'mollusc', 'coral', 'jellyfish'] # Add more as needed

marine_ids = set()
taxonomy_map = {}

print("Parsing taxonomy file...")
with open(tax_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        seq_id = parts[0]
        taxonomy = parts[1].lower() 
        taxonomy_map[seq_id] = line.strip() 

      
        if any(kw in taxonomy for kw in marine_keywords):
            marine_ids.add(seq_id)

print(f"Found {len(marine_ids)} sequences potentially marine based on keywords.")

print("Writing filtered FASTA and taxonomy files...")
from Bio import SeqIO

with open(output_fasta, 'w') as out_f, open(output_tax, 'w') as out_t:
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id in marine_ids:
            SeqIO.write(record, out_f, "fasta")
            out_t.write(taxonomy_map[record.id] + '\n')

print("Filtering complete.")


               #---method 2---

# Create a directory for your SILVA data
mkdir -p silva_marine_db
cd silva_marine_db

# Download the gzipped FASTA file
echo "Downloading SILVA 18S Eukaryotic FASTA..."
wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_Euk.fasta.gz

# Download the gzipped taxonomy file
echo "Downloading SILVA 18S Eukaryotic Taxonomy..."
wget https://www.arb-silva.de/fileadmin/silva_databases/release_138_1/Exports/SILVA_138.1_SSURef_NR99_Euk.tax.gz

# Unzip the files
echo "Unzipping files..."
gunzip SILVA_138.1_SSURef_NR99_Euk.fasta.gz
gunzip SILVA_138.1_SSURef_NR99_Euk.tax.gz

echo "Download and unzipping complete."

           #---filter the data---

import pandas as pd
from Bio import SeqIO
import gzip 

# --- Configuration ---
# Path to the unzipped SILVA files
SILVA_FASTA = 'silva_marine_db/SILVA_138.1_SSURef_NR99_Euk.fasta'
SILVA_TAX = 'silva_marine_db/SILVA_138.1_SSURef_NR99_Euk.tax'

OUTPUT_FASTA = 'filtered_marine_18S.fasta'
OUTPUT_TAX = 'filtered_marine_18S.tax'


marine_taxa_to_include = [
    'D_0__Eukaryota;D_1__Archaeplastida', 
    'D_0__Eukaryota;D_1__Opisthokonta;D_2__Metazoa', 
    'D_0__Eukaryota;D_1__Stramenopiles', 
    'D_0__Eukaryota;D_1__Alveolata', 
    'D_0__Eukaryota;D_1__Rhizaria', 
    'D_0__Eukaryota;D_1__Haptophyta', 
   
]

# --- 1. Load Taxonomy and filter for marine entries ---
print("Loading and filtering taxonomy data...")
marine_ids = set()
taxonomy_lines = {} 

with open(SILVA_TAX, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue 
        seq_id = parts[0]
        full_taxonomy_string = parts[1]
        taxonomy_lines[seq_id] = line.strip()

        is_marine = False
        for marine_prefix in marine_taxa_to_include:
            if full_taxonomy_string.startswith(marine_prefix):
                is_marine = True
                break
        
        if is_marine:
            marine_ids.add(seq_id)

print(f"Identified {len(marine_ids)} marine sequences based on defined taxonomic prefixes.")

# --- 2. Filter FASTA file ---
print(f"Filtering FASTA file and writing to {OUTPUT_FASTA}...")
with open(SILVA_FASTA, 'r') as fasta_in, open(OUTPUT_FASTA, 'w') as fasta_out, \
     open(OUTPUT_TAX, 'w') as tax_out:
    
    # Using SeqIO for robust FASTA parsing
    for record in SeqIO.parse(fasta_in, "fasta"):
        if record.id in marine_ids:
            SeqIO.write(record, fasta_out, "fasta")
            tax_out.write(taxonomy_lines[record.id] + '\n')

print("Filtering complete. Marine SILVA database subsets created.")
