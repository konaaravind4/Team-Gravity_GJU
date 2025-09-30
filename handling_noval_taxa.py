import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

# --- Configuration ---
ASV_TABLE_PATH = 'asv_table.csv'
TAXONOMY_PATH = 'taxonomy_assignments.csv'
TARGET_SAMPLE_COLUMN = 'Sample_A' 
SPECIES_CONFIDENCE_THRESHOLD = 0.85 

# --- 1. Load Data ---
try:
    asv_df = pd.read_csv(ASV_TABLE_PATH, index_col=0) # ASVs as index
    taxonomy_df = pd.read_csv(TAXONOMY_PATH, index_col=0) # ASVs as index
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure '{ASV_TABLE_PATH}' and '{TAXONOMY_PATH}' exist.")
    exit()

# Ensure both dataframes use the same ASV identifiers as their index
if not asv_df.index.is_unique or not taxonomy_df.index.is_unique:
    print("Warning: ASV indices are not unique. This might cause issues.")

# --- 2. Merge Taxonomy with ASV Counts ---
merged_df = asv_df.join(taxonomy_df, how='left')


taxonomic_levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']


for level in taxonomic_levels:
    if level not in merged_df.columns:
        merged_df[level] = np.nan # Add column if it doesn't exist
    merged_df[level] = merged_df[level].fillna(f'Unclassified_{level}')


# --- 3. Filter for Target Sample and Calculate Relative Abundance ---
if TARGET_SAMPLE_COLUMN not in merged_df.columns:
    print(f"Error: Target sample column '{TARGET_SAMPLE_COLUMN}' not found in ASV table.")
    print(f"Available columns are: {merged_df.columns.tolist()}")
    exit()

# Select relevant columns for the sample and taxonomy
sample_data = merged_df[[TARGET_SAMPLE_COLUMN] + taxonomic_levels + ['Confidence']].copy()

# Remove ASVs with 0 reads in the target sample
sample_data = sample_data[sample_data[TARGET_SAMPLE_COLUMN] > 0]

# Calculate total reads for the sample
total_reads = sample_data[TARGET_SAMPLE_COLUMN].sum()
print(f"\nTotal reads for {TARGET_SAMPLE_COLUMN}: {total_reads}")

if total_reads == 0:
    print(f"No reads found for {TARGET_SAMPLE_COLUMN}. Cannot calculate abundance.")
    exit()

# Calculate relative abundance
sample_data['Relative_Abundance'] = (sample_data[TARGET_SAMPLE_COLUMN] / total_reads) * 100

# --- 4. Identify Potential Novel Taxa ---




sample_data['Is_Novel_Species'] = False


novel_condition_unclassified = (sample_data['Species'] == 'Unclassified_Species')


novel_condition_confidence = pd.Series([False] * len(sample_data), index=sample_data.index)
if 'Confidence' in sample_data.columns:
   
    sample_data['Confidence'] = sample_data['Confidence'].fillna(0)
    novel_condition_confidence = (sample_data['Confidence'] < SPECIES_CONFIDENCE_THRESHOLD)


sample_data['Is_Novel_Species'] = novel_condition_unclassified | novel_condition_confidence

print(f"\n--- Potential Novel Taxa (at Species Level) in {TARGET_SAMPLE_COLUMN} ---")
novel_species_asvs = sample_data[sample_data['Is_Novel_Species']]
if not novel_species_asvs.empty:
    print(f"Found {len(novel_species_asvs)} ASVs potentially representing novel species.")
    print("Their higher-level classifications (if any) are:")
    
    novel_taxa_summary = novel_species_asvs.groupby(['Phylum', 'Class', 'Order', 'Family', 'Genus'])['Relative_Abundance'].sum().sort_values(ascending=False)
    print(novel_taxa_summary.head(10)) # Display top 10 novel groups
else:
    print("No ASVs identified as potentially novel species based on current criteria.")

# --- 5. Aggregate Abundance by Taxonomic Level (including 'Novel' category) ---

def aggregate_abundance_with_novelty(df, taxonomic_level):
    if taxonomic_level not in df.columns:
        print(f"Warning: Taxonomic level '{taxonomic_level}' not found in data. Skipping aggregation.")
        return None

  
    if taxonomic_level == 'Species':
        
        def get_species_label(row):
            if row['Is_Novel_Species']:
               
                if row['Genus'] == 'Unclassified_Genus':
                    return 'Unclassified Novel Species'
                else:
                    return f'Novel Species ({row["Genus"]})' 
            return row['Species'] 

        df['Aggregated_Species_Label'] = df.apply(get_species_label, axis=1)
        aggregated = df.groupby('Aggregated_Species_Label')['Relative_Abundance'].sum().sort_values(ascending=False)
    else:
        # For higher levels, use the 'Unclassified_Level' as placeholder for novelty
        df[taxonomic_level] = df[taxonomic_level].fillna(f'Unclassified_{taxonomic_level}')
        aggregated = df.groupby(taxonomic_level)['Relative_Abundance'].sum().sort_values(ascending=False)
    return aggregated

abundance_by_species_with_novelty = aggregate_abundance_with_novelty(sample_data, 'Species')
abundance_by_genus = aggregate_abundance_with_novelty(sample_data, 'Genus')
abundance_by_family = aggregate_abundance_with_novelty(sample_data, 'Family')
abundance_by_phylum = aggregate_abundance_with_novelty(sample_data, 'Phylum')


# --- 6. Display Results ---
print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Species (with Novelty) ---")
if abundance_by_species_with_novelty is not None:
    print(abundance_by_species_with_novelty.head(15)) # Display top 15 species/novel groups
    print(f"Total Species Abundance: {abundance_by_species_with_novelty.sum():.2f}%")

print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Genus ---")
if abundance_by_genus is not None:
    print(abundance_by_genus.head(10)) # Display top 10 genera
    print(f"Total Genus Abundance: {abundance_by_genus.sum():.2f}%")

print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Phylum ---")
if abundance_by_phylum is not None:
    print(abundance_by_phylum.head(10)) # Display top 10 phyla
    print(f"Total Phylum Abundance: {abundance_by_phylum.sum():.2f}%")

# --- 7. Visualization of Novel vs. Classified ---
if abundance_by_species_with_novelty is not None and not abundance_by_species_with_novelty.empty:
    plt.figure(figsize=(12, 7))
    plot_data = abundance_by_species_with_novelty.head(15) # Show top N categories

    colors = ['red' if 'Novel Species' in idx or 'Unclassified' in idx else 'skyblue' for idx in plot_data.index]

    sns.barplot(x=plot_data.index, y=plot_data.values, palette=colors)
    plt.title(f'Top Species/Novel Taxa Relative Abundance in {TARGET_SAMPLE_COLUMN}')
    plt.xlabel('Taxon/Novel Category')
    plt.ylabel('Relative Abundance (%)')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.show()


unclassified_phylum_abundance = sample_data[sample_data['Phylum'] == 'Unclassified_Phylum']['Relative_Abundance'].sum()
unclassified_class_abundance = sample_data[sample_data['Class'] == 'Unclassified_Class']['Relative_Abundance'].sum()
unclassified_order_abundance = sample_data[sample_data['Order'] == 'Unclassified_Order']['Relative_Abundance'].sum()
unclassified_family_abundance = sample_data[sample_data['Family'] == 'Unclassified_Family']['Relative_Abundance'].sum()
unclassified_genus_abundance = sample_data[sample_data['Genus'] == 'Unclassified_Genus']['Relative_Abundance'].sum()

print(f"\n--- Summary of Unclassified Reads by Higher Taxonomic Levels ---")
print(f"Unclassified Phylum: {unclassified_phylum_abundance:.2f}%")
print(f"Unclassified Class: {unclassified_class_abundance:.2f}%")
print(f"Unclassified Order: {unclassified_order_abundance:.2f}%")
print(f"Unclassified Family: {unclassified_family_abundance:.2f}%")
print(f"Unclassified Genus: {unclassified_genus_abundance:.2f}%")

if unclassified_phylum_abundance > 0 or unclassified_genus_abundance > 0:
    plt.figure(figsize=(8, 8))
    labels = ['Unclassified Phylum', 'Unclassified Class', 'Unclassified Order', 'Unclassified Family', 'Unclassified Genus', 'Classified']
    sizes = [unclassified_phylum_abundance,
             unclassified_class_abundance - unclassified_phylum_abundance, 
             unclassified_order_abundance - unclassified_class_abundance,
             unclassified_family_abundance - unclassified_order_abundance,
             unclassified_genus_abundance - unclassified_family_abundance,
             100 - unclassified_genus_abundance] 

    
    labels_filtered = [labels[i] for i, s in enumerate(sizes) if s > 0.01]
    sizes_filtered = [s for s in sizes if s > 0.01]


    plt.pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
            colors=sns.color_palette("rocket", len(sizes_filtered)))
    plt.title(f'Proportion of Reads Unclassified at Different Taxonomic Ranks in {TARGET_SAMPLE_COLUMN}')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
