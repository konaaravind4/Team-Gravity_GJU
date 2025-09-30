import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
ASV_TABLE_PATH = 'asv_table.csv'
TAXONOMY_PATH = 'taxonomy_assignments.csv'
TARGET_SAMPLE_COLUMN = 'Sample_A'

# --- 1. Load Data ---
try:
    asv_df = pd.read_csv(ASV_TABLE_PATH, index_col=0) 
    taxonomy_df = pd.read_csv(TAXONOMY_PATH, index_col=0) 
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure '{ASV_TABLE_PATH}' and '{TAXONOMY_PATH}' exist.")
    exit()


if not asv_df.index.is_unique or not taxonomy_df.index.is_unique:
    print("Warning: ASV indices are not unique. This might cause issues.")


merged_df = asv_df.join(taxonomy_df, how='left')

# --- 3. Calculate Relative Abundance for the Target Sample ---

if TARGET_SAMPLE_COLUMN not in merged_df.columns:
    print(f"Error: Target sample column '{TARGET_SAMPLE_COLUMN}' not found in ASV table.")
    print(f"Available columns are: {merged_df.columns.tolist()}")
    exit()

sample_data = merged_df[[TARGET_SAMPLE_COLUMN, 'Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum']].copy()


sample_data = sample_data[sample_data[TARGET_SAMPLE_COLUMN] > 0]


total_reads = sample_data[TARGET_SAMPLE_COLUMN].sum()
print(f"\nTotal reads for {TARGET_SAMPLE_COLUMN}: {total_reads}")

if total_reads == 0:
    print(f"No reads found for {TARGET_SAMPLE_COLUMN}. Cannot calculate abundance.")
    exit()


sample_data['Relative_Abundance'] = (sample_data[TARGET_SAMPLE_COLUMN] / total_reads) * 100

# --- 4. Aggregate Abundance by Taxonomic Level ---


def aggregate_abundance(df, taxonomic_level):
    if taxonomic_level not in df.columns:
        print(f"Warning: Taxonomic level '{taxonomic_level}' not found in data. Skipping aggregation.")
        return None
   
    df[taxonomic_level] = df[taxonomic_level].fillna(f'Unclassified_{taxonomic_level}')
    aggregated = df.groupby(taxonomic_level)['Relative_Abundance'].sum().sort_values(ascending=False)
    return aggregated

abundance_by_species = aggregate_abundance(sample_data, 'Species')
abundance_by_genus = aggregate_abundance(sample_data, 'Genus')
abundance_by_family = aggregate_abundance(sample_data, 'Family')
abundance_by_phylum = aggregate_abundance(sample_data, 'Phylum')


# --- 5. Display Results ---
print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Species ---")
if abundance_by_species is not None:
    print(abundance_by_species.head(10)) 
    print(f"Total Species Abundance: {abundance_by_species.sum():.2f}%")


print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Genus ---")
if abundance_by_genus is not None:
    print(abundance_by_genus.head(10))
    print(f"Total Genus Abundance: {abundance_by_genus.sum():.2f}%")

print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Family ---")
if abundance_by_family is not None:
    print(abundance_by_family.head(10)) 
    print(f"Total Family Abundance: {abundance_by_family.sum():.2f}%")

print(f"\n--- Abundance for {TARGET_SAMPLE_COLUMN} by Phylum ---")
if abundance_by_phylum is not None:
    print(abundance_by_phylum.head(10)) 
    print(f"Total Phylum Abundance: {abundance_by_phylum.sum():.2f}%")


# --- 6. Basic Visualization (e.g., Top N Species/Genera) ---
if abundance_by_genus is not None and not abundance_by_genus.empty:
    plt.figure(figsize=(12, 7))
    top_n = 10
    plot_data = abundance_by_genus.head(top_n)
   
    if len(abundance_by_genus) > top_n:
        other_sum = abundance_by_genus.iloc[top_n:].sum()
        plot_data.loc['Other'] = other_sum
    
    sns.barplot(x=plot_data.index, y=plot_data.values, palette='viridis')
    plt.title(f'Top {top_n} Genera Relative Abundance in {TARGET_SAMPLE_COLUMN}')
    plt.xlabel('Genus')
    plt.ylabel('Relative Abundance (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if abundance_by_phylum is not None and not abundance_by_phylum.empty:
    plt.figure(figsize=(10, 10))
    plot_data = abundance_by_phylum[abundance_by_phylum > 0.1]
    if len(plot_data) < len(abundance_by_phylum):
         plot_data.loc['Other Phyla'] = abundance_by_phylum[~abundance_by_phylum.index.isin(plot_data.index)].sum()
    
    plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    plt.title(f'Phylum-level Relative Abundance in {TARGET_SAMPLE_COLUMN}')
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()
