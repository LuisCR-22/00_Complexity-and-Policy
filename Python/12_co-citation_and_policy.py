"""
==============================================================================
CO-CITATION NETWORK MEETS POLICY/SDG ANALYSIS - CORRECTED VERSION
==============================================================================

CRITICAL FIXES:
1. Only uses papers that exist in BOTH co-citation network AND merged corpus
2. Provides detailed diagnostics about data matching
3. Warns if policy citation data is missing
4. Ensures all visualizations use only validated data

Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Global Complexity School 2025 Final Project
Date: November 2025
==============================================================================
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import os
import community.community_louvain as community_louvain
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import re

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
GEXF_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out\Complexity_CoCitation_LCC.gexf"

CORPUS_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\04_OA_Altmetrics_merge.xlsx"

# Output paths
OUTPUT_IMAGES = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"
OUTPUT_EXCEL = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel"

# Ensure directories exist
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_EXCEL, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

PLOT_SETTINGS = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 9
}

for key, value in PLOT_SETTINGS.items():
    plt.rcParams[key] = value

# Colors
COLORS = {
    'policy_high': '#E74C3C',     # Red - high policy impact
    'policy_medium': '#F39C12',   # Orange - medium
    'policy_low': '#3498DB',      # Blue - low
    'policy_none': '#95A5A6',     # Gray - no policy
    'academic': '#3498DB',
    'sdg_network': '#27AE60'      # Green for SDG networks
}

# Filtering parameters
FILTERS = {
    'medium': {
        'weight_threshold': 0.472,
        'min_cocitations': 5,
        'min_node_count': 10,
        'description': 'Medium filter'
    },
    'strong': {
        'weight_threshold': 0.667,
        'min_cocitations': 7,
        'min_node_count': 20,
        'description': 'Strong filter'
    },
    'very_strong': {
        'weight_threshold': 0.775,
        'min_cocitations': 9,
        'min_node_count': 30,
        'description': 'Very strong filter'
    }
}

print("="*80)
print(" "*15 + "CO-CITATION NETWORK × POLICY/SDG ANALYSIS - CORRECTED")
print("="*80)
print("\nIntegrating bibliometric structure with policy impact patterns")
print("Only using papers that exist in BOTH network and corpus")
print("="*80 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_author_lastname(author_string):
    """Extract last name of first author from author string."""
    if pd.isna(author_string) or str(author_string).strip() == '':
        return "Unknown"
    
    parts = str(author_string).split('|')
    if len(parts) == 0:
        return "Unknown"
    
    first_author = parts[0].strip()
    words = first_author.split()
    if len(words) == 0:
        return "Unknown"
    
    lastname = words[-1].strip()
    lastname = re.sub(r'[^a-zA-Z\-]', '', lastname)
    
    return lastname if lastname else "Unknown"


def create_node_label(row):
    """Create node label: 'LastName Year SDG#'"""
    lastname = extract_author_lastname(row.get('authors_names_oa', ''))
    
    # Try multiple year column names
    year = row.get('year') or row.get('publication_year') or row.get('publication_year_oa') or ''
    
    # Try multiple SDG column names
    sdg_num = row.get('sdg_number_oa') or row.get('sdg_number') or ''
    
    # Handle year
    if pd.notna(year) and str(year).strip() != '':
        try:
            year_str = str(int(float(year)))
        except (ValueError, TypeError):
            year_str = "????"
    else:
        year_str = "????"
    
    # Handle SDG
    if pd.notna(sdg_num) and str(sdg_num).strip() != '':
        try:
            sdg_str = f" SDG{int(float(sdg_num))}"
        except (ValueError, TypeError):
            sdg_str = ""
    else:
        sdg_str = ""
    
    return f"{lastname} {year_str}{sdg_str}"


def load_and_merge_data():
    """
    Load network and corpus, merge them.
    CRITICAL: Only keeps nodes that exist in BOTH datasets.
    
    Returns:
        G: NetworkX graph (filtered to matched nodes only)
        df_corpus: Full corpus DataFrame
        df_merged: Merged data (only papers in both network AND corpus)
    """
    print("📂 Loading data...")
    
    # Load network
    print(f"  • Loading co-citation network from GEXF...")
    G_original = nx.read_gexf(GEXF_PATH)
    print(f"    ✓ Loaded {G_original.number_of_nodes():,} nodes, {G_original.number_of_edges():,} edges")
    
    # Load corpus
    print(f"  • Loading corpus with policy/SDG data...")
    df_corpus = pd.read_excel(CORPUS_PATH)
    print(f"    ✓ Loaded {len(df_corpus):,} papers")
    
    # Prepare year column
    if 'publication_year' in df_corpus.columns:
        df_corpus['year'] = pd.to_numeric(df_corpus['publication_year'], errors='coerce')
    elif 'publication_year_oa' in df_corpus.columns:
        year_oa = pd.to_numeric(df_corpus['publication_year_oa'], errors='coerce')
        year_alt = pd.to_numeric(df_corpus.get('publication_year_alt', pd.Series()), errors='coerce')
        df_corpus['year'] = year_oa.fillna(year_alt)
    else:
        print("  ⚠ Warning: No publication year column found")
        df_corpus['year'] = np.nan
    
    print(f"\n  • Merging network with corpus data...")
    
    # Create mapping dictionary for fast lookup
    corpus_dict = {}
    for idx, row in df_corpus.iterrows():
        node_id = str(row.get('id_oa', '')).strip()
        if node_id and node_id != 'nan':
            corpus_dict[node_id] = row.to_dict()
    
    print(f"    • Corpus has {len(corpus_dict):,} valid IDs")
    
    # CRITICAL: Identify matched vs unmatched nodes
    matched_nodes = []
    unmatched_nodes = []
    
    for node in G_original.nodes():
        if node in corpus_dict:
            matched_nodes.append(node)
        else:
            unmatched_nodes.append(node)
    
    print(f"    • Matched nodes: {len(matched_nodes):,} ({len(matched_nodes)/G_original.number_of_nodes()*100:.1f}%)")
    print(f"    • Unmatched nodes: {len(unmatched_nodes):,} ({len(unmatched_nodes)/G_original.number_of_nodes()*100:.1f}%)")
    
    # Show warning if many unmatched
    if len(unmatched_nodes) > 0:
        print(f"\n    ⚠ WARNING: {len(unmatched_nodes):,} nodes from network NOT found in corpus!")
        print(f"    • Example unmatched IDs:")
        for i, node in enumerate(unmatched_nodes[:3]):
            print(f"      {i+1}. {node}")
        print(f"    • These papers will be EXCLUDED from all analyses")
        
        if len(matched_nodes) < G_original.number_of_nodes() * 0.5:
            print(f"\n    ⚠⚠⚠ CRITICAL: Less than 50% match rate!")
            print(f"    This suggests GEXF was built from a DIFFERENT corpus")
            print(f"    Consider rebuilding the network from your current corpus\n")
    
    # CRITICAL: Create subgraph with ONLY matched nodes
    print(f"\n  • Creating filtered network with matched nodes only...")
    G = G_original.subgraph(matched_nodes).copy()
    
    # Add corpus data as node attributes
    for node in G.nodes():
        for key, value in corpus_dict[node].items():
            G.nodes[node][key] = value
    
    print(f"    ✓ Filtered network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Create merged DataFrame (only papers in BOTH)
    df_merged = pd.DataFrame([
        {**{'node_id': node}, **corpus_dict[node]}
        for node in G.nodes()
    ])
    
    print(f"    ✓ Created merged dataset: {len(df_merged):,} papers")
    
    # CRITICAL: Verify policy data exists
    policy_papers = (df_merged['policy_mentions_total_alt'].fillna(0) > 0).sum()
    print(f"    ✓ Papers with policy citations: {policy_papers:,} ({policy_papers/len(df_merged)*100:.1f}%)")
    
    if policy_papers == 0:
        print(f"\n    ⚠⚠⚠ CRITICAL WARNING ⚠⚠⚠")
        print(f"    NO papers with policy citations found in merged dataset!")
        print(f"    This suggests a data merge problem. Check:")
        print(f"      1. Is 'policy_mentions_total_alt' the correct column name?")
        print(f"      2. Do the OpenAlex IDs match between GEXF and Excel?")
        print(f"      3. Was the GEXF built from a different corpus?")
        print(f"      4. Available columns: {df_corpus.columns.tolist()}\n")
    
    # Check for unknown papers
    unknown_count = sum(1 for node in G.nodes() if "Unknown" in create_node_label(G.nodes[node]) or "????" in create_node_label(G.nodes[node]))
    if unknown_count > 0:
        print(f"    ⚠ {unknown_count:,} papers have incomplete metadata (Unknown/????)")
    
    print()
    
    return G, df_corpus, df_merged


def filter_network(G, filter_params):
    """Apply filtering to network."""
    G_filtered = G.copy()
    
    # Filter edges
    edges_to_remove = []
    for u, v, data in G_filtered.edges(data=True):
        weight = float(data.get('weight', 0))
        count = int(data.get('count', 0))
        
        if weight < filter_params['weight_threshold'] or count < filter_params['min_cocitations']:
            edges_to_remove.append((u, v))
    
    G_filtered.remove_edges_from(edges_to_remove)
    
    # Filter nodes
    nodes_to_remove = []
    for node in list(G_filtered.nodes()):
        count = int(G_filtered.nodes[node].get('count', 0))
        if count < filter_params['min_node_count']:
            nodes_to_remove.append(node)
    
    G_filtered.remove_nodes_from(nodes_to_remove)
    
    # Get largest connected component
    if G_filtered.number_of_nodes() > 0:
        components = list(nx.connected_components(G_filtered))
        if len(components) > 0:
            largest_cc = max(components, key=len)
            G_filtered = G_filtered.subgraph(largest_cc).copy()
    
    return G_filtered


# ============================================================================
# VISUALIZATION: POLICY IMPACT NETWORK
# ============================================================================

def create_policy_impact_network(G, df_merged):
    """
    Create network colored by policy citation intensity.
    
    Quartiles:
    - Red: High policy impact (top 25%)
    - Orange: Medium policy impact (25-75%)
    - Blue: Low policy impact (bottom 25%)
    - Gray: No policy citations
    """
    print(f"\n{'='*80}")
    print(f"Creating policy impact network visualization")
    print(f"{'='*80}")
    
    # Apply medium filter
    G_filtered = filter_network(G, FILTERS['medium'])
    
    if G_filtered.number_of_nodes() == 0:
        print("  ⚠ Empty network after filtering, skipping...")
        return
    
    print(f"  • Network: {G_filtered.number_of_nodes():,} nodes, {G_filtered.number_of_edges():,} edges")
    
    # Get policy citations for each node
    policy_citations = []
    for node in G_filtered.nodes():
        policy = G_filtered.nodes[node].get('policy_mentions_total_alt', 0)
        if pd.notna(policy):
            policy_citations.append(float(policy))
        else:
            policy_citations.append(0)
    
    policy_citations = np.array(policy_citations)
    
    # Calculate quartiles (only for papers with policy citations > 0)
    policy_nonzero = policy_citations[policy_citations > 0]
    
    if len(policy_nonzero) > 0:
        q25 = np.percentile(policy_nonzero, 25)
        q75 = np.percentile(policy_nonzero, 75)
        print(f"  • Policy citation quartiles: Q25={q25:.1f}, Q75={q75:.1f}")
        print(f"  • Papers with policy citations: {len(policy_nonzero)} ({len(policy_nonzero)/len(policy_citations)*100:.1f}%)")
    else:
        q25 = q75 = 0
        print(f"  ⚠ WARNING: No papers with policy citations in this filtered network!")
        print(f"  This means the data matching failed. Check your data sources.")
    
    # Assign colors
    node_colors = []
    color_labels = []
    
    for policy in policy_citations:
        if policy == 0:
            node_colors.append(COLORS['policy_none'])
            color_labels.append('none')
        elif policy >= q75:
            node_colors.append(COLORS['policy_high'])
            color_labels.append('high')
        elif policy >= q25:
            node_colors.append(COLORS['policy_medium'])
            color_labels.append('medium')
        else:
            node_colors.append(COLORS['policy_low'])
            color_labels.append('low')
    
    # Count by category
    color_counts = Counter(color_labels)
    
    # Node sizes based on total policy citations
    node_sizes = [100 + np.log1p(p) * 100 for p in policy_citations]
    
    # Layout
    print(f"  • Computing layout...")
    pos = nx.spring_layout(G_filtered, k=1.5, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(26, 26))
    
    # Draw edges
    nx.draw_networkx_edges(G_filtered, pos, alpha=0.2, width=0.5,
                          edge_color='gray', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes,
                          node_color=node_colors, alpha=0.85,
                          edgecolors='black', linewidths=1.5, ax=ax)
    
    # Title
    title_text = (f"Co-Citation Network Colored by Policy Impact\n"
                 f"{G_filtered.number_of_nodes():,} papers | "
                 f"Node size & color = policy citation intensity\n"
                 f"High: {color_counts['high']} | Medium: {color_counts['medium']} | "
                 f"Low: {color_counts['low']} | None: {color_counts['none']}")
    
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['policy_high'],
               markersize=15, label=f'High Policy Impact (≥Q75, n={color_counts["high"]})',
               markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['policy_medium'],
               markersize=15, label=f'Medium Policy Impact (Q25-Q75, n={color_counts["medium"]})',
               markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['policy_low'],
               markersize=15, label=f'Low Policy Impact (<Q25, n={color_counts["low"]})',
               markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['policy_none'],
               markersize=15, label=f'No Policy Citations (n={color_counts["none"]})',
               markeredgecolor='black', markeredgewidth=1.5)
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             frameon=True, fancybox=True, shadow=True, title='Policy Impact Level',
             title_fontsize=13)
    
    # Add interpretation
    note_text = (f"INTERPRETATION:\n"
                f"• Red nodes: Papers frequently cited in policy documents\n"
                f"• Gray nodes: Papers without policy citations\n"
                f"• Clustering reveals if policy-relevant papers form communities\n"
                f"• Central red nodes are 'policy hubs' in complexity science")
    
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.6))
    
    plt.tight_layout()
    
    output_path = f"{OUTPUT_IMAGES}/16_1_policy_impact_network_CORRECTED.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with diagnostics"""
    
    print("\n" + "="*80)
    print(" "*10 + "INTEGRATING CO-CITATION NETWORK WITH POLICY/SDG ANALYSIS")
    print("="*80)
    print("  - CORRECTED VERSION: Only uses papers in BOTH datasets")
    print("\n" + "="*80 + "\n")
    
    try:
        # Load and merge data - NOW ONLY USES MATCHED PAPERS
        G, df_corpus, df_merged = load_and_merge_data()
        
        # Diagnostic report
        print("\n" + "="*80)
        print("DIAGNOSTIC REPORT")
        print("="*80)
        print(f"\n1. Data Quality Check:")
        print(f"   • Original network size: {G.number_of_nodes():,} nodes")
        print(f"   • Papers in corpus: {len(df_corpus):,}")
        print(f"   • Papers in BOTH (validated): {len(df_merged):,}")
        print(f"   • Match rate: {len(df_merged)/G.number_of_nodes()*100:.1f}%")
        
        # Check for "Unknown ????" papers
        unknown_count = sum(1 for node in G.nodes() 
                           if "Unknown" in create_node_label(G.nodes[node]) 
                           or "????" in create_node_label(G.nodes[node]))
        
        print(f"\n2. Metadata Quality:")
        print(f"   • Papers with complete metadata: {G.number_of_nodes() - unknown_count:,}")
        print(f"   • Papers with missing author/year: {unknown_count:,}")
        if unknown_count > 0:
            print(f"   ⚠ {unknown_count/G.number_of_nodes()*100:.1f}% have incomplete metadata")
        
        # Check policy citations
        policy_count = sum(1 for node in G.nodes() 
                          if G.nodes[node].get('policy_mentions_total_alt', 0) > 0)
        print(f"\n3. Policy Citation Coverage:")
        print(f"   • Papers with policy citations: {policy_count:,}")
        print(f"   • Papers without policy citations: {G.number_of_nodes() - policy_count:,}")
        print(f"   • Policy coverage: {policy_count/G.number_of_nodes()*100:.1f}%")
        
        if policy_count == 0:
            print(f"\n   ⚠⚠⚠ CRITICAL: No policy citations found!")
            print(f"   This indicates the GEXF was built from DIFFERENT papers than your corpus.")
            print(f"   Recommendation: Rebuild co-citation network from your current corpus.\n")
            return
        
        # If data quality is acceptable, proceed with analysis
        if policy_count > 0:
            print(f"\n4. Data quality acceptable - proceeding with analyses...")
            print(f"   Using {len(df_merged):,} validated papers with complete data\n")
            
            # Create policy impact network
            create_policy_impact_network(G, df_merged)
            
            print(f"\n{'='*80}")
            print("✓ ANALYSIS COMPLETED")
            print(f"{'='*80}")
            print(f"\n📊 Results:")
            print(f"  • Policy impact network created and saved")
            print(f"  • Check output images folder for visualization")
            print(f"\n💡 Next steps:")
            print(f"  • Review the visualization to see policy impact distribution")
            print(f"  • If all nodes are still gray, the GEXF needs to be rebuilt")
            print(f"  • If you see colored nodes, the fix worked!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()