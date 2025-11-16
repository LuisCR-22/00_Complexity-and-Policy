"""
==============================================================================
CO-CITATION NETWORK MEETS POLICY/SDG ANALYSIS
==============================================================================

This script integrates the co-citation network (Complexity_CoCitation_LCC.gexf)
with the policy and SDG impact analysis from the merged corpus.

Tasks:
1. Enhanced network visualizations with author-year-SDG labels (15_*.png)
2. Policy/SDG network analyses (16_*.png)
3. Comprehensive diagnostic file with policy/SDG data (14_*.xlsx)

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

# Filtering parameters (same as advanced visualization script)
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
print(" "*15 + "CO-CITATION NETWORK × POLICY/SDG ANALYSIS")
print("="*80)
print("\nIntegrating bibliometric structure with policy impact patterns")
print("="*80 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_author_lastname(author_string):
    """
    Extract last name of first author from author string.
    
    Format: "Author1 Name | Author2 Name | ..."
    Returns: Last word before first |
    
    Example: "Albert‐László Barabási | Réka Albert" → "Barabási"
    """
    if pd.isna(author_string) or str(author_string).strip() == '':
        return "Unknown"
    
    # Split by | to get first author
    parts = str(author_string).split('|')
    if len(parts) == 0:
        return "Unknown"
    
    first_author = parts[0].strip()
    
    # Get last word (last name)
    words = first_author.split()
    if len(words) == 0:
        return "Unknown"
    
    # Return last word, removing special characters
    lastname = words[-1].strip()
    # Remove non-alphanumeric except hyphens
    lastname = re.sub(r'[^a-zA-Z\-]', '', lastname)
    
    return lastname if lastname else "Unknown"


def create_node_label(row):
    """
    Create node label: "LastName Year SDG#"
    
    Args:
        row: DataFrame row or dict with author, year, sdg_number
        
    Returns:
        String label
    """
    lastname = extract_author_lastname(row.get('authors_names_oa', ''))
    
    # Try multiple year column names
    year = row.get('year') or row.get('publication_year') or row.get('publication_year_oa') or ''
    
    # Try multiple SDG column names
    sdg_num = row.get('sdg_number_oa') or row.get('sdg_number') or ''
    
    # Handle year - check for empty string, NaN, and valid numbers
    if pd.notna(year) and str(year).strip() != '':
        try:
            year_str = str(int(float(year)))
        except (ValueError, TypeError):
            year_str = "????"
    else:
        year_str = "????"
    
    # Handle SDG - check for empty string, NaN, and valid numbers
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
    
    Returns:
        G: NetworkX graph
        df_corpus: Full corpus DataFrame
        df_merged: Merged data (only papers in network)
    """
    print("📂 Loading data...")
    
    # Load network
    print(f"  • Loading co-citation network from GEXF...")
    G = nx.read_gexf(GEXF_PATH)
    print(f"    ✓ Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Load corpus
    print(f"  • Loading corpus with policy/SDG data...")
    df_corpus = pd.read_excel(CORPUS_PATH)
    print(f"    ✓ Loaded {len(df_corpus):,} papers")
    
    # Prepare year column - use the actual column name from Excel
    if 'publication_year' in df_corpus.columns:
        df_corpus['year'] = pd.to_numeric(df_corpus['publication_year'], errors='coerce')
    elif 'publication_year_oa' in df_corpus.columns:
        year_oa = pd.to_numeric(df_corpus['publication_year_oa'], errors='coerce')
        year_alt = pd.to_numeric(df_corpus.get('publication_year_alt', pd.Series()), errors='coerce')
        df_corpus['year'] = year_oa.fillna(year_alt)
    else:
        print("  ⚠ Warning: No publication year column found, using NaN")
        df_corpus['year'] = np.nan
    
    # Extract node IDs from network
    node_ids = list(G.nodes())
    
    # Match with corpus
    # Both should have format: "https://openalex.org/W..."
    print(f"  • Merging network with corpus data...")
    
    # Create mapping dictionary for fast lookup
    corpus_dict = {}
    for idx, row in df_corpus.iterrows():
        node_id = str(row.get('id_oa', '')).strip()
        if node_id and node_id != 'nan':
            corpus_dict[node_id] = row.to_dict()
    
    # Match nodes
    matched_count = 0
    for node in G.nodes():
        if node in corpus_dict:
            # Add corpus data as node attributes
            for key, value in corpus_dict[node].items():
                G.nodes[node][key] = value
            matched_count += 1
    
    print(f"    ✓ Matched {matched_count:,} nodes ({matched_count/G.number_of_nodes()*100:.1f}%)")
    
    # Create merged DataFrame (only papers in network)
    df_merged = pd.DataFrame([
        {**{'node_id': node}, **corpus_dict.get(node, {})}
        for node in G.nodes()
        if node in corpus_dict
    ])
    
    print(f"    ✓ Created merged dataset: {len(df_merged):,} papers\n")
    
    return G, df_corpus, df_merged


def filter_network(G, filter_params):
    """
    Apply filtering to network (same as advanced viz script).
    
    Args:
        G: NetworkX graph
        filter_params: Dictionary with thresholds
        
    Returns:
        Filtered graph (largest connected component)
    """
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
# TASK 1: ENHANCED NETWORK VISUALIZATIONS (15_*.png)
# ============================================================================

def create_enhanced_network_viz(G, df_merged, filter_name, filter_params):
    """
    Create network visualization with:
    - Community colors
    - Author-Year-SDG labels for top nodes
    - SDG legend
    
    Args:
        G: NetworkX graph
        df_merged: Merged data
        filter_name: Name of filter (medium/strong/very_strong)
        filter_params: Filter parameters
    """
    print(f"\n{'='*80}")
    print(f"Creating enhanced visualization: {filter_name} filter")
    print(f"{'='*80}")
    
    # Apply filter
    G_filtered = filter_network(G, filter_params)
    
    if G_filtered.number_of_nodes() == 0:
        print("  ⚠ Empty network after filtering, skipping...")
        return
    
    print(f"  • Filtered network: {G_filtered.number_of_nodes():,} nodes, {G_filtered.number_of_edges():,} edges")
    
    # Detect communities (Louvain algorithm)
    print(f"  • Detecting communities with Louvain algorithm...")
    partition = community_louvain.best_partition(G_filtered, weight='weight', random_state=42)
    modularity = community_louvain.modularity(partition, G_filtered, weight='weight')
    num_communities = len(set(partition.values()))
    
    print(f"    ✓ Found {num_communities} communities (modularity: {modularity:.4f})")
    
    # Layout
    print(f"  • Computing spring layout...")
    pos = nx.spring_layout(G_filtered, k=1.0, iterations=50, seed=42)
    
    # Prepare visualization data
    node_colors = [partition[node] for node in G_filtered.nodes()]
    
    # Node sizes based on citations (from network)
    node_sizes = []
    for node in G_filtered.nodes():
        citations = int(G_filtered.nodes[node].get('count', 1))
        size = 50 + np.log1p(citations) * 30
        node_sizes.append(size)
    
    # Identify top nodes for labels (by degree)
    degrees = dict(G_filtered.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    # Show labels for top nodes (at least 10, or top 15%)
    num_labels = max(10, int(len(G_filtered.nodes()) * 0.15))
    top_nodes = set([node for node, deg in sorted_nodes[:num_labels]])
    
    print(f"  • Will display labels for top {len(top_nodes)} nodes by degree")
    
    # Create labels for top nodes
    labels = {}
    for node in top_nodes:
        if node in G_filtered.nodes():
            # Get node data
            node_data = G_filtered.nodes[node]
            labels[node] = create_node_label(node_data)
    
    # Collect SDG information for legend
    sdg_info = defaultdict(int)
    for node in G_filtered.nodes():
        sdg_num = G_filtered.nodes[node].get('sdg_number_oa', None)
        sdg_name = G_filtered.nodes[node].get('sdg_name_oa', None)
        
        if pd.notna(sdg_num) and pd.notna(sdg_name):
            sdg_key = f"SDG{int(sdg_num)}: {sdg_name}"
            sdg_info[sdg_key] += 1
    
    # Sort SDGs by number
    sdg_info_sorted = sorted(sdg_info.items(), 
                            key=lambda x: int(x[0].split(':')[0].replace('SDG', '')))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 28))
    
    # Draw edges
    edge_widths = []
    for u, v in G_filtered.edges():
        weight = G_filtered[u][v].get('weight', 0.1)
        width = 0.3 + (weight * 2)
        edge_widths.append(width)
    
    nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, alpha=0.3,
                          edge_color='gray', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes,
                          node_color=node_colors, cmap='tab20',
                          alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
    
    # Draw labels for top nodes
    for node, label_text in labels.items():
        x, y = pos[node]
        ax.text(x, y, label_text, fontsize=7, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', alpha=0.8, linewidth=0.8))
    
    # Title
    title_text = (f"Co-Citation Network with Policy/SDG Context - {filter_params['description']}\n"
                 f"{G_filtered.number_of_nodes():,} papers | {G_filtered.number_of_edges():,} co-citations | "
                 f"{num_communities} communities (modularity: {modularity:.3f})\n"
                 f"Node size = citations | Colors = communities | Labels = Author Year SDG#")
    
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add SDG legend (if any SDGs present)
    if len(sdg_info_sorted) > 0:
        sdg_legend_text = "SDG Legend:\n" + "\n".join([
            f"{sdg} ({count} papers)" 
            for sdg, count in sdg_info_sorted[:10]  # Top 10 SDGs
        ])
        
        if len(sdg_info_sorted) > 10:
            sdg_legend_text += f"\n... and {len(sdg_info_sorted)-10} more SDGs"
        
        ax.text(0.02, 0.98, sdg_legend_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow',
                        alpha=0.9, pad=0.6, linewidth=1))
    
    # Add interpretation note
    note_text = (f"INTERPRETATION:\n"
                f"• Communities detected by Louvain algorithm\n"
                f"  (maximizes modularity, groups similar papers)\n"
                f"• Labels shown for top {len(top_nodes)} most connected papers\n"
                f"• SDG# indicates sustainability domain linkage\n"
                f"• Node clustering shows thematic similarity")
    
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat',
                    alpha=0.9, pad=0.6, linewidth=1))
    
    plt.tight_layout()
    
    # Save
    output_path = f"{OUTPUT_IMAGES}/15_network_{filter_name}_filter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}\n")


# ============================================================================
# TASK 2.1: NETWORK COLORED BY POLICY IMPACT (16_1_*.png)
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
    
    # Apply medium filter for this analysis
    G_filtered = filter_network(G, FILTERS['medium'])
    
    if G_filtered.number_of_nodes() == 0:
        print("  ⚠ Empty network, skipping...")
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
        print(f"  ⚠ No papers with policy citations in this network")
    
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
    
    output_path = f"{OUTPUT_IMAGES}/16_1_policy_impact_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}\n")


# ============================================================================
# TASK 2.2: SDG CO-OCCURRENCE NETWORK (16_2_*.png)
# ============================================================================

def create_sdg_cooccurrence_network(G, df_merged):
    """
    Create network showing which SDGs co-occur in connected papers.
    
    Method:
    - Nodes = SDGs
    - Edges = Papers linking two SDGs (co-citation relationship)
    - Edge weight = number of co-citation links
    """
    print(f"\n{'='*80}")
    print(f"Creating SDG co-occurrence network")
    print(f"{'='*80}")
    
    # Apply medium filter
    G_filtered = filter_network(G, FILTERS['medium'])
    
    if G_filtered.number_of_nodes() == 0:
        print("  ⚠ Empty network, skipping...")
        return
    
    # Build SDG co-occurrence network
    print(f"  • Analyzing SDG co-occurrence patterns...")
    
    # Create SDG network
    SDG_network = nx.Graph()
    
    # Track SDG names for labels
    sdg_names = {}
    
    # For each edge in citation network, check if it connects different SDGs
    sdg_connections = defaultdict(int)
    
    for u, v in G_filtered.edges():
        sdg_u = G_filtered.nodes[u].get('sdg_number_oa', None)
        sdg_v = G_filtered.nodes[v].get('sdg_number_oa', None)
        
        name_u = G_filtered.nodes[u].get('sdg_name_oa', None)
        name_v = G_filtered.nodes[v].get('sdg_name_oa', None)
        
        # If both have SDGs and they're different
        if pd.notna(sdg_u) and pd.notna(sdg_v):
            sdg_u_int = int(sdg_u)
            sdg_v_int = int(sdg_v)
            
            # Store names
            if pd.notna(name_u):
                sdg_names[sdg_u_int] = str(name_u)
            if pd.notna(name_v):
                sdg_names[sdg_v_int] = str(name_v)
            
            # Count connection (even if same SDG)
            if sdg_u_int != sdg_v_int:
                edge = tuple(sorted([sdg_u_int, sdg_v_int]))
                sdg_connections[edge] += 1
    
    # Add nodes and edges to network
    for (sdg1, sdg2), weight in sdg_connections.items():
        if weight >= 2:  # Threshold: at least 2 co-citation links
            SDG_network.add_edge(sdg1, sdg2, weight=weight)
    
    # Add isolated SDGs (those in network but without connections)
    for node in G_filtered.nodes():
        sdg_num = G_filtered.nodes[node].get('sdg_number_oa', None)
        if pd.notna(sdg_num):
            sdg_int = int(sdg_num)
            if sdg_int not in SDG_network.nodes():
                SDG_network.add_node(sdg_int)
    
    print(f"  • SDG network: {SDG_network.number_of_nodes()} SDGs, {SDG_network.number_of_edges()} connections")
    
    if SDG_network.number_of_nodes() == 0:
        print("  ⚠ No SDGs in network, skipping...")
        return
    
    # Calculate node sizes (by papers in that SDG)
    sdg_paper_counts = Counter()
    for node in G_filtered.nodes():
        sdg_num = G_filtered.nodes[node].get('sdg_number_oa', None)
        if pd.notna(sdg_num):
            sdg_paper_counts[int(sdg_num)] += 1
    
    node_sizes = [100 + sdg_paper_counts[node] * 50 for node in SDG_network.nodes()]
    
    # Layout
    if SDG_network.number_of_edges() > 0:
        pos = nx.spring_layout(SDG_network, k=3, iterations=100, seed=42, weight='weight')
    else:
        pos = nx.circular_layout(SDG_network)
    
    # Edge widths
    if SDG_network.number_of_edges() > 0:
        edge_widths = [SDG_network[u][v]['weight'] * 0.5 for u, v in SDG_network.edges()]
    else:
        edge_widths = []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Draw edges
    if SDG_network.number_of_edges() > 0:
        nx.draw_networkx_edges(SDG_network, pos, width=edge_widths, alpha=0.4,
                              edge_color='gray', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(SDG_network, pos, node_size=node_sizes,
                          node_color=COLORS['sdg_network'], alpha=0.8,
                          edgecolors='black', linewidths=2, ax=ax)
    
    # Draw labels
    labels = {}
    for node in SDG_network.nodes():
        name = sdg_names.get(node, f"SDG {node}")
        count = sdg_paper_counts[node]
        labels[node] = f"SDG{node}\n{name}\n({count} papers)"
    
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], fontsize=9, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='black', alpha=0.9, linewidth=1.2))
    
    # Title
    title_text = (f"SDG Co-Occurrence Network in Complexity Science Co-Citations\n"
                 f"{SDG_network.number_of_nodes()} SDGs | "
                 f"{SDG_network.number_of_edges()} connections | "
                 f"Node size = papers in SDG | Edge width = co-citation strength")
    
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add interpretation
    note_text = (f"INTERPRETATION:\n"
                f"• Nodes = Sustainable Development Goals\n"
                f"• Edges = Co-citation links between SDGs\n"
                f"• Reveals which sustainability domains are linked\n"
                f"  through complexity science research\n"
                f"• Thicker edges = stronger thematic connections")
    
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    alpha=0.9, pad=0.6, linewidth=1))
    
    plt.tight_layout()
    
    output_path = f"{OUTPUT_IMAGES}/16_2_sdg_cooccurrence_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}\n")


# ============================================================================
# TASK 2.3: CENTRALITY BY POLICY STATUS (16_3_*.png)
# ============================================================================

def create_centrality_comparison(G, df_merged):
    """
    Compare network centrality for papers with/without policy citations.
    
    Shows:
    - Degree centrality distribution
    - Betweenness centrality distribution
    - Clustering coefficient distribution
    """
    print(f"\n{'='*80}")
    print(f"Creating centrality comparison by policy status")
    print(f"{'='*80}")
    
    # Apply medium filter
    G_filtered = filter_network(G, FILTERS['medium'])
    
    if G_filtered.number_of_nodes() == 0:
        print("  ⚠ Empty network, skipping...")
        return
    
    print(f"  • Calculating network metrics...")
    
    # Debug: Check what policy data looks like
    print(f"\n  🔍 Debugging policy citation data:")
    sample_nodes = list(G_filtered.nodes())[:5]
    for node in sample_nodes:
        node_data = G_filtered.nodes[node]
        policy_val = node_data.get('policy_mentions_total_alt', 'NOT FOUND')
        print(f"    Node {node[:30]}... policy: {policy_val}")
    
    # Calculate metrics
    degree_cent = nx.degree_centrality(G_filtered)
    betweenness_cent = nx.betweenness_centrality(G_filtered, weight='weight')
    clustering = nx.clustering(G_filtered, weight='weight')
    
    # Separate by policy status
    policy_nodes = []
    no_policy_nodes = []
    
    # Try different column names for policy citations
    policy_col_names = [
        'policy_mentions_total_alt',
        'Policy_Citations',
        'policy_citations',
        'policy_mentions'
    ]
    
    for node in G_filtered.nodes():
        node_data = G_filtered.nodes[node]
        policy = None
        
        # Try to find policy data
        for col_name in policy_col_names:
            if col_name in node_data:
                policy = node_data[col_name]
                break
        
        # Categorize
        if policy is not None and pd.notna(policy) and float(policy) > 0:
            policy_nodes.append(node)
        else:
            no_policy_nodes.append(node)
    
    print(f"  • Papers with policy citations: {len(policy_nodes)} ({len(policy_nodes)/G_filtered.number_of_nodes()*100:.1f}%)")
    print(f"  • Papers without policy citations: {len(no_policy_nodes)}")
    
    # Check if we have enough data for comparison
    if len(policy_nodes) == 0:
        print(f"\n  ⚠ No papers with policy citations found in network!")
        print(f"  • This might mean:")
        print(f"    1. The merge didn't work correctly")
        print(f"    2. Policy-cited papers aren't in the co-citation network")
        print(f"    3. The filtering removed all policy-cited papers")
        print(f"\n  💡 Creating simplified visualization showing all papers...\n")
        
        # Create simplified single-distribution plot
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        all_nodes = list(G_filtered.nodes())
        degree_all = [degree_cent[n] for n in all_nodes]
        betweenness_all = [betweenness_cent[n] for n in all_nodes]
        clustering_all = [clustering[n] for n in all_nodes]
        
        # Panel 1: Degree
        ax1 = axes[0]
        ax1.hist(degree_all, bins=30, color=COLORS['academic'], alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Degree Centrality', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax1.set_title(f'Degree Centrality Distribution\n({len(all_nodes)} papers in network)',
                     fontweight='bold', fontsize=13)
        ax1.grid(alpha=0.3)
        
        # Panel 2: Betweenness
        ax2 = axes[1]
        ax2.hist(betweenness_all, bins=30, color=COLORS['academic'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Betweenness Centrality', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax2.set_title(f'Betweenness Centrality Distribution\n({len(all_nodes)} papers in network)',
                     fontweight='bold', fontsize=13)
        ax2.grid(alpha=0.3)
        
        # Panel 3: Clustering
        ax3 = axes[2]
        ax3.hist(clustering_all, bins=30, color=COLORS['academic'], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Clustering Coefficient', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax3.set_title(f'Clustering Coefficient Distribution\n({len(all_nodes)} papers in network)',
                     fontweight='bold', fontsize=13)
        ax3.grid(alpha=0.3)
        
        plt.suptitle('Network Centrality Metrics - All Papers\n' +
                    '(No policy-cited papers found in this network subset)',
                    fontsize=15, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        output_path = f"{OUTPUT_IMAGES}/16_3_centrality_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
        print(f"  📊 Summary statistics (all papers):")
        print(f"    Degree: mean={np.mean(degree_all):.4f}, median={np.median(degree_all):.4f}")
        print(f"    Betweenness: mean={np.mean(betweenness_all):.4f}, median={np.median(betweenness_all):.4f}")
        print(f"    Clustering: mean={np.mean(clustering_all):.4f}, median={np.median(clustering_all):.4f}\n")
        
        return
    
    # If we have both groups, proceed with comparison
    if len(no_policy_nodes) == 0:
        print(f"  ⚠ All papers have policy citations - cannot compare!")
        return
    
    # Extract metrics for each group
    degree_policy = [degree_cent[n] for n in policy_nodes]
    degree_no_policy = [degree_cent[n] for n in no_policy_nodes]
    
    betweenness_policy = [betweenness_cent[n] for n in policy_nodes]
    betweenness_no_policy = [betweenness_cent[n] for n in no_policy_nodes]
    
    clustering_policy = [clustering[n] for n in policy_nodes]
    clustering_no_policy = [clustering[n] for n in no_policy_nodes]
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Panel 1: Degree Centrality
    ax1 = axes[0]
    
    data_degree = [degree_policy, degree_no_policy]
    positions = [1, 2]
    parts = ax1.violinplot(data_degree, positions=positions, showmeans=True,
                           showmedians=True, widths=0.7)
    
    # Color violins
    colors_violin = [COLORS['policy_high'], COLORS['policy_none']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['With Policy\nCitations', 'Without Policy\nCitations'])
    ax1.set_ylabel('Degree Centrality', fontweight='bold', fontsize=12)
    ax1.set_title('Degree Centrality Distribution\n(How connected is the paper?)',
                 fontweight='bold', fontsize=13, pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistics
    if len(degree_policy) > 0 and len(degree_no_policy) > 0:
        from scipy import stats
        stat, pval = stats.mannwhitneyu(degree_policy, degree_no_policy, alternative='two-sided')
        ax1.text(0.5, 0.95, f'Mann-Whitney U test: p={pval:.4f}',
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 2: Betweenness Centrality
    ax2 = axes[1]
    
    data_betweenness = [betweenness_policy, betweenness_no_policy]
    parts = ax2.violinplot(data_betweenness, positions=positions, showmeans=True,
                           showmedians=True, widths=0.7)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['With Policy\nCitations', 'Without Policy\nCitations'])
    ax2.set_ylabel('Betweenness Centrality', fontweight='bold', fontsize=12)
    ax2.set_title('Betweenness Centrality Distribution\n(Does the paper bridge communities?)',
                 fontweight='bold', fontsize=13, pad=15)
    ax2.grid(axis='y', alpha=0.3)
    
    if len(betweenness_policy) > 0 and len(betweenness_no_policy) > 0:
        stat, pval = stats.mannwhitneyu(betweenness_policy, betweenness_no_policy, alternative='two-sided')
        ax2.text(0.5, 0.95, f'Mann-Whitney U test: p={pval:.4f}',
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 3: Clustering Coefficient
    ax3 = axes[2]
    
    data_clustering = [clustering_policy, clustering_no_policy]
    parts = ax3.violinplot(data_clustering, positions=positions, showmeans=True,
                           showmedians=True, widths=0.7)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_violin[i])
        pc.set_alpha(0.7)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['With Policy\nCitations', 'Without Policy\nCitations'])
    ax3.set_ylabel('Clustering Coefficient', fontweight='bold', fontsize=12)
    ax3.set_title('Clustering Coefficient Distribution\n(Is the paper in a tight community?)',
                 fontweight='bold', fontsize=13, pad=15)
    ax3.grid(alpha=0.3)
    
    if len(clustering_policy) > 0 and len(clustering_no_policy) > 0:
        stat, pval = stats.mannwhitneyu(clustering_policy, clustering_no_policy, alternative='two-sided')
        ax3.text(0.5, 0.95, f'Mann-Whitney U test: p={pval:.4f}',
                transform=ax3.transAxes, ha='center', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Overall title
    plt.suptitle('Network Centrality Comparison: Papers with vs without Policy Citations\n' +
                'Do policy-cited papers occupy different network positions?',
                fontsize=15, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    output_path = f"{OUTPUT_IMAGES}/16_3_centrality_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    # Print summary statistics
    print(f"\n  📊 Summary Statistics:")
    print(f"\n  Degree Centrality:")
    print(f"    With policy: mean={np.mean(degree_policy):.4f}, median={np.median(degree_policy):.4f}")
    print(f"    Without policy: mean={np.mean(degree_no_policy):.4f}, median={np.median(degree_no_policy):.4f}")
    
    print(f"\n  Betweenness Centrality:")
    print(f"    With policy: mean={np.mean(betweenness_policy):.4f}, median={np.median(betweenness_policy):.4f}")
    print(f"    Without policy: mean={np.mean(betweenness_no_policy):.4f}, median={np.median(betweenness_no_policy):.4f}")
    
    print(f"\n  Clustering Coefficient:")
    print(f"    With policy: mean={np.mean(clustering_policy):.4f}, median={np.median(clustering_policy):.4f}")
    print(f"    Without policy: mean={np.mean(clustering_no_policy):.4f}, median={np.median(clustering_no_policy):.4f}\n")


# ============================================================================
# TASK 3: COMPREHENSIVE DIAGNOSTIC FILE (14_*.xlsx)
# ============================================================================

def create_comprehensive_diagnostic(G, df_corpus):
    """
    Create comprehensive diagnostic file merging:
    - Network metrics (recalculated)
    - Full corpus data
    - Policy/SDG information
    
    Saves to: 14_network_policy_sdg_diagnostic.xlsx
    """
    print(f"\n{'='*80}")
    print(f"Creating comprehensive diagnostic file")
    print(f"{'='*80}")
    
    print(f"  • Recalculating network metrics for all nodes...")
    
    # Calculate metrics for full network (no filtering)
    print(f"    - Degree centrality...")
    degree_cent = nx.degree_centrality(G)
    
    print(f"    - Betweenness centrality (this may take a moment)...")
    if G.number_of_nodes() > 500:
        betweenness_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()), weight='weight')
    else:
        betweenness_cent = nx.betweenness_centrality(G, weight='weight')
    
    print(f"    - Clustering coefficient...")
    clustering = nx.clustering(G, weight='weight')
    
    print(f"    - Closeness centrality...")
    closeness_cent = nx.closeness_centrality(G)
    
    print(f"    - PageRank...")
    pagerank = nx.pagerank(G, weight='weight')
    
    # Calculate average edge weight for each node
    print(f"    - Average edge weights...")
    avg_edge_weights = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            weights = [G[node][neighbor].get('weight', 0) for neighbor in neighbors]
            avg_edge_weights[node] = np.mean(weights)
        else:
            avg_edge_weights[node] = 0
    
    # Get top 5 connected papers for each node
    print(f"    - Top connections...")
    top_connections = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        neighbor_weights = [(neighbor, G[node][neighbor].get('weight', 0)) for neighbor in neighbors]
        neighbor_weights.sort(key=lambda x: x[1], reverse=True)
        top_5 = neighbor_weights[:5]
        top_connections[node] = top_5
    
    # Create diagnostic data
    print(f"  • Building diagnostic dataset...")
    
    diagnostic_data = []
    
    for idx, row in df_corpus.iterrows():
        node_id = str(row.get('id_oa', '')).strip()
        
        # Network metrics (if node in network)
        if node_id and node_id in G.nodes():
            in_network = True
            node_degree = G.degree(node_id)
            node_citations = int(G.nodes[node_id].get('count', 0))
            degree_cent_val = degree_cent[node_id]
            betweenness_val = betweenness_cent[node_id]
            clustering_val = clustering[node_id]
            closeness_val = closeness_cent[node_id]
            pagerank_val = pagerank[node_id]
            avg_weight = avg_edge_weights[node_id]
            
            # Top connections
            top_5 = top_connections[node_id]
            top_1 = f"{top_5[0][0]} (weight: {top_5[0][1]:.4f})" if len(top_5) > 0 else "N/A"
            top_2 = f"{top_5[1][0]} (weight: {top_5[1][1]:.4f})" if len(top_5) > 1 else "N/A"
            top_3 = f"{top_5[2][0]} (weight: {top_5[2][1]:.4f})" if len(top_5) > 2 else "N/A"
            top_4 = f"{top_5[3][0]} (weight: {top_5[3][1]:.4f})" if len(top_5) > 3 else "N/A"
            top_5_str = f"{top_5[4][0]} (weight: {top_5[4][1]:.4f})" if len(top_5) > 4 else "N/A"
        else:
            in_network = False
            node_degree = 0
            node_citations = 0
            degree_cent_val = 0
            betweenness_val = 0
            clustering_val = 0
            closeness_val = 0
            pagerank_val = 0
            avg_weight = 0
            top_1 = top_2 = top_3 = top_4 = top_5_str = "N/A"
        
        # Corpus data
        diagnostic_record = {
            # Identifiers
            'Node_ID': node_id,
            'DOI': row.get('doi_oa', ''),
            'In_Network': in_network,
            
            # Network metrics
            'Degree': node_degree,
            'Network_Citations': node_citations,
            'Degree_Centrality': degree_cent_val,
            'Betweenness_Centrality': betweenness_val,
            'Closeness_Centrality': closeness_val,
            'Clustering_Coefficient': clustering_val,
            'PageRank': pagerank_val,
            'Average_Edge_Weight': avg_weight,
            
            # Top connections
            'Top_1_Connected_Paper': top_1,
            'Top_2_Connected_Paper': top_2,
            'Top_3_Connected_Paper': top_3,
            'Top_4_Connected_Paper': top_4,
            'Top_5_Connected_Paper': top_5_str,
            
            # Policy metrics
            'Policy_Citations': row.get('policy_mentions_total_alt', 0),
            'Academic_Citations_OA': row.get('cited_by_count_oa', 0),
            'Academic_Citations_Alt': row.get('citations_alt', 0),
            
            # Altmetric
            'Altmetric_Score': row.get('altmetric_score_alt', 0),
            'News_MSM': row.get('news_msm_total_alt', 0),
            'Blogs': row.get('blogs_total_alt', 0),
            'Twitter': row.get('twitter_total_alt', 0),
            'Wikipedia': row.get('wikipedia_mentions_alt', 0),
            'Mendeley_Readers': row.get('readers_mendeley_alt', 0),
            
            # SDG
            'SDG_Number': row.get('sdg_number_oa', ''),
            'SDG_Name': row.get('sdg_name_oa', ''),
            'SDG_Score': row.get('sdg_score_oa', ''),
            
            # Basic metadata
            'Title': row.get('title_oa', row.get('title_alt', '')),
            'Year': row.get('year', ''),
            'Authors': row.get('authors_names_oa', ''),
            'Journal': row.get('source_name_oa', row.get('journal_alt', '')),
            
            # Classification
            'Primary_Topic': row.get('primary_topic_oa', ''),
            'Primary_Field': row.get('primary_field_oa', ''),
            'Primary_Domain': row.get('primary_domain_oa', ''),
            
            # Additional
            'Keywords': row.get('keywords_all_oa', ''),
            'Institutions': row.get('institutions_oa', ''),
            'Countries': row.get('countries_oa', ''),
            'Is_OA': row.get('is_oa_oa', ''),
            'FWCI': row.get('fwci_oa', '')
        }
        
        diagnostic_data.append(diagnostic_record)
    
    # Create DataFrame
    df_diagnostic = pd.DataFrame(diagnostic_data)
    
    # Sort by degree (descending)
    df_diagnostic = df_diagnostic.sort_values('Degree', ascending=False).reset_index(drop=True)
    
    # Save to Excel
    print(f"  • Saving to Excel...")
    
    output_path = f"{OUTPUT_EXCEL}/14_network_policy_sdg_diagnostic.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main sheet
        df_diagnostic.to_excel(writer, sheet_name='Full Diagnostic', index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Papers in Corpus',
                'Papers in Co-Citation Network',
                '% in Network',
                'Papers with Policy Citations',
                'Papers with SDG',
                'Network Nodes',
                'Network Edges',
                'Network Density',
                'Average Degree',
                'Average Clustering'
            ],
            'Value': [
                len(df_corpus),
                df_diagnostic['In_Network'].sum(),
                f"{df_diagnostic['In_Network'].sum()/len(df_corpus)*100:.2f}%",
                (df_diagnostic['Policy_Citations'] > 0).sum(),
                df_diagnostic['SDG_Number'].notna().sum(),
                G.number_of_nodes(),
                G.number_of_edges(),
                f"{nx.density(G):.6f}",
                f"{sum(dict(G.degree()).values())/G.number_of_nodes():.2f}",
                f"{nx.average_clustering(G, weight='weight'):.4f}"
            ]
        }
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Top nodes by centrality
        top_degree = df_diagnostic.nlargest(50, 'Degree')[
            ['Node_ID', 'Title', 'Authors', 'Year', 'Degree', 'Policy_Citations', 'SDG_Name']
        ]
        top_degree.to_excel(writer, sheet_name='Top by Degree', index=False)
        
        top_betweenness = df_diagnostic.nlargest(50, 'Betweenness_Centrality')[
            ['Node_ID', 'Title', 'Authors', 'Year', 'Betweenness_Centrality', 'Policy_Citations', 'SDG_Name']
        ]
        top_betweenness.to_excel(writer, sheet_name='Top by Betweenness', index=False)
        
        # Policy-cited papers in network
        policy_in_network = df_diagnostic[
            (df_diagnostic['In_Network'] == True) & 
            (df_diagnostic['Policy_Citations'] > 0)
        ].sort_values('Policy_Citations', ascending=False)
        
        policy_in_network[
            ['Node_ID', 'Title', 'Authors', 'Year', 'Policy_Citations', 
             'Degree', 'Betweenness_Centrality', 'SDG_Name']
        ].to_excel(writer, sheet_name='Policy Papers in Network', index=False)
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    - Full Diagnostic sheet: {len(df_diagnostic):,} papers")
    print(f"    - Summary sheet: Network overview")
    print(f"    - Top by Degree: 50 most connected papers")
    print(f"    - Top by Betweenness: 50 most bridging papers")
    print(f"    - Policy Papers in Network: {len(policy_in_network):,} papers\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" "*10 + "INTEGRATING CO-CITATION NETWORK WITH POLICY/SDG ANALYSIS")
    print("="*80)
    print("  - Your comprehensive policy/SDG impact analysis")
    print("\n" + "="*80 + "\n")
    
    try:
        # Load and merge data
        G, df_corpus, df_merged = load_and_merge_data()
        
        # =====================================================================
        # TASK 1: Enhanced Network Visualizations (15_*.png)
        # =====================================================================
        print("\n" + "="*80)
        print("TASK 1: ENHANCED NETWORK VISUALIZATIONS WITH LABELS")
        print("="*80)
        
        for filter_name, filter_params in FILTERS.items():
            create_enhanced_network_viz(G, df_merged, filter_name, filter_params)
        
        # =====================================================================
        # TASK 2: Policy/SDG Network Analyses (16_*.png)
        # =====================================================================
        print("\n" + "="*80)
        print("TASK 2: POLICY/SDG NETWORK ANALYSES")
        print("="*80)
        
        # 2.1: Policy impact network
        create_policy_impact_network(G, df_merged)
        
        # 2.2: SDG co-occurrence network
        create_sdg_cooccurrence_network(G, df_merged)
        
        # 2.3: Centrality comparison
        create_centrality_comparison(G, df_merged)
        
        # =====================================================================
        # TASK 3: Comprehensive Diagnostic File (14_*.xlsx)
        # =====================================================================
        print("\n" + "="*80)
        print("TASK 3: COMPREHENSIVE DIAGNOSTIC FILE")
        print("="*80)
        
        create_comprehensive_diagnostic(G, df_corpus)
        
        # Final summary
        print("\n" + "="*80)
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print(f"\n📁 Outputs Created:")
        print(f"\nTask 1 - Enhanced Networks (15_*.png):")
        print(f"  • 15_network_medium_filter.png")
        print(f"  • 15_network_strong_filter.png")
        print(f"  • 15_network_very_strong_filter.png")
        
        print(f"\nTask 2 - Policy/SDG Analyses (16_*.png):")
        print(f"  • 16_1_policy_impact_network.png")
        print(f"  • 16_2_sdg_cooccurrence_network.png")
        print(f"  • 16_3_centrality_comparison.png")
        
        print(f"\nTask 3 - Diagnostic File (14_*.xlsx):")
        print(f"  • 14_network_policy_sdg_diagnostic.xlsx")
        
        print(f"\n💡 Key Insights to Explore:")
        print(f"  1. Do policy-cited papers cluster in specific communities?")
        print(f"  2. Which SDGs are connected through co-citations?")
        print(f"  3. Are policy-relevant papers more central in the network?")
        print(f"  4. Do certain communities have higher policy impact?")
        
        print("\n" + "="*80)
        print("Analysis complete! Files saved to:")
        print(f"  Images: {OUTPUT_IMAGES}")
        print(f"  Excel: {OUTPUT_EXCEL}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    main()