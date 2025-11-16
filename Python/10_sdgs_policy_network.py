"""
==============================================================================
FIXED CITATION NETWORK FOR SDG ANALYSIS - IMPROVED VISUALIZATION
==============================================================================

This script creates a citation network with larger SDG labels and explains
the node placement algorithm.

Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Global Complexity School 2025 Final Project
Date: November 2025
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\04_OA_Altmetrics_merge.xlsx"
OUTPUT_IMAGES = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

# Ensure output directory exists
os.makedirs(OUTPUT_IMAGES, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')

PLOT_SETTINGS = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 13  # Increased from 10 to 13
}

for key, value in PLOT_SETTINGS.items():
    plt.rcParams[key] = value

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print(" "*20 + "FIXED SDG CITATION NETWORK")
print("="*80)
print(f"\n📂 Loading data from:\n   {INPUT_PATH}\n")

df = pd.read_excel(INPUT_PATH)
print(f"✓ Loaded {len(df):,} papers")

# Filter papers with SDG information
df_sdg = df[df['sdg_name_oa'].notna() & (df['sdg_name_oa'] != '')].copy()
print(f"✓ Papers with SDG: {len(df_sdg):,}")

# ============================================================================
# CREATE CITATION NETWORK - FIXED VERSION
# ============================================================================

def create_fixed_citation_network(df_sdg):
    """
    Create citation network including ALL papers with SDG info,
    but highlighting those with policy citations.
    """
    print("="*80)
    print("CREATING FIXED CITATION NETWORK")
    print("="*80)
    
    # Use id_oa as identifier
    if 'id_oa' in df_sdg.columns:
        id_col = 'id_oa'
    elif 'openalex_id_oa' in df_sdg.columns:
        id_col = 'openalex_id_oa'
    else:
        print("⚠ Error: No OpenAlex ID column found")
        return
    
    print(f"Using '{id_col}' as node identifier")
    
    print(f"Total papers with SDG info: {len(df_sdg):,}")
    papers_with_policy = len(df_sdg[df_sdg['policy_mentions_total_alt'].fillna(0) > 0])
    print(f"Papers with policy citations: {papers_with_policy:,}")
    
    # Create set of all paper IDs in corpus for fast lookup
    paper_ids = set(df_sdg[id_col].dropna().astype(str).str.strip())
    print(f"Total unique paper IDs: {len(paper_ids)}")
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes with attributes - ALL papers with SDG
    print("Adding nodes to network...")
    node_count = 0
    for idx, row in df_sdg.iterrows():
        node_id = str(row[id_col]).strip()
        
        if node_id and node_id != 'nan':
            policy_cites = float(row.get('policy_mentions_total_alt', 0))
            if pd.isna(policy_cites):
                policy_cites = 0.0
                
            G.add_node(node_id,
                      policy_citations=policy_cites,
                      sdg=str(row.get('sdg_name_oa', 'Unknown')),
                      title=str(row.get('title_oa', row.get('title_alt', 'No title')))[:80],
                      year=str(row.get('publication_year', '')))
            node_count += 1
    
    print(f"✓ Added {node_count} nodes to network")
    
    # Check referenced_works column
    if 'referenced_works_oa' not in df_sdg.columns:
        print("⚠ Error: 'referenced_works_oa' column not found")
        return
    
    print("Processing citation relationships...")
    
    # Add edges
    edges_added = 0
    papers_with_refs = 0
    total_refs_found = 0
    matched_refs = 0
    
    for idx, row in df_sdg.iterrows():
        source_id = str(row[id_col]).strip()
        
        if source_id not in G.nodes():
            continue
        
        refs_raw = row.get('referenced_works_oa', '')
        
        if pd.notna(refs_raw) and str(refs_raw).strip() and str(refs_raw) != 'nan':
            papers_with_refs += 1
            
            # Split by semicolon
            refs_str = str(refs_raw).strip()
            ref_list = [r.strip() for r in refs_str.split(';') if r.strip()]
            
            total_refs_found += len(ref_list)
            
            # Check each reference
            for ref_id in ref_list:
                ref_id = ref_id.strip()
                
                if ref_id in paper_ids and ref_id in G.nodes():
                    G.add_edge(source_id, ref_id)
                    edges_added += 1
                    matched_refs += 1
    
    print(f"\n📊 Citation Network Statistics:")
    print(f"  • Papers with references: {papers_with_refs:,} ({papers_with_refs/len(df_sdg)*100:.1f}%)")
    print(f"  • Total references found: {total_refs_found:,}")
    print(f"  • References matching corpus papers: {matched_refs:,}")
    print(f"  • Citation edges added: {edges_added:,}")
    
    if edges_added == 0:
        print("\n⚠ WARNING: No citation edges found!")
        return
    
    # Remove isolated nodes
    if edges_added > 0:
        nodes_with_edges = set()
        for edge in G.edges():
            nodes_with_edges.add(edge[0])
            nodes_with_edges.add(edge[1])
        
        isolated_nodes = set(G.nodes()) - nodes_with_edges
        G.remove_nodes_from(isolated_nodes)
        print(f"✓ Removed {len(isolated_nodes)} isolated nodes")
        print(f"✓ Final network: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    if len(G.nodes()) == 0:
        print("⚠ No nodes in network after filtering. Skipping visualization.")
        return
    
    # Prepare visualization
    print("Creating network visualization...")
    print("\n📐 Layout Algorithm Explanation:")
    print("   Using Spring/Force-Directed Layout (ForceAtlas2 style):")
    print("   • Nodes REPEL each other (like magnets with same poles)")
    print("   • Edges ACT AS SPRINGS pulling connected nodes together")
    print("   • Algorithm iterates 50 times to reach equilibrium")
    print("   • Result: Highly connected papers cluster together")
    print("   •         Loosely connected papers spread apart")
    print("   • Papers citing each other → positioned close")
    print("   • Papers in different research areas → pushed apart\n")
    
    # Get unique SDGs and create color map
    sdgs_in_network = set(G.nodes[node]['sdg'] for node in G.nodes())
    sdg_list = sorted(sdgs_in_network)
    colors_palette = plt.cm.tab20(np.linspace(0, 1, min(len(sdg_list), 20)))
    sdg_color_map = {sdg: colors_palette[i] for i, sdg in enumerate(sdg_list)}
    
    # Calculate node sizes based on policy citations (log scale)
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        policy_cites = G.nodes[node].get('policy_citations', 0)
        sdg = G.nodes[node].get('sdg', 'Unknown')
        
        # Size: logarithmic scale, with minimum size for 0 citations
        if policy_cites > 0:
            size = 100 + np.log1p(policy_cites) * 150
        else:
            size = 50  # Smaller size for papers without policy citations
        node_sizes.append(size)
        
        # Color by SDG
        node_colors.append(sdg_color_map.get(sdg, (0.5, 0.5, 0.5, 1.0)))
    
    # Calculate layout - Spring-based force-directed
    print("Computing layout (this may take a moment for large networks)...")
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.kamada_kawai_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 24))
    
    # Draw edges
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                      alpha=0.5,          # CHANGE: from 0.2 to 0.5 (darker edges)
                      width=1.0,          # CHANGE: from 0.5 to 1.0 (thicker edges)
                      edge_color='darkgray',  # CHANGE: from 'gray' to 'darkgray'
                      arrows=True,
                      arrowsize=15,       # CHANGE: from 10 to 15 (larger arrows)
                      arrowstyle='->',
                      connectionstyle='arc3,rad=0.1',
                      ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors, 
                          alpha=0.75,
                          edgecolors='black', 
                          linewidths=1.2, 
                          ax=ax)
    
    # Create legend for SDGs with BIGGER font
    from matplotlib.patches import Patch
    
    if len(sdg_list) <= 20:
        legend_elements = [Patch(facecolor=sdg_color_map[sdg], 
                                edgecolor='black', 
                                label=sdg, linewidth=1)
                          for sdg in sorted(sdg_list)]
    else:
        sdg_counts = Counter(G.nodes[node]['sdg'] for node in G.nodes())
        top_sdgs = [sdg for sdg, count in sdg_counts.most_common(10)]
        legend_elements = [Patch(facecolor=sdg_color_map[sdg], 
                                edgecolor='black', 
                                label=sdg, linewidth=1)
                          for sdg in sorted(top_sdgs)]
        legend_elements.append(Patch(facecolor='white', 
                                     edgecolor='black',
                                     label=f'... and {len(sdg_list)-10} more', 
                                     linewidth=1))
    
    # BIGGER LEGEND: increased fontsize from 11 to 14
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             fontsize=18,  # INCREASED from 11
             title='SDGs', 
             title_fontsize=16,  # INCREASED from 13
             frameon=True, 
             fancybox=True, 
             shadow=True,
             ncol=2 if len(legend_elements) > 10 else 1)
    
    # Title
    title_text = ('Citation Network: Complexity Science Papers with SDG Classification\n' +
                 'Node size = Policy citations (smaller nodes = 0 citations) | ' +
                 'Node color = SDG | Edges = Citation relationships')
    
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add statistics with layout explanation
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    
    # Count nodes with and without policy citations
    nodes_with_policy = sum(1 for node in G.nodes() if G.nodes[node].get('policy_citations', 0) > 0)
    nodes_without_policy = len(G.nodes()) - nodes_with_policy
    
    stats_text = (f"Network Statistics:\n"
                 f"Total Nodes: {len(G.nodes()):,}\n"
                 f"  • With policy citations: {nodes_with_policy:,}\n"
                 f"  • Without policy citations: {nodes_without_policy:,}\n"
                 f"Edges: {len(G.edges()):,}\n"
                 f"Density: {density:.4f}\n"
                 f"Avg Degree: {avg_degree:.2f}\n\n"
                 f"Layout Algorithm:\n"
                 f"Spring/Force-Directed (k=2, iter=50)\n"
                 f"• Nodes repel (like magnets)\n"
                 f"• Edges attract (like springs)\n"
                 f"• Connected papers cluster together\n"
                 f"• Different domains spread apart")
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.7))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_policy_network_sdg_color_FIXED.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  • Visualized {len(G.nodes())} papers")
    print(f"  • {len(sdgs_in_network)} SDGs represented")
    print(f"  • Citation edges: {len(G.edges())}")
    
    # Show which papers have the most incoming citations
    in_degrees = dict(G.in_degree())
    if in_degrees:
        top_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  📌 Most cited papers in network:")
        for node_id, in_deg in top_cited:
            if in_deg > 0:
                title = G.nodes[node_id].get('title', 'Unknown')[:60]
                policy = G.nodes[node_id].get('policy_citations', 0)
                sdg = G.nodes[node_id].get('sdg', 'Unknown')
                print(f"     • {title}...")
                print(f"       SDG: {sdg} | Cited by {in_deg} papers | Policy citations: {policy:.0f}")
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    create_fixed_citation_network(df_sdg)
    
    print("\n" + "="*80)
    print(" "*20 + "✓ NETWORK VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\n📁 Output saved to: {OUTPUT_IMAGES}")
    print(f"   File: 12_policy_network_sdg_color_FIXED.png\n")
    
    print("📐 NODE PLACEMENT LOGIC (Spring/Force-Directed Layout):")
    print("   " + "="*70)
    print("   The algorithm simulates a physical system:")
    print()
    print("   1. REPULSION: All nodes push away from each other")
    print("      → Like magnets with the same pole")
    print("      → Prevents nodes from overlapping")
    print("      → Creates space in the visualization")
    print()
    print("   2. ATTRACTION: Edges act as springs")
    print("      → Papers that cite each other are pulled together")
    print("      → Strength controlled by parameter k=2")
    print("      → Creates clusters of related papers")
    print()
    print("   3. ITERATION: Runs 50 times (iterations=50)")
    print("      → Each iteration adjusts positions slightly")
    print("      → Converges to stable equilibrium")
    print("      → Balances repulsion and attraction forces")
    print()
    print("   4. RESULT:")
    print("      ✓ Tightly connected papers → cluster together")
    print("      ✓ Weakly connected papers → pushed apart")
    print("      ✓ Papers from same SDG → often group (if they cite each other)")
    print("      ✓ Different research communities → separated in space")
    print()
    print("   This is why the visualization looks 'organic' - it reveals")
    print("   the natural community structure through citation patterns!")
    print("   " + "="*70)
    
    print("\n" + "="*80 + "\n")