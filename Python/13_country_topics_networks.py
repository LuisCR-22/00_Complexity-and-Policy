"""
==============================================================================
ENHANCED NETWORK ANALYSIS - INSTITUTIONS, COUNTRIES, AND TOPICS
==============================================================================

This script creates enhanced versions of collaboration networks with:
- 17_: Country collaboration networks (enhanced visualization)
- 18_: Institution collaboration networks  
- 19_: Topic co-occurrence networks weighted by policy citations

Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Global Complexity School 2025 Final Project
Date: November 2025
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\04_OA_Altmetrics_merge.xlsx"
OUTPUT_IMAGES = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

os.makedirs(OUTPUT_IMAGES, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

PLOT_SETTINGS = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 12,  # Increased from 11
    'axes.labelsize': 14,  # Increased from 13
    'axes.titlesize': 16,  # Increased from 15
    'legend.fontsize': 14,  # Increased from 13
}

for key, value in PLOT_SETTINGS.items():
    plt.rcParams[key] = value

# Darker, more distinguishable colors
COLORS = {
    'network_high': '#B71C1C',    # Darker red
    'network_mid': '#E65100',     # Darker orange  
    'network_low': '#616161',     # Darker gray
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print(" "*20 + "ENHANCED NETWORK ANALYSIS")
print("="*80)
print(f"\n📂 Loading data from:\n   {INPUT_PATH}\n")

df = pd.read_excel(INPUT_PATH)
print(f"✓ Loaded {len(df):,} papers")

# Ensure policy citations column exists
if 'policy_mentions_total_alt' not in df.columns:
    df['policy_mentions_total_alt'] = 0
else:
    df['policy_mentions_total_alt'] = df['policy_mentions_total_alt'].fillna(0)

# ============================================================================
# SECTION 1: ENHANCED COUNTRY COLLABORATION NETWORKS (17_)
# ============================================================================

def plot_enhanced_country_network(country_counter, collab_counter, title, filename):
    """
    Create enhanced country collaboration network with bigger nodes/labels
    and darker colors
    """
    print(f"\n   Creating enhanced network: {title}")
    
    # Select top countries
    top_countries = [c[0] for c in country_counter.most_common(40)]
    
    # Create network
    G = nx.Graph()
    G.add_nodes_from(top_countries)
    
    # Add edges
    for (c1, c2), weight in collab_counter.items():
        if c1 in top_countries and c2 in top_countries and weight >= 2:
            G.add_edge(c1, c2, weight=weight)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes()) == 0:
        print(f"   ⚠ No network to plot for {title}")
        return
    
    print(f"   • {title}: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Calculate betweenness centrality
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Node sizes - INCREASED (multiply by 8 instead of 5)
    node_sizes = []
    for node in G.nodes():
        count = country_counter[node]
        size = 200 + (count / max(country_counter.values())) * 4000  # Increased from 100-3000
        node_sizes.append(size)
    
    # Node colors - DARKER
    node_colors = []
    centrality_values = list(betweenness_cent.values())
    if centrality_values:
        threshold_high = np.percentile(centrality_values, 90)
        threshold_med = np.percentile(centrality_values, 75)
        
        for node in G.nodes():
            cent = betweenness_cent[node]
            if cent >= threshold_high:
                node_colors.append(COLORS['network_high'])
            elif cent >= threshold_med:
                node_colors.append(COLORS['network_mid'])
            else:
                node_colors.append(COLORS['network_low'])
    else:
        node_colors = [COLORS['network_low']] * len(G.nodes())
    
    # Edge widths
    edge_widths = []
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    for u, v in G.edges():
        weight = G[u][v]['weight']
        width = 1.0 + (weight / max_weight) * 9.0  # Increased from 0.5-8
        edge_widths.append(width)
    
    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, weight='weight')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(26, 26))  # Increased from 24,24
    
    # Draw edges - DARKER
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,  # Increased alpha from 0.3
                          edge_color='#424242', ax=ax)  # Darker gray
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          edgecolors='black', linewidths=2.5, alpha=0.9, ax=ax)
    
    # Labels - BIGGER
    degree_cent = nx.degree_centrality(G)
    for node, (x, y) in pos.items():
        size = 12 + degree_cent[node] * 28  # Increased from 8-20
        ax.text(x, y, node, fontsize=size, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor='black', alpha=0.85, linewidth=1.5))
    
    # Title
    title_text = f'{title}\nInternational Collaboration Network - Complexity Science'
    ax.set_title(title_text, fontsize=20, fontweight='bold', pad=30)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_high'],
              markersize=18, label='High Betweenness Centrality (Top 10%)',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_mid'],
              markersize=18, label='Medium Betweenness Centrality (Top 25%)',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_low'],
              markersize=18, label='Lower Betweenness Centrality',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], color='#424242', linewidth=6, alpha=0.6, label='Strong Collaboration'),
        Line2D([0], [0], color='#424242', linewidth=2, alpha=0.6, label='Weak Collaboration')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
             frameon=True, fancybox=True, shadow=True, title='Legend',
             title_fontsize=16)
    
    # Explanation with betweenness centrality info
    explanation = (
        "Node Size = Total Papers\n"
        "Node Color = Betweenness Centrality\n"
        "Edge Width = Collaboration Strength\n"
        f"Network: {len(G.nodes())} countries, {len(G.edges())} collaborations\n\n"
        "Betweenness Centrality measures how often a country\n"
        "acts as a bridge between other countries.\n"
        "High betweenness = crucial connector/broker role."
    )
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_IMAGES}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")

def analyze_enhanced_countries():
    """Analyze country patterns with enhanced visualization"""
    print("\n" + "="*80)
    print("[1/3] ENHANCED COUNTRY COLLABORATION NETWORKS (17_)")
    print("="*80)
    
    all_countries = []
    policy_countries = []
    academic_countries = []
    
    all_collabs = []
    policy_collabs = []
    academic_collabs = []
    
    for idx, row in df.iterrows():
        country_str = row.get('countries_oa', '')
        if pd.notna(country_str) and country_str != '':
            countries = list(set([c.strip() for c in str(country_str).split(';') if c.strip()]))
            all_countries.extend(countries)
            
            # Track collaborations
            if len(countries) > 1:
                for i in range(len(countries)):
                    for j in range(i+1, len(countries)):
                        collab = tuple(sorted([countries[i], countries[j]]))
                        all_collabs.append(collab)
                        
                        if row.get('policy_mentions_total_alt', 0) > 0:
                            policy_collabs.append(collab)
                        
                        if row.get('cited_by_count_oa', 0) > 0:
                            academic_collabs.append(collab)
            
            if row.get('policy_mentions_total_alt', 0) > 0:
                policy_countries.extend(countries)
            
            if row.get('cited_by_count_oa', 0) > 0:
                academic_countries.extend(countries)
    
    country_counter = Counter(all_countries)
    policy_counter = Counter(policy_countries)
    academic_counter = Counter(academic_countries)
    collab_counter = Counter(all_collabs)
    policy_collab_counter = Counter(policy_collabs)
    academic_collab_counter = Counter(academic_collabs)
    
    print(f"\n   • Total countries: {len(country_counter)}")
    print(f"   • Total collaborations: {len(collab_counter)}")
    
    # Create networks
    plot_enhanced_country_network(
        country_counter, collab_counter,
        'Full Network - Enhanced', '17_collaboration_network_full.png')
    
    plot_enhanced_country_network(
        policy_counter, policy_collab_counter,
        'Policy Impact Network - Enhanced', '17_collaboration_network_policy.png')
    
    plot_enhanced_country_network(
        academic_counter, academic_collab_counter,
        'Academic Impact Network - Enhanced', '17_collaboration_network_academic.png')
    
    print("\n   ✓ Enhanced country networks complete")
    print("\n   📖 BETWEENNESS CENTRALITY INTERPRETATION:")
    print("   " + "="*70)
    print("   Betweenness centrality measures how often a node lies on the")
    print("   shortest path between two other nodes in the network.")
    print()
    print("   High betweenness centrality means:")
    print("   • The country is a BRIDGE connecting different research clusters")
    print("   • It has BROKERAGE POWER in the collaboration network")
    print("   • Removing it would DISCONNECT parts of the network")
    print()
    print("   Example: If USA has high betweenness, it means USA often")
    print("   connects researchers from different countries who wouldn't")
    print("   otherwise collaborate directly.")
    print("   " + "="*70)

# ============================================================================
# SECTION 2: INSTITUTION COLLABORATION NETWORKS (18_)
# ============================================================================

def plot_institution_network(inst_counter, collab_counter, title, filename):
    """Create institution collaboration network"""
    print(f"\n   Creating network: {title}")
    
    # Select top institutions
    top_insts = [inst[0] for inst in inst_counter.most_common(40)]
    
    # Create network
    G = nx.Graph()
    G.add_nodes_from(top_insts)
    
    # Add edges
    for (i1, i2), weight in collab_counter.items():
        if i1 in top_insts and i2 in top_insts and weight >= 2:
            G.add_edge(i1, i2, weight=weight)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes()) == 0:
        print(f"   ⚠ No network to plot for {title}")
        return
    
    print(f"   • {title}: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    # Calculate betweenness centrality
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Node sizes
    node_sizes = []
    for node in G.nodes():
        count = inst_counter[node]
        size = 200 + (count / max(inst_counter.values())) * 4000
        node_sizes.append(size)
    
    # Node colors
    node_colors = []
    centrality_values = list(betweenness_cent.values())
    if centrality_values:
        threshold_high = np.percentile(centrality_values, 90)
        threshold_med = np.percentile(centrality_values, 75)
        
        for node in G.nodes():
            cent = betweenness_cent[node]
            if cent >= threshold_high:
                node_colors.append(COLORS['network_high'])
            elif cent >= threshold_med:
                node_colors.append(COLORS['network_mid'])
            else:
                node_colors.append(COLORS['network_low'])
    else:
        node_colors = [COLORS['network_low']] * len(G.nodes())
    
    # Edge widths
    edge_widths = []
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    for u, v in G.edges():
        weight = G[u][v]['weight']
        width = 1.0 + (weight / max_weight) * 9.0
        edge_widths.append(width)
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42, weight='weight')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 28))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                          edge_color='#424242', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          edgecolors='black', linewidths=2.5, alpha=0.9, ax=ax)
    
    # Labels - adjusted for longer institution names
    degree_cent = nx.degree_centrality(G)
    for node, (x, y) in pos.items():
        # Truncate long names
        label = node if len(node) <= 30 else node[:27] + "..."
        size = 10 + degree_cent[node] * 20
        ax.text(x, y, label, fontsize=size, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='black', alpha=0.85, linewidth=1.5))
    
    # Title
    title_text = f'{title}\nInstitutional Collaboration Network - Complexity Science'
    ax.set_title(title_text, fontsize=20, fontweight='bold', pad=30)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_high'],
              markersize=18, label='High Betweenness Centrality (Top 10%)',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_mid'],
              markersize=18, label='Medium Betweenness Centrality (Top 25%)',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_low'],
              markersize=18, label='Lower Betweenness Centrality',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], color='#424242', linewidth=6, alpha=0.6, label='Strong Collaboration'),
        Line2D([0], [0], color='#424242', linewidth=2, alpha=0.6, label='Weak Collaboration')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14,
             frameon=True, fancybox=True, shadow=True, title='Legend',
             title_fontsize=16)
    
    # Explanation
    explanation = (
        "Node Size = Total Papers\n"
        "Node Color = Betweenness Centrality\n"
        "Edge Width = Collaboration Strength\n"
        f"Network: {len(G.nodes())} institutions, {len(G.edges())} collaborations\n\n"
        "Betweenness Centrality: Measures institutional\n"
        "brokerage power in connecting different research groups."
    )
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_IMAGES}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")

def analyze_institutions():
    """Analyze institution collaboration patterns"""
    print("\n" + "="*80)
    print("[2/3] INSTITUTION COLLABORATION NETWORKS (18_)")
    print("="*80)
    
    all_institutions = []
    policy_institutions = []
    academic_institutions = []
    
    all_collabs = []
    policy_collabs = []
    academic_collabs = []
    
    for idx, row in df.iterrows():
        inst_str = row.get('institutions_oa', '')
        if pd.notna(inst_str) and inst_str != '':
            institutions = list(set([i.strip() for i in str(inst_str).split(';') if i.strip()]))
            all_institutions.extend(institutions)
            
            # Track collaborations
            if len(institutions) > 1:
                for i in range(len(institutions)):
                    for j in range(i+1, len(institutions)):
                        collab = tuple(sorted([institutions[i], institutions[j]]))
                        all_collabs.append(collab)
                        
                        if row.get('policy_mentions_total_alt', 0) > 0:
                            policy_collabs.append(collab)
                        
                        if row.get('cited_by_count_oa', 0) > 0:
                            academic_collabs.append(collab)
            
            if row.get('policy_mentions_total_alt', 0) > 0:
                policy_institutions.extend(institutions)
            
            if row.get('cited_by_count_oa', 0) > 0:
                academic_institutions.extend(institutions)
    
    inst_counter = Counter(all_institutions)
    policy_counter = Counter(policy_institutions)
    academic_counter = Counter(academic_institutions)
    collab_counter = Counter(all_collabs)
    policy_collab_counter = Counter(policy_collabs)
    academic_collab_counter = Counter(academic_collabs)
    
    print(f"\n   • Total institutions: {len(inst_counter)}")
    print(f"   • Total collaborations: {len(collab_counter)}")
    
    # Create networks
    plot_institution_network(
        inst_counter, collab_counter,
        'Full Network', '18_collaboration_network_full.png')
    
    plot_institution_network(
        policy_counter, policy_collab_counter,
        'Policy Impact Network', '18_collaboration_network_policy.png')
    
    plot_institution_network(
        academic_counter, academic_collab_counter,
        'Academic Impact Network', '18_collaboration_network_academic.png')
    
    print("\n   ✓ Institution networks complete")

# ============================================================================
# SECTION 3: TOPIC CO-OCCURRENCE NETWORKS BY POLICY CITATIONS (19_)
# ============================================================================

def plot_topic_network(topic_policy_citations, topic_cooccurrence, title, filename):
    """
    Create topic co-occurrence network weighted by policy citations
    
    Args:
        topic_policy_citations: Counter of total policy citations per topic
        topic_cooccurrence: Counter of topic pairs (co-occurrence)
        title: Plot title
        filename: Output filename
    """
    print(f"\n   Creating topic network: {title}")
    
    # Select top topics by policy citations
    top_topics = [topic[0] for topic in topic_policy_citations.most_common(50)]
    
    # Create network
    G = nx.Graph()
    G.add_nodes_from(top_topics)
    
    # Add edges (co-occurrences)
    for (t1, t2), weight in topic_cooccurrence.items():
        if t1 in top_topics and t2 in top_topics and weight >= 3:  # Threshold
            G.add_edge(t1, t2, weight=weight)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes()) == 0:
        print(f"   ⚠ No network to plot for {title}")
        return
    
    print(f"   • {title}: {len(G.nodes())} topics, {len(G.edges())} connections")
    
    # Detect communities
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        print(f"   • Detected {len(communities)} topic communities")
    except:
        communities = []
    
    # Assign colors to communities
    if len(communities) > 0:
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = []
        for node in G.nodes():
            for idx, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(community_colors[idx])
                    break
    else:
        node_colors = ['#3498DB'] * len(G.nodes())
    
    # Node sizes based on policy citations
    node_sizes = []
    max_citations = max(topic_policy_citations.values()) if topic_policy_citations else 1
    for node in G.nodes():
        citations = topic_policy_citations.get(node, 0)
        size = 300 + (citations / max_citations) * 5000  # Large range for visibility
        node_sizes.append(size)
    
    # Edge widths
    edge_widths = []
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    for u, v in G.edges():
        weight = G[u][v]['weight']
        width = 0.5 + (weight / max_weight) * 8.0
        edge_widths.append(width)
    
    # Layout
    pos = nx.spring_layout(G, k=3.5, iterations=100, seed=42, weight='weight')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(30, 30))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                          edge_color='#757575', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          edgecolors='black', linewidths=2, alpha=0.85, ax=ax)
    
    # Labels - ALL TOPICS LABELED
    for node, (x, y) in pos.items():
        # Truncate very long topic names
        label = node if len(node) <= 40 else node[:37] + "..."
        
        # Size based on policy citations
        citations = topic_policy_citations.get(node, 0)
        size = 10 + (citations / max_citations) * 18
        
        ax.text(x, y, label, fontsize=size, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='black', alpha=0.9, linewidth=1.5))
    
    # Title
    title_text = f'{title}\nTopic Co-occurrence Network - Weighted by Policy Citations'
    ax.set_title(title_text, fontsize=22, fontweight='bold', pad=30)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
              markersize=20, label='Topic (color = community)',
              markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], color='#757575', linewidth=6, alpha=0.5,
              label='Strong Co-occurrence'),
        Line2D([0], [0], color='#757575', linewidth=2, alpha=0.5,
              label='Weak Co-occurrence'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=16,
             frameon=True, fancybox=True, shadow=True, title='Legend',
             title_fontsize=18)
    
    # Explanation
    explanation = (
        f"Node Size = Total Policy Citations for Topic\n"
        f"Node Color = Topic Community (Louvain algorithm)\n"
        f"Edge Width = Papers where topics co-occur\n"
        f"Label Size = Proportional to policy citations\n\n"
        f"Network: {len(G.nodes())} topics, {len(G.edges())} co-occurrences\n"
        f"Communities detected: {len(communities)}\n\n"
        f"Interpretation: Topics connected by edges appear\n"
        f"together in papers. Larger nodes = more policy impact.\n"
        f"Clustered topics = frequently studied together."
    )
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
           fontsize=13, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.92, pad=0.9))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_IMAGES}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")

def analyze_topics():
    """Analyze topic co-occurrence patterns weighted by policy citations"""
    print("\n" + "="*80)
    print("[3/3] TOPIC CO-OCCURRENCE NETWORKS (19_)")
    print("="*80)
    
    # Collect topic data
    topic_policy_citations = Counter()  # Total policy citations per topic
    topic_cooccurrence = Counter()  # Count of papers where topics co-occur
    
    topic_papers = defaultdict(list)  # topic -> list of paper indices
    
    for idx, row in df.iterrows():
        policy_cites = row.get('policy_mentions_total_alt', 0)
        
        # Get primary topic
        topic = row.get('primary_topic_oa', '')
        
        if pd.notna(topic) and topic != '':
            topic = str(topic).strip()
            
            # Track policy citations
            topic_policy_citations[topic] += policy_cites
            
            # Track which papers have this topic
            topic_papers[topic].append(idx)
    
    # Calculate co-occurrence (topics appearing together in papers)
    # For this, we need to identify which papers share multiple topics
    # Since we only have primary_topic_oa, we'll use a different approach:
    # Topics co-occur when papers from the same subfield/field cite each other
    
    # Alternative: Use secondary and tertiary topics if available
    for idx, row in df.iterrows():
        topics_in_paper = []
        
        for col in ['primary_topic_oa', 'secondary_topic_oa', 'tertiary_topic_oa']:
            if col in df.columns:
                topic = row.get(col, '')
                if pd.notna(topic) and topic != '':
                    topics_in_paper.append(str(topic).strip())
        
        # Remove duplicates
        topics_in_paper = list(set(topics_in_paper))
        
        # Create co-occurrence edges
        if len(topics_in_paper) > 1:
            for i in range(len(topics_in_paper)):
                for j in range(i+1, len(topics_in_paper)):
                    edge = tuple(sorted([topics_in_paper[i], topics_in_paper[j]]))
                    topic_cooccurrence[edge] += 1
    
    print(f"\n   • Total unique topics: {len(topic_policy_citations)}")
    print(f"   • Topic co-occurrences: {len(topic_cooccurrence)}")
    print(f"   • Total policy citations across all topics: {sum(topic_policy_citations.values()):,.0f}")
    
    # Create networks
    
    # Full network (all papers)
    plot_topic_network(
        topic_policy_citations, topic_cooccurrence,
        'Full Topic Network', '19_topic_network_full.png')
    
    # Network for papers with policy citations only
    topic_policy_only = Counter()
    topic_cooccur_policy = Counter()
    
    for idx, row in df.iterrows():
        if row.get('policy_mentions_total_alt', 0) > 0:
            policy_cites = row.get('policy_mentions_total_alt', 0)
            
            topics_in_paper = []
            for col in ['primary_topic_oa', 'secondary_topic_oa', 'tertiary_topic_oa']:
                if col in df.columns:
                    topic = row.get(col, '')
                    if pd.notna(topic) and topic != '':
                        topics_in_paper.append(str(topic).strip())
            
            topics_in_paper = list(set(topics_in_paper))
            
            for topic in topics_in_paper:
                topic_policy_only[topic] += policy_cites
            
            if len(topics_in_paper) > 1:
                for i in range(len(topics_in_paper)):
                    for j in range(i+1, len(topics_in_paper)):
                        edge = tuple(sorted([topics_in_paper[i], topics_in_paper[j]]))
                        topic_cooccur_policy[edge] += 1
    
    plot_topic_network(
        topic_policy_only, topic_cooccur_policy,
        'Policy Papers Topic Network', '19_topic_network_policy.png')
    
    # Print top topics by policy citations
    print(f"\n   📊 Top 10 Topics by Policy Citations:")
    for topic, citations in topic_policy_citations.most_common(10):
        print(f"      • {topic}: {citations:,.0f} policy citations")
    
    print("\n   ✓ Topic networks complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute all network analyses"""
    print("\n" + "="*80)
    print(" "*15 + "STARTING ENHANCED NETWORK ANALYSIS")
    print("="*80)
    
    try:
        # Section 1: Enhanced country networks
        analyze_enhanced_countries()
        
        # Section 2: Institution networks
        analyze_institutions()
        
        # Section 3: Topic networks
        analyze_topics()
        
        print("\n" + "="*80)
        print(" "*25 + "✓ ALL ANALYSES COMPLETE")
        print("="*80)
        print(f"\n📁 All outputs saved to: {OUTPUT_IMAGES}")
        print("\nGenerated files:")
        print("   17_collaboration_network_full.png")
        print("   17_collaboration_network_policy.png")
        print("   17_collaboration_network_academic.png")
        print("   18_collaboration_network_full.png")
        print("   18_collaboration_network_policy.png")
        print("   18_collaboration_network_academic.png")
        print("   19_topic_network_full.png")
        print("   19_topic_network_policy.png")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()