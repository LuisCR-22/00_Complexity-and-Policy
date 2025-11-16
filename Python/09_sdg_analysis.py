"""
==============================================================================
SDG ANALYSIS - EXTENDED VERSION
==============================================================================

This script extends the SDG analysis from the main complexity science
citation analysis, focusing on:
1. Enhanced SDG distribution with policy citation percentage
2. Top papers per SDG in policy citations  
3. Scatter plot: SDG corpus share vs policy citation share
4. Citation network colored by SDG (requires reference column clarification)
5. SDG Impact Profile analysis

Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Global Complexity School 2025 Final Project
Date: November 2025

To run: use python 3.13 or below with required libraries installed. (3.14 has no wheel support yet for seborn)
For example run in the powershell: py -3.13 .../Github/Complexity-and-Policy/Python/09_sdg_analysis.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
OUTPUT_EXCEL = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel"

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_EXCEL, exist_ok=True)

# Plot settings for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

PLOT_SETTINGS = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
}

for key, value in PLOT_SETTINGS.items():
    plt.rcParams[key] = value

# Color scheme
COLORS = {
    'policy': '#E74C3C',      # Red
    'academic': '#3498DB',     # Blue  
    'media': '#9B59B6',        # Purple
    'sdg_high': '#27AE60',     # Green
    'sdg_medium': '#F39C12',   # Orange
    'sdg_low': '#95A5A6'       # Gray
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load the merged OpenAlex-Altmetric dataset and prepare SDG statistics
    
    Returns:
        df: Full dataset
        df_sdg: Papers with SDG information
        sdg_stats: Aggregated statistics by SDG
        total_policy_mentions: Total policy citations in corpus
    """
    print("="*80)
    print(" "*25 + "SDG ANALYSIS - EXTENDED VERSION")
    print("="*80)
    print(f"\n📂 Loading data from:\n   {INPUT_PATH}")
    
    df = pd.read_excel(INPUT_PATH)
    print(f"\n✓ Loaded {len(df):,} papers with {len(df.columns)} columns")
    
    # Filter papers with SDG information
    df_sdg = df[df['sdg_name_oa'].notna() & (df['sdg_name_oa'] != '')].copy()
    print(f"✓ Papers with SDG information: {len(df_sdg):,} ({len(df_sdg)/len(df)*100:.1f}%)")
    
    # Calculate total policy mentions in corpus
    total_policy_mentions = df['policy_mentions_total_alt'].fillna(0).sum()
    print(f"✓ Total policy mentions in corpus: {total_policy_mentions:,.0f}")
    
    # Calculate total media mentions (News + Blogs + Videos + Podcasts)
    media_cols = ['news_msm_total_alt', 'blogs_total_alt', 'video_mentions_alt', 'podcast_mentions_alt']
    df['total_media'] = sum(df[col].fillna(0) for col in media_cols if col in df.columns)
    total_media_mentions = df['total_media'].sum()
    print(f"✓ Total media mentions in corpus: {total_media_mentions:,.0f}")
    
    # Group by SDG and calculate statistics
    sdg_stats = df_sdg.groupby('sdg_name_oa').agg({
        'doi_oa': 'count',  # Number of papers
        'policy_mentions_total_alt': lambda x: x.fillna(0).sum(),  # Total policy citations
        'cited_by_count_oa': lambda x: x.fillna(0).sum(),  # Total academic citations
        'news_msm_total_alt': lambda x: x.fillna(0).sum()  # Total news mentions
    }).reset_index()
    
    # Add media total for each SDG
    df_sdg['total_media'] = sum(df_sdg[col].fillna(0) for col in media_cols if col in df_sdg.columns)
    media_by_sdg = df_sdg.groupby('sdg_name_oa')['total_media'].sum()
    
    sdg_stats.columns = ['SDG', 'Papers', 'Policy_Citations', 'Academic_Citations', 'News_Mentions']
    sdg_stats['Media_Mentions'] = sdg_stats['SDG'].map(media_by_sdg).fillna(0)
    
    # Calculate percentages
    sdg_stats['Corpus_Share_%'] = (sdg_stats['Papers'] / len(df) * 100).round(2)
    sdg_stats['Policy_Share_%'] = (sdg_stats['Policy_Citations'] / total_policy_mentions * 100).round(2) if total_policy_mentions > 0 else 0
    sdg_stats['Media_Share_%'] = (sdg_stats['Media_Mentions'] / total_media_mentions * 100).round(2) if total_media_mentions > 0 else 0
    
    # Sort by number of papers
    sdg_stats = sdg_stats.sort_values('Papers', ascending=True)
    
    print(f"✓ Analyzed {len(sdg_stats)} SDGs")
    print(f"✓ Data preparation complete\n")
    
    return df, df_sdg, sdg_stats, total_policy_mentions, total_media_mentions

# ============================================================================
# 1. ENHANCED SDG DISTRIBUTION GRAPH
# ============================================================================

def create_enhanced_distribution(sdg_stats, total_papers, total_policy_mentions, total_media_mentions):
    """
    Create enhanced SDG distribution with dual axis showing:
    - Primary axis: Number of papers (with % of corpus)
    - Secondary axis: % of total policy citations AND % of total media mentions
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        total_papers: Total number of papers in corpus
        total_policy_mentions: Total policy citations
        total_media_mentions: Total media mentions
    """
    print("="*80)
    print("[1/7] CREATING ENHANCED SDG DISTRIBUTION GRAPH")
    print("="*80)
    
    fig, ax1 = plt.subplots(figsize=(16, 12))
    
    # Primary axis - Number of papers
    y_pos = np.arange(len(sdg_stats))
    bars = ax1.barh(y_pos, sdg_stats['Papers'], 
                    color='steelblue', edgecolor='black', 
                    linewidth=0.8, alpha=0.7, label='Number of Papers')
    
    # Add labels with paper count and corpus share
    for idx, (papers, corpus_share) in enumerate(zip(sdg_stats['Papers'], sdg_stats['Corpus_Share_%'])):
        ax1.text(papers, idx, f' {papers} ({corpus_share}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sdg_stats['SDG'], fontsize=10)
    ax1.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
    ax1.set_title('SDG Distribution in Complexity Science Papers\n' +
                  'Bars: Paper count (% of corpus) | Points: % of total policy citations (red) & media mentions (purple)',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Secondary axis - Percentage of policy mentions and media mentions
    ax2 = ax1.twiny()
    
    # Policy points (red)
    policy_points = ax2.scatter(sdg_stats['Policy_Share_%'], y_pos, 
                         color=COLORS['policy'], s=150, 
                         edgecolors='black', linewidths=1.5,
                         alpha=0.8, zorder=5, label='% of Policy Citations')
    
    # Media points (purple)
    media_points = ax2.scatter(sdg_stats['Media_Share_%'], y_pos, 
                        color=COLORS['media'], s=150, 
                        edgecolors='black', linewidths=1.5,
                        alpha=0.8, zorder=5, label='% of Media Mentions')
    
    # Set axis properties
    max_val = max(sdg_stats['Policy_Share_%'].max(), sdg_stats['Media_Share_%'].max())
    ax2.set_xlim(0, max_val * 1.1)
    ax2.set_xlabel('Percentage of Total Citations/Mentions (%)', 
                  fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x')
    
    # Add legend - positioned at bottom right
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='steelblue', edgecolor='black', 
                      label='Papers (% of corpus)', alpha=0.7),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['policy'],
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='% of Policy Citations'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['media'],
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='% of Media Mentions')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add note with key statistics - moved to top left to avoid overlap
    papers_with_sdg = sdg_stats['Papers'].sum()
    note_text = (f"Dataset: {total_papers:,} papers | With SDG: {papers_with_sdg:,} ({papers_with_sdg/total_papers*100:.1f}%)\n"
                f"Policy citations: {total_policy_mentions:,.0f} | Media mentions: {total_media_mentions:,.0f}\n"
                f"Number of SDGs: {len(sdg_stats)}\n"
                f"Media = News + Blogs + Videos + Podcasts")
    ax1.text(0.02, 0.98, note_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_sdg_distribution_V2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  • Chart shows {len(sdg_stats)} SDGs")
    print(f"  • Triple metrics: papers (blue bars) + policy % (red) + media % (purple)\n")

# ============================================================================
# 2. TOP 2 PAPERS PER SDG IN POLICY CITATIONS
# ============================================================================

def extract_top_papers_per_sdg(df_sdg):
    """
    Extract top 2 papers per SDG by policy citations with full metadata
    
    Args:
        df_sdg: DataFrame with SDG papers
        
    Returns:
        DataFrame with top papers and their metadata
    """
    print("="*80)
    print("[2/7] EXTRACTING TOP PAPERS PER SDG BY POLICY CITATIONS")
    print("="*80)
    
    top_papers_list = []
    
    sdg_list = df_sdg['sdg_name_oa'].unique()
    print(f"Processing {len(sdg_list)} SDGs...")
    
    for sdg in sdg_list:
        sdg_papers = df_sdg[df_sdg['sdg_name_oa'] == sdg].copy()
        
        # Ensure policy mentions column exists and has numeric values
        if 'policy_mentions_total_alt' in sdg_papers.columns:
            sdg_papers['policy_mentions_total_alt'] = pd.to_numeric(
                sdg_papers['policy_mentions_total_alt'], errors='coerce').fillna(0)
        else:
            sdg_papers['policy_mentions_total_alt'] = 0
        
        # Sort by policy citations and get top 2
        top_2 = sdg_papers.nlargest(2, 'policy_mentions_total_alt')
        
        for idx, row in top_2.iterrows():
            paper_info = {
                'SDG': sdg,
                'Rank_in_SDG': len(top_2) - list(top_2.index).index(idx),  # 1 or 2
                
                # Identifiers
                'DOI_OA': row.get('doi_oa', ''),
                'DOI_Alt': row.get('doi_alt', ''),
                'OpenAlex_ID': row.get('id_oa', ''),
                
                # Basic metadata
                'Title_OA': row.get('title_oa', ''),
                'Title_Alt': row.get('title_alt', ''),
                'Year': row.get('year', ''),
                'Publication_Date_OA': row.get('publication_date_oa', ''),
                'Journal_OA': row.get('source_name_oa', ''),
                'Journal_Alt': row.get('journal_alt', ''),
                
                # Authors and affiliations
                'Authors': row.get('authors_names_oa', ''),
                'Authors_Count': row.get('authors_count_oa', ''),
                'Institutions': row.get('institutions_oa', ''),
                'Countries': row.get('countries_oa', ''),
                
                # Citation metrics
                'Policy_Citations': row.get('policy_mentions_total_alt', 0),
                'Academic_Citations_OA': row.get('cited_by_count_oa', 0),
                'Academic_Citations_Alt': row.get('citations_alt', 0),
                
                # Altmetric indicators
                'Altmetric_Score': row.get('altmetric_score_alt', 0),
                'News_MSM': row.get('news_msm_total_alt', 0),
                'Blogs': row.get('blogs_total_alt', 0),
                'Twitter': row.get('twitter_total_alt', 0),
                'Facebook': row.get('facebook_total_alt', 0),
                'Reddit': row.get('reddit_total_alt', 0),
                'Wikipedia': row.get('wikipedia_mentions_alt', 0),
                'Mendeley_Readers': row.get('readers_mendeley_alt', 0),
                'Peer_Reviews': row.get('peer_reviews_alt', 0),
                
                # Classification
                'Primary_Topic': row.get('primary_topic_oa', ''),
                'Primary_Subfield': row.get('primary_subfield_oa', ''),
                'Primary_Field': row.get('primary_field_oa', ''),
                'Primary_Domain': row.get('primary_domain_oa', ''),
                'SDG_Score': row.get('sdg_score_oa', ''),
                
                # Content
                'Keywords': row.get('keywords_all_oa', ''),
                'Concepts': row.get('concepts_oa', ''),
                'Abstract_OA': row.get('abstract_oa', ''),
                
                # Open Access
                'Is_OA': row.get('is_oa_oa', ''),
                'OA_Status': row.get('oa_status_oa', ''),
                'OA_URL': row.get('oa_url_oa', ''),
                
                # Quality metrics
                'FWCI': row.get('fwci_oa', ''),
                'Cited_by_Percentile': row.get('cited_by_percentile_year_oa', '')
            }
            top_papers_list.append(paper_info)
    
    top_papers_df = pd.DataFrame(top_papers_list)
    
    # Sort by SDG and rank
    top_papers_df = top_papers_df.sort_values(['SDG', 'Rank_in_SDG'])
    
    # Save to Excel
    excel_path = f"{OUTPUT_EXCEL}/12_sdg_top_papers.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main sheet with all data
        top_papers_df.to_excel(writer, sheet_name='All Top Papers', index=False)
        
        # Summary sheet
        summary_data = []
        for sdg in top_papers_df['SDG'].unique():
            sdg_papers = top_papers_df[top_papers_df['SDG'] == sdg]
            summary_data.append({
                'SDG': sdg,
                'Papers_Analyzed': len(sdg_papers),
                'Total_Policy_Citations': sdg_papers['Policy_Citations'].sum(),
                'Avg_Policy_Citations': sdg_papers['Policy_Citations'].mean(),
                'Total_Academic_Citations': sdg_papers['Academic_Citations_OA'].sum(),
                'Avg_Altmetric_Score': sdg_papers['Altmetric_Score'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary by SDG', index=False)
    
    print(f"✓ Saved: {excel_path}")
    print(f"  • Extracted top 2 papers for {len(sdg_list)} SDGs")
    print(f"  • Total papers in file: {len(top_papers_df)}")
    print(f"  • Includes {len(top_papers_df.columns)} metadata fields")
    print(f"  • Two sheets: 'All Top Papers' and 'Summary by SDG'\n")
    
    return top_papers_df

# ============================================================================
# 3. SCATTER PLOT: SDG CORPUS SHARE VS POLICY CITATION SHARE
# ============================================================================

def create_scatter_plot(sdg_stats, total_papers, total_policy_mentions):
    """
    Create scatter plot showing SDG representation in corpus vs policy citations
    Similar to the reference image provided
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        total_papers: Total papers in corpus
        total_policy_mentions: Total policy citations
    """
    print("="*80)
    print("[3/7] CREATING SCATTER PLOT: CORPUS SHARE VS POLICY SHARE")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create scatter plot with size based on total papers
    scatter = ax.scatter(sdg_stats['Corpus_Share_%'], 
                        sdg_stats['Policy_Share_%'],
                        s=sdg_stats['Papers']*3,  # Size proportional to papers
                        alpha=0.6, 
                        c=sdg_stats['Policy_Share_%'],  # Color by policy share
                        cmap='RdYlGn', 
                        edgecolors='black', 
                        linewidths=1.5,
                        zorder=3)
    
    # Add SDG labels with smart positioning
    texts = []
    for idx, row in sdg_stats.iterrows():
        # Offset text slightly to avoid overlap with points
        offset = 0.15
        ax.annotate(row['SDG'], 
                   (row['Corpus_Share_%'], row['Policy_Share_%']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', 
                            edgecolor='gray', 
                            alpha=0.8, linewidth=1))
    
    # Add diagonal reference line (equal representation)
    max_val = max(sdg_stats['Corpus_Share_%'].max(), 
                  sdg_stats['Policy_Share_%'].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 
           'r--', alpha=0.6, linewidth=2.5, 
           label='Equal representation\n(Corpus share = Policy share)',
           zorder=1)
    
    # Set axis limits with margins
    x_min, x_max = sdg_stats['Corpus_Share_%'].min(), sdg_stats['Corpus_Share_%'].max()
    y_min, y_max = sdg_stats['Policy_Share_%'].min(), sdg_stats['Policy_Share_%'].max()
    
    x_margin = (x_max - x_min) * 0.15
    y_margin = (y_max - y_min) * 0.15
    
    ax.set_xlim(max(0, x_min - x_margin), x_max + x_margin)
    ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
    
    # Labels and title
    ax.set_xlabel('Share in Complexity Science Corpus (%)', 
                 fontweight='bold', fontsize=13)
    ax.set_ylabel('Share in Policy Citations (%)', 
                 fontweight='bold', fontsize=13)
    ax.set_title('SDG Representation: Complexity Science Corpus vs Policy Impact\n' +
                'Point size = Number of papers | Color intensity = Policy citation share\n' +
                'Points above diagonal line are over-represented in policy citations',
                fontsize=15, fontweight='bold', pad=20)
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11, loc='upper left', frameon=True, 
             fancybox=True, shadow=True)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Policy Citation Share (%)', fontweight='bold', fontsize=11)
    
    # Add statistics note
    over_represented = sdg_stats[sdg_stats['Policy_Share_%'] > sdg_stats['Corpus_Share_%']]
    note = (f"Dataset: {total_papers:,} papers | SDGs: {len(sdg_stats)}\n"
           f"Policy citations: {total_policy_mentions:,.0f}\n"
           f"SDGs over-represented in policy: {len(over_represented)}/{len(sdg_stats)}")
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_sdg_policy_vs_corpus.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  • Plotted {len(sdg_stats)} SDGs")
    print(f"  • {len(over_represented)} SDGs over-represented in policy citations")
    
    # Print over-represented SDGs
    if len(over_represented) > 0:
        print(f"  • Top over-represented SDGs:")
        over_represented['Over_Rep_Ratio'] = over_represented['Policy_Share_%'] / over_represented['Corpus_Share_%']
        for idx, row in over_represented.nlargest(3, 'Over_Rep_Ratio').iterrows():
            print(f"    - {row['SDG']}: {row['Over_Rep_Ratio']:.2f}x")
    print()

# ============================================================================
# 4. CITATION NETWORK COLORED BY SDG
# ============================================================================

def create_citation_network(df_sdg):
    """
    Create citation network where:
    - Nodes = papers with policy citations AND SDG information
    - Edges = citation relationships (one paper references another within corpus)
    - Node size = policy citations
    - Node color = SDG
    
    Args:
        df_sdg: DataFrame with SDG papers
    """
    print("="*80)
    print("[4/7] CREATING CITATION NETWORK")
    print("="*80)
    
    # Filter papers with policy citations
    df_policy = df_sdg[df_sdg['policy_mentions_total_alt'].fillna(0) > 0].copy()
    print(f"Papers with policy citations: {len(df_policy):,}")
    
    # Create a mapping of OpenAlex IDs to make lookups faster
    # Use id_oa (or openalex_id_oa) as the primary identifier
    if 'id_oa' in df_policy.columns:
        id_col = 'id_oa'
    elif 'openalex_id_oa' in df_policy.columns:
        id_col = 'openalex_id_oa'
    else:
        print("⚠ Error: No OpenAlex ID column found")
        return
    
    print(f"Using '{id_col}' as node identifier")
    
    # Create set of all paper IDs in our corpus for fast lookup
    paper_ids = set(df_policy[id_col].dropna())
    print(f"Total unique paper IDs in policy corpus: {len(paper_ids)}")
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes with attributes
    print("Adding nodes to network...")
    node_count = 0
    for idx, row in df_policy.iterrows():
        node_id = str(row[id_col]).strip()
        
        if node_id and node_id != 'nan':
            G.add_node(node_id,
                      policy_citations=float(row.get('policy_mentions_total_alt', 0)),
                      sdg=str(row.get('sdg_name_oa', 'Unknown')),
                      title=str(row.get('title_oa', row.get('title_alt', 'No title')))[:80],
                      year=str(row.get('year', '')))
            node_count += 1
    
    print(f"✓ Added {node_count} nodes to network")
    
    # Check if referenced_works_oa column exists
    if 'referenced_works_oa' not in df_policy.columns:
        print("⚠ Error: 'referenced_works_oa' column not found")
        print(f"Available columns: {', '.join(df_policy.columns[:20])}...")
        return
    
    print(f"✓ Found 'referenced_works_oa' column")
    print("Processing citation relationships...")
    
    # Add edges
    edges_added = 0
    papers_with_refs = 0
    total_refs_found = 0
    matched_refs = 0
    
    for idx, row in df_policy.iterrows():
        source_id = str(row[id_col]).strip()
        
        if source_id not in G.nodes():
            continue
        
        refs_raw = row.get('referenced_works_oa', '')
        
        if pd.notna(refs_raw) and str(refs_raw).strip() and str(refs_raw) != 'nan':
            papers_with_refs += 1
            
            # Split by semicolon (with or without space)
            # Example: "https://openalex.org/W123; https://openalex.org/W456"
            refs_str = str(refs_raw).strip()
            ref_list = [r.strip() for r in refs_str.split(';') if r.strip()]
            
            total_refs_found += len(ref_list)
            
            # Check each reference
            for ref_id in ref_list:
                # Make sure it's a clean ID
                ref_id = ref_id.strip()
                
                # Check if this referenced paper is in our corpus
                if ref_id in paper_ids and ref_id in G.nodes():
                    G.add_edge(source_id, ref_id)
                    edges_added += 1
                    matched_refs += 1
    
    print(f"\n📊 Citation Network Statistics:")
    print(f"  • Papers with references: {papers_with_refs:,} ({papers_with_refs/len(df_policy)*100:.1f}%)")
    print(f"  • Total references found: {total_refs_found:,}")
    print(f"  • References matching corpus papers: {matched_refs:,}")
    print(f"  • Citation edges added: {edges_added:,}")
    
    if edges_added == 0:
        print("\n⚠ WARNING: No citation edges found!")
        print("  Possible reasons:")
        print("  1. Papers with policy citations don't cite each other within the corpus")
        print("  2. Referenced_works_oa format doesn't match id_oa format")
        print("  3. The corpus is from different time periods or fields")
        
        # Show sample for debugging
        sample_paper = df_policy.iloc[0]
        print(f"\n  Sample paper ID format: {sample_paper[id_col]}")
        sample_refs = str(sample_paper.get('referenced_works_oa', ''))[:200]
        print(f"  Sample refs format: {sample_refs}...")
        
        # Try to find if ANY references match
        print(f"\n  Checking first 10 papers for reference matches...")
        for i in range(min(10, len(df_policy))):
            paper = df_policy.iloc[i]
            refs = str(paper.get('referenced_works_oa', ''))
            if pd.notna(refs) and refs != 'nan':
                ref_list = [r.strip() for r in refs.split(';') if r.strip()]
                matches = [r for r in ref_list if r in paper_ids]
                if matches:
                    print(f"  ✓ Paper {i+1}: {len(matches)} matches out of {len(ref_list)} refs")
                    print(f"    Paper ID: {paper[id_col]}")
                    print(f"    Example match: {matches[0]}")
                    break
        
        print("\n  Creating visualization with nodes only (no edges)...")
    
    # Remove isolated nodes if we have edges
    if edges_added > 0:
        # Keep only nodes with at least one connection
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
        
        # Size: logarithmic scale for better visualization
        size = 100 + np.log1p(policy_cites) * 150
        node_sizes.append(size)
        
        # Color by SDG
        node_colors.append(sdg_color_map.get(sdg, (0.5, 0.5, 0.5, 1.0)))
    
    # Calculate layout
    print("Computing layout (this may take a moment for large networks)...")
    if G.number_of_edges() > 0:
        # Use spring layout for networks with edges
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.kamada_kawai_layout(G)
    else:
        # Use circular layout if no edges
        pos = nx.circular_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 24))
    
    # Draw edges (if any)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, 
                              alpha=0.2, 
                              width=0.5,
                              edge_color='gray', 
                              arrows=True,
                              arrowsize=10,
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
    
    # Create legend for SDGs (limit to top SDGs if too many)
    from matplotlib.patches import Patch
    
    if len(sdg_list) <= 15:
        legend_elements = [Patch(facecolor=sdg_color_map[sdg], 
                                edgecolor='black', 
                                label=sdg, linewidth=1)
                          for sdg in sorted(sdg_list)]
    else:
        # Show only top 10 SDGs by node count
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
    
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             fontsize=11, 
             title='SDGs', 
             title_fontsize=13,
             frameon=True, 
             fancybox=True, 
             shadow=True,
             ncol=2 if len(legend_elements) > 10 else 1)
    
    # Title
    if G.number_of_edges() > 0:
        title_text = ('Citation Network: Complexity Science Papers with Policy Impact\n' +
                     'Node size = Policy citations | Node color = SDG | Edges = Citation relationships')
    else:
        title_text = ('Papers with Policy Impact by SDG\n' +
                     'Node size = Policy citations | Node color = SDG | (No internal citations found)')
    
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add statistics
    if G.number_of_edges() > 0:
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        
        stats_text = (f"Network Statistics:\n"
                     f"Nodes: {len(G.nodes()):,}\n"
                     f"Edges: {len(G.edges()):,}\n"
                     f"Density: {density:.4f}\n"
                     f"Avg Degree: {avg_degree:.2f}")
    else:
        stats_text = (f"Network Statistics:\n"
                     f"Nodes: {len(G.nodes()):,}\n"
                     f"No citation edges\n"
                     f"(Papers don't cite each other)")
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_policy_network_sdg_color.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  • Visualized {len(G.nodes())} papers")
    print(f"  • {len(sdgs_in_network)} SDGs represented")
    if G.number_of_edges() > 0:
        print(f"  • Citation edges: {len(G.edges())}")
    else:
        print(f"  • No citation edges (papers in corpus don't cite each other)")
    print()
    """
    Create citation network where:
    - Nodes = papers with policy citations
    - Edges = citation relationships (one paper references another)
    - Node size = policy citations
    - Node color = SDG
    
    Note: This function needs clarification on which column contains
    referenced works. It will try common column names.
    
    Args:
        df_sdg: DataFrame with SDG papers
    """
    print("="*80)
    print("[4/7] CREATING CITATION NETWORK")
    print("="*80)
    
    # Filter papers with policy citations
    df_policy = df_sdg[df_sdg['policy_mentions_total_alt'].fillna(0) > 0].copy()
    print(f"Papers with policy citations: {len(df_policy):,}")
    
    # Create network
    G = nx.DiGraph()
    
    # Add nodes with attributes
    node_count = 0
    for idx, row in df_policy.iterrows():
        # Use DOI as node ID, fallback to index if no DOI
        node_id = row.get('doi_oa', '') or row.get('doi_alt', '') or row.get('id_oa', '') or f"paper_{idx}"
        
        if node_id and str(node_id).strip() and str(node_id) != 'nan':
            G.add_node(str(node_id),
                      policy_citations=float(row.get('policy_mentions_total_alt', 0)),
                      sdg=str(row.get('sdg_name_oa', 'Unknown')),
                      title=str(row.get('title_oa', row.get('title_alt', 'No title')))[:80],
                      year=str(row.get('year', '')))
            node_count += 1
    
    print(f"✓ Added {node_count} nodes to network")
    
    # Try to find references column
    possible_ref_columns = [
        'referenced_works',
        'referenced_works_oa', 
        'references_oa',
        'cited_references',
        'references',
        'cited_works'
    ]
    
    ref_column = None
    for col in possible_ref_columns:
        if col in df_policy.columns:
            ref_column = col
            print(f"✓ Found reference column: '{ref_column}'")
            break
    
    if ref_column:
        print("Processing citation relationships...")
        edges_added = 0
        
        for idx, row in df_policy.iterrows():
            source = str(row.get('doi_oa', '') or row.get('doi_alt', '') or row.get('id_oa', '') or f"paper_{idx}")
            
            if source not in G.nodes():
                continue
            
            refs = row.get(ref_column, '')
            
            if pd.notna(refs) and str(refs).strip() and str(refs) != 'nan':
                # Try different separators
                for separator in [';', '|', ',', '\n']:
                    if separator in str(refs):
                        ref_list = [r.strip() for r in str(refs).split(separator) if r.strip()]
                        break
                else:
                    ref_list = [str(refs).strip()]
                
                for ref in ref_list:
                    if ref and ref in G.nodes():
                        G.add_edge(source, ref)
                        edges_added += 1
        
        print(f"✓ Added {edges_added} citation edges")
    else:
        print("⚠ Warning: No reference column found in the data")
        print(f"  Available columns: {', '.join(df_policy.columns[:10])}...")
        print("  Network will show nodes only (no citation edges)")
    
    # Remove isolated nodes if there are edges
    if G.number_of_edges() > 0:
        # Keep largest connected component or nodes with edges
        nodes_to_keep = set()
        for node in G.nodes():
            if G.degree(node) > 0:
                nodes_to_keep.add(node)
        
        nodes_to_remove = set(G.nodes()) - nodes_to_keep
        G.remove_nodes_from(nodes_to_remove)
        print(f"✓ Removed {len(nodes_to_remove)} isolated nodes")
        print(f"✓ Network size: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    if len(G.nodes()) == 0:
        print("⚠ No nodes in network after filtering. Skipping visualization.")
        return
    
    # Prepare visualization
    print("Creating network visualization...")
    
    # Get unique SDGs and create color map
    sdgs_in_network = set(G.nodes[node]['sdg'] for node in G.nodes())
    sdg_list = sorted(sdgs_in_network)
    colors_palette = plt.cm.tab20(np.linspace(0, 1, len(sdg_list)))
    sdg_color_map = {sdg: colors_palette[i] for i, sdg in enumerate(sdg_list)}
    
    # Calculate node sizes based on policy citations (log scale)
    node_sizes = []
    node_colors = []
    
    for node in G.nodes():
        policy_cites = G.nodes[node].get('policy_citations', 0)
        sdg = G.nodes[node].get('sdg', 'Unknown')
        
        # Size: logarithmic scale for better visualization
        size = 100 + np.log1p(policy_cites) * 150
        node_sizes.append(size)
        
        # Color by SDG
        node_colors.append(sdg_color_map.get(sdg, (0.5, 0.5, 0.5, 1.0)))
    
    # Calculate layout
    print("Computing layout (this may take a moment for large networks)...")
    if G.number_of_edges() > 0:
        # Use spring layout for networks with edges
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    else:
        # Use random layout if no edges
        pos = nx.circular_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 24))
    
    # Draw edges (if any)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, 
                              alpha=0.15, 
                              width=0.5,
                              edge_color='gray', 
                              arrows=True,
                              arrowsize=8,
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
    
    # Create legend for SDGs (limit to top SDGs if too many)
    from matplotlib.patches import Patch
    
    if len(sdg_list) <= 15:
        legend_elements = [Patch(facecolor=sdg_color_map[sdg], 
                                edgecolor='black', 
                                label=sdg, linewidth=1)
                          for sdg in sorted(sdg_list)]
    else:
        # Show only top 10 SDGs by node count
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
    
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             fontsize=11, 
             title='SDGs', 
             title_fontsize=13,
             frameon=True, 
             fancybox=True, 
             shadow=True,
             ncol=2 if len(legend_elements) > 10 else 1)
    
    # Title
    ax.set_title('Citation Network: Complexity Science Papers with Policy Impact\n' +
                'Node size = Policy citations | Node color = SDG | Edges = Citation relationships',
                fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')
    
    # Add statistics
    if G.number_of_edges() > 0:
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        
        # Calculate network metrics
        if nx.is_strongly_connected(G):
            diameter = nx.diameter(G)
            stats_text = (f"Network Statistics:\n"
                         f"Nodes: {len(G.nodes()):,}\n"
                         f"Edges: {len(G.edges()):,}\n"
                         f"Density: {density:.4f}\n"
                         f"Avg Degree: {avg_degree:.2f}\n"
                         f"Diameter: {diameter}")
        else:
            stats_text = (f"Network Statistics:\n"
                         f"Nodes: {len(G.nodes()):,}\n"
                         f"Edges: {len(G.edges()):,}\n"
                         f"Density: {density:.4f}\n"
                         f"Avg Degree: {avg_degree:.2f}\n"
                         f"(Disconnected)")
    else:
        stats_text = (f"Network Statistics:\n"
                     f"Nodes: {len(G.nodes()):,}\n"
                     f"No citation edges\n"
                     f"(Reference data not available)")
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_policy_network_sdg_color.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  • Visualized {len(G.nodes())} papers")
    print(f"  • {len(sdgs_in_network)} SDGs represented")
    if G.number_of_edges() > 0:
        print(f"  • Citation edges: {len(G.edges())}\n")
    else:
        print(f"  • No citation edges (reference column not found)\n")

# ============================================================================
# 5. SDG IMPACT PROFILES - MULTI-DIMENSIONAL ANALYSIS
# ============================================================================

def create_impact_profiles(sdg_stats, total_papers):
    """
    Create comprehensive impact profile analysis showing:
    - Policy vs Academic citations per paper
    - Media mentions per paper  
    - Papers vs Total Impact scatter
    - Normalized impact heatmap
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        total_papers: Total papers in corpus
    """
    print("="*80)
    print("[5/7] CREATING SDG IMPACT PROFILES")
    print("="*80)
    
    # Calculate normalized metrics (per paper)
    sdg_stats_norm = sdg_stats.copy()
    sdg_stats_norm['Policy_per_Paper'] = sdg_stats_norm['Policy_Citations'] / sdg_stats_norm['Papers']
    sdg_stats_norm['Academic_per_Paper'] = sdg_stats_norm['Academic_Citations'] / sdg_stats_norm['Papers']
    sdg_stats_norm['Media_per_Paper'] = sdg_stats_norm['Media_Mentions'] / sdg_stats_norm['Papers']
    sdg_stats_norm['Total_Impact'] = (sdg_stats_norm['Policy_Citations'] + 
                                       sdg_stats_norm['Academic_Citations'] + 
                                       sdg_stats_norm['Media_Mentions'])
    
    # Sort by papers for consistent display
    sdg_stats_norm = sdg_stats_norm.sort_values('Papers', ascending=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # -------------------------------------------------------------------------
    # Panel 1: Policy vs Academic per paper (TOP LEFT)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    x_pos = np.arange(len(sdg_stats_norm))
    width = 0.35
    
    bars1 = ax1.barh(x_pos - width/2, sdg_stats_norm['Policy_per_Paper'], 
                    width, label='Policy', color=COLORS['policy'], 
                    edgecolor='black', linewidth=0.8, alpha=0.8)
    bars2 = ax1.barh(x_pos + width/2, sdg_stats_norm['Academic_per_Paper'], 
                    width, label='Academic', color=COLORS['academic'],
                    edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(sdg_stats_norm['SDG'], fontsize=9)
    ax1.set_xlabel('Average Citations per Paper', fontweight='bold', fontsize=11)
    ax1.set_title('Policy vs Academic Impact per Paper\n' +
                  'Shows which SDGs get more policy vs academic attention',
                  fontweight='bold', fontsize=12, pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # -------------------------------------------------------------------------
    # Panel 2: Media attention per paper (TOP RIGHT)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    bars = ax2.barh(x_pos, sdg_stats_norm['Media_per_Paper'],
                   color=COLORS['media'], edgecolor='black',
                   linewidth=0.8, alpha=0.8)
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(sdg_stats_norm['SDG'], fontsize=9)
    ax2.set_xlabel('Average Media Mentions per Paper', fontweight='bold', fontsize=11)
    ax2.set_title('Media Visibility per Paper\n' +
                  'Identifies SDGs with high public/press coverage',
                  fontweight='bold', fontsize=12, pad=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # -------------------------------------------------------------------------
    # Panel 3: Papers vs Total Impact (BOTTOM LEFT)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    scatter = ax3.scatter(sdg_stats_norm['Papers'], 
                         sdg_stats_norm['Total_Impact'],
                         s=400, alpha=0.65, 
                         c=sdg_stats_norm['Policy_Share_%'],
                         cmap='RdYlGn', 
                         edgecolors='black', 
                         linewidths=1.5)
    
    # Add SDG labels
    for idx, row in sdg_stats_norm.iterrows():
        ax3.annotate(row['SDG'], 
                    (row['Papers'], row['Total_Impact']),
                    fontsize=8, 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor='gray',
                             alpha=0.7, linewidth=0.8))
    
    ax3.set_xlabel('Number of Papers', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Total Impact\n(Policy + Academic + Media)', fontweight='bold', fontsize=11)
    ax3.set_title('Volume vs Total Impact\n' +
                  'Color shows % of policy citations (darker green = more policy impact)',
                  fontweight='bold', fontsize=12, pad=10)
    ax3.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.02)
    cbar.set_label('% of Policy Citations', fontweight='bold', fontsize=10)
    
    # -------------------------------------------------------------------------
    # Panel 4: Normalized Impact Heatmap (BOTTOM RIGHT)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    # Create normalized impact matrix
    impact_matrix = sdg_stats_norm[['Policy_per_Paper', 'Academic_per_Paper', 'Media_per_Paper']].values
    
    # Normalize each column to 0-1 range
    impact_matrix_norm = np.zeros_like(impact_matrix)
    for i in range(impact_matrix.shape[1]):
        col = impact_matrix[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            impact_matrix_norm[:, i] = (col - col_min) / (col_max - col_min)
        else:
            impact_matrix_norm[:, i] = 0
    
    im = ax4.imshow(impact_matrix_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(['Policy', 'Academic', 'Media'], fontweight='bold', fontsize=11)
    ax4.set_yticks(range(len(sdg_stats_norm)))
    ax4.set_yticklabels(sdg_stats_norm['SDG'], fontsize=9)
    ax4.set_title('Normalized Impact Profile Heatmap\n' +
                  'Darker colors = Higher relative impact in that dimension',
                  fontweight='bold', fontsize=12, pad=10)
    
    # Add colorbar
    cbar2 = plt.colorbar(im, ax=ax4, pad=0.02)
    cbar2.set_label('Normalized Impact\n(0=lowest, 1=highest)', 
                   fontweight='bold', fontsize=10)
    
    # Add values to heatmap cells
    for i in range(len(sdg_stats_norm)):
        for j in range(3):
            text = ax4.text(j, i, f'{impact_matrix_norm[i, j]:.2f}',
                          ha="center", va="center", 
                          color="black" if impact_matrix_norm[i, j] < 0.5 else "white",
                          fontsize=8, fontweight='bold')
    
    # Overall title
    plt.suptitle('SDG Impact Profiles: Multi-Dimensional Analysis\n' +
                'Understanding how different SDGs achieve visibility across impact channels',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_sdg_impact_profiles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # GENERATE SEPARATE IMAGE: Volume vs Impact (Bottom Left)
    # =========================================================================
    print("   📊 Creating separate volume vs impact scatter...")
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(14, 12))
    
    scatter = ax_scatter.scatter(sdg_stats_norm['Papers'], 
                         sdg_stats_norm['Total_Impact'],
                         s=400, alpha=0.65, 
                         c=sdg_stats_norm['Policy_Share_%'],
                         cmap='RdYlGn', 
                         edgecolors='black', 
                         linewidths=1.5)
    
    # Add SDG labels
    for idx, row in sdg_stats_norm.iterrows():
        ax_scatter.annotate(row['SDG'], 
                    (row['Papers'], row['Total_Impact']),
                    fontsize=9, 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor='gray',
                             alpha=0.8, linewidth=0.8))
    
    ax_scatter.set_xlabel('Number of Papers', fontweight='bold', fontsize=13)
    ax_scatter.set_ylabel('Total Impact\n(Policy + Academic + Media)', fontweight='bold', fontsize=13)
    ax_scatter.set_title('SDG Volume vs Total Impact\n' +
                'Color shows % of policy citations (darker green = more policy impact)',
                fontweight='bold', fontsize=14, pad=20)
    ax_scatter.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar_scatter = plt.colorbar(scatter, ax=ax_scatter, pad=0.02)
    cbar_scatter.set_label('% of Policy Citations', fontweight='bold', fontsize=12)
    
    # Add note
    note_scatter = (f"Dataset: {total_papers:,} papers | SDGs: {len(sdg_stats_norm)}\n"
                   f"Total Impact = Policy Citations + Academic Citations + Media Mentions\n"
                   f"Larger/greener bubbles indicate higher policy impact")
    ax_scatter.text(0.02, 0.98, note_scatter, transform=ax_scatter.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=0.5))
    
    plt.tight_layout()
    output_path_scatter = f"{OUTPUT_IMAGES}/12_sdg_volume_impact.png"
    plt.savefig(output_path_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path_scatter}")
    
    # =========================================================================
    # GENERATE SEPARATE IMAGE: Heatmap (Bottom Right) - NO GRID LINES
    # =========================================================================
    print("   📊 Creating separate impact heatmap...")
    
    fig_heat, ax_heat = plt.subplots(figsize=(12, 14))
    
    # Create normalized impact matrix
    impact_matrix = sdg_stats_norm[['Policy_per_Paper', 'Academic_per_Paper', 'Media_per_Paper']].values
    
    # Normalize each column to 0-1 range
    impact_matrix_norm = np.zeros_like(impact_matrix)
    for i in range(impact_matrix.shape[1]):
        col = impact_matrix[:, i]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            impact_matrix_norm[:, i] = (col - col_min) / (col_max - col_min)
        else:
            impact_matrix_norm[:, i] = 0
    
    im = ax_heat.imshow(impact_matrix_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    ax_heat.set_xticks([0, 1, 2])
    ax_heat.set_xticklabels(['Policy', 'Academic', 'Media'], fontweight='bold', fontsize=12)
    ax_heat.set_yticks(range(len(sdg_stats_norm)))
    ax_heat.set_yticklabels(sdg_stats_norm['SDG'], fontsize=10)
    ax_heat.set_title('Normalized Impact Profile Heatmap\n' +
                  'Darker colors = Higher relative impact in that dimension',
                  fontweight='bold', fontsize=14, pad=20)
    
    # Add colorbar
    cbar_heat = plt.colorbar(im, ax=ax_heat, pad=0.02)
    cbar_heat.set_label('Normalized Impact\n(0=lowest, 1=highest)', 
                   fontweight='bold', fontsize=11)
    
    # Add values to heatmap cells WITHOUT grid lines
    for i in range(len(sdg_stats_norm)):
        for j in range(3):
            text = ax_heat.text(j, i, f'{impact_matrix_norm[i, j]:.2f}',
                          ha="center", va="center", 
                          color="black" if impact_matrix_norm[i, j] < 0.5 else "white",
                          fontsize=9, fontweight='bold')
    
    # NO grid lines - remove them explicitly
    ax_heat.set_xticks([0, 1, 2])
    ax_heat.set_yticks(range(len(sdg_stats_norm)))
    ax_heat.grid(False)  # Explicitly disable grid
    
    # Add explanatory note
    note_heat = ("Normalized Impact: Each column scaled 0-1 independently\n"
                "• 0.00 = Lowest impact for that dimension across all SDGs\n"
                "• 1.00 = Highest impact for that dimension across all SDGs\n"
                "• Allows comparison across different impact types\n"
                "• Darker red = relatively more impact in that channel")
    ax_heat.text(0.5, -0.08, note_heat, transform=ax_heat.transAxes,
                fontsize=9, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', 
                         alpha=0.9, pad=0.5, linewidth=1))
    
    plt.tight_layout()
    output_path_heat = f"{OUTPUT_IMAGES}/12_heatmap_sgd_impact.png"
    plt.savefig(output_path_heat, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path_heat}")
    
    print(f"\n📊 Impact Profile Insights:")
    
    # Find SDGs with different impact patterns
    policy_leaders = sdg_stats_norm.nlargest(3, 'Policy_per_Paper')
    print(f"\n  Top 3 SDGs by Policy Impact per Paper:")
    for idx, row in policy_leaders.iterrows():
        print(f"    {row['SDG']}: {row['Policy_per_Paper']:.2f} policy citations/paper")
    
    academic_leaders = sdg_stats_norm.nlargest(3, 'Academic_per_Paper')
    print(f"\n  Top 3 SDGs by Academic Impact per Paper:")
    for idx, row in academic_leaders.iterrows():
        print(f"    {row['SDG']}: {row['Academic_per_Paper']:.2f} academic citations/paper")
    
    media_leaders = sdg_stats_norm.nlargest(3, 'Media_per_Paper')
    print(f"\n  Top 3 SDGs by Media Visibility per Paper:")
    for idx, row in media_leaders.iterrows():
        print(f"    {row['SDG']}: {row['Media_per_Paper']:.2f} media mentions/paper")
    
    print("\n  This visualization helps identify:")
    print("    • Which SDGs 'punch above their weight' in policy impact")
    print("    • SDGs with balanced vs specialized impact patterns")
    print("    • Opportunities to increase complexity science's policy relevance\n")

# ============================================================================
# 6. ADDITIONAL INSIGHT: POLICY EFFECTIVENESS INDEX
# ============================================================================

def create_policy_effectiveness_analysis(sdg_stats, df_sdg):
    """
    Create additional analysis: Policy Effectiveness Index
    
    This shows which SDGs are most effective at translating complexity
    science research into policy impact, normalized by volume of papers.
    
    Generates 3 images:
    - Combined view
    - Left panel only (_1)
    - Right panel only (_2)
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        df_sdg: DataFrame with SDG papers
    """
    print("="*80)
    print("[6/7] CREATING POLICY EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # Calculate Policy Effectiveness Index (PEI)
    # PEI = (Policy Share % / Corpus Share %) 
    # Values > 1 mean over-represented in policy relative to corpus size
    
    sdg_stats_pei = sdg_stats.copy()
    sdg_stats_pei['Policy_Effectiveness_Index'] = (
        sdg_stats_pei['Policy_Share_%'] / sdg_stats_pei['Corpus_Share_%']
    )
    
    # Sort by PEI
    sdg_stats_pei = sdg_stats_pei.sort_values('Policy_Effectiveness_Index', ascending=True)
    
    # =========================================================================
    # COMBINED FIGURE
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    
    # -------------------------------------------------------------------------
    # Left panel: Policy Effectiveness Index ranking
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    y_pos = np.arange(len(sdg_stats_pei))
    
    # Color bars based on effectiveness
    colors = ['#27AE60' if pei > 1.5 else '#F39C12' if pei > 0.75 else '#E74C3C' 
             for pei in sdg_stats_pei['Policy_Effectiveness_Index']]
    
    bars = ax1.barh(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'],
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Add reference line at 1.0 (equal representation)
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, 
               label='Equal representation (PEI = 1.0)')
    
    # Add values
    for idx, (y, pei) in enumerate(zip(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'])):
        ax1.text(pei, y, f' {pei:.2f}x', 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sdg_stats_pei['SDG'], fontsize=10)
    ax1.set_xlabel('Policy Effectiveness Index\n(Policy Share % / Corpus Share %)', 
                  fontweight='bold', fontsize=12)
    ax1.set_title('Policy Effectiveness Index by SDG\n' +
                  'Green: High effectiveness (>1.5x) | Orange: Moderate | Red: Low (<0.75x)',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # -------------------------------------------------------------------------
    # Right panel: Bubble chart - Papers vs Policy Impact vs Effectiveness
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Create bubble chart
    scatter = ax2.scatter(sdg_stats_pei['Papers'],
                         sdg_stats_pei['Policy_Citations'],
                         s=sdg_stats_pei['Policy_Effectiveness_Index'] * 300,
                         c=sdg_stats_pei['Policy_Effectiveness_Index'],
                         cmap='RdYlGn',
                         alpha=0.6,
                         edgecolors='black',
                         linewidths=1.5)
    
    # Add SDG labels ABOVE dots to prevent overlap
    for idx, row in sdg_stats_pei.iterrows():
        ax2.annotate(row['SDG'],
                    (row['Papers'], row['Policy_Citations']),
                    fontsize=8,
                    xytext=(0, 5),  # Offset above the dot
                    textcoords='offset points',
                    ha='center',  # Center horizontally
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.8, linewidth=0.8))
    
    # Set x-axis max to 500 to prevent labels going out
    ax2.set_xlim(0, 500)
    ax2.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Total Policy Citations', fontweight='bold', fontsize=12)
    ax2.set_title('Papers vs Policy Impact\n' +
                  'Bubble size and color = Policy Effectiveness Index',
                  fontweight='bold', fontsize=13, pad=15)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label('Policy Effectiveness Index', fontweight='bold', fontsize=11)
    
    # Add note explaining PEI - moved to BOTTOM RIGHT
    note = ("Policy Effectiveness Index (PEI):\n"
           "• PEI > 1.5: High policy impact relative to corpus size\n"
           "• PEI = 1.0: Policy share matches corpus share\n"
           "• PEI < 0.75: Low policy impact relative to corpus size\n"
           "\nLarger bubbles = more effective policy translation")
    ax2.text(0.98, 0.02, note, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, pad=0.5, linewidth=1))
    
    plt.suptitle('Policy Effectiveness Analysis: Which SDGs Translate Complexity Science into Policy Impact?\n' +
                'Understanding which sustainability domains benefit most from complexity approaches',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_policy_effectiveness_index.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # =========================================================================
    # SEPARATE IMAGE 1: Left panel only
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 12)) 
    
    y_pos = np.arange(len(sdg_stats_pei))
    colors = ['#27AE60' if pei > 1.5 else '#F39C12' if pei > 0.75 else '#E74C3C' 
             for pei in sdg_stats_pei['Policy_Effectiveness_Index']]
    
    bars = ax1.barh(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'],
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, 
               label='Equal representation (PEI = 1.0)')
    
    for idx, (y, pei) in enumerate(zip(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'])):
        ax1.text(pei, y, f' {pei:.2f}x', 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sdg_stats_pei['SDG'], fontsize=10)
    ax1.set_xlabel('Policy Effectiveness Index\n(Policy Share % / Corpus Share %)', 
                  fontweight='bold', fontsize=12)
    ax1.set_title('Policy Effectiveness Index by SDG\n' +
                  'Green: High effectiveness (>1.5x) | Orange: Moderate | Red: Low (<0.75x)',
                  fontweight='bold', fontsize=14, pad=15)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path_1 = f"{OUTPUT_IMAGES}/12_policy_effectiveness_index_1.png"
    plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path_1}")
    
    # =========================================================================
    # SEPARATE IMAGE 2: Right panel only  
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    
    scatter = ax2.scatter(sdg_stats_pei['Papers'],
                         sdg_stats_pei['Policy_Citations'],
                         s=sdg_stats_pei['Policy_Effectiveness_Index'] * 300,
                         c=sdg_stats_pei['Policy_Effectiveness_Index'],
                         cmap='RdYlGn',
                         alpha=0.6,
                         edgecolors='black',
                         linewidths=1.5)
    
    # Add SDG labels ABOVE dots
    for idx, row in sdg_stats_pei.iterrows():
        ax2.annotate(row['SDG'],
                    (row['Papers'], row['Policy_Citations']),
                    fontsize=9,
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.8, linewidth=0.8))
    
    ax2.set_xlim(0, 500)
    ax2.set_xlabel('Number of Papers', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Total Policy Citations', fontweight='bold', fontsize=13)
    ax2.set_title('Papers vs Policy Impact\n' +
                  'Bubble size and color = Policy Effectiveness Index',
                  fontweight='bold', fontsize=14, pad=15)
    ax2.grid(alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label('Policy Effectiveness Index', fontweight='bold', fontsize=12)
    
    note = ("Policy Effectiveness Index (PEI):\n"
           "• PEI > 1.5: High policy impact relative to corpus size\n"
           "• PEI = 1.0: Policy share matches corpus share\n"
           "• PEI < 0.75: Low policy impact relative to corpus size\n"
           "\nLarger bubbles = more effective policy translation")
    ax2.text(0.98, 0.02, note, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, pad=0.5, linewidth=1))
    
    plt.tight_layout()
    output_path_2 = f"{OUTPUT_IMAGES}/12_policy_effectiveness_index_2.png"
    plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path_2}")
    
    # Print insights
    print(f"\n📊 Policy Effectiveness Insights:")
    
    high_effectiveness = sdg_stats_pei[sdg_stats_pei['Policy_Effectiveness_Index'] > 1.5]
    print(f"\n  🟢 High Effectiveness SDGs (PEI > 1.5): {len(high_effectiveness)}")
    for idx, row in high_effectiveness.nlargest(5, 'Policy_Effectiveness_Index').iterrows():
        print(f"    {row['SDG']}: PEI = {row['Policy_Effectiveness_Index']:.2f}x")
        print(f"      ({row['Papers']} papers → {row['Policy_Citations']:.0f} policy citations)")
    
    low_effectiveness = sdg_stats_pei[sdg_stats_pei['Policy_Effectiveness_Index'] < 0.75]
    print(f"\n  🔴 Lower Effectiveness SDGs (PEI < 0.75): {len(low_effectiveness)}")
    for idx, row in low_effectiveness.nsmallest(3, 'Policy_Effectiveness_Index').iterrows():
        print(f"    {row['SDG']}: PEI = {row['Policy_Effectiveness_Index']:.2f}x")
        print(f"      ({row['Papers']} papers → {row['Policy_Citations']:.0f} policy citations)")
    
    print(f"\n  💡 Interpretation:")
    print(f"    • High PEI SDGs: Complexity science research effectively reaches policymakers")
    print(f"    • Low PEI SDGs: Opportunity to strengthen science-policy translation")
    print(f"    • This reveals where complexity methods are most policy-relevant\n")
    """
    Create additional analysis: Policy Effectiveness Index
    
    This shows which SDGs are most effective at translating complexity
    science research into policy impact, normalized by volume of papers.
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        df_sdg: DataFrame with SDG papers
    """
    print("="*80)
    print("[6/7] CREATING POLICY EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # Calculate Policy Effectiveness Index (PEI)
    # PEI = (Policy Share % / Corpus Share %) 
    # Values > 1 mean over-represented in policy relative to corpus size
    
    sdg_stats_pei = sdg_stats.copy()
    sdg_stats_pei['Policy_Effectiveness_Index'] = (
        sdg_stats_pei['Policy_Share_%'] / sdg_stats_pei['Corpus_Share_%']
    )
    
    # Sort by PEI
    sdg_stats_pei = sdg_stats_pei.sort_values('Policy_Effectiveness_Index', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    
    # -------------------------------------------------------------------------
    # Left panel: Policy Effectiveness Index ranking
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    y_pos = np.arange(len(sdg_stats_pei))
    
    # Color bars based on effectiveness
    colors = ['#27AE60' if pei > 1.5 else '#F39C12' if pei > 0.75 else '#E74C3C' 
             for pei in sdg_stats_pei['Policy_Effectiveness_Index']]
    
    bars = ax1.barh(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'],
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)
    
    # Add reference line at 1.0 (equal representation)
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, 
               label='Equal representation (PEI = 1.0)')
    
    # Add values
    for idx, (y, pei) in enumerate(zip(y_pos, sdg_stats_pei['Policy_Effectiveness_Index'])):
        ax1.text(pei, y, f' {pei:.2f}x', 
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sdg_stats_pei['SDG'], fontsize=10)
    ax1.set_xlabel('Policy Effectiveness Index\n(Policy Share % / Corpus Share %)', 
                  fontweight='bold', fontsize=12)
    ax1.set_title('Policy Effectiveness Index by SDG\n' +
                  'Green: High effectiveness (>1.5x) | Orange: Moderate | Red: Low (<0.75x)',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # -------------------------------------------------------------------------
    # Right panel: Bubble chart - Papers vs Policy Impact vs Effectiveness
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Create bubble chart
    scatter = ax2.scatter(sdg_stats_pei['Papers'],
                         sdg_stats_pei['Policy_Citations'],
                         s=sdg_stats_pei['Policy_Effectiveness_Index'] * 300,
                         c=sdg_stats_pei['Policy_Effectiveness_Index'],
                         cmap='RdYlGn',
                         alpha=0.6,
                         edgecolors='black',
                         linewidths=1.5)
    
    # Add SDG labels
    for idx, row in sdg_stats_pei.iterrows():
        ax2.annotate(row['SDG'],
                    (row['Papers'], row['Policy_Citations']),
                    fontsize=8,
                    xytext=(3, 3),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.7, linewidth=0.8))
    
    ax2.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Total Policy Citations', fontweight='bold', fontsize=12)
    ax2.set_title('Papers vs Policy Impact\n' +
                  'Bubble size and color = Policy Effectiveness Index',
                  fontweight='bold', fontsize=13, pad=15)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label('Policy Effectiveness Index', fontweight='bold', fontsize=11)
    
    # Add note explaining PEI
    note = ("Policy Effectiveness Index (PEI):\n"
           "• PEI > 1.5: High policy impact relative to corpus size\n"
           "• PEI = 1.0: Policy share matches corpus share\n"
           "• PEI < 0.75: Low policy impact relative to corpus size\n"
           "\nLarger bubbles = more effective policy translation")
    ax2.text(0.02, 0.98, note, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, pad=0.5, linewidth=1))
    
    plt.suptitle('Policy Effectiveness Analysis: Which SDGs Translate Complexity Science into Policy Impact?\n' +
                'Understanding which sustainability domains benefit most from complexity approaches',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_IMAGES}/12_policy_effectiveness_index.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Print insights
    print(f"\n📊 Policy Effectiveness Insights:")
    
    high_effectiveness = sdg_stats_pei[sdg_stats_pei['Policy_Effectiveness_Index'] > 1.5]
    print(f"\n  🟢 High Effectiveness SDGs (PEI > 1.5): {len(high_effectiveness)}")
    for idx, row in high_effectiveness.nlargest(5, 'Policy_Effectiveness_Index').iterrows():
        print(f"    {row['SDG']}: PEI = {row['Policy_Effectiveness_Index']:.2f}x")
        print(f"      ({row['Papers']} papers → {row['Policy_Citations']:.0f} policy citations)")
    
    low_effectiveness = sdg_stats_pei[sdg_stats_pei['Policy_Effectiveness_Index'] < 0.75]
    print(f"\n  🔴 Lower Effectiveness SDGs (PEI < 0.75): {len(low_effectiveness)}")
    for idx, row in low_effectiveness.nsmallest(3, 'Policy_Effectiveness_Index').iterrows():
        print(f"    {row['SDG']}: PEI = {row['Policy_Effectiveness_Index']:.2f}x")
        print(f"      ({row['Papers']} papers → {row['Policy_Citations']:.0f} policy citations)")
    
    print(f"\n  💡 Interpretation:")
    print(f"    • High PEI SDGs: Complexity science research effectively reaches policymakers")
    print(f"    • Low PEI SDGs: Opportunity to strengthen science-policy translation")
    print(f"    • This reveals where complexity methods are most policy-relevant\n")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

def generate_summary_report(sdg_stats, df, df_sdg, total_policy_mentions):
    """
    Generate comprehensive summary report of all analyses
    
    Args:
        sdg_stats: DataFrame with SDG statistics
        df: Full dataset
        df_sdg: Papers with SDG
        total_policy_mentions: Total policy citations
    """
    print("="*80)
    print("[7/7] GENERATING SUMMARY REPORT")
    print("="*80)
    
    # Calculate key statistics
    papers_with_policy = len(df_sdg[df_sdg['policy_mentions_total_alt'].fillna(0) > 0])
    
    over_represented = sdg_stats[sdg_stats['Policy_Share_%'] > sdg_stats['Corpus_Share_%']]
    under_represented = sdg_stats[sdg_stats['Policy_Share_%'] < sdg_stats['Corpus_Share_%']]
    
    # Calculate Policy Effectiveness Index for report
    sdg_stats_report = sdg_stats.copy()
    sdg_stats_report['PEI'] = sdg_stats_report['Policy_Share_%'] / sdg_stats_report['Corpus_Share_%']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📚 CORPUS OVERVIEW:")
    print(f"  • Total papers: {len(df):,}")
    print(f"  • Papers with SDG information: {len(df_sdg):,} ({len(df_sdg)/len(df)*100:.1f}%)")
    print(f"  • Papers with policy citations: {papers_with_policy:,} ({papers_with_policy/len(df)*100:.1f}%)")
    print(f"  • Number of SDGs represented: {len(sdg_stats)}")
    
    print(f"\n📊 POLICY IMPACT:")
    print(f"  • Total policy citations: {total_policy_mentions:,.0f}")
    print(f"  • Average policy citations per paper: {total_policy_mentions/len(df):.2f}")
    print(f"  • Average policy citations per SDG paper: {total_policy_mentions/len(df_sdg):.2f}")
    
    print(f"\n🏆 TOP SDGS BY PAPER COUNT:")
    for idx, row in sdg_stats.nlargest(5, 'Papers').iterrows():
        print(f"  {idx+1}. {row['SDG']}: {row['Papers']:,} papers ({row['Corpus_Share_%']}% of corpus)")
    
    print(f"\n🎯 TOP SDGS BY POLICY IMPACT:")
    for idx, row in sdg_stats.nlargest(5, 'Policy_Citations').iterrows():
        print(f"  {idx+1}. {row['SDG']}: {row['Policy_Citations']:,.0f} citations ({row['Policy_Share_%']}% of total)")
    
    print(f"\n📈 POLICY EFFECTIVENESS (Top 5 by PEI):")
    for idx, row in sdg_stats_report.nlargest(5, 'PEI').iterrows():
        print(f"  {idx+1}. {row['SDG']}: PEI = {row['PEI']:.2f}x")
        print(f"     ({row['Papers']} papers, {row['Policy_Citations']:.0f} policy citations)")
    
    print(f"\n🔍 REPRESENTATION ANALYSIS:")
    print(f"  • SDGs over-represented in policy: {len(over_represented)}/{len(sdg_stats)}")
    if len(over_represented) > 0:
        print(f"    Top 3:")
        for idx, row in over_represented.nlargest(3, 'Policy_Share_%').iterrows():
            ratio = row['Policy_Share_%'] / row['Corpus_Share_%']
            print(f"      - {row['SDG']}: {ratio:.2f}x over-represented")
    
    print(f"  • SDGs under-represented in policy: {len(under_represented)}/{len(sdg_stats)}")
    if len(under_represented) > 0:
        print(f"    Bottom 3:")
        for idx, row in under_represented.nsmallest(3, 'Policy_Share_%').iterrows():
            ratio = row['Policy_Share_%'] / row['Corpus_Share_%']
            print(f"      - {row['SDG']}: {ratio:.2f}x representation")
    
    print(f"\n💡 KEY INSIGHTS FOR COMPLEXITY SCIENCE:")
    print(f"  1. SDGs with high policy effectiveness show where complexity methods")
    print(f"     are most relevant for policymakers")
    print(f"  2. Over-represented SDGs indicate successful science-policy translation")
    print(f"  3. Under-represented SDGs suggest opportunities to strengthen")
    print(f"     complexity science's policy impact")
    print(f"  4. The diversity of SDG representation shows complexity science's")
    print(f"     broad applicability to sustainability challenges")
    
    print("\n" + "="*80)
    print("FILES GENERATED")
    print("="*80)
    
    files_generated = [
        "12_sdg_distribution_V2.png - Enhanced SDG distribution with policy %",
        "12_sdg_top_papers.xlsx - Top 2 papers per SDG by policy citations",
        "12_sdg_policy_vs_corpus.png - Scatter plot: corpus vs policy share",
        "12_policy_network_sdg_color.png - Citation network colored by SDG",
        "12_sdg_impact_profiles.png - Multi-dimensional impact analysis",
        "12_policy_effectiveness_index.png - Policy effectiveness ranking"
    ]
    
    for file in files_generated:
        print(f"  ✓ {file}")
    
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80)
    
    print("""
1. DEEP DIVE ON HIGH-EFFECTIVENESS SDGS:
   • Examine what makes papers in these SDGs policy-relevant
   • Identify methodological patterns (ABM, network analysis, etc.)
   • Study language and framing used in these papers

2. ANALYZE LOW-EFFECTIVENESS SDGS:
   • Investigate barriers to policy impact
   • Explore whether research is too theoretical
   • Consider if policy translation strategies are needed

3. TEMPORAL ANALYSIS:
   • Track how SDG representation has changed over time
   • Identify emerging SDG trends in complexity science
   • Correlate with global sustainability agenda evolution

4. CROSS-SDG ANALYSIS:
   • Identify papers addressing multiple SDGs
   • Map interdependencies between SDGs in research
   • Find synergies and trade-offs

5. QUALITATIVE FOLLOW-UP:
   • Interview authors of high-impact SDG papers
   • Survey policymakers on complexity science utility
   • Case studies of successful science-policy translation
    """)
    
    print("="*80)
    print(" "*25 + "✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 All outputs saved to:")
    print(f"   Images: {OUTPUT_IMAGES}")
    print(f"   Excel: {OUTPUT_EXCEL}")
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs all analyses in sequence
    """
    print("\n" + "="*80)
    print(" "*15 + "SDG EXTENDED ANALYSIS - MODULAR VERSION")
    print(" "*10 + "Complexity Science & Sustainable Development Goals")
    print("="*80)
    print("\nBased on: Grauwin et al. (2012)")
    print("'Complex Systems Science: Dreams of Universality, Interdisciplinarity Reality'")
    print("Journal of the American Society for Information Science and Technology")
    print("\n" + "="*80 + "\n")
    
    try:
        # Load and prepare data
        df, df_sdg, sdg_stats, total_policy_mentions, total_media_mentions = load_and_prepare_data()
        
        # Run all analyses
        print("\n" + "="*80)
        print("STARTING ANALYSIS PIPELINE")
        print("="*80 + "\n")
        
        # 1. Enhanced distribution graph
        create_enhanced_distribution(sdg_stats, len(df), total_policy_mentions, total_media_mentions)
        
        # 2. Extract top papers per SDG
        top_papers_df = extract_top_papers_per_sdg(df_sdg)
        
        # 3. Scatter plot: corpus vs policy share
        create_scatter_plot(sdg_stats, len(df), total_policy_mentions)
        
        # 4. Citation network (will show warning if reference column not found)
        create_citation_network(df_sdg)
        
        # 5. Impact profiles
        create_impact_profiles(sdg_stats, len(df))
        
        # 6. Policy effectiveness analysis
        create_policy_effectiveness_analysis(sdg_stats, df_sdg)
        
        # 7. Summary report
        generate_summary_report(sdg_stats, df, df_sdg, total_policy_mentions)
        
        print("\n" + "="*80)
        print(" "*20 + "🎉 ALL ANALYSES COMPLETED SUCCESSFULLY 🎉")
        print("="*80 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found")
        print(f"   Expected: {INPUT_PATH}")
        print(f"\n   Please check:")
        print(f"   1. File path is correct")
        print(f"   2. File exists in the specified location")
        print(f"   3. You have read permissions for the file\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred")
        print(f"   {str(e)}")
        print(f"\n   Please check:")
        print(f"   1. Input data format matches expected structure")
        print(f"   2. All required columns are present")
        print(f"   3. Data types are compatible")
        import traceback
        print(f"\n   Full error traceback:")
        traceback.print_exc()
        print()

if __name__ == "__main__":
    main()
