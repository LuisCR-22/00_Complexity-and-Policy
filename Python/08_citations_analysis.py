"""
==============================================================================
COMPLEXITY SCIENCE CITATION ANALYSIS - COMPREHENSIVE VERSION
==============================================================================

This script analyzes the merged OpenAlex-Altmetric dataset for complexity 
science papers, focusing on citation patterns across academic and policy domains.

Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Global Complexity School 2025 Final Project
Date: November 2025

==============================================================================
OUTPUT FILES STRUCTURE:
==============================================================================

EXCEL FILES (in Excel directory):
- 05_citation_metrics.xlsx: Descriptive statistics for all citation types
- 06_top_papers.xlsx: Top 10 papers for each citation type (separate sheets)
- 07_journal_analysis.xlsx: Journal patterns (policy vs academic)
- 08_institution_analysis.xlsx: Top institutions by impact type
- 09_country_analysis.xlsx: Country patterns and collaborations
- 10_topic_analysis.xlsx: Topics/subfields/fields/domains distributions
- 11_keyword_analysis.xlsx: Keyword frequencies and co-occurrences
- 12_sdg_analysis.xlsx: SDG associations and frequencies
- 13_temporal_analysis.xlsx: Time series data

IMAGE FILES (in Images directory):
- 05_*.png: Citation distribution visualizations
- 07_*.png: Journal comparisons
- 08_*.png: Institution comparisons
- 09_*.png: Country analysis and collaboration networks
- 10_*.png: Topic/field/domain comparisons
- 11_*.png: Keyword networks and word clouds
- 12_*.png: SDG visualizations and networks
- 13_*.png: Temporal trends

==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import re
from typing import Dict, List, Tuple, Set
import networkx as nx
from wordcloud import WordCloud
import json
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Global plot settings for publication quality
PLOT_SETTINGS = {
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}

for key, value in PLOT_SETTINGS.items():
    plt.rcParams[key] = value

# Color schemes for different analysis types
COLORS = {
    'policy': '#E74C3C',      # Red for policy
    'academic': '#3498DB',     # Blue for academic
    'media': '#9B59B6',        # Purple for media
    'social': '#F39C12',       # Orange for social media
    'readers': '#1ABC9C',      # Teal for readers
    'other': '#95A5A6',        # Gray for other
    'network_high': '#E74C3C', # High centrality
    'network_mid': '#F39C12',  # Medium centrality
    'network_low': '#95A5A6'   # Low centrality
}


class ComplexityScienceAnalyzer:
    """
    Main analyzer class for complexity science citation data
    
    This class performs comprehensive analysis of merged OpenAlex-Altmetric
    data, including:
    - Citation distribution analysis
    - Network visualizations (countries, keywords, SDGs, topics)
    - Temporal trend analysis
    - Comparative analysis (policy vs academic impact)
    """
    
    def __init__(self, input_path: str, output_excel_dir: str, output_images_dir: str):
        """
        Initialize analyzer with file paths
        
        Args:
            input_path: Path to merged Excel file (04_OA_Altmetrics_merge.xlsx)
            output_excel_dir: Directory for Excel outputs
            output_images_dir: Directory for PNG images
        """
        self.input_path = input_path
        self.output_excel_dir = output_excel_dir
        self.output_images_dir = output_images_dir
        
        print("="*80)
        print(" "*20 + "COMPLEXITY SCIENCE CITATION ANALYSIS")
        print("="*80)
        print(f"\n📂 Loading data from:\n   {input_path}")
        
        # Load data
        self.df = pd.read_excel(input_path)
        print(f"\n✓ Loaded {len(self.df):,} papers with {len(self.df.columns)} columns")
        
        # Define citation columns with their display names and categories
        self.citation_columns = {
            'Academic Citations (OA)': ('cited_by_count_oa', 'academic'),
            'Academic Citations (Alt)': ('citations_alt', 'academic'),
            'Policy Mentions': ('policy_mentions_total_alt', 'policy'),
            'Mainstream Media': ('news_msm_total_alt', 'media'),
            'Blogs': ('blogs_total_alt', 'media'),
            'Reddit': ('reddit_total_alt', 'social'),
            'Twitter/X': ('twitter_total_alt', 'social'),
            'Facebook': ('facebook_total_alt', 'social'),
            'Bluesky': ('bluesky_total_alt', 'social'),
            'Wikipedia': ('wikipedia_mentions_alt', 'other'),
            'Videos': ('video_mentions_alt', 'media'),
            'Podcasts': ('podcast_mentions_alt', 'media'),
            'Q&A Sites': ('qna_mentions_alt', 'other'),
            'Guidelines': ('guideline_mentions_alt', 'other'),
            'Mendeley Readers': ('readers_mendeley_alt', 'readers'),
            'Peer Reviews': ('peer_reviews_alt', 'academic'),
            'Patent Mentions': ('patent_mentions_alt', 'other')
        }
        
        # Ensure all citation columns exist (fill with 0 if missing)
        for display_name, (col_name, category) in self.citation_columns.items():
            if col_name not in self.df.columns:
                print(f"⚠ Warning: Column '{col_name}' not found, creating with zeros")
                self.df[col_name] = 0
            else:
                self.df[col_name] = self.df[col_name].fillna(0)
        
        # Extract publication years from both sources
        self._prepare_temporal_data()
        
        print(f"\n✓ Data preparation complete")
        print(f"  • Years covered: {self.df['year'].min():.0f} - {self.df['year'].max():.0f}")
        print(f"  • Papers with policy citations: {(self.df['policy_mentions_total_alt'] > 0).sum():,}")
        print(f"  • Papers with academic citations: {(self.df['cited_by_count_oa'] > 0).sum():,}")
        
        print("\n" + "="*80)
        print(" "*25 + "INITIALIZATION COMPLETE")
        print("="*80)
    
    def _prepare_temporal_data(self):
        """Prepare temporal data using all available years"""
        # Try to get publication year from OA first, then Altmetric
        year_oa = pd.to_numeric(self.df.get('publication_year_oa', pd.Series()), errors='coerce')
        year_alt = pd.to_numeric(self.df.get('publication_year_alt', pd.Series()), errors='coerce')
        
        # Use OA year if available, otherwise Altmetric
        self.df['year'] = year_oa.fillna(year_alt)
        
        # For papers with no year, try to extract from publication_date
        if 'publication_date_oa' in self.df.columns:
            missing_year = self.df['year'].isna()
            date_series = pd.to_datetime(self.df.loc[missing_year, 'publication_date_oa'], errors='coerce')
            self.df.loc[missing_year, 'year'] = date_series.dt.year
        
        # Filter to reasonable years (1950-2025)
        self.df.loc[self.df['year'] < 1950, 'year'] = np.nan
        self.df.loc[self.df['year'] > 2025, 'year'] = np.nan
    
    def run_all_analyses(self):
        """
        Execute all analysis steps in sequence
        
        This is the main execution method that runs all analyses:
        1. Citation distributions (05)
        2. Top papers (06)
        3. Journal analysis (07)
        4. Institution analysis (08)
        5. Country analysis with networks (09)
        6. Topic analysis with networks (10)
        7. Keyword analysis with networks (11)
        8. SDG analysis with networks (12)
        9. Temporal trends (13)
        """
        print("\n" + "="*80)
        print(" "*25 + "STARTING COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        analyses = [
            (1, "Citation Distributions", self.analyze_citation_distributions),
            (2, "Top Papers by Citation Type", self.create_top_papers_file),
            (3, "Journal Patterns", self.analyze_journals),
            (4, "Institutional Affiliations", self.analyze_institutions),
            (5, "Country Patterns & Collaboration Networks", self.analyze_countries),
            (6, "Topic/Field/Domain Distributions & Networks", self.analyze_topics),
            (7, "Keyword Analysis & Co-occurrence Networks", self.analyze_keywords),
            (8, "SDG Associations & Networks", self.analyze_sdgs),
            (9, "Temporal Trends", self.analyze_temporal_trends)
        ]
        
        for num, name, func in analyses:
            print(f"\n{'='*80}")
            print(f"[{num}/9] {name}")
            print(f"{'='*80}")
            try:
                func()
                print(f"✓ Completed: {name}")
            except Exception as e:
                print(f"✗ Error in {name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print(" "*30 + "✓ ALL ANALYSES COMPLETE")
        print("="*80)
        print(f"\n📁 Outputs saved to:")
        print(f"   Excel files: {self.output_excel_dir}")
        print(f"   Images: {self.output_images_dir}")
        print("\n" + "="*80)
    
    # ========================================================================
    # SECTION 1: CITATION DISTRIBUTIONS (File 05)
    # ========================================================================
    
    def analyze_citation_distributions(self):
        """
        Analyze and visualize citation distributions across all types
        
        Outputs:
        - 05_citation_metrics.xlsx: Descriptive statistics
        - 05_citation_comparison.png: Overview comparison
        - 05_citation_distributions.png: Distribution histograms
        - 05_citation_boxplots.png: Box plot comparison
        - 05_citation_correlations.png: Correlation heatmap
        - 05_papers_by_category.png: Papers with citations by category
        """
        print("\n   📊 Computing descriptive statistics...")
        
        # Calculate descriptive statistics
        stats_dict = {}
        
        for display_name, (col_name, category) in self.citation_columns.items():
            data = self.df[col_name].dropna()
            
            stats_dict[display_name] = {
                'Category': category.title(),
                'Count': len(data),
                'Papers with >0': (data > 0).sum(),
                'Percentage >0': f"{(data > 0).sum() / len(data) * 100:.2f}%",
                'Mean': data.mean(),
                'Median': data.median(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Q25': data.quantile(0.25),
                'Q50': data.quantile(0.50),
                'Q75': data.quantile(0.75),
                'Q90': data.quantile(0.90),
                'Q95': data.quantile(0.95),
                'Q99': data.quantile(0.99),
                'Total': data.sum()
            }
        
        stats_df = pd.DataFrame(stats_dict).T
        
        # Save to Excel
        excel_path = f"{self.output_excel_dir}/05_citation_metrics.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Descriptive Statistics')
            
            # Add raw data for each citation type
            for display_name, (col_name, category) in self.citation_columns.items():
                sheet_name = display_name[:31]
                data_df = pd.DataFrame({
                    'Value': self.df[col_name].dropna()
                }).sort_values('Value', ascending=False).reset_index(drop=True)
                data_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"   ✓ Saved: {excel_path}")
        
        # Create visualizations
        print("   📈 Creating visualizations...")
        self._plot_citation_comparison(stats_df)
        self._plot_citation_distributions()
        self._plot_citation_boxplots()
        self._plot_citation_correlations()
        self._plot_papers_by_category()
        
        print("   ✓ All distribution visualizations complete")
    
    def _plot_citation_comparison(self, stats_df: pd.DataFrame):
        """Create comprehensive comparison of citation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Define colors by category
        colors_map = {display: COLORS.get(cat, COLORS['other']) 
                     for display, (col, cat) in self.citation_columns.items()}
        
        # 1. Mean citations
        ax1 = axes[0, 0]
        means = stats_df['Mean'].sort_values(ascending=True)
        colors = [colors_map.get(idx, COLORS['other']) for idx in means.index]
        means.plot(kind='barh', ax=ax1, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Mean Citations', fontweight='bold')
        ax1.set_title('Mean Citations by Type', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 2. Median citations
        ax2 = axes[0, 1]
        medians = stats_df['Median'].sort_values(ascending=True)
        colors = [colors_map.get(idx, COLORS['other']) for idx in medians.index]
        medians.plot(kind='barh', ax=ax2, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Median Citations', fontweight='bold')
        ax2.set_title('Median Citations by Type', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 3. Total citations (log scale for visibility)
        ax3 = axes[1, 0]
        totals = stats_df['Total'].sort_values(ascending=True)
        colors = [colors_map.get(idx, COLORS['other']) for idx in totals.index]
        ax3.barh(range(len(totals)), totals, color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_yticks(range(len(totals)))
        ax3.set_yticklabels(totals.index)
        ax3.set_xlabel('Total Citations (log scale)', fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_title('Total Citations by Type', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 4. Percentage with citations
        ax4 = axes[1, 1]
        pct_data = stats_df['Papers with >0'] / stats_df['Count'] * 100
        pct_data = pct_data.sort_values(ascending=True)
        colors = [colors_map.get(idx, COLORS['other']) for idx in pct_data.index]
        pct_data.plot(kind='barh', ax=ax4, color=colors, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Percentage of Papers (%)', fontweight='bold')
        ax4.set_title('Papers with at Least One Citation', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.suptitle('Citation Metrics Overview - Complexity Science Papers',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/05_citation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 05_citation_comparison.png")
    
    def _plot_citation_distributions(self):
        """Plot distribution histograms for key citation types"""
        main_types = {
            'Academic Citations (OA)': ('cited_by_count_oa', 'academic'),
            'Policy Mentions': ('policy_mentions_total_alt', 'policy'),
            'Mainstream Media': ('news_msm_total_alt', 'media'),
            'Twitter/X': ('twitter_total_alt', 'social'),
            'Mendeley Readers': ('readers_mendeley_alt', 'readers'),
            'Blogs': ('blogs_total_alt', 'media')
        }
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        for idx, (display_name, (col_name, category)) in enumerate(main_types.items()):
            ax = axes[idx]
            data = self.df[col_name].dropna()
            data_nonzero = data[data > 0]
            
            if len(data_nonzero) > 0:
                color = COLORS.get(category, COLORS['other'])
                
                # Histogram with log scale
                ax.hist(np.log10(data_nonzero + 1), bins=50, 
                       color=color, edgecolor='black', alpha=0.7, linewidth=0.5)
                ax.set_xlabel('Log₁₀(Citations + 1)', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'{display_name}\n(n={len(data_nonzero):,} papers with citations)',
                           fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3, linestyle='--')
                
                # Add statistics
                median_val = data_nonzero.median()
                mean_val = data_nonzero.mean()
                ax.axvline(np.log10(median_val + 1), color='red', 
                          linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
                ax.axvline(np.log10(mean_val + 1), color='darkred', 
                          linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}')
                ax.legend(loc='upper right', framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'No citations', ha='center', va='center',
                       fontsize=14, fontweight='bold')
                ax.set_title(display_name, fontsize=12, fontweight='bold')
        
        plt.suptitle('Citation Distributions - Main Types',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/05_citation_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 05_citation_distributions.png")
    
    def _plot_citation_boxplots(self):
        """Create box plots for citation distributions"""
        plot_data = []
        plot_labels = []
        plot_colors = []
        
        for display_name, (col_name, category) in self.citation_columns.items():
            data = self.df[col_name].dropna()
            data_nonzero = data[data > 0]
            if len(data_nonzero) > 10:  # Only include if sufficient data
                plot_data.append(np.log10(data_nonzero + 1))
                plot_labels.append(display_name)
                plot_colors.append(COLORS.get(category, COLORS['other']))
        
        fig, ax = plt.subplots(figsize=(16, 12))
        bp = ax.boxplot(plot_data, labels=plot_labels, vert=False, 
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Log₁₀(Citations + 1)', fontweight='bold', fontsize=13)
        ax.set_title('Citation Distributions by Type\n(Papers with >0 citations, outliers removed)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/05_citation_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 05_citation_boxplots.png")
    
    def _plot_citation_correlations(self):
        """Plot correlation heatmap between citation types"""
        cols_to_use = [col for col, cat in self.citation_columns.values() if col in self.df.columns]
        
        # Calculate correlations
        corr_matrix = self.df[cols_to_use].corr()
        
        # Rename for display
        rename_dict = {col: display for display, (col, cat) in self.citation_columns.items() 
                      if col in cols_to_use}
        corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)
        
        # Plot
        fig, ax = plt.subplots(figsize=(18, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                   annot_kws={"size": 8}, ax=ax)
        
        ax.set_title('Correlation Matrix: Citation Types\n(Pearson Correlation)',
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/05_citation_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 05_citation_correlations.png")
    
    def _plot_papers_by_category(self):
        """Plot number of papers by citation category"""
        categories = {
            'Academic': ['cited_by_count_oa', 'citations_alt', 'peer_reviews_alt'],
            'Policy': ['policy_mentions_total_alt'],
            'Traditional Media': ['news_msm_total_alt'],
            'Social Media': ['twitter_total_alt', 'facebook_total_alt', 
                           'reddit_total_alt', 'bluesky_total_alt'],
            'Online Content': ['blogs_total_alt', 'wikipedia_mentions_alt', 
                             'video_mentions_alt', 'podcast_mentions_alt'],
            'Academic Platforms': ['readers_mendeley_alt'],
            'Other': ['qna_mentions_alt', 'guideline_mentions_alt', 
                     'patent_mentions_alt']
        }
        
        category_counts = {}
        for cat_name, cols in categories.items():
            mask = pd.Series(False, index=self.df.index)
            for col in cols:
                if col in self.df.columns:
                    mask = mask | (self.df[col] > 0)
            category_counts[cat_name] = mask.sum()
        
        # Sort by count
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1])
        cats = [c[0] for c in sorted_cats]
        counts = [c[1] for c in sorted_cats]
        percentages = [c / len(self.df) * 100 for c in counts]
        
        # Colors
        cat_colors = {
            'Academic': COLORS['academic'],
            'Policy': COLORS['policy'],
            'Traditional Media': COLORS['media'],
            'Social Media': COLORS['social'],
            'Online Content': COLORS['media'],
            'Academic Platforms': COLORS['readers'],
            'Other': COLORS['other']
        }
        colors = [cat_colors.get(c, COLORS['other']) for c in cats]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        bars = ax.barh(cats, counts, color=colors, edgecolor='black', linewidth=1)
        
        # Add labels
        for idx, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {count:,} ({pct:.1f}%)',
                   ha='left', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Number of Papers', fontweight='bold', fontsize=13)
        ax.set_title('Papers with Citations by Category\n(Papers may appear in multiple categories)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/05_papers_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 05_papers_by_category.png")
    
    # ========================================================================
    # SECTION 2: TOP PAPERS (File 06)
    # ========================================================================
    
    def create_top_papers_file(self):
        """
        Create Excel file with top 10 papers for each citation type
        
        Output:
        - 06_top_papers.xlsx: Multiple sheets, one per citation type
        
        Each sheet shows:
        1. Title (from both sources)
        2. Policy citations (highlighted first)
        3. All other citation types
        4. Metadata (authors, institutions, topics, etc.)
        """
        print("\n   📝 Identifying top papers...")
        
        # Priority columns for display
        priority_cols = []
        
        # Titles first
        for col in ['title_oa', 'title_alt']:
            if col in self.df.columns:
                priority_cols.append(col)
        
        # Policy citations next
        if 'policy_mentions_total_alt' in self.df.columns:
            priority_cols.append('policy_mentions_total_alt')
        
        # All other citations
        for display_name, (col_name, category) in self.citation_columns.items():
            if col_name in self.df.columns and col_name not in priority_cols:
                priority_cols.append(col_name)
        
        # Metadata columns
        metadata_cols = [
            'doi_oa', 'doi_alt', 'year', 'publication_year_oa', 'publication_date_alt',
            'source_name_oa', 'journal_alt',
            'authors_names_oa', 'authors_count_oa',
            'institutions_oa', 'countries_oa',
            'primary_topic_oa', 'primary_subfield_oa', 'primary_field_oa', 'primary_domain_oa',
            'keywords_all_oa', 'concepts_oa',
            'sdg_name_oa', 'sdg_score_oa',
            'is_oa_oa', 'oa_status_oa',
            'altmetric_score_alt',
            'fwci_oa', 'cited_by_count_oa'
        ]
        
        # Combine columns (only those that exist)
        all_cols = []
        for col in priority_cols + metadata_cols:
            if col in self.df.columns and col not in all_cols:
                all_cols.append(col)
        
        # Add match info if exists
        if 'match_type' in self.df.columns:
            all_cols.insert(0, 'match_type')
        
        excel_path = f"{self.output_excel_dir}/06_top_papers.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for display_name, (col_name, category) in self.citation_columns.items():
                if col_name in self.df.columns:
                    # Get top 10 papers (or fewer if not enough data)
                    n_top = min(10, (self.df[col_name] > 0).sum())
                    if n_top > 0:
                        top_papers = self.df.nlargest(n_top, col_name)[all_cols].copy()
                        
                        # Clean sheet name (max 31 chars for Excel)
                        sheet_name = display_name[:31]
                        
                        # Save to sheet
                        top_papers.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        print(f"   ✓ Created sheet: {sheet_name} ({n_top} papers)")
        
        print(f"   ✓ Saved: {excel_path}")
    
    # ========================================================================
    # SECTION 3: JOURNAL ANALYSIS (File 07)
    # ========================================================================
    
    def analyze_journals(self):
        """
        Analyze journal patterns in policy vs academic citations
        
        Outputs:
        - 07_journal_analysis.xlsx: Top journals overall, policy, academic
        - 07_journal_comparison.png: Visual comparison
        """
        print("\n   📚 Analyzing journal distributions...")
        
        # Prepare journal data
        self.df['journal'] = self.df.get('source_name_oa', '').fillna(
            self.df.get('journal_alt', ''))
        
        # Create citation categories
        self.df['has_policy'] = self.df['policy_mentions_total_alt'] > 0
        self.df['has_academic'] = self.df['cited_by_count_oa'] > 0
        
        # Top journals
        journal_counts = self.df['journal'].value_counts().head(30)
        policy_papers = self.df[self.df['has_policy']]
        policy_journals = policy_papers['journal'].value_counts().head(30)
        academic_papers = self.df[self.df['has_academic']]
        academic_journals = academic_papers['journal'].value_counts().head(30)
        
        # Save to Excel
        excel_path = f"{self.output_excel_dir}/07_journal_analysis.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame({
                'Journal': journal_counts.index,
                'Paper Count': journal_counts.values,
                'Percentage': (journal_counts.values / len(self.df) * 100).round(2)
            }).to_excel(writer, sheet_name='Top Journals Overall', index=False)
            
            pd.DataFrame({
                'Journal': policy_journals.index,
                'Papers with Policy Citations': policy_journals.values,
                'Percentage': (policy_journals.values / len(policy_papers) * 100).round(2)
            }).to_excel(writer, sheet_name='Top Journals Policy', index=False)
            
            pd.DataFrame({
                'Journal': academic_journals.index,
                'Papers with Academic Citations': academic_journals.values,
                'Percentage': (academic_journals.values / len(academic_papers) * 100).round(2)
            }).to_excel(writer, sheet_name='Top Journals Academic', index=False)
        
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualization
        self._plot_journal_comparison(policy_journals, academic_journals)
        print(f"   ✓ Saved: 07_journal_comparison.png")
    
    def _plot_journal_comparison(self, policy_journals, academic_journals):
        """Plot comparison of top journals"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
        # Policy journals
        ax1 = axes[0]
        top_policy = policy_journals.head(20)
        ax1.barh(range(len(top_policy)), top_policy.values, 
                color=COLORS['policy'], edgecolor='black', linewidth=0.8, alpha=0.8)
        ax1.set_yticks(range(len(top_policy)))
        ax1.set_yticklabels(top_policy.index, fontsize=10)
        ax1.set_xlabel('Number of Papers with Policy Citations', fontweight='bold', fontsize=12)
        ax1.set_title('Top 20 Journals: Policy Impact', fontweight='bold', fontsize=14, pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add counts
        for idx, (j, count) in enumerate(top_policy.items()):
            ax1.text(count, idx, f' {count}', va='center', fontweight='bold')
        
        # Academic journals  
        ax2 = axes[1]
        top_academic = academic_journals.head(20)
        ax2.barh(range(len(top_academic)), top_academic.values,
                color=COLORS['academic'], edgecolor='black', linewidth=0.8, alpha=0.8)
        ax2.set_yticks(range(len(top_academic)))
        ax2.set_yticklabels(top_academic.index, fontsize=10)
        ax2.set_xlabel('Number of Papers with Academic Citations', fontweight='bold', fontsize=12)
        ax2.set_title('Top 20 Journals: Academic Impact', fontweight='bold', fontsize=14, pad=15)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add counts
        for idx, (j, count) in enumerate(top_academic.items()):
            ax2.text(count, idx, f' {count}', va='center', fontweight='bold')
        
        plt.suptitle('Journal Comparison: Policy vs Academic Impact',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/07_journal_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # SECTION 4: INSTITUTION ANALYSIS (File 08)
    # ========================================================================
    
    def analyze_institutions(self):
        """
        Analyze institutional patterns
        
        Outputs:
        - 08_institution_analysis.xlsx: Top institutions
        - 08_institution_comparison.png: Visual comparison
        """
        print("\n   🏛️ Analyzing institutional affiliations...")
        
        all_institutions = []
        policy_institutions = []
        academic_institutions = []
        
        for idx, row in self.df.iterrows():
            inst_str = row.get('institutions_oa', '')
            if pd.notna(inst_str) and inst_str != '':
                institutions = [i.strip() for i in str(inst_str).split(';') if i.strip()]
                all_institutions.extend(institutions)
                
                if row.get('policy_mentions_total_alt', 0) > 0:
                    policy_institutions.extend(institutions)
                
                if row.get('cited_by_count_oa', 0) > 0:
                    academic_institutions.extend(institutions)
        
        inst_counter = Counter(all_institutions)
        policy_counter = Counter(policy_institutions)
        academic_counter = Counter(academic_institutions)
        
        top_overall = inst_counter.most_common(50)
        top_policy = policy_counter.most_common(50)
        top_academic = academic_counter.most_common(50)
        
        # Save to Excel
        excel_path = f"{self.output_excel_dir}/08_institution_analysis.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(top_overall, columns=['Institution', 'Paper Count']).to_excel(
                writer, sheet_name='Top Institutions Overall', index=False)
            pd.DataFrame(top_policy, columns=['Institution', 'Papers in Policy']).to_excel(
                writer, sheet_name='Top Institutions Policy', index=False)
            pd.DataFrame(top_academic, columns=['Institution', 'Papers in Academic']).to_excel(
                writer, sheet_name='Top Institutions Academic', index=False)
        
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualization
        self._plot_institution_comparison(top_policy, top_academic)
        print(f"   ✓ Saved: 08_institution_comparison.png")
    
    def _plot_institution_comparison(self, policy_insts, academic_insts):
        """Plot institutional comparison"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 20))
        
        # Policy institutions
        ax1 = axes[0]
        insts = [i[0] for i in policy_insts[:25]]
        counts = [i[1] for i in policy_insts[:25]]
        ax1.barh(range(len(insts)), counts, color=COLORS['policy'], 
                edgecolor='black', linewidth=0.8, alpha=0.8)
        ax1.set_yticks(range(len(insts)))
        ax1.set_yticklabels(insts, fontsize=9)
        ax1.set_xlabel('Number of Papers with Policy Citations', fontweight='bold', fontsize=12)
        ax1.set_title('Top 25 Institutions: Policy Impact', fontweight='bold', fontsize=14, pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        for idx, count in enumerate(counts):
            ax1.text(count, idx, f' {count}', va='center', fontweight='bold', fontsize=9)
        
        # Academic institutions
        ax2 = axes[1]
        insts = [i[0] for i in academic_insts[:25]]
        counts = [i[1] for i in academic_insts[:25]]
        ax2.barh(range(len(insts)), counts, color=COLORS['academic'],
                edgecolor='black', linewidth=0.8, alpha=0.8)
        ax2.set_yticks(range(len(insts)))
        ax2.set_yticklabels(insts, fontsize=9)
        ax2.set_xlabel('Number of Papers with Academic Citations', fontweight='bold', fontsize=12)
        ax2.set_title('Top 25 Institutions: Academic Impact', fontweight='bold', fontsize=14, pad=15)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        for idx, count in enumerate(counts):
            ax2.text(count, idx, f' {count}', va='center', fontweight='bold', fontsize=9)
        
        plt.suptitle('Institution Comparison: Policy vs Academic Impact',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/08_institution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # SECTION 5: COUNTRY ANALYSIS WITH ADVANCED NETWORKS (File 09)
    # ========================================================================
    
    def analyze_countries(self):
        """
        Analyze country patterns and create sophisticated collaboration networks
        
        Outputs:
        - 09_country_analysis.xlsx: Country statistics and collaborations
        - 09_country_comparison.png: Policy vs academic comparison
        - 09_collaboration_network_full.png: Full collaboration network
        - 09_collaboration_network_policy.png: Policy-focused network
        - 09_collaboration_network_academic.png: Academic-focused network
        
        Network Design Philosophy:
        - Node size = total paper count (shows productivity)
        - Node color = centrality (red=high, orange=medium, gray=low)
        - Edge width = number of collaborations (shows strength)
        - Layout = force-directed (clusters collaborators)
        """
        print("\n   🌍 Analyzing country patterns...")
        
        # Parse countries
        all_countries = []
        policy_countries = []
        academic_countries = []
        
        # Collaborations: (country1, country2, type)
        all_collabs = []
        policy_collabs = []
        academic_collabs = []
        
        for idx, row in self.df.iterrows():
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
        
        # Count everything
        country_counter = Counter(all_countries)
        policy_counter = Counter(policy_countries)
        academic_counter = Counter(academic_countries)
        collab_counter = Counter(all_collabs)
        policy_collab_counter = Counter(policy_collabs)
        academic_collab_counter = Counter(academic_collabs)
        
        print(f"   • Total countries: {len(country_counter)}")
        print(f"   • Total collaborations: {len(collab_counter)}")
        
        # Top items
        top_overall = country_counter.most_common(50)
        top_policy = policy_counter.most_common(50)
        top_academic = academic_counter.most_common(50)
        top_collabs = collab_counter.most_common(100)
        
        # Save to Excel
        excel_path = f"{self.output_excel_dir}/09_country_analysis.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(top_overall, columns=['Country', 'Paper Count']).to_excel(
                writer, sheet_name='Top Countries Overall', index=False)
            pd.DataFrame(top_policy, columns=['Country', 'Papers in Policy']).to_excel(
                writer, sheet_name='Top Countries Policy', index=False)
            pd.DataFrame(top_academic, columns=['Country', 'Papers in Academic']).to_excel(
                writer, sheet_name='Top Countries Academic', index=False)
            
            # Collaboration data
            collab_df = pd.DataFrame([
                {
                    'Country 1': c[0], 
                    'Country 2': c[1], 
                    'Total Collaborations': collab_counter[c],
                    'Policy Collaborations': policy_collab_counter.get(c, 0),
                    'Academic Collaborations': academic_collab_counter.get(c, 0)
                }
                for c in top_collabs
            ])
            collab_df.to_excel(writer, sheet_name='Top Collaborations', index=False)
        
        print(f"   ✓ Saved: {excel_path}")
        
        # Create visualizations
        print("   📊 Creating visualizations...")
        self._plot_country_comparison(top_policy, top_academic)
        
        print("   🔗 Creating collaboration networks...")
        self._plot_collaboration_network(
            country_counter, collab_counter, 
            'Full Network', '09_collaboration_network_full.png')
        
        self._plot_collaboration_network(
            policy_counter, policy_collab_counter,
            'Policy Impact Network', '09_collaboration_network_policy.png')
        
        self._plot_collaboration_network(
            academic_counter, academic_collab_counter,
            'Academic Impact Network', '09_collaboration_network_academic.png')
        
        print("   ✓ All country analyses complete")
    
    def _plot_country_comparison(self, policy_countries, academic_countries):
        """Plot country comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
        # Policy countries
        ax1 = axes[0]
        countries = [c[0] for c in policy_countries[:25]]
        counts = [c[1] for c in policy_countries[:25]]
        ax1.barh(range(len(countries)), counts, color=COLORS['policy'],
                edgecolor='black', linewidth=0.8, alpha=0.8)
        ax1.set_yticks(range(len(countries)))
        ax1.set_yticklabels(countries, fontsize=10)
        ax1.set_xlabel('Number of Papers with Policy Citations', fontweight='bold', fontsize=12)
        ax1.set_title('Top 25 Countries: Policy Impact', fontweight='bold', fontsize=14, pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        for idx, count in enumerate(counts):
            ax1.text(count, idx, f' {count}', va='center', fontweight='bold')
        
        # Academic countries
        ax2 = axes[1]
        countries = [c[0] for c in academic_countries[:25]]
        counts = [c[1] for c in academic_countries[:25]]
        ax2.barh(range(len(countries)), counts, color=COLORS['academic'],
                edgecolor='black', linewidth=0.8, alpha=0.8)
        ax2.set_yticks(range(len(countries)))
        ax2.set_yticklabels(countries, fontsize=10)
        ax2.set_xlabel('Number of Papers with Academic Citations', fontweight='bold', fontsize=12)
        ax2.set_title('Top 25 Countries: Academic Impact', fontweight='bold', fontsize=14, pad=15)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        for idx, count in enumerate(counts):
            ax2.text(count, idx, f' {count}', va='center', fontweight='bold')
        
        plt.suptitle('Country Comparison: Policy vs Academic Impact',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/09_country_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 09_country_comparison.png")
    
    def _plot_collaboration_network(self, country_counter: Counter, 
                                   collab_counter: Counter, 
                                   title: str, filename: str):
        """
        Create sophisticated collaboration network
        
        Visual Design:
        - Node size: Proportional to paper count (min 100, max 3000)
        - Node color: Based on betweenness centrality
          * Red (high): Central connectors (top 10%)
          * Orange (medium): Important nodes (top 25%)
          * Light gray (low): Peripheral nodes
        - Edge width: Proportional to collaboration count (min 0.5, max 8)
        - Edge color: Semi-transparent gray
        - Layout: Force-directed spring layout for natural clustering
        - Labels: Country codes, sized by importance
        """
        # Select top countries to show
        top_countries = [c[0] for c in country_counter.most_common(40)]
        
        # Create network
        G = nx.Graph()
        G.add_nodes_from(top_countries)
        
        # Add edges with weights
        for (c1, c2), weight in collab_counter.items():
            if c1 in top_countries and c2 in top_countries and weight >= 2:
                G.add_edge(c1, c2, weight=weight)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if len(G.nodes()) == 0:
            print(f"   ⚠ No network to plot for {title}")
            return
        
        print(f"   • {title}: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Calculate network metrics
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        
        # Node sizes based on paper count
        node_sizes = []
        for node in G.nodes():
            count = country_counter[node]
            # Scale: 100 to 3000
            size = 100 + (count / max(country_counter.values())) * 2900
            node_sizes.append(size)
        
        # Node colors based on betweenness centrality
        node_colors = []
        centrality_values = list(betweenness_cent.values())
        if centrality_values:
            threshold_high = np.percentile(centrality_values, 90)
            threshold_med = np.percentile(centrality_values, 75)
            
            for node in G.nodes():
                cent = betweenness_cent[node]
                if cent >= threshold_high:
                    node_colors.append(COLORS['network_high'])  # Red - highly central
                elif cent >= threshold_med:
                    node_colors.append(COLORS['network_mid'])   # Orange - medium
                else:
                    node_colors.append(COLORS['network_low'])   # Gray - peripheral
        else:
            node_colors = [COLORS['network_low']] * len(G.nodes())
        
        # Edge widths based on collaboration count
        edge_widths = []
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        
        for u, v in G.edges():
            weight = G[u][v]['weight']
            # Scale: 0.5 to 8
            width = 0.5 + (weight / max_weight) * 7.5
            edge_widths.append(width)
        
        # Layout
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42, weight='weight')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(24, 24))
        
        # Draw network
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, 
                              edge_color='gray', ax=ax)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              edgecolors='black', linewidths=2, alpha=0.9, ax=ax)
        
        # Labels - size based on degree centrality
        labels = {node: node for node in G.nodes()}
        for node, (x, y) in pos.items():
            size = 8 + degree_cent[node] * 20  # 8 to 28
            ax.text(x, y, labels[node], fontsize=size, fontweight='bold',
                   ha='center', va='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='black', alpha=0.7, linewidth=1))
        
        # Title and legend
        title_text = f'{title}\nInternational Collaboration Network - Complexity Science'
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=30)
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_high'],
                  markersize=15, label='High Centrality (Top 10%)', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_mid'],
                  markersize=15, label='Medium Centrality (Top 25%)', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['network_low'],
                  markersize=15, label='Lower Centrality', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], color='gray', linewidth=5, alpha=0.5, label='Strong Collaboration'),
            Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Weak Collaboration')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
                 frameon=True, fancybox=True, shadow=True, title='Legend',
                 title_fontsize=13)
        
        # Add explanation
        explanation = (
            "Node Size = Total Papers\n"
            "Node Color = Betweenness Centrality\n"
            "Edge Width = Collaboration Strength\n"
            f"Network: {len(G.nodes())} countries, {len(G.edges())} collaborations"
        )
        ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {filename}")
    
    # ========================================================================
    # SECTION 6: TOPIC ANALYSIS WITH NETWORKS (File 10)
    # ========================================================================
    
    def analyze_topics(self):
        """
        Analyze topic distributions and create topic networks
        
        Outputs:
        - 10_topic_analysis.xlsx: All levels of topic classification
        - 10_primary_topic_comparison.png
        - 10_primary_subfield_comparison.png
        - 10_primary_field_comparison.png
        - 10_primary_domain_comparison.png
        - 10_field_network.png: Network of fields
        - 10_domain_field_network.png: Bipartite network
        """
        print("\n   🎯 Analyzing topic distributions...")
        
        levels = {
            'Primary Topic': 'primary_topic_oa',
            'Primary Subfield': 'primary_subfield_oa',
            'Primary Field': 'primary_field_oa',
            'Primary Domain': 'primary_domain_oa'
        }
        
        excel_path = f"{self.output_excel_dir}/10_topic_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for level_name, col_name in levels.items():
                if col_name in self.df.columns:
                    # Overall distribution
                    overall = self.df[col_name].value_counts().head(50)
                    
                    # Policy papers
                    policy_papers = self.df[self.df['policy_mentions_total_alt'] > 0]
                    policy_dist = policy_papers[col_name].value_counts().head(50)
                    
                    # Academic papers
                    academic_papers = self.df[self.df['cited_by_count_oa'] > 0]
                    academic_dist = academic_papers[col_name].value_counts().head(50)
                    
                    # Combine
                    combined = pd.DataFrame({
                        level_name: overall.index,
                        'Overall Count': overall.values,
                        'Overall %': (overall.values / len(self.df) * 100).round(2),
                        'Policy Count': [policy_dist.get(t, 0) for t in overall.index],
                        'Policy %': [(policy_dist.get(t, 0) / len(policy_papers) * 100) 
                                    if len(policy_papers) > 0 else 0 for t in overall.index],
                        'Academic Count': [academic_dist.get(t, 0) for t in overall.index],
                        'Academic %': [(academic_dist.get(t, 0) / len(academic_papers) * 100) 
                                      if len(academic_papers) > 0 else 0 for t in overall.index]
                    })
                    
                    combined.to_excel(writer, sheet_name=level_name[:31], index=False)
        
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualizations
        print("   📊 Creating comparison visualizations...")
        self._plot_topic_distributions(levels)
        
        print("   🔗 Creating topic networks...")
        self._plot_field_network()
        self._plot_domain_field_network()
        
        print("   ✓ All topic analyses complete")
    
    def _plot_topic_distributions(self, levels):
        """Plot topic distribution comparisons"""
        for level_name, col_name in levels.items():
            if col_name not in self.df.columns:
                continue
            
            # Get distributions
            policy_papers = self.df[self.df['policy_mentions_total_alt'] > 0]
            policy_dist = policy_papers[col_name].value_counts().head(20)
            
            academic_papers = self.df[self.df['cited_by_count_oa'] > 0]
            academic_dist = academic_papers[col_name].value_counts().head(20)
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(20, 12))
            
            # Policy
            ax1 = axes[0]
            ax1.barh(range(len(policy_dist)), policy_dist.values, 
                    color=COLORS['policy'], edgecolor='black', linewidth=0.8, alpha=0.8)
            ax1.set_yticks(range(len(policy_dist)))
            ax1.set_yticklabels(policy_dist.index, fontsize=10)
            ax1.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
            ax1.set_title(f'Top 20 {level_name}: Policy Citations', 
                         fontweight='bold', fontsize=14, pad=15)
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            
            for idx, count in enumerate(policy_dist.values):
                ax1.text(count, idx, f' {count}', va='center', fontweight='bold')
            
            # Academic
            ax2 = axes[1]
            ax2.barh(range(len(academic_dist)), academic_dist.values,
                    color=COLORS['academic'], edgecolor='black', linewidth=0.8, alpha=0.8)
            ax2.set_yticks(range(len(academic_dist)))
            ax2.set_yticklabels(academic_dist.index, fontsize=10)
            ax2.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
            ax2.set_title(f'Top 20 {level_name}: Academic Citations',
                         fontweight='bold', fontsize=14, pad=15)
            ax2.invert_yaxis()
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            for idx, count in enumerate(academic_dist.values):
                ax2.text(count, idx, f' {count}', va='center', fontweight='bold')
            
            plt.suptitle(f'{level_name} Comparison: Policy vs Academic Impact',
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            filename = level_name.lower().replace(' ', '_')
            plt.savefig(f"{self.output_images_dir}/10_{filename}_comparison.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: 10_{filename}_comparison.png")
    
    def _plot_field_network(self):
        """
        Create network of primary fields showing co-occurrence in papers
        
        Visual Design:
        - Nodes = Primary fields
        - Edges = Papers that mention both fields (via secondary/tertiary topics)
        - Node size = Number of papers
        - Node color = Policy vs academic emphasis
        - Edge width = Co-occurrence strength
        """
        if 'primary_field_oa' not in self.df.columns:
            return
        
        # Get field data
        fields = self.df['primary_field_oa'].value_counts().head(20).index.tolist()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes with attributes
        for field in fields:
            field_papers = self.df[self.df['primary_field_oa'] == field]
            G.add_node(field,
                      papers=len(field_papers),
                      policy=field_papers['policy_mentions_total_alt'].sum(),
                      academic=field_papers['cited_by_count_oa'].sum())
        
        # Add edges based on co-occurrence in secondary/tertiary topics
        # (Papers may have multiple topic levels)
        for field1 in fields:
            for field2 in fields:
                if field1 < field2:  # Avoid duplicates
                    # Count papers that have both fields in any topic level
                    common = 0
                    for col in ['primary_field_oa', 'secondary_field_oa', 'tertiary_field_oa']:
                        if col in self.df.columns:
                            mask = (self.df[col] == field1) | (self.df[col] == field2)
                            common += (mask.sum() // 2)  # Rough estimate
                    
                    if common > 10:
                        G.add_edge(field1, field2, weight=common)
        
        if len(G.edges()) == 0:
            print("   ⚠ No field co-occurrences found for network")
            return
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        # Node sizes
        node_sizes = [G.nodes[node]['papers'] * 5 for node in G.nodes()]
        
        # Node colors based on policy/academic ratio
        node_colors = []
        for node in G.nodes():
            policy = G.nodes[node]['policy']
            academic = G.nodes[node]['academic']
            total = policy + academic
            if total > 0:
                ratio = policy / total
                if ratio > 0.3:
                    node_colors.append(COLORS['policy'])
                elif ratio < 0.1:
                    node_colors.append(COLORS['academic'])
                else:
                    node_colors.append(COLORS['other'])
            else:
                node_colors.append(COLORS['other'])
        
        # Edge widths
        edge_widths = [(G[u][v]['weight'] / 50) for u, v in G.edges()]
        
        # Plot
        fig, ax = plt.subplots(figsize=(20, 20))
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              edgecolors='black', linewidths=2, alpha=0.8, ax=ax)
        
        # Labels
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='black', alpha=0.8))
        
        ax.set_title('Primary Field Network\n(Node size = papers, Color = policy vs academic emphasis)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/10_field_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 10_field_network.png")
    
    def _plot_domain_field_network(self):
        """
        Create bipartite network of domains and fields
        
        Shows the hierarchical structure of complexity science
        """
        if 'primary_domain_oa' not in self.df.columns or 'primary_field_oa' not in self.df.columns:
            return
        
        # Get top domains and fields
        top_domains = self.df['primary_domain_oa'].value_counts().head(5).index.tolist()
        top_fields = self.df['primary_field_oa'].value_counts().head(15).index.tolist()
        
        # Create bipartite graph
        B = nx.Graph()
        
        # Add nodes
        B.add_nodes_from(top_domains, bipartite=0)  # Domains
        B.add_nodes_from(top_fields, bipartite=1)    # Fields
        
        # Add edges
        for domain in top_domains:
            for field in top_fields:
                count = len(self.df[(self.df['primary_domain_oa'] == domain) & 
                                   (self.df['primary_field_oa'] == field)])
                if count > 5:
                    B.add_edge(domain, field, weight=count)
        
        # Layout - bipartite
        pos = {}
        domain_y = np.linspace(0, 1, len(top_domains))
        field_y = np.linspace(0, 1, len(top_fields))
        
        for idx, domain in enumerate(top_domains):
            pos[domain] = (0, domain_y[idx])
        
        for idx, field in enumerate(top_fields):
            pos[field] = (1, field_y[idx])
        
        # Plot
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw edges
        for u, v, data in B.edges(data=True):
            width = data['weight'] / 20
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   'gray', linewidth=width, alpha=0.4, zorder=1)
        
        # Draw nodes
        for node in top_domains:
            ax.scatter(*pos[node], s=2000, c=COLORS['policy'], 
                      edgecolors='black', linewidths=2, zorder=2, alpha=0.8)
            ax.text(pos[node][0]-0.02, pos[node][1], node,
                   ha='right', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='black', alpha=0.9))
        
        for node in top_fields:
            ax.scatter(*pos[node], s=1500, c=COLORS['academic'],
                      edgecolors='black', linewidths=2, zorder=2, alpha=0.8)
            ax.text(pos[node][0]+0.02, pos[node][1], node,
                   ha='left', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='black', alpha=0.9))
        
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Domain-Field Hierarchy\n(Edge width = number of papers)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add labels
        ax.text(0, 1.05, 'DOMAINS', ha='center', va='bottom',
               fontsize=14, fontweight='bold', color=COLORS['policy'])
        ax.text(1, 1.05, 'FIELDS', ha='center', va='bottom',
               fontsize=14, fontweight='bold', color=COLORS['academic'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/10_domain_field_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 10_domain_field_network.png")
    
    # ========================================================================
    # SECTION 7: KEYWORD ANALYSIS WITH NETWORKS (File 11)
    # ========================================================================
    
    def analyze_keywords(self):
        """
        Analyze keywords with focus on co-occurrence networks
        
        Outputs:
        - 11_keyword_analysis.xlsx: Keyword frequencies
        - 11_keyword_wordcloud.png: Word cloud
        - 11_keyword_cooccurrence_full.png: Full co-occurrence network
        - 11_keyword_cooccurrence_policy.png: Policy papers network
        - 11_keyword_cooccurrence_academic.png: Academic papers network
        """
        print("\n   🔤 Analyzing keywords...")
        
        # Collect keywords
        all_keywords = []
        keyword_papers = defaultdict(list)  # keyword -> list of paper indices
        
        for idx, row in self.df.iterrows():
            keywords_str = row.get('keywords_all_oa', '')
            if pd.notna(keywords_str) and keywords_str != '':
                keywords = [k.strip().lower() for k in str(keywords_str).split(';') if k.strip()]
                all_keywords.extend(keywords)
                
                for kw in keywords:
                    keyword_papers[kw].append(idx)
        
        keyword_counter = Counter(all_keywords)
        top_keywords = keyword_counter.most_common(200)
        
        print(f"   • Total unique keywords: {len(keyword_counter)}")
        print(f"   • Analyzing top 200 keywords")
        
        # Save to Excel
        excel_path = f"{self.output_excel_dir}/11_keyword_analysis.xlsx"
        
        # Add paper counts with policy/academic breakdown
        keyword_data = []
        for kw, count in top_keywords:
            papers = keyword_papers[kw]
            policy_count = sum(1 for p in papers if self.df.iloc[p]['policy_mentions_total_alt'] > 0)
            academic_count = sum(1 for p in papers if self.df.iloc[p]['cited_by_count_oa'] > 0)
            
            keyword_data.append({
                'Keyword': kw,
                'Total Frequency': count,
                'Papers with Policy': policy_count,
                'Papers with Academic': academic_count
            })
        
        pd.DataFrame(keyword_data).to_excel(excel_path, index=False)
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualizations
        print("   📊 Creating visualizations...")
        self._plot_keyword_wordcloud(keyword_counter)
        
        print("   🔗 Creating co-occurrence networks...")
        self._plot_keyword_cooccurrence_network(keyword_papers, top_keywords,
                                               'All Papers', '11_keyword_cooccurrence_full.png')
        
        # Policy papers network
        policy_indices = set(self.df[self.df['policy_mentions_total_alt'] > 0].index)
        policy_kw_papers = {kw: [p for p in papers if p in policy_indices] 
                           for kw, papers in keyword_papers.items()}
        self._plot_keyword_cooccurrence_network(policy_kw_papers, top_keywords,
                                               'Policy Papers', '11_keyword_cooccurrence_policy.png')
        
        # Academic papers network
        academic_indices = set(self.df[self.df['cited_by_count_oa'] > 0].index)
        academic_kw_papers = {kw: [p for p in papers if p in academic_indices]
                             for kw, papers in keyword_papers.items()}
        self._plot_keyword_cooccurrence_network(academic_kw_papers, top_keywords,
                                               'Academic Papers', '11_keyword_cooccurrence_academic.png')
        
        print("   ✓ All keyword analyses complete")
    
    def _plot_keyword_wordcloud(self, keyword_counter):
        """Create word cloud of keywords"""
        wordcloud = WordCloud(width=2400, height=1200,
                             background_color='white',
                             colormap='viridis',
                             relative_scaling=0.5,
                             min_font_size=12,
                             max_words=200).generate_from_frequencies(keyword_counter)
        
        fig, ax = plt.subplots(figsize=(24, 12))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Keyword Word Cloud - Top 200 Keywords',
                    fontsize=22, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/11_keyword_wordcloud.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 11_keyword_wordcloud.png")
    
    def _plot_keyword_cooccurrence_network(self, keyword_papers, top_keywords,
                                          subset_name, filename):
        """
        Create keyword co-occurrence network
        
        Visual Design:
        - Nodes = Keywords
        - Edges = Co-occurrence in same papers
        - Node size = Keyword frequency
        - Node color = Community detection
        - Edge width = Co-occurrence strength
        """
        # Use top 60 keywords
        top_kws = [kw[0] for kw in top_keywords[:60] if len(keyword_papers.get(kw[0], [])) > 0]
        
        # Calculate co-occurrences
        cooccurrence = {}
        for kw1 in top_kws:
            for kw2 in top_kws:
                if kw1 < kw2:
                    papers1 = set(keyword_papers.get(kw1, []))
                    papers2 = set(keyword_papers.get(kw2, []))
                    common = len(papers1 & papers2)
                    if common >= 5:  # Threshold
                        cooccurrence[(kw1, kw2)] = common
        
        # Create network
        G = nx.Graph()
        G.add_nodes_from(top_kws)
        
        for (kw1, kw2), weight in cooccurrence.items():
            G.add_edge(kw1, kw2, weight=weight)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if len(G.nodes()) == 0:
            print(f"   ⚠ No keyword network for {subset_name}")
            return
        
        print(f"   • {subset_name}: {len(G.nodes())} keywords, {len(G.edges())} co-occurrences")
        
        # Detect communities
        communities = nx.community.louvain_communities(G, seed=42)
        
        # Assign colors to communities
        community_colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = []
        for node in G.nodes():
            for idx, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(community_colors[idx])
                    break
        
        # Node sizes
        kw_freq = {kw[0]: kw[1] for kw in top_keywords if kw[0] in G.nodes()}
        node_sizes = [kw_freq.get(kw, 1) * 3 for kw in G.nodes()]
        
        # Edge widths
        edge_widths = [(G[u][v]['weight'] * 0.15) for u, v in G.edges()]
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42, weight='weight')
        
        # Plot
        fig, ax = plt.subplots(figsize=(26, 26))
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              edgecolors='black', linewidths=1.5, alpha=0.8, ax=ax)
        
        # Labels
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=7, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', alpha=0.7, linewidth=0.8))
        
        ax.set_title(f'Keyword Co-occurrence Network - {subset_name}\n'
                    f'(Node size = frequency, Edge width = co-occurrences, '
                    f'Colors = communities)',
                    fontsize=18, fontweight='bold', pad=30)
        ax.axis('off')
        
        # Add stats
        stats_text = (f"Keywords: {len(G.nodes())}\n"
                     f"Co-occurrences: {len(G.edges())}\n"
                     f"Communities: {len(communities)}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {filename}")
    
    # ========================================================================
    # SECTION 8: SDG ANALYSIS WITH NETWORKS (File 12)
    # ========================================================================
    
    def analyze_sdgs(self):
        """
        Analyze SDG associations with network visualization
        
        Outputs:
        - 12_sdg_analysis.xlsx: SDG statistics
        - 12_sdg_distribution.png: Bar chart
        - 12_sdg_field_network.png: SDG-Field bipartite network
        """
        print("\n   🌱 Analyzing SDG associations...")
        
        # Count SDGs
        sdg_counter = Counter()
        sdg_policy = Counter()
        sdg_academic = Counter()
        sdg_fields = defaultdict(lambda: Counter())  # SDG -> field -> count
        
        for idx, row in self.df.iterrows():
            sdg = row.get('sdg_name_oa', '')
            field = row.get('primary_field_oa', '')
            
            if pd.notna(sdg) and sdg != '':
                sdg_counter[sdg] += 1
                
                if pd.notna(field) and field != '':
                    sdg_fields[sdg][field] += 1
                
                if row.get('policy_mentions_total_alt', 0) > 0:
                    sdg_policy[sdg] += 1
                
                if row.get('cited_by_count_oa', 0) > 0:
                    sdg_academic[sdg] += 1
        
        # Create dataframe
        sdgs = sorted(sdg_counter.keys())
        sdg_df = pd.DataFrame({
            'SDG': sdgs,
            'Total Papers': [sdg_counter[s] for s in sdgs],
            'Total %': [(sdg_counter[s] / len(self.df) * 100) for s in sdgs],
            'Papers with Policy': [sdg_policy[s] for s in sdgs],
            'Policy %': [(sdg_policy[s] / sdg_counter[s] * 100) if sdg_counter[s] > 0 else 0 
                        for s in sdgs],
            'Papers with Academic': [sdg_academic[s] for s in sdgs],
            'Academic %': [(sdg_academic[s] / sdg_counter[s] * 100) if sdg_counter[s] > 0 else 0
                          for s in sdgs]
        })
        
        # Save
        excel_path = f"{self.output_excel_dir}/12_sdg_analysis.xlsx"
        sdg_df.to_excel(excel_path, index=False)
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualizations
        print("   📊 Creating visualizations...")
        self._plot_sdg_distribution(sdg_df)
        self._plot_sdg_field_network(sdg_counter, sdg_fields)
        
        print("   ✓ All SDG analyses complete")
    
    def _plot_sdg_distribution(self, sdg_df):
        """Plot SDG distribution"""
        sdg_df_sorted = sdg_df.sort_values('Total Papers', ascending=True)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create bars
        y_pos = range(len(sdg_df_sorted))
        bars = ax.barh(y_pos, sdg_df_sorted['Total Papers'],
                      color='mediumseagreen', edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Color by policy emphasis
        for idx, (_, row) in enumerate(sdg_df_sorted.iterrows()):
            if row['Policy %'] > 30:
                bars[idx].set_color(COLORS['policy'])
            elif row['Policy %'] < 10:
                bars[idx].set_color(COLORS['academic'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sdg_df_sorted['SDG'], fontsize=10)
        ax.set_xlabel('Number of Papers', fontweight='bold', fontsize=12)
        ax.set_title('Papers by Sustainable Development Goal\n'
                    '(Color: Red=High Policy, Blue=High Academic, Green=Balanced)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add labels
        for i, (idx, row) in enumerate(sdg_df_sorted.iterrows()):
            ax.text(row['Total Papers'], i,
                   f" {row['Total Papers']} ({row['Total %']:.1f}%)",
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/12_sdg_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 12_sdg_distribution.png")
    
    def _plot_sdg_field_network(self, sdg_counter, sdg_fields):
        """
        Create bipartite network of SDGs and Fields
        
        Shows which fields contribute to which SDGs
        """
        # Get top SDGs (by paper count)
        top_sdgs = [sdg for sdg, count in sdg_counter.most_common(10)]
        
        # Get top fields for these SDGs
        all_fields = Counter()
        for sdg in top_sdgs:
            all_fields.update(sdg_fields[sdg])
        top_fields = [field for field, count in all_fields.most_common(15)]
        
        # Create bipartite graph
        B = nx.Graph()
        B.add_nodes_from(top_sdgs, bipartite=0)
        B.add_nodes_from(top_fields, bipartite=1)
        
        # Add edges
        for sdg in top_sdgs:
            for field in top_fields:
                count = sdg_fields[sdg][field]
                if count > 2:
                    B.add_edge(sdg, field, weight=count)
        
        # Layout
        pos = {}
        sdg_y = np.linspace(0, 1, len(top_sdgs))
        field_y = np.linspace(0, 1, len(top_fields))
        
        for idx, sdg in enumerate(top_sdgs):
            pos[sdg] = (0, sdg_y[idx])
        
        for idx, field in enumerate(top_fields):
            pos[field] = (1, field_y[idx])
        
        # Plot
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Draw edges
        for u, v, data in B.edges(data=True):
            width = data['weight'] / 5
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                   'gray', linewidth=width, alpha=0.4, zorder=1)
        
        # Draw nodes
        for node in top_sdgs:
            ax.scatter(*pos[node], s=2500, c='mediumseagreen',
                      edgecolors='black', linewidths=2, zorder=2, alpha=0.8)
            ax.text(pos[node][0]-0.02, pos[node][1], node,
                   ha='right', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='black', alpha=0.9))
        
        for node in top_fields:
            ax.scatter(*pos[node], s=2000, c=COLORS['academic'],
                      edgecolors='black', linewidths=2, zorder=2, alpha=0.8)
            ax.text(pos[node][0]+0.02, pos[node][1], node,
                   ha='left', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='black', alpha=0.9))
        
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('SDG-Field Network\n(Edge width = number of papers linking SDG to Field)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Labels
        ax.text(0, 1.05, 'SDGs', ha='center', va='bottom',
               fontsize=14, fontweight='bold', color='mediumseagreen')
        ax.text(1, 1.05, 'PRIMARY FIELDS', ha='center', va='bottom',
               fontsize=14, fontweight='bold', color=COLORS['academic'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/12_sdg_field_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 12_sdg_field_network.png")
    
    # ========================================================================
    # SECTION 9: TEMPORAL ANALYSIS (File 13)
    # ========================================================================
    
    def analyze_temporal_trends(self):
        """
        Analyze temporal trends in citations
        
        Outputs:
        - 13_temporal_analysis.xlsx: Time series data
        - 13_temporal_trends.png: Overall trends
        - 13_temporal_by_field.png: Trends by field
        - 13_temporal_by_sdg.png: Trends by SDG
        """
        print("\n   📅 Analyzing temporal trends...")
        
        # Filter to papers with years
        df_temporal = self.df[self.df['year'].notna()].copy()
        
        if len(df_temporal) == 0:
            print("   ⚠ No temporal data available")
            return
        
        print(f"   • Analyzing {len(df_temporal)} papers from {df_temporal['year'].min():.0f} to {df_temporal['year'].max():.0f}")
        
        # Calculate metrics by year
        yearly_stats = df_temporal.groupby('year').agg({
            'cited_by_count_oa': ['count', 'sum', 'mean', 'median'],
            'policy_mentions_total_alt': ['sum', 'mean'],
            'news_msm_total_alt': ['sum', 'mean'],
            'twitter_total_alt': ['sum', 'mean'],
            'readers_mendeley_alt': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        yearly_stats.columns = ['_'.join(str(c) for c in col).strip('_') 
                               for col in yearly_stats.columns.values]
        yearly_stats = yearly_stats.rename(columns={'year_': 'year'})
        
        # Save
        excel_path = f"{self.output_excel_dir}/13_temporal_analysis.xlsx"
        yearly_stats.to_excel(excel_path, index=False)
        print(f"   ✓ Saved: {excel_path}")
        
        # Visualizations
        print("   📊 Creating temporal visualizations...")
        self._plot_temporal_trends(yearly_stats)
        self._plot_temporal_by_field(df_temporal)
        self._plot_temporal_by_sdg(df_temporal)
        
        print("   ✓ All temporal analyses complete")
    
    def _plot_temporal_trends(self, yearly_stats):
        """Plot overall temporal trends"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        years = yearly_stats['year']
        
        # 1. Number of papers
        ax1 = axes[0, 0]
        ax1.plot(years, yearly_stats['cited_by_count_oa_count'],
                marker='o', linewidth=3, markersize=8, color=COLORS['academic'])
        ax1.fill_between(years, yearly_stats['cited_by_count_oa_count'],
                        alpha=0.3, color=COLORS['academic'])
        ax1.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Number of Papers', fontweight='bold', fontsize=12)
        ax1.set_title('Publications Over Time', fontweight='bold', fontsize=14, pad=15)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # 2. Mean academic citations
        ax2 = axes[0, 1]
        ax2.plot(years, yearly_stats['cited_by_count_oa_mean'],
                marker='o', linewidth=3, markersize=8, color=COLORS['academic'])
        ax2.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Mean Citations', fontweight='bold', fontsize=12)
        ax2.set_title('Mean Academic Citations Over Time', fontweight='bold', fontsize=14, pad=15)
        ax2.grid(alpha=0.3, linestyle='--')
        
        # 3. Mean policy mentions
        ax3 = axes[1, 0]
        ax3.plot(years, yearly_stats['policy_mentions_total_alt_mean'],
                marker='o', linewidth=3, markersize=8, color=COLORS['policy'])
        ax3.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Mean Policy Mentions', fontweight='bold', fontsize=12)
        ax3.set_title('Mean Policy Mentions Over Time', fontweight='bold', fontsize=14, pad=15)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # 4. Multiple metrics
        ax4 = axes[1, 1]
        ax4.plot(years, yearly_stats['news_msm_total_alt_mean'],
                marker='o', linewidth=2, label='News Media', color=COLORS['media'])
        ax4.plot(years, yearly_stats['twitter_total_alt_mean'],
                marker='s', linewidth=2, label='Twitter/X', color=COLORS['social'])
        ax4.plot(years, yearly_stats['readers_mendeley_alt_mean'],
                marker='^', linewidth=2, label='Mendeley', color=COLORS['readers'])
        ax4.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Mean Mentions', fontweight='bold', fontsize=12)
        ax4.set_title('Mean Media Mentions Over Time', fontweight='bold', fontsize=14, pad=15)
        ax4.legend(loc='best', fontsize=11)
        ax4.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Temporal Trends - Complexity Science Papers',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/13_temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 13_temporal_trends.png")
    
    def _plot_temporal_by_field(self, df_temporal):
        """Plot temporal trends by primary field"""
        # Get top 5 fields
        top_fields = df_temporal['primary_field_oa'].value_counts().head(5).index
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 14))
        
        # Academic citations
        ax1 = axes[0]
        for field in top_fields:
            field_data = df_temporal[df_temporal['primary_field_oa'] == field]
            yearly = field_data.groupby('year')['cited_by_count_oa'].mean()
            if len(yearly) > 0:
                ax1.plot(yearly.index, yearly.values, marker='o',
                        linewidth=2, label=field, markersize=6)
        
        ax1.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Mean Academic Citations', fontweight='bold', fontsize=12)
        ax1.set_title('Mean Academic Citations by Primary Field',
                     fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Policy mentions
        ax2 = axes[1]
        for field in top_fields:
            field_data = df_temporal[df_temporal['primary_field_oa'] == field]
            yearly = field_data.groupby('year')['policy_mentions_total_alt'].mean()
            if len(yearly) > 0:
                ax2.plot(yearly.index, yearly.values, marker='o',
                        linewidth=2, label=field, markersize=6)
        
        ax2.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Mean Policy Mentions', fontweight='bold', fontsize=12)
        ax2.set_title('Mean Policy Mentions by Primary Field',
                     fontweight='bold', fontsize=14, pad=15)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Temporal Trends by Field',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/13_temporal_by_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 13_temporal_by_field.png")
    
    def _plot_temporal_by_sdg(self, df_temporal):
        """Plot temporal trends by SDG"""
        # Get top 5 SDGs
        top_sdgs = df_temporal['sdg_name_oa'].value_counts().head(5).index
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 14))
        
        # Academic citations
        ax1 = axes[0]
        for sdg in top_sdgs:
            sdg_data = df_temporal[df_temporal['sdg_name_oa'] == sdg]
            yearly = sdg_data.groupby('year')['cited_by_count_oa'].mean()
            if len(yearly) > 0:
                ax1.plot(yearly.index, yearly.values, marker='o',
                        linewidth=2, label=sdg, markersize=6)
        
        ax1.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Mean Academic Citations', fontweight='bold', fontsize=12)
        ax1.set_title('Mean Academic Citations by SDG',
                     fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Policy mentions
        ax2 = axes[1]
        for sdg in top_sdgs:
            sdg_data = df_temporal[df_temporal['sdg_name_oa'] == sdg]
            yearly = sdg_data.groupby('year')['policy_mentions_total_alt'].mean()
            if len(yearly) > 0:
                ax2.plot(yearly.index, yearly.values, marker='o',
                        linewidth=2, label=sdg, markersize=6)
        
        ax2.set_xlabel('Year', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Mean Policy Mentions', fontweight='bold', fontsize=12)
        ax2.set_title('Mean Policy Mentions by SDG',
                     fontweight='bold', fontsize=14, pad=15)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle('Temporal Trends by SDG',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{self.output_images_dir}/13_temporal_by_sdg.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: 13_temporal_by_sdg.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Define paths
    INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel\04_OA_Altmetrics_merge.xlsx"
    
    OUTPUT_EXCEL_DIR = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel"
    
    OUTPUT_IMAGES_DIR = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"
    
    print("\n" + "="*80)
    print(" "*15 + "COMPLEXITY SCIENCE CITATION ANALYSIS - FINAL VERSION")
    print("="*80)
    print("\nBased on: Grauwin et al. (2012)")
    print("'Complex Systems Science: Dreams of Universality, Interdisciplinarity Reality'")
    print("\n" + "="*80)
    
    # Create analyzer
    analyzer = ComplexityScienceAnalyzer(INPUT_PATH, OUTPUT_EXCEL_DIR, OUTPUT_IMAGES_DIR)
    
    # Run all analyses
    analyzer.run_all_analyses()
    
    # Final summary
    print("\n" + "="*80)
    print(" "*20 + "ANALYSIS COMPLETE - RESEARCH INSIGHTS")
    print("="*80)
    print("""
KEY RESEARCH QUESTIONS ADDRESSED:

1. ✓ Citation Distributions (Files 05)
   - Academic vs policy citation patterns
   - Media and social media impact
   - Correlation analysis

2. ✓ Top Papers Analysis (File 06)
   - Leaders by citation type
   - Cross-impact comparison

3. ✓ Journal Patterns (File 07)
   - Policy vs academic journals
   - Publication venue analysis

4. ✓ Institutional Analysis (File 08)
   - Leading institutions by impact type
   
5. ✓ Country Collaboration Networks (Files 09)
   - International cooperation patterns
   - Network centrality analysis
   - Visual networks with detailed design

6. ✓ Topic/Field/Domain Analysis (Files 10)
   - Disciplinary distributions
   - Field networks
   - Hierarchical relationships

7. ✓ Keyword Analysis (Files 11)
   - Co-occurrence networks
   - Conceptual structure
   - Policy vs academic keywords

8. ✓ SDG Analysis (Files 12)
   - Sustainable development associations
   - SDG-Field networks

9. ✓ Temporal Trends (Files 13)
   - Evolution over time
   - Field-specific trends
   - SDG-specific trends

VISUALIZATION FEATURES:
- Professional color schemes
- Network centrality indicators
- Community detection
- Comparative layouts
- Publication-quality images

FOLLOWING UP ON GRAUWIN ET AL. (2012):
Consider additional analyses for:
- Trading zones identification
- Self-organization concept tracking
- Methodological vs theoretical classification
- Bibliographic coupling networks
- Universality claims vs actual impact
    """)
    
    print("\n" + "="*80)
    print(" "*30 + "ANALYSIS SESSION COMPLETE")
    print("="*80)
    print(f"\n📊 All outputs saved to:")
    print(f"   Excel: {OUTPUT_EXCEL_DIR}")
    print(f"   Images: {OUTPUT_IMAGES_DIR}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()