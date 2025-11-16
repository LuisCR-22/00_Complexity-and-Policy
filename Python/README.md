# Complexity Science and Policy Impact Analysis

**Author:** Luis Castellanos (le.castellanos10@uniandes.edu.co)
**Project:** Global Complexity School 2025 Final Project
**Date:** November 2025

## Overview

This repository contains Python scripts for analyzing the relationship between complexity science research and policy impact. The analysis combines bibliometric data from OpenAlex and Altmetric to explore citation patterns, policy mentions, SDG associations, and collaboration networks.

## Pipeline Structure

The scripts are numbered sequentially to reflect the recommended execution order:

### Phase 1: Network Visualization (01-03)
- **01_Visualizing_Joes_networks.py** - Primary co-citation network visualization
- **02_Explore_network_file.py** - Network data exploration and statistics
- **03_detailed_visualization.py** - Advanced network visualizations with filtering

### Phase 2: Data Collection (04-06)
- **04_test_altmetric_api.py** - Test Altmetric API connection
- **05_altmetric_extraction_api.py** - Extract policy mentions and altmetric data
- **06_test_OpenAlex_api.py** / **06_OpenAlex_api.py** - Extract bibliometric data from OpenAlex

### Phase 3: Data Integration (07)
- **07_merge_openalex_altmetric.py** - Merge OpenAlex and Altmetric datasets with deduplication

### Phase 4: Citation Analysis (08)
- **08_citations_analysis.py** - Comprehensive citation analysis including:
  - Citation metrics (academic vs policy)
  - Journal analysis
  - Institution and country patterns
  - Topic and keyword analysis
  - Temporal trends

### Phase 5: SDG Analysis (09-10)
- **09_sdg_analysis.py** - Extended SDG distribution and impact analysis
- **10_sdgs_policy_network.py** - Citation networks colored by SDG associations

### Phase 6: Policy Analysis (11)
- **11_policy_only_analysis.py** - Enhanced visualizations focusing on policy impact

### Phase 7: Advanced Network Analysis (12-13)
- **12_co-citation_and_policy.py** - Integrate co-citation networks with policy/SDG data
- **13_country_topics_networks.py** - Country, institution, and topic collaboration networks

## Requirements

### Python Version
Python 3.13 or below (3.14 lacks wheel support for some dependencies)

### Core Libraries
```bash
pip install pandas numpy matplotlib seaborn networkx
```

### Network Analysis
```bash
pip install python-louvain pyvis
```

### API & Data Processing
```bash
pip install requests openpyxl
```

### Text Analysis
```bash
pip install wordcloud
```

## Input Data Requirements

1. **Co-citation Network** (GEXF format)
   - File: `Complexity_CoCitation_LCC.gexf`
   - Largest connected component of co-citation network

2. **OpenAlex Data** (Excel format)
   - Bibliometric metadata for complexity science papers

3. **Altmetric Data** (Excel format)
   - Policy mentions, news, blogs, and social media metrics

## Output Structure

### Excel Files
Generated in the Excel output directory:
- `04_OA_Altmetrics_merge.xlsx` - Merged dataset
- `05_citation_metrics.xlsx` - Descriptive statistics
- `06_top_papers.xlsx` - Top papers by citation type
- `07-13_*.xlsx` - Journal, institution, country, topic, keyword, SDG, and temporal analyses

### Visualizations
Generated in the Images output directory with numbered prefixes:
- `01-04_*.png` - Network visualizations
- `05_*.png` - Citation distributions
- `07-13_*.png` - Various thematic analyses
- `14-19_*.png` - SDG, policy, and collaboration networks

## Configuration

Each script contains a configuration section at the top with paths that can be modified:

```python
INPUT_PATH = r"C:\path\to\input\files"
OUTPUT_PATH = r"C:\path\to\output\images"
EXCEL_PATH = r"C:\path\to\output\excel"
```

## Running the Scripts

### Quick Start
Execute scripts in numerical order for the full pipeline:

```bash
# 1. Visualize networks
python 01_Visualizing_Joes_networks.py
python 02_Explore_network_file.py
python 03_detailed_visualization.py

# 2. Collect API data
python 04_test_altmetric_api.py
python 05_altmetric_extraction_api.py
python 06_OpenAlex_api.py

# 3. Merge datasets
python 07_merge_openalex_altmetric.py

# 4. Run analyses (requires merged data)
python 08_citations_analysis.py
python 09_sdg_analysis.py
python 10_sdgs_policy_network.py
python 11_policy_only_analysis.py
python 12_co-citation_and_policy.py
python 13_country_topics_networks.py
```

### Python Version Specification
If Python 3.14 is installed, explicitly use 3.13:
```bash
py -3.13 script_name.py
```

## Key Features

### Network Analysis
- Co-citation network exploration
- Community detection using Louvain algorithm
- Multi-level filtering (medium, strong, very strong)
- Interactive and static visualizations

### Citation Metrics
- Academic citations (traditional impact)
- Policy citations (real-world impact)
- Comparison of impact types across:
  - Journals
  - Institutions
  - Countries
  - Research topics
  - Time periods

### SDG Analysis
- SDG association frequencies
- Policy citation patterns by SDG
- Citation networks colored by SDG
- SDG impact profiles

### Collaboration Networks
- Country collaboration weighted by citations
- Institution collaboration patterns
- Topic co-occurrence networks

## Methodology Notes

### Data Matching
The merger script (07) uses a two-phase approach:
1. DOI-based matching (primary)
2. Title + year matching (secondary)
3. Deduplication at all stages

### Network Filtering
Filtering parameters based on:
- Edge weight (normalized co-citation similarity)
- Minimum co-citations
- Minimum node citations

### Visualization Settings
All scripts use publication-quality settings:
- DPI: 300 for saved images
- Style: seaborn-v0_8-whitegrid
- Consistent color schemes

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install --upgrade [missing-package]
```

**Path errors:**
- Use raw strings: `r"C:\path\to\file"`
- Or use forward slashes: `"C:/path/to/file"`

**Python version conflicts:**
```bash
py -3.13 -m pip install [package]
```

**Seaborn style warnings:**
Use `seaborn-v0_8-whitegrid` instead of deprecated `seaborn-whitegrid`

## Citation

If you use this code, please cite:
```
Castellanos, L. (2025). Complexity Science and Policy Impact Analysis.
Global Complexity School Final Project, Universidad de los Andes.
```

## Additional Resources

- See [03_README_INSTRUCTIONS.md](03_README_INSTRUCTIONS.md) for detailed network visualization guidance
- Scripts include extensive inline comments explaining methodology
- Each major section is clearly marked with separators
