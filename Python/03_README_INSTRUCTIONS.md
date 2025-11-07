# Network Visualization Scripts - Usage Guide
==========================================

## 📦 FILES INCLUDED

1. **explore_network.py** - Data exploration (run this FIRST)
2. **visualize_network.py** - Advanced visualization (run SECOND)

---

## 🚀 QUICK START

### Step 1: Install dependencies
```bash
pip install networkx matplotlib numpy pandas python-louvain
```

### Step 2: Explore your data
```bash
python explore_network.py
```
- This analyzes your network
- Shows statistics
- **Suggests optimal filtering parameters**
- Creates distribution plots

### Step 3: Create visualizations
```bash
python visualize_network.py
```
- Uses the recommended parameters from Step 2
- Creates 3 filtered versions
- Generates comparison chart
- All in English with interpretation notes

---

## 📊 OUTPUT FILES (in viewing order)

The script creates files numbered for easy exploration:

1. **01_comparison_all_filters.png**
   - Compare all 3 filtering levels side-by-side
   - **START HERE** to choose your preferred filter

2. **02_network_medium_filter.png / .pdf**
   - Recommended filter (best balance)
   - Good for seeing general structure
   - ~20,000 edges

3. **03_network_strong_filter.png / .pdf**
   - Only important connections
   - Cleaner visualization
   - ~8,000 edges

4. **04_network_very_strong_filter.png / .pdf**
   - Core of the field
   - Most essential connections only
   - ~4,000 edges

---

## 🎨 HOW TO READ THE VISUALIZATIONS

### Each visualization has 2 panels:

**LEFT PANEL (By Communities):**
- Each node = one paper
- Node size = number of citations
- Node color = thematic community
- Edge thickness = co-citation strength
- Nodes close together = frequently co-cited (related topics)

**RIGHT PANEL (By Importance):**
- Darker/larger nodes = more cited papers (more influential)
- Lighter/smaller nodes = less cited
- Edge thickness = co-citation frequency
- Central positions = papers cited across multiple topics

---

## ⚙️ CUSTOMIZING PATHS

### In explore_network.py (lines 23-26):
```python
NETWORK_FILE = "Complexity_CoCitation_LCC.gexf"  # Your input file
OUTPUT_DIR = "./visualizations"                   # Output folder
```

### In visualize_network.py (lines 29-33):
```python
NETWORK_FILE = "Complexity_CoCitation_LCC.gexf"   # Your input file
OUTPUT_DIR = "./visualizations_output"            # Output folder
```

**Examples:**

Windows full path:
```python
NETWORK_FILE = r"C:\Users\YourName\Documents\Data\file.gexf"
OUTPUT_DIR = r"C:\Users\YourName\Documents\Results"
```

Mac/Linux full path:
```python
NETWORK_FILE = "/Users/YourName/Documents/Data/file.gexf"
OUTPUT_DIR = "/Users/YourName/Documents/Results"
```

---

## 🤔 UNDERSTANDING YOUR NETWORK

### Why is degree > citations?

**DEGREE** = How many other papers this paper is connected to in the co-citation network
**CITATIONS** = How many times this paper was cited by papers in your corpus

Example:
- Paper cited by 538 papers (citations = 538)
- Those 538 papers also cite many other different papers
- This creates 758 unique connections (degree = 758)

**It's normal for degree > citations** because:
- Each citing paper typically cites 10-50 other papers
- Those create multiple connections in the co-citation network

---

## 📝 FILTERING PARAMETERS EXPLAINED

The script uses 3 filtering levels based on your data analysis:

| Filter | Weight | Co-citations | Node citations | Result |
|--------|--------|--------------|----------------|--------|
| Medium | ≥ 0.472 | ≥ 5 | ≥ 10 | ~20,000 edges |
| Strong | ≥ 0.667 | ≥ 7 | ≥ 20 | ~8,000 edges |
| Very Strong | ≥ 0.775 | ≥ 9 | ≥ 30 | ~4,000 edges |

**Weight** = Normalized co-citation similarity (0 to 1)
**Co-citations** = Raw number of times papers are cited together
**Node citations** = Minimum citations required to include a paper

---

## 🎯 FOR YOUR PAPER

### Recommended usage:

1. **Main text**: Use `02_network_medium_filter.png`
   - Best balance between detail and clarity
   - Shows overall field structure

2. **Appendix**: Include `01_comparison_all_filters.png`
   - Shows robustness of community structure
   - Different levels of granularity

3. **Save PDFs**: For publication-quality figures
   - Infinitely scalable
   - No pixelation

---

## 💡 TIPS

- Start with the comparison (file 01)
- Choose the filter that best shows your communities
- Use PNG for slides, PDF for papers
- The interpretation notes are built into each graph
- Hub analysis is printed to console (save it!)

---

## ❓ TROUBLESHOOTING

**Error: "python-louvain not installed"**
```bash
pip install python-louvain
```

**Visualizations too dense:**
- Use a stronger filter (03 or 04)
- Or adjust parameters in lines 39-59 of visualize_network.py

**Visualizations too sparse:**
- Use a lighter filter (02)
- Or reduce threshold values

**File not found:**
- Check NETWORK_FILE path in lines 29-33
- Use `r"..."` for Windows paths
- Use absolute paths if relative paths fail

---

## 📚 REPLICATING GRAUWIN ET AL. (2012)

This code is inspired by:
"Complex Systems Science: Dreams of Universality, Interdisciplinarity Reality"
by Grauwin et al., 2012

Key differences:
- Their data: ISI Web of Science, 215K articles
- Your data: OpenAlex, ~5K articles (focused corpus)
- Their method: Keyword-based selection
- Your method: Author-based selection (more precise)

---

## ✉️ QUESTIONS?

All labels, titles, and interpretation notes are now in English.
Each file is numbered (01, 02, 03, 04) for easy sequential viewing.
Each graph includes a small interpretation note at the bottom.

Good luck with your analysis! 🚀
