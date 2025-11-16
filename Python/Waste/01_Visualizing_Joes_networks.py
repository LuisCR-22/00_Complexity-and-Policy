"""
Network Visualization Script for Complex Systems Analysis
Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Date: 2025

This script visualizes three network files from the Complex Systems research paper:
1. Complexity_BiblioCoupling.gexf - Bibliographic coupling network
2. Complexity_Bibliographic_Coupling.graphml - Same data, different format
3. Complexity_CoCitation_LCC.gexf - Co-citation network (Largest Connected Component)
"""

# ============================================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================================

import networkx as nx           # For network analysis and manipulation
import matplotlib.pyplot as plt  # For creating static visualizations
from pyvis.network import Network  # For creating interactive HTML visualizations
import os                       # For file path operations
import warnings                 # To suppress unnecessary warnings
warnings.filterwarnings('ignore')  # Keep output clean

print("✓ All libraries imported successfully!")

# ============================================================================
# SECTION 2: CONFIGURE PATHS (CHANGE THESE TO YOUR DIRECTORIES)
# ============================================================================

# INPUT PATH: Where your network files are stored
# Example for Windows: r"C:\Users\YourName\Documents\NetworkFiles"
# Example for Mac/Linux: "/Users/YourName/Documents/NetworkFiles"
INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out"

# OUTPUT PATH: Where you want to save the visualizations
OUTPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"✓ Created output directory: {OUTPUT_PATH}")

# Define the three files we want to analyze
FILES = {
    'biblio_coupling_gexf': 'Complexity_BiblioCoupling.gexf',
    'biblio_coupling_graphml': 'Complexity_Bibliographic_Coupling.graphml',
    'cocitation_lcc': 'Complexity_CoCitation_LCC.gexf'
}

print(f"✓ Paths configured successfully!")
print(f"  Input folder: {INPUT_PATH}")
print(f"  Output folder: {OUTPUT_PATH}")

# ============================================================================
# SECTION 3: DEFINE VISUALIZATION FUNCTIONS
# ============================================================================

def load_network(file_path, file_type='gexf'):
    """
    Load a network file into a NetworkX graph object.
    
    Parameters:
    -----------
    file_path : str
        Full path to the network file
    file_type : str
        Type of file ('gexf' or 'graphml')
    
    Returns:
    --------
    G : NetworkX Graph object
        The loaded network
    """
    print(f"\n📊 Loading network from: {os.path.basename(file_path)}")
    
    try:
        # Load the appropriate file format
        if file_type == 'gexf':
            G = nx.read_gexf(file_path)
        elif file_type == 'graphml':
            G = nx.read_graphml(file_path)
        else:
            raise ValueError("File type must be 'gexf' or 'graphml'")
        
        # Print network statistics
        print(f"  ✓ Network loaded successfully!")
        print(f"  • Number of nodes: {G.number_of_nodes()}")
        print(f"  • Number of edges: {G.number_of_edges()}")
        print(f"  • Network type: {'Directed' if G.is_directed() else 'Undirected'}")
        
        # Calculate and print density (how connected the network is)
        # Density = actual edges / possible edges (ranges from 0 to 1)
        if G.number_of_nodes() > 1:
            density = nx.density(G)
            print(f"  • Network density: {density:.4f}")
        
        return G
    
    except FileNotFoundError:
        print(f"  ✗ ERROR: File not found at {file_path}")
        print(f"  Please check that the file exists in the INPUT_PATH directory.")
        return None
    except Exception as e:
        print(f"  ✗ ERROR loading file: {str(e)}")
        return None


def visualize_network_matplotlib(G, title, output_filename, layout_type='spring'):
    """
    Create a static visualization using Matplotlib and save as PNG.
    
    Parameters:
    -----------
    G : NetworkX Graph
        The network to visualize
    title : str
        Title for the plot
    output_filename : str
        Name for the output PNG file (without extension)
    layout_type : str
        Type of layout algorithm ('spring', 'circular', 'kamada_kawai')
    
    Why PNG?
    --------
    PNG (Portable Network Graphics) is ideal for academic papers because:
    - Lossless compression: No quality degradation
    - High resolution: Can be scaled for print
    - Universal compatibility: Accepted by all journals
    - Transparent backgrounds: Can be placed on any background
    - No artifacts: Unlike JPEG, no compression artifacts around text/lines
    """
    print(f"\n🎨 Creating static visualization: {output_filename}.png")
    
    if G is None or G.number_of_nodes() == 0:
        print("  ✗ Cannot visualize empty network")
        return
    
    # Create a large figure (12x10 inches) for better quality
    # DPI=300 gives publication-quality resolution
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Calculate node positions using the specified layout algorithm
    print(f"  • Computing {layout_type} layout...")
    
    if layout_type == 'spring':
        # Spring layout: Nodes repel each other, edges act like springs
        # k controls the optimal distance between nodes
        # iterations: more iterations = better layout but slower
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout_type == 'circular':
        # Circular layout: Nodes arranged in a circle
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        # Kamada-Kawai: Another force-directed algorithm, often better for small networks
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate node sizes based on degree (number of connections)
    # More connected nodes = larger size
    node_sizes = [G.degree(node) * 20 + 100 for node in G.nodes()]
    
    # Calculate node colors based on degree
    node_colors = [G.degree(node) for node in G.nodes()]
    
    # Draw the network
    print("  • Drawing network...")
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,           # Transparency (0=invisible, 1=opaque)
        edge_color='gray',   # Color of edges
        width=0.5,           # Thickness of edges
        ax=ax
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,      # Size of each node
        node_color=node_colors,    # Color based on degree
        cmap='viridis',            # Color map (blue to yellow)
        alpha=0.8,                 # Slight transparency
        ax=ax
    )
    
    # Add labels only if network is small (< 50 nodes)
    if G.number_of_nodes() < 50:
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
    
    # Add a colorbar to show what node colors mean
    plt.colorbar(nodes, ax=ax, label='Node Degree (Number of Connections)')
    
    # Set title and remove axes
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')  # Hide axes for cleaner look
    plt.tight_layout()
    
    # Save as PNG
    output_path = os.path.join(OUTPUT_PATH, f"{output_filename}.png")
    plt.savefig(
        output_path,
        dpi=300,              # High resolution for publication
        bbox_inches='tight',  # Remove extra whitespace
        facecolor='white',    # White background
        edgecolor='none'      # No border
    )
    
    print(f"  ✓ Saved PNG to: {output_path}")
    
    # Also save as PDF (vector format, infinitely scalable)
    pdf_path = os.path.join(OUTPUT_PATH, f"{output_filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF to: {pdf_path}")
    
    plt.close()  # Close the figure to free memory


def visualize_network_interactive(G, title, output_filename):
    """
    Create an interactive HTML visualization using PyVis.
    
    This creates a web page where you can:
    - Zoom in/out
    - Drag nodes around
    - Click on nodes to see information
    - Pan around the network
    
    Parameters:
    -----------
    G : NetworkX Graph
        The network to visualize
    title : str
        Title for the visualization
    output_filename : str
        Name for the output HTML file (without extension)
    """
    print(f"\n🌐 Creating interactive visualization: {output_filename}.html")
    
    if G is None or G.number_of_nodes() == 0:
        print("  ✗ Cannot visualize empty network")
        return
    
    # Create PyVis network object
    # height and width control the size in the HTML page
    net = Network(
        height='750px',
        width='100%',
        bgcolor='#ffffff',      # White background
        font_color='black',
        notebook=False          # Set to False for standalone HTML
    )
    
    # Convert NetworkX graph to PyVis format
    print("  • Converting network to interactive format...")
    net.from_nx(G)
    
    # Configure physics for better visualization
    # This controls how nodes move and settle
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "nodes": {
        "font": {"size": 12}
      },
      "edges": {
        "smooth": {
          "type": "continuous"
        }
      }
    }
    """)
    
    # Save as HTML file
    output_path = os.path.join(OUTPUT_PATH, f"{output_filename}.html")
    net.show(output_path)
    
    print(f"  ✓ Saved interactive HTML to: {output_path}")
    print(f"  → Open this file in a web browser to interact with the network!")


def analyze_network_properties(G, network_name):
    """
    Calculate and print important network properties.
    
    This helps you understand the structure of your network beyond just visualizing it.
    """
    print(f"\n📈 Analyzing network properties for: {network_name}")
    
    if G is None or G.number_of_nodes() == 0:
        print("  ✗ Cannot analyze empty network")
        return
    
    # Basic properties
    print(f"\n  BASIC PROPERTIES:")
    print(f"  • Nodes: {G.number_of_nodes()}")
    print(f"  • Edges: {G.number_of_edges()}")
    print(f"  • Density: {nx.density(G):.4f}")
    
    # Degree statistics (how many connections each node has)
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\n  DEGREE STATISTICS:")
    print(f"  • Average degree: {sum(degrees)/len(degrees):.2f}")
    print(f"  • Max degree: {max(degrees)}")
    print(f"  • Min degree: {min(degrees)}")
    
    # Connectivity (for undirected graphs)
    if not G.is_directed():
        print(f"\n  CONNECTIVITY:")
        print(f"  • Connected: {nx.is_connected(G)}")
        
        if not nx.is_connected(G):
            # Find connected components (isolated groups)
            components = list(nx.connected_components(G))
            print(f"  • Number of components: {len(components)}")
            print(f"  • Largest component size: {len(max(components, key=len))}")
    
    # Clustering coefficient (how clustered/grouped the network is)
    try:
        avg_clustering = nx.average_clustering(G)
        print(f"\n  CLUSTERING:")
        print(f"  • Average clustering coefficient: {avg_clustering:.4f}")
        print(f"    (0 = no clustering, 1 = highly clustered)")
    except:
        print(f"\n  CLUSTERING: Cannot compute (might be directed)")


# ============================================================================
# SECTION 4: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that runs all the analysis and visualization.
    """
    print("="*70)
    print("COMPLEX SYSTEMS NETWORK VISUALIZATION")
    print("="*70)
    
    # Process each file
    for key, filename in FILES.items():
        print("\n" + "="*70)
        print(f"PROCESSING: {filename}")
        print("="*70)
        
        # Construct full file path
        file_path = os.path.join(INPUT_PATH, filename)
        
        # Determine file type from extension
        file_type = 'gexf' if filename.endswith('.gexf') else 'graphml'
        
        # Load the network
        G = load_network(file_path, file_type)
        
        if G is not None:
            # Analyze network properties
            analyze_network_properties(G, filename)
            
            # Create visualizations with descriptive names
            base_name = filename.replace('.gexf', '').replace('.graphml', '')
            
            # Static visualization (PNG for papers)
            visualize_network_matplotlib(
                G,
                title=f"Network Visualization: {base_name}",
                output_filename=f"{base_name}_static",
                layout_type='spring'
            )
            
            # Interactive visualization (HTML for exploration)
            # Only create if network is not too large (< 1000 nodes)
            if G.number_of_nodes() < 1000:
                visualize_network_interactive(
                    G,
                    title=f"Interactive Network: {base_name}",
                    output_filename=f"{base_name}_interactive"
                )
            else:
                print(f"\n  ⚠ Skipping interactive visualization (network too large)")
                print(f"    Interactive visualizations work best with < 1000 nodes")
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETED!")
    print("="*70)
    print(f"\nYour files are saved in: {OUTPUT_PATH}")
    print("\nFiles created:")
    print("  • PNG files - High-resolution images for papers (300 DPI)")
    print("  • PDF files - Vector format, infinitely scalable")
    print("  • HTML files - Interactive visualizations (open in browser)")


# ============================================================================
# SECTION 5: RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # This ensures the script only runs when executed directly
    # (not when imported as a module)
    main()