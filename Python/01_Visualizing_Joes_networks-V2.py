"""
Network Visualization Script for Complex Systems Analysis
Author: Luis Castellanos - le.castellanos10@uniandes.edu.co
Date: 2025

This script visualizes the Co-Citation network (Largest Connected Component)
from the Complex Systems research paper.
"""

# ============================================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================================

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!")

# ============================================================================
# SECTION 2: CONFIGURE PATHS
# ============================================================================

# INPUT PATH: Where your network file is stored
INPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out"

# OUTPUT PATH: Where visualizations will be saved
OUTPUT_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

# EXCEL OUTPUT PATH: Where diagnostic file will be saved
EXCEL_PATH = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Excel"

# Create output directories if they don't exist
for path in [OUTPUT_PATH, EXCEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Created output directory: {path}")

# Define the file to analyze
INPUT_FILE = 'Complexity_CoCitation_LCC.gexf'

print(f"✓ Paths configured successfully!")
print(f"  Input folder: {INPUT_PATH}")
print(f"  Output folder: {OUTPUT_PATH}")
print(f"  Excel folder: {EXCEL_PATH}")

# ============================================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================================

def load_network(file_path):
    """
    Load a network file into a NetworkX graph object.
    
    Parameters:
    -----------
    file_path : str
        Full path to the network file
    
    Returns:
    --------
    G : NetworkX Graph object
        The loaded network
    """
    print(f"\n📊 Loading network from: {os.path.basename(file_path)}")
    
    try:
        G = nx.read_gexf(file_path)
        
        # Print network statistics
        print(f"  ✓ Network loaded successfully!")
        print(f"  • Number of nodes: {G.number_of_nodes()}")
        print(f"  • Number of edges: {G.number_of_edges()}")
        print(f"  • Network type: {'Directed' if G.is_directed() else 'Undirected'}")
        
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


def calculate_average_edge_weight(G, node):
    """
    Calculate the average weight of all edges connected to a node.
    
    Parameters:
    -----------
    G : NetworkX Graph
        The network
    node : node identifier
        The node to analyze
    
    Returns:
    --------
    float : Average edge weight for this node
    """
    neighbors = list(G.neighbors(node))
    if len(neighbors) == 0:
        return 0
    
    total_weight = 0
    for neighbor in neighbors:
        # Check if edge has a weight attribute
        if G.has_edge(node, neighbor):
            edge_data = G[node][neighbor]
            weight = edge_data.get('weight', 1.0)  # Default to 1.0 if no weight
            total_weight += weight
    
    return total_weight / len(neighbors)


def get_top_connected_papers(G, node, top_n=5):
    """
    Get the top N most strongly connected papers to a given node.
    
    Parameters:
    -----------
    G : NetworkX Graph
        The network
    node : node identifier
        The node to analyze
    top_n : int
        Number of top connections to return
    
    Returns:
    --------
    list : List of tuples (neighbor_id, weight)
    """
    neighbors = list(G.neighbors(node))
    
    # Get weights for all neighbors
    neighbor_weights = []
    for neighbor in neighbors:
        if G.has_edge(node, neighbor):
            edge_data = G[node][neighbor]
            weight = edge_data.get('weight', 1.0)
            neighbor_weights.append((neighbor, weight))
    
    # Sort by weight (descending) and return top N
    neighbor_weights.sort(key=lambda x: x[1], reverse=True)
    return neighbor_weights[:top_n]


def create_diagnostic_dataframe(G):
    """
    Create a comprehensive diagnostic dataframe for all nodes in the network.
    
    Parameters:
    -----------
    G : NetworkX Graph
        The network to analyze
    
    Returns:
    --------
    pandas.DataFrame : Diagnostic information for each node
    """
    print("\n📋 Creating diagnostic dataframe...")
    
    data = []
    
    # Calculate clustering coefficients once (it's expensive)
    print("  • Calculating clustering coefficients...")
    try:
        clustering_dict = nx.clustering(G)
    except:
        clustering_dict = {node: 0 for node in G.nodes()}
    
    # Calculate betweenness centrality (can be slow for large networks)
    print("  • Calculating betweenness centrality...")
    try:
        # For large networks, use approximate betweenness
        if G.number_of_nodes() > 500:
            betweenness_dict = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        else:
            betweenness_dict = nx.betweenness_centrality(G)
    except:
        betweenness_dict = {node: 0 for node in G.nodes()}
    
    print("  • Processing each node...")
    for node in G.nodes():
        # Basic properties
        degree = G.degree(node)
        avg_weight = calculate_average_edge_weight(G, node)
        clustering = clustering_dict.get(node, 0)
        betweenness = betweenness_dict.get(node, 0)
        
        # Get top 5 connected papers
        top_connections = get_top_connected_papers(G, node, top_n=5)
        
        # Format top connections as strings
        top_1 = f"{top_connections[0][0]} (weight: {top_connections[0][1]:.4f})" if len(top_connections) > 0 else "N/A"
        top_2 = f"{top_connections[1][0]} (weight: {top_connections[1][1]:.4f})" if len(top_connections) > 1 else "N/A"
        top_3 = f"{top_connections[2][0]} (weight: {top_connections[2][1]:.4f})" if len(top_connections) > 2 else "N/A"
        top_4 = f"{top_connections[3][0]} (weight: {top_connections[3][1]:.4f})" if len(top_connections) > 3 else "N/A"
        top_5 = f"{top_connections[4][0]} (weight: {top_connections[4][1]:.4f})" if len(top_connections) > 4 else "N/A"
        
        # Append to data
        data.append({
            'Node_ID': node,
            'Degree': degree,
            'Average_Edge_Weight': avg_weight,
            'Clustering_Coefficient': clustering,
            'Betweenness_Centrality': betweenness,
            'Top_1_Connected_Paper': top_1,
            'Top_2_Connected_Paper': top_2,
            'Top_3_Connected_Paper': top_3,
            'Top_4_Connected_Paper': top_4,
            'Top_5_Connected_Paper': top_5
        })
    
    df = pd.DataFrame(data)
    
    # Sort by degree (descending) for easier analysis
    df = df.sort_values('Degree', ascending=False).reset_index(drop=True)
    
    print(f"  ✓ Diagnostic dataframe created with {len(df)} nodes")
    
    return df


def visualize_network_matplotlib(G, title, output_filename, layout_type='spring'):
    """
    Create a static visualization using Matplotlib and save as PNG.
    
    KEY FEATURES:
    - Node size: Based on average edge weight (stronger connections = larger nodes)
    - Node color: Based on degree (more connections = darker color)
    - Color scheme: Light to dark (less connected to more connected)
    
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
    """
    print(f"\n🎨 Creating static visualization: {output_filename}.png")
    
    if G is None or G.number_of_nodes() == 0:
        print("  ✗ Cannot visualize empty network")
        return
    
    # Create a large figure for better quality
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    
    # Calculate node positions with more spacing
    print(f"  • Computing {layout_type} layout with increased spacing...")
    
    if layout_type == 'spring':
        # Increased k for more space between nodes
        # k=1.5 provides good spacing while maintaining structure
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Calculate node sizes based on AVERAGE EDGE WEIGHT
    print("  • Calculating node sizes based on average edge weights...")
    node_sizes = []
    for node in G.nodes():
        avg_weight = calculate_average_edge_weight(G, node)
        # Scale the size: multiply by a factor and add minimum size
        size = avg_weight * 5000 + 50  # Adjust multiplier as needed
        node_sizes.append(size)
    
    # Calculate node colors based on DEGREE
    print("  • Calculating node colors based on degree...")
    node_colors = [G.degree(node) for node in G.nodes()]
    
    # Draw the network
    print("  • Drawing network...")
    
    # Draw edges first
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.15,          # More transparent for cleaner look
        edge_color='gray',
        width=0.3,
        ax=ax
    )
    
    # Draw nodes with new color scheme
    # Using 'YlOrBr' (Yellow-Orange-Brown): light for low degree, dark for high degree
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='YlOrBr',  # Light yellow to dark brown
        alpha=0.85,
        edgecolors='black',  # Add black outline for better visibility
        linewidths=0.5,
        ax=ax
    )
    
    # Add labels only if network is small
    if G.number_of_nodes() < 50:
        nx.draw_networkx_labels(
            G, pos,
            font_size=7,
            font_weight='bold',
            ax=ax
        )
    
    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax, label='Node Degree (Number of Connections)', 
                       fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Set title and styling
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save as PNG
    output_path = os.path.join(OUTPUT_PATH, f"{output_filename}.png")
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    print(f"  ✓ Saved PNG to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(OUTPUT_PATH, f"{output_filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved PDF to: {pdf_path}")
    
    plt.close()


def visualize_network_interactive(G, title, output_filename):
    """
    Create an interactive HTML visualization using PyVis.
    
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
    net = Network(
        height='800px',
        width='100%',
        bgcolor='#ffffff',
        font_color='black',
        notebook=False
    )
    
    # Convert NetworkX graph to PyVis format
    print("  • Converting network to interactive format...")
    net.from_nx(G)
    
    # Customize node sizes based on average edge weight
    print("  • Customizing node properties...")
    for node in net.nodes:
        node_id = node['id']
        avg_weight = calculate_average_edge_weight(G, node_id)
        node['value'] = avg_weight * 10  # Scale for visualization
        node['title'] = f"Node: {node_id}<br>Degree: {G.degree(node_id)}<br>Avg Weight: {avg_weight:.4f}"
    
    # Configure physics with more spacing
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.05,
          "damping": 0.5
        },
        "maxVelocity": 40,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 200}
      },
      "nodes": {
        "font": {"size": 14}
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
    
    # Degree statistics
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\n  DEGREE STATISTICS:")
    print(f"  • Average degree: {sum(degrees)/len(degrees):.2f}")
    print(f"  • Max degree: {max(degrees)}")
    print(f"  • Min degree: {min(degrees)}")
    
    # Weight statistics
    print(f"\n  EDGE WEIGHT STATISTICS:")
    weights = []
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        weights.append(weight)
    
    if weights:
        print(f"  • Average edge weight: {sum(weights)/len(weights):.4f}")
        print(f"  • Max edge weight: {max(weights):.4f}")
        print(f"  • Min edge weight: {min(weights):.4f}")
    
    # Connectivity
    if not G.is_directed():
        print(f"\n  CONNECTIVITY:")
        print(f"  • Connected: {nx.is_connected(G)}")
        
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            print(f"  • Number of components: {len(components)}")
            print(f"  • Largest component size: {len(max(components, key=len))}")
    
    # Clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G)
        print(f"\n  CLUSTERING:")
        print(f"  • Average clustering coefficient: {avg_clustering:.4f}")
        print(f"    (0 = no clustering, 1 = highly clustered)")
    except:
        print(f"\n  CLUSTERING: Cannot compute")


# ============================================================================
# SECTION 4: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that runs all the analysis and visualization.
    """
    print("="*70)
    print("CO-CITATION NETWORK ANALYSIS")
    print("="*70)
    
    # Construct full file path
    file_path = os.path.join(INPUT_PATH, INPUT_FILE)
    
    # Load the network
    G = load_network(file_path)
    
    if G is not None:
        # Analyze network properties
        analyze_network_properties(G, INPUT_FILE)
        
        # Create diagnostic dataframe and save to Excel
        print("\n" + "="*70)
        print("CREATING DIAGNOSTIC EXCEL FILE")
        print("="*70)
        
        df_diagnostic = create_diagnostic_dataframe(G)
        excel_output_path = os.path.join(EXCEL_PATH, "00_OA_corpus_LCC_diagnostic.xlsx")
        
        try:
            df_diagnostic.to_excel(excel_output_path, index=False, engine='openpyxl')
            print(f"\n✓ Diagnostic Excel file saved to: {excel_output_path}")
            print(f"  • Total nodes analyzed: {len(df_diagnostic)}")
            print(f"  • Columns: {', '.join(df_diagnostic.columns)}")
        except Exception as e:
            print(f"\n✗ Error saving Excel file: {str(e)}")
            # Fallback to CSV
            csv_output_path = os.path.join(EXCEL_PATH, "00_OA_corpus_LCC_diagnostic.csv")
            df_diagnostic.to_csv(csv_output_path, index=False)
            print(f"  ✓ Saved as CSV instead: {csv_output_path}")
        
        # Create visualizations
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # Static visualization
        visualize_network_matplotlib(
            G,
            title="Co-Citation Network - Largest Connected Component",
            output_filename="00_Co_citation_LCC_static",
            layout_type='spring'
        )
        
        # Interactive visualization
        if G.number_of_nodes() < 1000:
            visualize_network_interactive(
                G,
                title="Co-Citation Network - Largest Connected Component (Interactive)",
                output_filename="00_Co_citation_LCC_interactive"
            )
        else:
            print(f"\n  ⚠ Skipping interactive visualization (network too large)")
            print(f"    Interactive visualizations work best with < 1000 nodes")
    
    print("\n" + "="*70)
    print("✓ ALL ANALYSIS COMPLETED!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  • Diagnostic Excel: {EXCEL_PATH}\\00_OA_corpus_LCC_diagnostic.xlsx")
    print(f"  • Static PNG: {OUTPUT_PATH}\\00_Co_citation_LCC_static.png")
    print(f"  • Static PDF: {OUTPUT_PATH}\\00_Co_citation_LCC_static.pdf")
    print(f"  • Interactive HTML: {OUTPUT_PATH}\\00_Co_citation_LCC_interactive.html")


# ============================================================================
# SECTION 5: RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()