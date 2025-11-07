"""
ADVANCED NETWORK VISUALIZATION - With optimized parameters
===========================================================

This script creates visualizations with filters based on previous analysis.

CONFIGURED WITH YOUR DATA:
- Minimum weight: 0.472 (75th percentile)
- Minimum co-citations: 5
- Minimum citations per node: 10

USAGE:
------
1. Place this script in the same folder as your .gexf file
2. Run: python visualize_network.py
3. Check the ./visualizations_output/ folder
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict

# Community detection (optional)
try:
    import community.community_louvain as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️  python-louvain not installed. Install with: pip install python-louvain")

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS AS NEEDED
# ============================================================================

# ARCHIVO DE ENTRADA
NETWORK_FILE = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out\Complexity_CoCitation_LCC.gexf"

# CARPETA DE SALIDA
OUTPUT_DIR = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

# ============================================================================
# FILTERING PARAMETERS (based on your results)
# ============================================================================

# We'll create 3 versions with different filtering levels:

FILTROS = {
    'medium': {
        'weight_threshold': 0.472,  # 75th percentile
        'min_cocitations': 5,
        'min_node_count': 10,
        'descripcion': 'Medium filter - Good for general structure',
        'filename': '02_network_medium_filter'
    },
    'strong': {
        'weight_threshold': 0.667,  # 90th percentile
        'min_cocitations': 7,
        'min_node_count': 20,
        'descripcion': 'Strong filter - Only important connections',
        'filename': '03_network_strong_filter'
    },
    'very_strong': {
        'weight_threshold': 0.775,  # 95th percentile
        'min_cocitations': 9,
        'min_node_count': 30,
        'descripcion': 'Very strong filter - Core of the field',
        'filename': '04_network_very_strong_filter'
    }
}

# ============================================================================
# FUNCIONES
# ============================================================================

def cargar_y_filtrar(archivo, filtro):
    """Loads and filters the network according to parameters"""
    print(f"\n{'='*70}")
    print(f"APPLYING FILTER: {filtro['descripcion']}")
    print(f"{'='*70}")
    print(f"  • Minimum weight: {filtro['weight_threshold']}")
    print(f"  • Minimum co-citations: {filtro['min_cocitations']}")
    print(f"  • Minimum citations per node: {filtro['min_node_count']}")
    
    # Load
    G = nx.read_gexf(archivo)
    print(f"\nOriginal network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Filter edges
    aristas_eliminar = []
    for u, v, data in G.edges(data=True):
        peso = float(data.get('weight', 0))
        cocit = int(data.get('count', 0))
        if peso < filtro['weight_threshold'] or cocit < filtro['min_cocitations']:
            aristas_eliminar.append((u, v))
    
    G.remove_edges_from(aristas_eliminar)
    print(f"After filtering edges: {G.number_of_edges():,} edges")
    
    # Filter nodes
    nodos_eliminar = []
    for node in list(G.nodes()):
        cit = int(G.nodes[node].get('count', 0))
        if cit < filtro['min_node_count']:
            nodos_eliminar.append(node)
    
    G.remove_nodes_from(nodos_eliminar)
    print(f"After filtering nodes: {G.number_of_nodes():,} nodes")
    
    # Largest connected component
    if G.number_of_nodes() > 0:
        componentes = list(nx.connected_components(G))
        if len(componentes) > 0:
            mayor = max(componentes, key=len)
            G = G.subgraph(mayor).copy()
            print(f"Largest connected component (LCC): {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    if G.number_of_nodes() > 0:
        print(f"Final density: {nx.density(G):.6f}")
    
    return G


def detectar_comunidades(G):
    """Detects communities with Louvain algorithm"""
    if not LOUVAIN_AVAILABLE or G.number_of_nodes() == 0:
        return {node: 0 for node in G.nodes()}
    
    print(f"\nDetecting communities...")
    partition = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    comunidades = Counter(partition.values())
    print(f"  • Communities: {len(comunidades)}")
    print(f"  • Modularity: {modularity:.4f}")
    
    # Top 3 communities
    print(f"\n  Top 3 largest communities:")
    for i, (comm_id, size) in enumerate(comunidades.most_common(3), 1):
        pct = (size / G.number_of_nodes()) * 100
        print(f"    {i}. Community {comm_id}: {size} nodes ({pct:.1f}%)")
    
    return partition


def visualizar(G, partition, nombre, descripcion):
    """Creates high-quality visualization with interpretation notes"""
    if G.number_of_nodes() == 0:
        print(f"  ⚠️  Empty network, skipping visualization")
        return
    
    print(f"\n  📊 Creating visualization: {nombre}")
    
    # Large figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
    
    # Layout
    print(f"      Computing layout...")
    pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    
    # Visual attributes
    tamaños = [50 + np.log1p(float(G.nodes[n].get('count', 1))) * 30 for n in G.nodes()]
    colores_comunidad = [partition.get(node, 0) for node in G.nodes()]
    citaciones = [float(G.nodes[n].get('count', 1)) for n in G.nodes()]
    num_comunidades = len(set(partition.values()))
    
    # Edge widths
    anchos = []
    for u, v, data in G.edges(data=True):
        peso = float(data.get('weight', 0.1))
        ancho = 0.3 + (peso * 2)
        anchos.append(ancho)
    
    # PANEL 1: By community
    ax1 = axes[0]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=anchos, 
                           edge_color='gray', ax=ax1)
    
    nx.draw_networkx_nodes(G, pos, node_size=tamaños,
                           node_color=colores_comunidad,
                           cmap='tab20', alpha=0.9,
                           linewidths=1, edgecolors='black', ax=ax1)
    
    ax1.set_title(
        f"Network by Communities - {descripcion}\n"
        f"{G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges | "
        f"{num_comunidades} communities",
        fontsize=12, fontweight='bold', pad=20
    )
    
    # Interpretation note for panel 1
    note1 = ("HOW TO READ: Each node = one paper. Node size = # citations. "
             "Node color = thematic community. \n"
             "Edge thickness = co-citation strength. "
             "Nodes close together = frequently co-cited (thematically related).")
    ax1.text(0.5, -0.05, note1, transform=ax1.transAxes, 
             fontsize=7, ha='center', va='top', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    ax1.axis('off')
    
    # PANEL 2: By importance
    ax2 = axes[1]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=anchos,
                           edge_color='gray', ax=ax2)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=tamaños,
                                    node_color=citaciones,
                                    cmap='viridis', alpha=0.9,
                                    linewidths=1, edgecolors='black',
                                    vmin=min(citaciones),
                                    vmax=np.percentile(citaciones, 95),
                                    ax=ax2)
    
    plt.colorbar(nodes, ax=ax2, label='Citations', fraction=0.046, pad=0.04)
    
    ax2.set_title(
        f"Network by Importance - {descripcion}\n"
        f"Size and color = citations | Thickness = connection weight",
        fontsize=12, fontweight='bold', pad=20
    )
    
    # Interpretation note for panel 2
    note2 = ("HOW TO READ: Darker/larger nodes = more cited papers (more influential). "
             "Lighter nodes = less cited. \n"
             "Edge thickness = how frequently papers are co-cited together. "
             "Central positions = papers cited across multiple topics.")
    ax2.text(0.5, -0.05, note2, transform=ax2.transAxes, 
             fontsize=7, ha='center', va='top', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    archivo_png = os.path.join(OUTPUT_DIR, f"{nombre}.png")
    plt.savefig(archivo_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"      ✓ Saved: {archivo_png}")
    
    archivo_pdf = os.path.join(OUTPUT_DIR, f"{nombre}.pdf")
    plt.savefig(archivo_pdf, bbox_inches='tight', facecolor='white')
    print(f"      ✓ Saved: {archivo_pdf}")
    
    plt.close()


def analizar_hubs(G, partition, top_n=10):
    """Analyzes the most important hub nodes"""
    if G.number_of_nodes() == 0:
        return
    
    print(f"\n{'='*70}")
    print("HUB NODE ANALYSIS")
    print(f"{'='*70}")
    
    # Calculate metrics
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Combine information
    nodos_info = []
    for node in G.nodes():
        nodos_info.append({
            'id': node.split('/')[-1],
            'citaciones': int(G.nodes[node].get('count', 0)),
            'grado': G.degree(node),
            'degree_cent': degree_cent[node],
            'betweenness': betweenness[node],
            'comunidad': partition.get(node, 0)
        })
    
    # Top by degree
    print(f"\n🌟 TOP {top_n} BY NUMBER OF CONNECTIONS (degree):")
    top_grado = sorted(nodos_info, key=lambda x: x['grado'], reverse=True)[:top_n]
    for i, info in enumerate(top_grado, 1):
        print(f"  {i:2d}. {info['id']}: "
              f"{info['grado']} connections | "
              f"{info['citaciones']} citations | "
              f"Community {info['comunidad']}")
    
    # Top by betweenness (bridges)
    print(f"\n🌉 TOP {top_n} BRIDGES (betweenness - connect communities):")
    top_between = sorted(nodos_info, key=lambda x: x['betweenness'], reverse=True)[:top_n]
    for i, info in enumerate(top_between, 1):
        print(f"  {i:2d}. {info['id']}: "
              f"betweenness={info['betweenness']:.4f} | "
              f"{info['grado']} connections | "
              f"Community {info['comunidad']}")


def crear_comparacion(grafos_data):
    """Creates comparative visualization"""
    if len(grafos_data) == 0:
        return
    
    print(f"\n{'='*70}")
    print("CREATING COMPARISON")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    
    for idx, (nombre, G, partition, desc) in enumerate(grafos_data):
        ax = axes[idx]
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f'Empty network\n{desc}',
                   ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue
        
        # Quick layout
        pos = nx.spring_layout(G, k=0.8, iterations=30, seed=42)
        
        # Simple visualization
        tamaños = [20 + np.log1p(float(G.nodes[n].get('count', 1))) * 10 
                   for n in G.nodes()]
        colores = [partition.get(node, 0) for node in G.nodes()]
        
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=tamaños,
                               node_color=colores, cmap='tab20',
                               alpha=0.8, ax=ax)
        
        num_com = len(set(partition.values()))
        ax.set_title(
            f"{desc}\n"
            f"{G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges\n"
            f"{num_com} communities",
            fontsize=9
        )
        ax.axis('off')
    
    # Overall interpretation note
    fig.text(0.5, 0.02, 
             "HOW TO READ: Each panel shows the same network with different filtering levels. "
             "Left = more nodes/edges (general view). Right = fewer nodes/edges (core structure). "
             "Node size = citations, color = community.",
             ha='center', fontsize=8, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    archivo = os.path.join(OUTPUT_DIR, "01_comparison_all_filters.png")
    plt.savefig(archivo, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Comparison saved: {archivo}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("ADVANCED CO-CITATION NETWORK VISUALIZATION")
    print("="*70)
    print(f"\n📁 Configuration:")
    print(f"  • File: {NETWORK_FILE}")
    print(f"  • Output: {OUTPUT_DIR}/")
    
    if not os.path.exists(NETWORK_FILE):
        print(f"\n✗ ERROR: File not found: {NETWORK_FILE}")
        return
    
    # Process each filter
    grafos_para_comparar = []
    
    for nombre_filtro, filtro in FILTROS.items():
        # Load and filter
        G = cargar_y_filtrar(NETWORK_FILE, filtro)
        
        if G.number_of_nodes() == 0:
            print(f"  ⚠️  Empty network with this filter")
            grafos_para_comparar.append((nombre_filtro, G, {}, filtro['descripcion']))
            continue
        
        # Detect communities
        partition = detectar_comunidades(G)
        
        # Visualize
        visualizar(G, partition, filtro['filename'], filtro['descripcion'])
        
        # Analyze hubs (only for medium filter)
        if nombre_filtro == 'medium':
            analizar_hubs(G, partition)
        
        # Save for comparison
        grafos_para_comparar.append((nombre_filtro, G, partition, filtro['descripcion']))
    
    # Create comparison
    crear_comparacion(grafos_para_comparar)
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETED")
    print("="*70)
    print(f"\n📊 RESULTS IN: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nFiles generated (in recommended viewing order):")
    print("  • 01_comparison_all_filters.png - Compare all 3 filters side-by-side")
    print("  • 02_network_medium_filter.png / .pdf - Recommended filter (best balance)")
    print("  • 03_network_strong_filter.png / .pdf - Only strong connections")
    print("  • 04_network_very_strong_filter.png / .pdf - Core of the field")
    print("\n💡 Recommendation: Start by reviewing '01_comparison_all_filters.png',")
    print("   then explore '02_network_medium_filter.png' in detail")


if __name__ == "__main__":
    main()
