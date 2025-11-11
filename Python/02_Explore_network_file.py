"""
EXPLORADOR DE RED - Entiende tu archivo GEXF
==============================================

Este script te ayuda a entender qué contiene exactamente
tu archivo de red antes de visualizarlo.
Author: Luis Castellanos - le.castellanos10@uniandes.edu.co

USO:
----
1. Coloca este script en la misma carpeta que tu archivo .gexf
2. Ejecuta: python explore_network.py
3. Lee las sugerencias de thresholds al final
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NETWORK_FILE = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out\Complexity_CoCitation_LCC.gexf"

# ============================================================================
# FUNCIONES DE EXPLORACIÓN
# ============================================================================

def explore_network_structure(G):
    """
    Explora la estructura básica de la red.
    """
    print("\n" + "="*70)
    print("ESTRUCTURA BÁSICA DE LA RED")
    print("="*70)
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"  • Tipo: {'Dirigida' if G.is_directed() else 'No dirigida'}")
    print(f"  • Número de nodos: {G.number_of_nodes():,}")
    print(f"  • Número de aristas: {G.number_of_edges():,}")
    print(f"  • Densidad: {nx.density(G):.6f}")
    print(f"    (0 = sin conexiones, 1 = totalmente conectada)")
    
    # Componentes
    if not G.is_directed():
        components = list(nx.connected_components(G))
        print(f"\n🔗 CONECTIVIDAD:")
        print(f"  • ¿Está conectada?: {'Sí' if nx.is_connected(G) else 'No'}")
        print(f"  • Número de componentes: {len(components)}")
        if len(components) > 1:
            sizes = sorted([len(c) for c in components], reverse=True)
            print(f"  • Tamaño del componente más grande: {sizes[0]:,}")
            print(f"  • Tamaños de componentes: {sizes[:10]}")


def explore_node_attributes(G):
    """
    Explora los atributos de los nodos.
    """
    print("\n" + "="*70)
    print("ATRIBUTOS DE NODOS")
    print("="*70)
    
    # Obtener lista de todos los atributos
    sample_node = list(G.nodes())[0]
    attributes = list(G.nodes[sample_node].keys())
    
    print(f"\n📌 ATRIBUTOS DISPONIBLES:")
    for attr in attributes:
        print(f"  • {attr}")
    
    # Analizar el atributo 'count' si existe
    if 'count' in attributes:
        counts = [int(G.nodes[n].get('count', 0)) for n in G.nodes()]
        
        print(f"\n📈 DISTRIBUCIÓN DE 'count' (citaciones):")
        print(f"  • Mínimo: {min(counts)}")
        print(f"  • Máximo: {max(counts)}")
        print(f"  • Promedio: {np.mean(counts):.2f}")
        print(f"  • Mediana: {np.median(counts):.0f}")
        print(f"  • Desviación estándar: {np.std(counts):.2f}")
        
        # Percentiles
        print(f"\n  PERCENTILES:")
        for p in [25, 50, 75, 90, 95, 99]:
            value = np.percentile(counts, p)
            print(f"  • {p}%: {value:.0f}")
        
        # Top nodos más citados
        top_nodes = sorted(
            [(n, G.nodes[n].get('count', 0)) for n in G.nodes()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print(f"\n  🏆 TOP 10 NODOS MÁS CITADOS:")
        for i, (node, count) in enumerate(top_nodes, 1):
            node_short = node.split('/')[-1]
            print(f"  {i:2d}. {node_short}: {count} citaciones")


def explore_edge_attributes(G):
    """
    Explora los atributos de las aristas.
    """
    print("\n" + "="*70)
    print("ATRIBUTOS DE ARISTAS")
    print("="*70)
    
    # Obtener muestra de arista
    sample_edge = list(G.edges(data=True))[0]
    attributes = list(sample_edge[2].keys())
    
    print(f"\n🔗 ATRIBUTOS DISPONIBLES:")
    for attr in attributes:
        print(f"  • {attr}")
    
    # Analizar 'weight' y 'count'
    if 'weight' in attributes:
        weights = [float(data.get('weight', 0)) for u, v, data in G.edges(data=True)]
        
        print(f"\n⚖️  DISTRIBUCIÓN DE 'weight' (peso normalizado):")
        print(f"  • Mínimo: {min(weights):.6f}")
        print(f"  • Máximo: {max(weights):.6f}")
        print(f"  • Promedio: {np.mean(weights):.6f}")
        print(f"  • Mediana: {np.median(weights):.6f}")
        
        print(f"\n  PERCENTILES:")
        for p in [25, 50, 75, 90, 95, 99]:
            value = np.percentile(weights, p)
            print(f"  • {p}%: {value:.6f}")
    
    if 'count' in attributes:
        counts = [int(data.get('count', 0)) for u, v, data in G.edges(data=True)]
        
        print(f"\n🔢 DISTRIBUCIÓN DE 'count' (co-citaciones):")
        print(f"  • Mínimo: {min(counts)}")
        print(f"  • Máximo: {max(counts)}")
        print(f"  • Promedio: {np.mean(counts):.2f}")
        print(f"  • Mediana: {np.median(counts):.0f}")
        
        print(f"\n  PERCENTILES:")
        for p in [25, 50, 75, 90, 95, 99]:
            value = np.percentile(counts, p)
            print(f"  • {p}%: {value:.0f}")


def explore_degree_distribution(G):
    """
    Analiza la distribución de grados (conexiones).
    """
    print("\n" + "="*70)
    print("DISTRIBUCIÓN DE GRADO")
    print("="*70)
    
    degrees = [G.degree(n) for n in G.nodes()]
    
    print(f"\n📊 ESTADÍSTICAS DE GRADO:")
    print(f"  • Grado mínimo: {min(degrees)}")
    print(f"  • Grado máximo: {max(degrees)}")
    print(f"  • Grado promedio: {np.mean(degrees):.2f}")
    print(f"  • Grado mediano: {np.median(degrees):.0f}")
    
    # Nodos más conectados (hubs)
    top_degree_nodes = sorted(
        [(n, G.degree(n)) for n in G.nodes()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    print(f"\n  🌟 TOP 10 NODOS MÁS CONECTADOS (HUBS):")
    for i, (node, degree) in enumerate(top_degree_nodes, 1):
        node_short = node.split('/')[-1]
        count = G.nodes[node].get('count', 'N/A')
        print(f"  {i:2d}. {node_short}: {degree} conexiones (citado {count} veces)")


def create_distribution_plots(G, output_dir=r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"):
    """
    Crea gráficos de las distribuciones.
    """
    print("\n" + "="*70)
    print("CREANDO GRÁFICOS DE DISTRIBUCIÓN")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de grados
    ax1 = axes[0, 0]
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Grade (number of connections)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Degree Distribution', fontweight='bold')
    ax1.set_yscale('log')
    
    # 2. Distribución de citaciones (node count)
    ax2 = axes[0, 1]
    counts = [int(G.nodes[n].get('count', 0)) for n in G.nodes()]
    ax2.hist(counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Number of citations', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Citation Degree', fontweight='bold')
    ax2.set_yscale('log')
    
    # 3. Distribución de pesos de aristas
    ax3 = axes[1, 0]
    weights = [float(data.get('weight', 0)) for u, v, data in G.edges(data=True)]
    ax3.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax3.set_xlabel('Weight (similarity)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Edge Weight Distribution', fontweight='bold')
    
    # 4. Distribución de co-citaciones
    ax4 = axes[1, 1]
    cocitations = [int(data.get('count', 0)) for u, v, data in G.edges(data=True)]
    ax4.hist(cocitations, bins=50, edgecolor='black', alpha=0.7, color='red')
    ax4.set_xlabel('Number of co-citations', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Distribution of Co-citations', fontweight='bold')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "distributions_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Gráficos guardados en: {output_path}")
    plt.close()


def suggest_thresholds(G):
    """
    Sugiere thresholds óptimos basados en los datos.
    """
    print("\n" + "="*70)
    print("SUGERENCIAS DE THRESHOLDS")
    print("="*70)
    
    weights = [float(data.get('weight', 0)) for u, v, data in G.edges(data=True)]
    counts = [int(data.get('count', 0)) for u, v, data in G.edges(data=True)]
    
    print("\n💡 THRESHOLDS SUGERIDOS PARA FILTRADO:")
    print("\n  BASADOS EN PESO (weight):")
    
    for p in [50, 75, 90, 95, 99]:
        value = np.percentile(weights, p)
        remaining = sum(1 for w in weights if w >= value)
        percentage = (remaining / len(weights)) * 100
        print(f"  • Percentil {p}% = {value:.4f}")
        print(f"    → Mantendría {remaining:,} aristas ({percentage:.1f}%)")
    
    print("\n  BASADOS EN CO-CITACIONES (count):")
    for p in [50, 75, 90, 95, 99]:
        value = np.percentile(counts, p)
        remaining = sum(1 for c in counts if c >= value)
        percentage = (remaining / len(counts)) * 100
        print(f"  • Percentil {p}% = {value:.0f} co-citaciones")
        print(f"    → Mantendría {remaining:,} aristas ({percentage:.1f}%)")
    
    print("\n  💡 RECOMENDACIÓN:")
    print("  Para una visualización clara con comunidades visibles:")
    weight_75 = np.percentile(weights, 75)
    print(f"  • Usa weight_threshold = {weight_75:.3f} (percentil 75)")
    print(f"  • O min_cocitations = 5-10")
    print(f"  • Y min_node_count = 10-20")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPLORADOR DE RED DE CO-CITACIÓN")
    print("="*70)
    
    if not os.path.exists(NETWORK_FILE):
        print(f"\n✗ ERROR: Archivo no encontrado: {NETWORK_FILE}")
        print("  Coloca el archivo .gexf en la misma carpeta que este script")
        return
    
    print(f"\n📂 Cargando: {NETWORK_FILE}")
    G = nx.read_gexf(NETWORK_FILE)
    print(f"✓ Red cargada exitosamente")
    
    # Explorar estructura
    explore_network_structure(G)
    
    # Explorar nodos
    explore_node_attributes(G)
    
    # Explorar aristas
    explore_edge_attributes(G)
    
    # Distribución de grados
    explore_degree_distribution(G)
    
    # Crear gráficos
    create_distribution_plots(G)
    
    # Sugerir thresholds
    suggest_thresholds(G)
    
    print("\n" + "="*70)
    print("✓ EXPLORACIÓN COMPLETADA")
    print("="*70)
    print("\nAhora puedes usar estos datos para configurar")
    print("el script de visualización avanzada")


if __name__ == "__main__":
    main()