"""
VISUALIZACIÓN AVANZADA DE RED - Con parámetros optimizados
=============================================================

Este script crea visualizaciones con filtros basados en el análisis previo.

CONFIGURADO CON TUS DATOS:
- Peso mínimo: 0.472 (percentil 75)
- Co-citaciones mínimas: 5
- Citaciones mínimas por nodo: 10

USO:
----
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter, defaultdict

# Detección de comunidades (opcional)
try:
    import community.community_louvain as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️  python-louvain no instalado. Instalar con: pip install python-louvain")

# ============================================================================
# CONFIGURACIÓN - CAMBIA ESTAS RUTAS SEGÚN TUS NECESIDADES
# ============================================================================

# ARCHIVO DE ENTRADA
NETWORK_FILE = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Bibliography\Joe\ComplexPolicyImpact\out\Complexity_CoCitation_LCC.gexf"

# CARPETA DE SALIDA
OUTPUT_DIR = r"C:\Users\User\OneDrive\OneDrive - Universidad de los andes\Global Complexity School\Final project\Images"

# ============================================================================
# PARÁMETROS DE FILTRADO (basados en tus resultados)
# ============================================================================

# Vamos a crear 3 versiones con diferentes niveles de filtrado:

FILTROS = {
    'medio': {
        'weight_threshold': 0.472,  # Percentil 75
        'min_cocitations': 5,
        'min_node_count': 10,
        'descripcion': 'Filtro medio - Bueno para ver estructura general'
    },
    'fuerte': {
        'weight_threshold': 0.667,  # Percentil 90
        'min_cocitations': 7,
        'min_node_count': 20,
        'descripcion': 'Filtro fuerte - Solo conexiones importantes'
    },
    'muy_fuerte': {
        'weight_threshold': 0.775,  # Percentil 95
        'min_cocitations': 9,
        'min_node_count': 30,
        'descripcion': 'Filtro muy fuerte - Núcleo del campo'
    }
}

# ============================================================================
# FUNCIONES
# ============================================================================

def cargar_y_filtrar(archivo, filtro):
    """Carga y filtra la red según parámetros"""
    print(f"\n{'='*70}")
    print(f"APLICANDO FILTRO: {filtro['descripcion']}")
    print(f"{'='*70}")
    print(f"  • Peso mínimo: {filtro['weight_threshold']}")
    print(f"  • Co-citaciones mínimas: {filtro['min_cocitations']}")
    print(f"  • Citaciones mínimas por nodo: {filtro['min_node_count']}")
    
    # Cargar
    G = nx.read_gexf(archivo)
    print(f"\nRed original: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    
    # Filtrar aristas
    aristas_eliminar = []
    for u, v, data in G.edges(data=True):
        peso = float(data.get('weight', 0))
        cocit = int(data.get('count', 0))
        if peso < filtro['weight_threshold'] or cocit < filtro['min_cocitations']:
            aristas_eliminar.append((u, v))
    
    G.remove_edges_from(aristas_eliminar)
    print(f"Después de filtrar aristas: {G.number_of_edges():,} aristas")
    
    # Filtrar nodos
    nodos_eliminar = []
    for node in list(G.nodes()):
        cit = int(G.nodes[node].get('count', 0))
        if cit < filtro['min_node_count']:
            nodos_eliminar.append(node)
    
    G.remove_nodes_from(nodos_eliminar)
    print(f"Después de filtrar nodos: {G.number_of_nodes():,} nodos")
    
    # Componente gigante
    if G.number_of_nodes() > 0:
        componentes = list(nx.connected_components(G))
        if len(componentes) > 0:
            mayor = max(componentes, key=len)
            G = G.subgraph(mayor).copy()
            print(f"Componente gigante (LCC): {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    
    if G.number_of_nodes() > 0:
        print(f"Densidad final: {nx.density(G):.6f}")
    
    return G


def detectar_comunidades(G):
    """Detecta comunidades con Louvain"""
    if not LOUVAIN_AVAILABLE or G.number_of_nodes() == 0:
        return {node: 0 for node in G.nodes()}
    
    print(f"\nDetectando comunidades...")
    partition = community_louvain.best_partition(G, weight='weight')
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    comunidades = Counter(partition.values())
    print(f"  • Comunidades: {len(comunidades)}")
    print(f"  • Modularidad: {modularity:.4f}")
    
    # Top 3 comunidades
    print(f"\n  Top 3 comunidades más grandes:")
    for i, (comm_id, size) in enumerate(comunidades.most_common(3), 1):
        pct = (size / G.number_of_nodes()) * 100
        print(f"    {i}. Comunidad {comm_id}: {size} nodos ({pct:.1f}%)")
    
    return partition


def visualizar(G, partition, nombre, descripcion):
    """Crea visualización de alta calidad"""
    if G.number_of_nodes() == 0:
        print(f"  ⚠️  Red vacía, saltando visualización")
        return
    
    print(f"\n  📊 Creando visualización: {nombre}")
    
    # Figura grande
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
    
    # Layout
    print(f"      Calculando layout...")
    pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)
    
    # Atributos visuales
    tamaños = [50 + np.log1p(float(G.nodes[n].get('count', 1))) * 30 for n in G.nodes()]
    colores_comunidad = [partition.get(node, 0) for node in G.nodes()]
    citaciones = [float(G.nodes[n].get('count', 1)) for n in G.nodes()]
    num_comunidades = len(set(partition.values()))
    
    # Anchos de aristas
    anchos = []
    for u, v, data in G.edges(data=True):
        peso = float(data.get('weight', 0.1))
        ancho = 0.3 + (peso * 2)
        anchos.append(ancho)
    
    # PANEL 1: Por comunidad
    ax1 = axes[0]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=anchos, 
                           edge_color='gray', ax=ax1)
    
    nx.draw_networkx_nodes(G, pos, node_size=tamaños,
                           node_color=colores_comunidad,
                           cmap='tab20', alpha=0.9,
                           linewidths=1, edgecolors='black', ax=ax1)
    
    ax1.set_title(
        f"Red por Comunidades - {descripcion}\n"
        f"{G.number_of_nodes():,} nodos | {G.number_of_edges():,} aristas | "
        f"{num_comunidades} comunidades",
        fontsize=12, fontweight='bold', pad=20
    )
    ax1.axis('off')
    
    # PANEL 2: Por importancia
    ax2 = axes[1]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=anchos,
                           edge_color='gray', ax=ax2)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=tamaños,
                                    node_color=citaciones,
                                    cmap='viridis', alpha=0.9,
                                    linewidths=1, edgecolors='black',
                                    vmin=min(citaciones),
                                    vmax=np.percentile(citaciones, 95),  # Limitar para mejor escala
                                    ax=ax2)
    
    plt.colorbar(nodes, ax=ax2, label='Citaciones', fraction=0.046, pad=0.04)
    
    ax2.set_title(
        f"Red por Importancia - {descripcion}\n"
        f"Tamaño y color = citaciones | Grosor = peso de conexión",
        fontsize=12, fontweight='bold', pad=20
    )
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    archivo_png = os.path.join(OUTPUT_DIR, f"{nombre}.png")
    plt.savefig(archivo_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"      ✓ Guardado: {archivo_png}")
    
    archivo_pdf = os.path.join(OUTPUT_DIR, f"{nombre}.pdf")
    plt.savefig(archivo_pdf, bbox_inches='tight', facecolor='white')
    print(f"      ✓ Guardado: {archivo_pdf}")
    
    plt.close()


def analizar_hubs(G, partition, top_n=10):
    """Analiza los nodos hub más importantes"""
    if G.number_of_nodes() == 0:
        return
    
    print(f"\n{'='*70}")
    print("ANÁLISIS DE NODOS HUB")
    print(f"{'='*70}")
    
    # Calcular métricas
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Combinar información
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
    
    # Top por grado
    print(f"\n🌟 TOP {top_n} POR NÚMERO DE CONEXIONES (grado):")
    top_grado = sorted(nodos_info, key=lambda x: x['grado'], reverse=True)[:top_n]
    for i, info in enumerate(top_grado, 1):
        print(f"  {i:2d}. {info['id']}: "
              f"{info['grado']} conexiones | "
              f"{info['citaciones']} citaciones | "
              f"Comunidad {info['comunidad']}")
    
    # Top por betweenness (puentes)
    print(f"\n🌉 TOP {top_n} PUENTES (betweenness - conectan comunidades):")
    top_between = sorted(nodos_info, key=lambda x: x['betweenness'], reverse=True)[:top_n]
    for i, info in enumerate(top_between, 1):
        print(f"  {i:2d}. {info['id']}: "
              f"betweenness={info['betweenness']:.4f} | "
              f"{info['grado']} conexiones | "
              f"Comunidad {info['comunidad']}")


def crear_comparacion(grafos_data):
    """Crea visualización comparativa"""
    if len(grafos_data) == 0:
        return
    
    print(f"\n{'='*70}")
    print("CREANDO COMPARACIÓN")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    
    for idx, (nombre, G, partition, desc) in enumerate(grafos_data):
        ax = axes[idx]
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f'Red vacía\n{desc}',
                   ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue
        
        # Layout rápido
        pos = nx.spring_layout(G, k=0.8, iterations=30, seed=42)
        
        # Visualización simple
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
            f"{G.number_of_nodes():,} nodos | {G.number_of_edges():,} aristas\n"
            f"{num_com} comunidades",
            fontsize=9
        )
        ax.axis('off')
    
    plt.tight_layout()
    
    archivo = os.path.join(OUTPUT_DIR, "comparacion_filtros.png")
    plt.savefig(archivo, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Comparación guardada: {archivo}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("VISUALIZACIÓN AVANZADA DE RED DE CO-CITACIÓN")
    print("="*70)
    print(f"\n📁 Configuración:")
    print(f"  • Archivo: {NETWORK_FILE}")
    print(f"  • Salida: {OUTPUT_DIR}/")
    
    if not os.path.exists(NETWORK_FILE):
        print(f"\n✗ ERROR: No se encuentra {NETWORK_FILE}")
        return
    
    # Procesar cada filtro
    grafos_para_comparar = []
    
    for nombre_filtro, filtro in FILTROS.items():
        # Cargar y filtrar
        G = cargar_y_filtrar(NETWORK_FILE, filtro)
        
        if G.number_of_nodes() == 0:
            print(f"  ⚠️  Red vacía con este filtro")
            grafos_para_comparar.append((nombre_filtro, G, {}, filtro['descripcion']))
            continue
        
        # Detectar comunidades
        partition = detectar_comunidades(G)
        
        # Visualizar
        visualizar(G, partition, f"red_{nombre_filtro}", filtro['descripcion'])
        
        # Analizar hubs (solo para el filtro medio)
        if nombre_filtro == 'medio':
            analizar_hubs(G, partition)
        
        # Guardar para comparación
        grafos_para_comparar.append((nombre_filtro, G, partition, filtro['descripcion']))
    
    # Crear comparación
    crear_comparacion(grafos_para_comparar)
    
    print("\n" + "="*70)
    print("✓ VISUALIZACIÓN COMPLETADA")
    print("="*70)
    print(f"\n📊 RESULTADOS EN: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nArchivos generados:")
    print("  • red_medio.png / .pdf - Filtro recomendado (mejor balance)")
    print("  • red_fuerte.png / .pdf - Solo conexiones fuertes")
    print("  • red_muy_fuerte.png / .pdf - Núcleo del campo")
    print("  • comparacion_filtros.png - Comparación de los 3 filtros")
    print("\n💡 Recomendación: Empieza revisando 'red_medio.png'")


if __name__ == "__main__":
    main()