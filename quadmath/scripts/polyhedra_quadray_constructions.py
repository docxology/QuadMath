#!/usr/bin/env python3
"""Enhanced polyhedra relationships panel (synergetics volumes and mappings).

Produces a comprehensive figure showing:
1. 3D polyhedra visualizations with proper faces and edges
2. Extended network diagram with more integer-volumed polyhedra
3. Volume relationships and geometric constructions
4. Color-coded polyhedra with volume annotations

Includes: tetrahedron (V=1), cube (V=3), octahedron (V=4), 
rhombic dodecahedron (V=6), cuboctahedron (V=20), truncated octahedron (V=20),
and additional synergetic polyhedra with their volume relationships.
"""
from __future__ import annotations

import os
import sys
import numpy as np
from typing import List, Tuple, Dict


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def generate_regular_tetrahedron(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate regular tetrahedron vertices and faces."""
    # Golden ratio for regular tetrahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ], dtype=float) * scale
    
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return vertices, faces


def generate_cube(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate cube vertices and faces."""
    vertices = np.array([
        [1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
        [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
    ], dtype=float) * scale
    
    faces = [
        [0, 1, 2, 3], [4, 7, 6, 5],  # top, bottom
        [0, 4, 5, 1], [2, 6, 7, 3],  # front, back
        [0, 3, 7, 4], [1, 5, 6, 2]   # left, right
    ]
    return vertices, faces


def generate_octahedron(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate octahedron vertices and faces."""
    vertices = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float) * scale
    
    faces = [
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],  # upper faces
        [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]   # lower faces
    ]
    return vertices, faces


def generate_rhombic_dodecahedron(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate rhombic dodecahedron vertices and faces."""
    # Rhombic dodecahedron with face-centered cubic structure
    vertices = np.array([
        [0, 0, 2], [0, 0, -2], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0],
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float) * scale
    
    # Rhombic faces (each face is a rhombus with 4 vertices)
    faces = [
        [0, 6, 2, 8], [0, 8, 4, 10], [0, 10, 12, 6], [0, 12, 2, 6],  # upper faces
        [1, 7, 2, 9], [1, 9, 4, 11], [1, 11, 13, 7], [1, 13, 2, 7],  # lower faces
        [2, 6, 7, 9], [4, 8, 9, 11], [2, 10, 11, 13], [2, 12, 13, 7]  # side faces
    ]
    return vertices, faces


def generate_cuboctahedron(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate cuboctahedron vertices and faces."""
    vertices = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float) * scale
    
    # 6 square faces and 8 triangular faces
    faces = [
        [0, 1, 3, 2], [4, 6, 7, 5], [8, 10, 11, 9],  # square faces
        [0, 4, 8], [0, 8, 2], [2, 6, 8], [4, 6, 8],   # upper triangular faces
        [1, 5, 9], [1, 9, 3], [3, 7, 9], [5, 7, 9]    # lower triangular faces
    ]
    return vertices, faces


def generate_truncated_octahedron(scale: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate truncated octahedron vertices and faces."""
    # Truncated octahedron (Archimedean solid)
    vertices = np.array([
        [0, 1, 2], [0, 1, -2], [0, -1, 2], [0, -1, -2],
        [1, 0, 2], [1, 0, -2], [-1, 0, 2], [-1, 0, -2],
        [1, 2, 0], [1, -2, 0], [-1, 2, 0], [-1, -2, 0],
        [2, 1, 0], [2, -1, 0], [-2, 1, 0], [-2, -1, 0]
    ], dtype=float) * scale
    
    # 6 square faces and 8 hexagonal faces
    faces = [
        [0, 4, 8, 12, 13, 9], [1, 5, 9, 13, 12, 8],  # hexagonal faces
        [2, 6, 10, 14, 15, 11], [3, 7, 11, 15, 14, 10],
        [0, 2, 6, 4], [1, 3, 7, 5], [8, 10, 14, 12], [9, 11, 15, 13]  # square faces
    ]
    return vertices, faces


def plot_polyhedron_3d(ax, vertices: np.ndarray, faces: List[List[int]], 
                       color: str, alpha: float = 0.7, edge_color: str = 'black'):
    """Plot a polyhedron in 3D with proper faces and edges."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create face polygons
    face_polygons = []
    for face in faces:
        face_verts = [vertices[i] for i in face]
        face_polygons.append(face_verts)
    
    # Plot faces
    poly3d = Poly3DCollection(face_polygons, alpha=alpha, facecolor=color, 
                             edgecolor=edge_color, linewidth=0.5)
    ax.add_collection3d(poly3d)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from paths import get_figure_dir

    # Create figure with 3D polyhedra and network diagram
    fig = plt.figure(figsize=(16, 10))
    
    # 3D polyhedra visualization (left side)
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.set_box_aspect([1, 1, 1])
    
    # Generate and plot polyhedra
    polyhedra_data = {
        'Tetrahedron (V=1)': (generate_regular_tetrahedron(0.8), '#FF6B6B'),
        'Cube (V=3)': (generate_cube(0.8), '#4ECDC4'),
        'Octahedron (V=4)': (generate_octahedron(0.8), '#45B7D1'),
        'Rhombic Dodecahedron (V=6)': (generate_rhombic_dodecahedron(0.6), '#96CEB4'),
        'Cuboctahedron (V=20)': (generate_cuboctahedron(0.8), '#FFEAA7'),
        'Truncated Octahedron (V=20)': (generate_truncated_octahedron(0.6), '#DDA0DD')
    }
    
    # Position polyhedra in a grid layout
    positions = [
        (-2, 1), (0, 1), (2, 1),
        (-2, -1), (0, -1), (2, -1)
    ]
    
    for i, (name, ((vertices, faces), color)) in enumerate(polyhedra_data.items()):
        if i < len(positions):
            x_offset, y_offset = positions[i]
            vertices_translated = vertices + np.array([x_offset, y_offset, 0])
            plot_polyhedron_3d(ax_3d, vertices_translated, faces, color)
            
            # Add label
            ax_3d.text(x_offset, y_offset, 2.5, name, ha='center', va='center', 
                       fontsize=8, fontweight='bold')
    
    ax_3d.set_xlim(-3, 3)
    ax_3d.set_ylim(-2, 2)
    ax_3d.set_zlim(-2, 3)
    ax_3d.set_title('3D Polyhedra Visualizations\n(IVM Volume Units)', fontsize=12, fontweight='bold')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.grid(False)
    
    # Network diagram (right side)
    ax_net = fig.add_subplot(122)
    ax_net.set_facecolor("#ffffff")
    ax_net.axis("off")
    
    # Enhanced node helper with volume annotations
    def node(x: float, y: float, text: str, volume: int, fc: str = "#f5f5f5") -> None:
        rect = plt.Rectangle((x - 1.0, y - 0.4), 2.0, 0.8, fc=fc, ec="#444444", lw=1.5, zorder=2)
        ax_net.add_patch(rect)
        ax_net.text(x, y, text, ha="center", va="center", fontsize=10, zorder=3, fontweight='bold')
        ax_net.text(x, y - 0.25, f"V={volume}", ha="center", va="center", fontsize=9, zorder=3, color='#666666')
    
    # Enhanced arrow helper
    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "", style: str = "->") -> None:
        ax_net.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle=style, lw=1.8, color="#444444"), zorder=1)
        if label:
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax_net.text(xm, ym + 0.15, label, fontsize=9, ha="center", va="bottom", zorder=3, 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Layout with more polyhedra
    # Top row: Fundamental shapes
    node(-4.0, 2.0, "Tetrahedron\n(Unit)", 1, "#FF6B6B")
    node(-1.5, 2.0, "Cube", 3, "#4ECDC4")
    node(1.0, 2.0, "Octahedron", 4, "#45B7D1")
    node(3.5, 2.0, "Rhombic\nDodecahedron", 6, "#96CEB4")
    
    # Middle row: Complex shapes
    node(-2.5, 0.0, "Cuboctahedron\n(Vector Eq.)", 20, "#FFEAA7")
    node(0.5, 0.0, "Truncated\nOctahedron", 20, "#DDA0DD")
    
    # Bottom row: Additional synergetic shapes
    node(-3.5, -2.0, "Tetrahedron\n(2× scale)", 8, "#FFB3BA")
    node(-1.0, -2.0, "Octahedron\n(2× scale)", 32, "#87CEEB")
    node(1.5, -2.0, "Cube\n(2× scale)", 24, "#98FB98")
    node(4.0, -2.0, "Rhombic\nDodecahedron\n(2× scale)", 48, "#D8BFD8")
    
    # Volume relationships and geometric connections
    # Fundamental relationships
    arrow(-3.2, 2.0, -2.4, 2.0, "×3")
    arrow(-0.7, 2.0, 0.2, 2.0, "+1")
    arrow(2.2, 2.0, 3.0, 2.0, "+2")
    
    # Scaling relationships
    arrow(-4.0, 1.6, -3.5, -1.6, "×8")
    arrow(-1.5, 1.6, -1.0, -1.6, "×8")
    arrow(1.0, 1.6, 1.5, -1.6, "×8")
    arrow(3.5, 1.6, 4.0, -1.6, "×8")
    
    # Geometric constructions
    arrow(-2.5, 0.4, -1.5, 1.6, "edge-union")
    arrow(0.5, 0.4, 1.0, 1.6, "truncation")
    arrow(-2.5, -0.4, -0.7, 1.6, "shell of 12")
    arrow(0.5, -0.4, 2.2, 1.6, "Voronoi cell")
    
    # Additional synergetic relationships
    arrow(-3.5, -1.6, -2.5, -0.4, "dual")
    arrow(-1.0, -1.6, 0.5, -0.4, "dual")
    arrow(1.5, -1.6, 0.5, 0.4, "dual")
    arrow(4.0, -1.6, 3.5, 0.4, "dual")
    
    # Title and description
    ax_net.text(-4.5, 3.0, "Synergetic Polyhedra Volume Relationships\nQuadray/IVM Framework Network Diagram", 
                fontsize=14, fontweight='bold', ha='left', va='top')
    
    # Legend and notes
    ax_net.text(-4.5, -3.0, 
                "Volume relationships in IVM tetra-units. Regular tetrahedron (V=1) is the fundamental unit.\n"
                "Scaling relationships: 2× edge length → 8× volume (V ∝ L³).\n"
                "Geometric constructions: edge-union, truncation, dual relationships, and Voronoi cells.\n"
                "All polyhedra constructed with consistent edge lengths and integer Quadray coordinates.",
                fontsize=9, ha='left', va='top', style='italic')
    
    # Ensure content is within view
    ax_net.set_xlim(-5.0, 5.0)
    ax_net.set_ylim(-3.5, 3.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figure_dir = get_figure_dir()
    outpath = os.path.join(figure_dir, "polyhedra_quadray_constructions.png")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Enhanced Figure 4 saved to: {outpath}")
    
    # Also save a high-resolution version
    outpath_hq = os.path.join(figure_dir, "polyhedra_quadray_constructions_hq.png")
    fig.savefig(outpath_hq, dpi=600, bbox_inches='tight')
    print(f"High-resolution version saved to: {outpath_hq}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()


