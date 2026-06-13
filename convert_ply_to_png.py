#!/usr/bin/env python3
"""
Convert PLY files to PNG images for visualization
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

def render_ply_to_png(ply_path, output_path, width=12, height=9):
    """
    Render a PLY file to PNG image using matplotlib
    
    Args:
        ply_path: Path to input PLY file
        output_path: Path to output PNG file
        width: Figure width in inches
        height: Figure height in inches
    """
    vertices = None
    faces = None

    if HAS_TRIMESH:
        try:
            mesh = trimesh.load(str(ply_path))
            if isinstance(mesh, trimesh.PointCloud):
                vertices = mesh.vertices
            else:
                vertices = mesh.vertices
                faces = mesh.faces
        except Exception as e:
            print(f"trimesh error: {e}")

    if vertices is None:
        vertices_raw = read_ply_simple(ply_path)
        if vertices_raw is None:
            print(f"Error: could not read {ply_path}")
            return
        vertices = vertices_raw

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection='3d')

    if faces is not None:
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                       cmap='viridis', alpha=0.8, edgecolor='none')
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                   c=vertices[:, 2], cmap='viridis', s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(ply_path.stem)

    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def read_ply_simple(ply_path):
    """Simple PLY reader for vertex coordinates"""
    if HAS_OPEN3D:
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            vertices = np.asarray(pcd.points)
            if len(vertices) > 0:
                return vertices
            # Try reading as mesh
            mesh = o3d.io.read_triangle_mesh(str(ply_path))
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                return vertices
        except Exception as e:
            print(f"open3d read error: {e}")
    try:
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_path))
        vertices = np.vstack([plydata['vertex']['x'],
                             plydata['vertex']['y'],
                             plydata['vertex']['z']]).T
        return vertices
    except Exception:
        return None

def main():
    # Find all PLY files in volume_output
    base_dir = Path(__file__).parent / "volume_output"
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return
    
    ply_files = list(base_dir.glob("**/*.ply"))
    
    if not ply_files:
        print("No PLY files found in volume_output/")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    for ply_file in ply_files:
        # Create output PNG path (same location, replace .ply with .png)
        png_file = ply_file.with_suffix('.png')
        
        try:
            print(f"Converting: {ply_file.relative_to(base_dir)}")
            render_ply_to_png(ply_file, png_file)
        except Exception as e:
            print(f"Error converting {ply_file}: {e}")
            continue
    
    print(f"\nConversion complete! Check volume_output/ for PNG files.")

if __name__ == "__main__":
    main()
