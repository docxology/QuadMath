from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import csv
import os

from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING
from nelder_mead_quadray import SimplexState
from paths import get_output_dir, get_data_dir, get_figure_dir
from discrete_variational import DiscretePath


def _set_axes_equal(ax) -> None:
    """Set 3D axes to equal scale for better geometric interpretation.

    Matplotlib's 3D axes do not support 'equal' aspect natively across versions.
    This helper normalizes the axis ranges to the same span.
    """
    try:
        # Newer Matplotlib supports box aspect directly
        ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    spans = [abs(l[1] - l[0]) for l in limits]
    centers = [(l[0] + l[1]) / 2.0 for l in limits]
    radius = max(spans) / 2.0
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def plot_ivm_neighbors(embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING, save: bool = True) -> str:
    """Scatter the 12 IVM neighbor points in 3D.

    Parameters
    - embedding: 3x4 mapping from A,B,C,D to X,Y,Z (defaults to symmetric embedding).
    - save: If True, write PNG to `quadmath/output/`, else return empty string.

    Returns
    - str: Output file path if saved, else "".
    """
    import itertools

    base = [2, 1, 1, 0]
    perms = sorted({p for p in itertools.permutations(base)})
    points = [Quadray(*p) for p in perms]
    xyz = [to_xyz(q, embedding) for q in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = zip(*xyz)
    ax.scatter(xs, ys, zs, c="tab:blue")
    ax.set_title("IVM neighbors: permutations of {2,1,1,0}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    outpath = ""
    if save:
        figure_dir = get_figure_dir()
        data_dir = get_data_dir()
        outpath = f"{figure_dir}/ivm_neighbors.png"
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        # Save full raw data alongside the figure
        q_arr = np.array([p.as_tuple() for p in points], dtype=int)
        xyz_arr = np.array(xyz, dtype=float)
        emb_arr = np.array(embedding, dtype=float)
        np.savez(os.path.join(data_dir, "ivm_neighbors_data.npz"), quadrays=q_arr, xyz=xyz_arr, embedding=emb_arr)
        with open(os.path.join(data_dir, "ivm_neighbors_data.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c", "d", "x", "y", "z"])
            for (a, b, c, d), (x, y, z) in zip(q_arr.tolist(), xyz_arr.tolist()):
                writer.writerow([a, b, c, d, x, y, z])
    plt.close(fig)
    return outpath


def animate_simplex(vertices_list, embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING, save: bool = True) -> str:
    """Animate simplex evolution across iterations.

    Parameters
    - vertices_list: Sequence of vertex lists (each of length 4) from optimization.
    - embedding: 3x4 mapping to XYZ for plotting.
    - save: If True, write MP4 to `quadmath/output/`, else return empty string.

    Returns
    - str: Output file path if saved, else "".
    """
    # Avoid creating an Animation when not saving to prevent Matplotlib warnings
    if not save:
        return ""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.clear()
        verts = vertices_list[frame_idx]
        pts = [to_xyz(v, embedding) for v in verts]
        xs, ys, zs = zip(*pts)
        ax.scatter(xs, ys, zs, c="tab:red")
        ax.set_title(f"Simplex iteration {frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_axes_equal(ax)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(vertices_list), interval=400, blit=False)
    outpath = ""
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    outpath = f"{figure_dir}/simplex_animation.mp4"
    ani.save(outpath, writer="ffmpeg", fps=2)
    # Save raw vertices and xyz trajectory
    verts_ivm = np.array([[v.as_tuple() for v in verts] for verts in vertices_list], dtype=int)
    verts_xyz = np.array(
        [[[to_xyz(v, embedding)[i] for i in range(3)] for v in verts] for verts in vertices_list],
        dtype=float,
    )
    emb_arr = np.array(embedding, dtype=float)
    np.savez(os.path.join(data_dir, "simplex_animation_vertices.npz"), vertices_ivm=verts_ivm, vertices_xyz=verts_xyz, embedding=emb_arr)
    # CSV (one row per vertex per frame)
    with open(os.path.join(data_dir, "simplex_animation_vertices.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vertex_index", "a", "b", "c", "d", "x", "y", "z"])
        for t, verts in enumerate(vertices_list):
            for j, v in enumerate(verts):
                x, y, z = to_xyz(v, embedding)
                a, b, c, d = v.as_tuple()
                writer.writerow([t, j, a, b, c, d, x, y, z])
    plt.close(fig)
    return outpath


def plot_simplex_trace(state: SimplexState, save: bool = True) -> str:
    """Plot per-iteration diagnostics for Nelder–Mead.

    Shows best/worst objective values and spread on the left axis and integer
    IVM tetra-volume on the right axis across iterations. Saves PNG and raw
    CSV/NPZ data under `quadmath/output/` when save=True.

    Parameters
    - state: SimplexState from `nelder_mead_quadray` containing diagnostics.
    - save: If True, write outputs; else return empty string.

    Returns
    - str: Output PNG path if saved, else "".
    """
    if not save:
        return ""

    iterations = list(range(len(state.volumes)))
    fig, ax1 = plt.subplots()
    ax1.plot(iterations, state.best_values, label="best f", color="tab:green")
    ax1.plot(iterations, state.worst_values, label="worst f", color="tab:red", alpha=0.6)
    ax1.plot(iterations, state.spreads, label="spread", color="tab:orange", linestyle="--")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("objective value")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(iterations, state.volumes, label="volume (IVM)", color="tab:blue", where="post")
    ax2.set_ylabel("integer volume")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()

    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    png_path = os.path.join(figure_dir, "simplex_trace.png")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")

    # Save raw arrays
    import numpy as np  # local import to keep module imports minimal
    import csv
    np.savez(
        os.path.join(data_dir, "simplex_trace.npz"),
        iterations=np.array(iterations, dtype=int),
        best_values=np.array(state.best_values, dtype=float),
        worst_values=np.array(state.worst_values, dtype=float),
        spreads=np.array(state.spreads, dtype=float),
        volumes=np.array(state.volumes, dtype=int),
    )
    with open(os.path.join(data_dir, "simplex_trace.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best", "worst", "spread", "volume"])
        for i, b, w, s, v in zip(iterations, state.best_values, state.worst_values, state.spreads, state.volumes):
            writer.writerow([i, b, w, s, v])

    plt.close(fig)
    return png_path

def plot_partition_tetrahedron(
    mu: Iterable[int],
    s: Iterable[int],
    a: Iterable[int],
    psi: Iterable[int],
    embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
) -> str:
    """Plot the four-fold partition as a labeled tetrahedron in 3D.

    Parameters
    - mu, s, a, psi: 4-tuples (A,B,C,D) of nonnegative integers mapped to Quadrays.
    - embedding: 3x4 mapping from A,B,C,D to X,Y,Z.
    - save: If True, write PNG to `quadmath/output/partition_tetrahedron.png`.

    Returns
    - str: Output file path if saved, else "".
    """
    points = {
        "mu": Quadray(*mu),
        "s": Quadray(*s),
        "a": Quadray(*a),
        "psi": Quadray(*psi),
    }
    xyz = {name: to_xyz(q, embedding) for name, q in points.items()}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter points and labels
    for name, (x, y, z) in xyz.items():
        ax.scatter([x], [y], [z], s=60)
        ax.text(x, y, z, name, fontsize=10)

    # Draw edges of the tetrahedron
    names = list(xyz.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x0, y0, z0 = xyz[names[i]]
            x1, y1, z1 = xyz[names[j]]
            ax.plot([x0, x1], [y0, y1], [z0, z1], c="gray", linewidth=1.0)

    ax.set_title("Four-fold partition mapped to Quadray tetrahedron")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    outpath = ""
    if save:
        figure_dir = get_figure_dir()
        data_dir = get_data_dir()
        outpath = f"{figure_dir}/partition_tetrahedron.png"
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        # Save raw named points as CSV and NPZ
        emb_arr = np.array(embedding, dtype=float)
        names = list(points.keys())
        q_arr = np.array([points[n].as_tuple() for n in names], dtype=int)
        xyz_arr = np.array([xyz[n] for n in names], dtype=float)
        np.savez(os.path.join(data_dir, "partition_tetrahedron_data.npz"), names=np.array(names), quadrays=q_arr, xyz=xyz_arr, embedding=emb_arr)
        with open(os.path.join(data_dir, "partition_tetrahedron_data.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "a", "b", "c", "d", "x", "y", "z"])
            for name, (a, b, c, d), (x, y, z) in zip(names, q_arr.tolist(), xyz_arr.tolist()):
                writer.writerow([name, a, b, c, d, x, y, z])
    plt.close(fig)
    return outpath


def animate_discrete_path(
    path: DiscretePath,
    embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
) -> str:
    """Animate a point moving along a discrete quadray path.

    Saves MP4 and CSV/NPZ trajectory data under `quadmath/output/` when save=True.
    """
    if not save:
        return ""
    # Gracefully handle empty paths by skipping animation work
    if len(path.path) == 0:
        return ""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        ax.clear()
        q = path.path[frame_idx]
        x, y, z = to_xyz(q, embedding)
        ax.scatter([x], [y], [z], c="tab:purple")
        ax.set_title(f"Discrete path step {frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_axes_equal(ax)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(path.path), interval=300, blit=False)
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    outpath = f"{figure_dir}/discrete_path.mp4"
    ani.save(outpath, writer="ffmpeg", fps=3)

    # Save raw data
    import numpy as np  # local to avoid polluting module scope earlier
    import csv
    import os

    q_arr = np.array([q.as_tuple() for q in path.path], dtype=int)
    xyz_arr = np.array([to_xyz(q, embedding) for q in path.path], dtype=float)
    vals = np.array(path.values, dtype=float)
    emb_arr = np.array(embedding, dtype=float)
    np.savez(os.path.join(data_dir, "discrete_path.npz"), quadrays=q_arr, xyz=xyz_arr, values=vals, embedding=emb_arr)
    with open(os.path.join(data_dir, "discrete_path.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "a", "b", "c", "d", "x", "y", "z", "value"])
        for i, (q, (x, y, z), v) in enumerate(zip(path.path, xyz_arr.tolist(), vals.tolist())):
            a, b, c, d = q.as_tuple()
            writer.writerow([i, a, b, c, d, x, y, z, v])

    # Also save a static PNG of the final step for inclusion in PDFs
    ax.clear()
    qf = path.path[-1]
    xf, yf, zf = to_xyz(qf, embedding)
    ax.scatter([xf], [yf], [zf], c="tab:purple")
    ax.set_title("Discrete path (final state)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)
    static_png = os.path.join(figure_dir, "discrete_path_final.png")
    fig.savefig(static_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outpath
