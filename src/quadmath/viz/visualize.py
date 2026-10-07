from __future__ import annotations

from typing import Optional, Sequence

from matplotlib import animation
import numpy as np
import csv
import os

from quadmath.core.quadray import Quadray, to_xyz, DEFAULT_EMBEDDING
from quadmath.optimize.nelder_mead_quadray import SimplexState
from quadmath.paths import get_data_dir, get_figure_dir
from quadmath.optimize.discrete_variational import DiscretePath
from quadmath.viz._common import (
    atomic_target,
    embedding_array,
    figure_scope,
    mp4_writer,
    resolve_output_path,
    save_figure,
    set_axes_equal,
)

__all__ = [
    "plot_ivm_neighbors",
    "animate_simplex",
    "plot_simplex_trace",
    "plot_partition_tetrahedron",
    "animate_discrete_path",
]


def plot_ivm_neighbors(
    embedding: Sequence[Sequence[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
    out_path: Optional[str] = None,
) -> str:
    """Scatter the 12 IVM neighbor points in 3D.

    Parameters
    - embedding: 3x4 mapping from A,B,C,D to X,Y,Z (defaults to symmetric embedding).
    - save: If True, write the PNG and the CSV/NPZ data; else write nothing.
    - out_path: PNG destination. A bare file name goes in `quadmath/output/figures/`;
      a path with a directory component is used as given. Defaults to `ivm_neighbors.png`.
      The data files always go to `quadmath/output/data/`.

    Returns
    - str: The PNG path when saved, else "" (the figure is closed; no handle is returned).
    """
    import itertools

    emb = embedding_array(embedding)
    base = [2, 1, 1, 0]
    perms = sorted({p for p in itertools.permutations(base)})
    points = [Quadray(*p) for p in perms]
    xyz = [to_xyz(q, emb) for q in points]

    with figure_scope() as fig:
        ax = fig.add_subplot(111, projection="3d")
        xs, ys, zs = zip(*xyz)
        ax.scatter(xs, ys, zs, c="tab:blue")
        ax.set_title("IVM neighbors: permutations of {2,1,1,0}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)

        outpath = ""
        if save:
            data_dir = get_data_dir()
            outpath = resolve_output_path(out_path or "ivm_neighbors.png", get_figure_dir)
            save_figure(fig, outpath, dpi=160, bbox_inches="tight")
            # Save full raw data alongside the figure
            q_arr = np.array([p.as_tuple() for p in points], dtype=int)
            xyz_arr = np.array(xyz, dtype=float)
            with atomic_target(os.path.join(data_dir, "ivm_neighbors_data.npz")) as tmp:
                np.savez(tmp, quadrays=q_arr, xyz=xyz_arr, embedding=emb)
            with atomic_target(os.path.join(data_dir, "ivm_neighbors_data.csv")) as tmp, open(tmp, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["a", "b", "c", "d", "x", "y", "z"])
                for (a, b, c, d), (x, y, z) in zip(q_arr.tolist(), xyz_arr.tolist()):
                    writer.writerow([a, b, c, d, x, y, z])
    return outpath


def animate_simplex(
    vertices_list,
    embedding: Sequence[Sequence[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
    out_path: Optional[str] = None,
) -> str:
    """Animate simplex evolution across iterations.

    Parameters
    - vertices_list: Sequence of vertex lists (each of length 4) from optimization.
    - embedding: 3x4 mapping to XYZ for plotting.
    - save: If True, write the MP4 and the NPZ/CSV data; else write nothing.
    - out_path: MP4 destination. A bare file name goes in `quadmath/output/figures/`;
      a path with a directory component is used as given. Defaults to `simplex_animation.mp4`.
      The data files always go to `quadmath/output/data/`.

    Returns
    - str: The MP4 path when saved, else "" (no animation is built).
    """
    # Avoid creating an Animation when not saving to prevent Matplotlib warnings
    if not save:
        return ""
    emb = embedding_array(embedding)

    with figure_scope() as fig:
        ax = fig.add_subplot(111, projection="3d")

        def update(frame_idx):
            ax.clear()
            verts = vertices_list[frame_idx]
            pts = [to_xyz(v, emb) for v in verts]
            xs, ys, zs = zip(*pts)
            ax.scatter(xs, ys, zs, c="tab:red")
            ax.set_title(f"Simplex iteration {frame_idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            set_axes_equal(ax)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(vertices_list), interval=400, blit=False)
        data_dir = get_data_dir()
        outpath = resolve_output_path(out_path or "simplex_animation.mp4", get_figure_dir)
        with atomic_target(outpath) as tmp:
            ani.save(tmp, writer=mp4_writer(fps=2))
        # Save raw vertices and xyz trajectory
        verts_ivm = np.array([[v.as_tuple() for v in verts] for verts in vertices_list], dtype=int)
        verts_xyz = np.array(
            [[[to_xyz(v, emb)[i] for i in range(3)] for v in verts] for verts in vertices_list],
            dtype=float,
        )
        with atomic_target(os.path.join(data_dir, "simplex_animation_vertices.npz")) as tmp:
            np.savez(tmp, vertices_ivm=verts_ivm, vertices_xyz=verts_xyz, embedding=emb)
        # CSV (one row per vertex per frame)
        with atomic_target(os.path.join(data_dir, "simplex_animation_vertices.csv")) as tmp, open(tmp, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["frame", "vertex_index", "a", "b", "c", "d", "x", "y", "z"])
            for t, verts in enumerate(vertices_list):
                for j, v in enumerate(verts):
                    x, y, z = to_xyz(v, emb)
                    a, b, c, d = v.as_tuple()
                    writer.writerow([t, j, a, b, c, d, x, y, z])
    return outpath


def plot_simplex_trace(state: SimplexState, save: bool = True, out_path: Optional[str] = None) -> str:
    """Plot per-iteration diagnostics for Nelder–Mead.

    Shows best/worst objective values and spread on the left axis and exact
    IVM tetra-volume on the right axis across iterations. Saves PNG and raw
    CSV/NPZ data when save=True.

    Parameters
    - state: SimplexState from `nelder_mead_quadray` containing diagnostics.
    - save: If True, write outputs; else write nothing.
    - out_path: PNG destination. A bare file name goes in `quadmath/output/figures/`;
      a path with a directory component is used as given. Defaults to `simplex_trace.png`.
      The data files always go to `quadmath/output/data/`.

    Returns
    - str: The PNG path when saved, else "" (the figure is closed; no handle is returned).
    """
    if not save:
        return ""

    iterations = list(range(len(state.volumes)))
    with figure_scope() as fig:
        ax1 = fig.add_subplot(111)
        ax1.plot(iterations, state.best_values, label="best f", color="tab:green")
        ax1.plot(iterations, state.worst_values, label="worst f", color="tab:red", alpha=0.6)
        ax1.plot(iterations, state.spreads, label="spread", color="tab:orange", linestyle="--")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("objective value")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        volumes = [float(v) for v in state.volumes]  # exact Fractions -> floats for plotting
        ax2.step(iterations, volumes, label="volume (IVM)", color="tab:blue", where="post")
        ax2.set_ylabel("IVM volume")

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        fig.tight_layout()

        data_dir = get_data_dir()
        png_path = resolve_output_path(out_path or "simplex_trace.png", get_figure_dir)
        save_figure(fig, png_path, dpi=160, bbox_inches="tight")

        # Save raw arrays
        with atomic_target(os.path.join(data_dir, "simplex_trace.npz")) as tmp:
            np.savez(
                tmp,
                iterations=np.array(iterations, dtype=int),
                best_values=np.array(state.best_values, dtype=float),
                worst_values=np.array(state.worst_values, dtype=float),
                spreads=np.array(state.spreads, dtype=float),
                volumes=np.array(volumes, dtype=float),
            )
        with atomic_target(os.path.join(data_dir, "simplex_trace.csv")) as tmp, open(tmp, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["iteration", "best", "worst", "spread", "volume"])
            for i, b, w, s, v in zip(iterations, state.best_values, state.worst_values, state.spreads, volumes):
                writer.writerow([i, b, w, s, v])

    return png_path

def plot_partition_tetrahedron(
    mu: Sequence[int],
    s: Sequence[int],
    a: Sequence[int],
    psi: Sequence[int],
    embedding: Sequence[Sequence[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
    out_path: Optional[str] = None,
) -> str:
    """Plot the four-fold partition as a labeled tetrahedron in 3D.

    Parameters
    - mu, s, a, psi: 4-tuples (A,B,C,D) of nonnegative integers mapped to Quadrays.
    - embedding: 3x4 mapping from A,B,C,D to X,Y,Z.
    - save: If True, write the PNG and the CSV/NPZ data; else write nothing.
    - out_path: PNG destination. A bare file name goes in `quadmath/output/figures/`;
      a path with a directory component is used as given. Defaults to `partition_tetrahedron.png`.
      The data files always go to `quadmath/output/data/`.

    Returns
    - str: The PNG path when saved, else "" (the figure is closed; no handle is returned).
    """
    emb = embedding_array(embedding)
    points = {
        "mu": Quadray(*mu),
        "s": Quadray(*s),
        "a": Quadray(*a),
        "psi": Quadray(*psi),
    }
    xyz = {name: to_xyz(q, emb) for name, q in points.items()}

    with figure_scope() as fig:
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
        set_axes_equal(ax)

        outpath = ""
        if save:
            data_dir = get_data_dir()
            outpath = resolve_output_path(out_path or "partition_tetrahedron.png", get_figure_dir)
            save_figure(fig, outpath, dpi=160, bbox_inches="tight")
            # Save raw named points as CSV and NPZ
            names = list(points.keys())
            q_arr = np.array([points[n].as_tuple() for n in names], dtype=int)
            xyz_arr = np.array([xyz[n] for n in names], dtype=float)
            with atomic_target(os.path.join(data_dir, "partition_tetrahedron_data.npz")) as tmp:
                np.savez(tmp, names=np.array(names), quadrays=q_arr, xyz=xyz_arr, embedding=emb)
            with atomic_target(os.path.join(data_dir, "partition_tetrahedron_data.csv")) as tmp, open(tmp, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["name", "a", "b", "c", "d", "x", "y", "z"])
                for name, (a, b, c, d), (x, y, z) in zip(names, q_arr.tolist(), xyz_arr.tolist()):
                    writer.writerow([name, a, b, c, d, x, y, z])
    return outpath


def animate_discrete_path(
    path: DiscretePath,
    embedding: Sequence[Sequence[float]] = DEFAULT_EMBEDDING,
    save: bool = True,
    out_path: Optional[str] = None,
) -> str:
    """Animate a point moving along a discrete quadray path.

    When save=True, writes the MP4 (`out_path`, default `discrete_path.mp4`),
    a static PNG of the final step in `quadmath/output/figures/`, and CSV/NPZ
    trajectory data in `quadmath/output/data/`.  A bare MP4 name goes in the
    figure directory; a path with a directory component is used as given.

    Returns
    - str: The MP4 path when saved, else "" (also "" for an empty path).
    """
    if not save:
        return ""
    # Gracefully handle empty paths by skipping animation work
    if len(path.path) == 0:
        return ""
    emb = embedding_array(embedding)

    with figure_scope() as fig:
        ax = fig.add_subplot(111, projection="3d")

        def update(frame_idx):
            ax.clear()
            q = path.path[frame_idx]
            x, y, z = to_xyz(q, emb)
            ax.scatter([x], [y], [z], c="tab:purple")
            ax.set_title(f"Discrete path step {frame_idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            set_axes_equal(ax)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(path.path), interval=300, blit=False)
        figure_dir = get_figure_dir()
        data_dir = get_data_dir()
        outpath = resolve_output_path(out_path or "discrete_path.mp4", get_figure_dir)
        with atomic_target(outpath) as tmp:
            ani.save(tmp, writer=mp4_writer(fps=3))

        # Save raw data

        q_arr = np.array([q.as_tuple() for q in path.path], dtype=int)
        xyz_arr = np.array([to_xyz(q, emb) for q in path.path], dtype=float)
        vals = np.array(path.values, dtype=float)
        with atomic_target(os.path.join(data_dir, "discrete_path.npz")) as tmp:
            np.savez(tmp, quadrays=q_arr, xyz=xyz_arr, values=vals, embedding=emb)
        with atomic_target(os.path.join(data_dir, "discrete_path.csv")) as tmp, open(tmp, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["step", "a", "b", "c", "d", "x", "y", "z", "value"])
            for i, (q, (x, y, z), v) in enumerate(zip(path.path, xyz_arr.tolist(), vals.tolist())):
                a, b, c, d = q.as_tuple()
                writer.writerow([i, a, b, c, d, x, y, z, v])

        # Also save a static PNG of the final step for inclusion in PDFs
        ax.clear()
        qf = path.path[-1]
        xf, yf, zf = to_xyz(qf, emb)
        ax.scatter([xf], [yf], [zf], c="tab:purple")
        ax.set_title("Discrete path (final state)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_equal(ax)
        static_png = os.path.join(figure_dir, "discrete_path_final.png")
        save_figure(fig, static_png, dpi=160, bbox_inches="tight")
    return outpath
