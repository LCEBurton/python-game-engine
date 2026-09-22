"""
Visualization helpers for Profiler field snapshots.
"""
import matplotlib.pyplot as plt


def plot_field_snapshot(profiler, name: str, title: str = None, cmap: str = "inferno", save_path: str = None): # type: ignore
    """
    Plot a captured field snapshot as a heatmap.

    Args:
        profiler: Profiler instance containing the captured field.
        name: Name of the registered field snapshot to plot.
        title: Optional plot title (defaults to `name`).
        cmap: Matplotlib colormap to use.
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    field = profiler.get_field(name)
    if field is None:
        print(f"No field snapshot captured for '{name}' (frame may not have been reached yet).")
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(field, cmap=cmap, origin="upper")
    plt.colorbar(label=name)
    plt.title(title or name)
    plt.xlabel("i (column)")
    plt.ylabel("j (row)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_field_comparison(fields: dict, title: str = None, cmap: str = "inferno", save_path: str = None): # type: ignore
    """
    Plot multiple named 2D fields side-by-side for comparison (e.g. across different
    pressure_iterations or solver methods).

    Args:
        fields: Dict mapping a label (e.g. "10 iters") to a 2D numpy array.
        cmap: Matplotlib colormap to use.
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    vmax = max(f.max() for f in fields.values())

    for i, (label, field) in enumerate(fields.items()):
        im = axes[i].imshow(field, cmap=cmap, origin="upper", vmax=vmax)
        axes[i].set_title(label)
        fig.colorbar(im, ax=axes[i])

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
