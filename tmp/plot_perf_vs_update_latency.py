import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

latte_style = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"
plt.style.use(["ggplot", latte_style])
plt.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#648FFF",
        "#E84494",
        "#9BC750",
        "#229487",
        "#785EF0",
        "#FF924E",
        "#5BC5DB",
        "#F0434F",
        "#AD6F50",
    ]
)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# plt.rcParams["font.family"] = "Ubuntu"
# plt.rcParams["font.size"] = 14
# plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.facecolor"] = "F7FBFC"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["grid.color"] = "cccccc"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["axes.edgecolor"] = "ccd0da"
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13


# Reproducibility
rng = np.random.default_rng(42)

# Define mock performances vs update latency (in seconds)
# X-axis: update latency in seconds (log scale)
# Y-axis: F1 performance


def combine_std_for_sum(std_list):
    """
    Compute combined standard deviation when adding independent random variables.

    For independent random variables X1, X2, ..., Xn:
    Var(X1 + X2 + ... + Xn) = Var(X1) + Var(X2) + ... + Var(Xn)
    Therefore: std(sum) = sqrt(std1² + std2² + ... + stdn²)

    Args:
        std_list: List of standard deviations

    Returns:
        Combined standard deviation for the sum
    """
    return np.sqrt(sum(std**2 for std in std_list))


datasets = {
    "SQuAD": {
        "Base model w/o context": [(0.0, 0.0, 0.1866)],  # No update needed
        "CD (oracle)": [
            (
                39.974 + 456.989 * 1e-3,
                combine_std_for_sum([965.326 * 1e-3, 259.130 * 1e-3]),
                0.859,
            )
        ],
        "CD (generated queries)": [(21.254 * 4 + 49.358 + 7.450, 10, 0.5986)],
        "Ours": [(85.954 * 1e-3, 31.057 * 1e-3, 0.7174)],
        "T2L": [(35.480 * 1e-3, 22.193 * 1e-3, 0.1761)],
    },
    # "DROP": {
    #     "Base model w/o context": [(0.0, 0.1417)],
    #     "CD (oracle)": [(300, 0.443)],
    #     "CD (generated queries)": [(600, 0.3986)],
    #     "Ours": [(120, 0.4174)],
    #     "T2L": [(180, 0.1561)],
    # },
}


def plot_performance_comparison(datasets):
    """Plot performance of methods vs update latency."""
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6))

    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]

    for idx, (dataset_name, methods) in enumerate(datasets.items()):
        ax = axes[idx]

        # Use a fixed reference performance for ICL (since we removed it from the plot)
        ref_performance = 0.8692 if dataset_name == "SQuAD" else 0.4541

        # Get base model w/o context performance for reference line
        no_context_performance = (
            methods["Base model w/o context"][0][2] / ref_performance
        )

        # All methods are now in-parameter methods
        in_param_methods = {
            "Base model w/o context",
            "Ours",
            "T2L",
            "CD (oracle)",
            "CD (generated queries)",
        }

        # Plot each method
        for method_name, data_points in methods.items():
            x_vals = [point[0] for point in data_points]
            y_vals = [
                point[2] / ref_performance for point in data_points
            ]  # Relative to reference (third element)
            std_vals = [
                point[1] / ref_performance for point in data_points
            ]  # Standard deviation (second element)

            ax.errorbar(
                x_vals,
                y_vals,
                xerr=std_vals,
                label=method_name,
                linewidth=2,
                markersize=8,
                marker="o",
                alpha=1,
                markeredgecolor="black",
                markeredgewidth=1,
                markerfacecolor=None,
                capsize=4,
                capthick=1,
            )

        # Calculate max x value for dynamic xlim
        max_x = max(
            point[0] for data_points in methods.values() for point in data_points
        )

        # Formatting
        ax.set_xlabel("Update Latency (seconds)", fontweight="bold")
        ax.set_ylabel("Relative Performance vs ICL", fontweight="bold")
        ax.set_title(f"{dataset_name}", fontweight="bold", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, max_x * 1.2)

        # Add horizontal line at y=1 (reference performance)
        ax.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=-1,
        )

        # Add horizontal line for base model w/o context
        ax.axhline(
            y=no_context_performance,
            color="red",
            linestyle="--",
            alpha=0.7,
            zorder=-1,
        )

        # Create single legend for in-parameter methods
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(
            handles,
            labels,
            title="In-parameter knowledge",
            bbox_to_anchor=(1.05, 0.6),
            loc="upper left",
            ncol=1,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            fancybox=True,
        ).get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend on the right
    return fig


# Create the plot
fig = plot_performance_comparison(datasets)
plt.show()

# Optionally save the figure
plt.savefig(
    "/home/tan/research/ctx-to-lora/tmp/performance_vs_latency.png",
    dpi=300,
    bbox_inches="tight",
)
