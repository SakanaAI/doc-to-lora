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
        # "#DC4BDC",
        "#229487",
        "#785EF0",
        "#FF924E",
        "#5BC5DB",
        "#F0434F",
        "#AD6F50",
        # "#1982C4",
        # "#6A4C93",
        # "#56CCF2",
        # "#2F52E0",
        # "#FF006E",
        # "#FF8A00",
        # "#FFD300",
        # "#00A6ED",
        # "#FF7A5A",
        # "#FFAA92",
        # "#FFB7B2",
        # "#FFDAC1",
        # "#E2F0CB",
        # "#B5EAD7",
        # "#C7CEEA",
        # "#D4A5A5",
        # "#E9B0DF",
        # "#F8D3DA",
        # "#FBE0C3",
        # "#FFF0B8",
        # "#E6E2AF",
        # "#A2D2FF",
        # "#B4F8C8",
        # "#FFD6E0",
        # "#FFEFB5",
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

# Reproducibility
rng = np.random.default_rng(42)

# X-axis: context length as a fraction of full ICL
x_inparam = np.array([0.0])  # in-parameter updates
x_icl = np.array([1.0])  # full ICL

# Define mock performances (F1) per dataset, consistent with the paper narrative:
# - ICL highest
# - Compression approaches ICL as ratio increases

datasets = {
    "SQuAD": {
        "Base model w/ ICL": [(1.0, 0.8692)],
        "Base model w/o context": [(0.0, 0.1866)],
        "CD (oracle)": [(0.0, 0.859)],
        "CD (generated queries)": [(0.0, 0.5986)],
        "Ours": [(0.0, 0.7174)],
        "T2L": [(0.0, 0.1761)],
        "LLMLingua-2": [
            (0.9, 0.8492),
            (0.8, 0.8278),
            (0.6, 0.7749),
            (0.4, 0.6624),
            (0.2, 0.4271),
            (0.1, 0.3192),
        ],
    },
    # "DROP": {
    #     "Base model w/ ICL": [(1.0, 0.4541)],
    #     "Base model w/o context": [(0.0, 0.1417)],
    #     "CD (oracle)": [(0.0, 0.443)],
    #     "CD (generated queries)": [(0.0, X)],
    #     "Ours": [(0.0, 0.7174)],
    #     "T2L": [(0.0, 0.1761)],
    #     "LLMLingua-2": [
    #         (0.9, 0.8492),
    #         (0.8, 0.8278),
    #         (0.6, 0.7749),
    #         (0.4, 0.6624),
    #         (0.2, 0.4271),
    #         (0.1, 0.3192),
    #     ],
    # },
}


def plot_performance_comparison(datasets):
    """Plot performance of methods relative to Base model w/ ICL reference."""
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6))

    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]

    global_methods = set()  # collect methods for global legends

    for idx, (dataset_name, methods) in enumerate(datasets.items()):
        ax = axes[idx]

        # Get reference performance from "Base model w/ ICL"
        ref_performance = methods["Base model w/ ICL"][0][1]

        # Get base model w/o context performance for reference line
        no_context_performance = (
            methods["Base model w/o context"][0][1] / ref_performance
        )

        # Define method groups
        in_context_methods = {"Base model w/ ICL", "LLMLingua-2"}
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
                point[1] / ref_performance for point in data_points
            ]  # Relative to reference

            ax.plot(
                x_vals,
                y_vals,
                label=method_name,
                linewidth=2,
                markersize=8,
                marker="o",
                alpha=1,
                markeredgecolor="black",
                markeredgewidth=1,
                markerfacecolor=None,
            )
            global_methods.add(method_name)

        # Formatting
        ax.set_xlabel("Context Length Ratio vs ICL", fontweight="bold")
        ax.set_ylabel("Relative Performance vs ICL)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

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
            linestyle=":",
            alpha=0.7,
            zorder=-1,
        )

        # Add highlight rectangle: fixed 20% of axis width (axes coords), y from 0.8 to 1.0
        ax.add_patch(
            mpl.patches.Rectangle(
                (0.0, 0.8),
                0.2,
                0.2,
                transform=ax.transAxes,
                facecolor="green",
                edgecolor="none",
                alpha=0.15,
                zorder=-2,
            )
        )

    # Global legends (after all subplots)
    in_context = [
        m for m in ["Base model w/ ICL", "LLMLingua-2"] if m in global_methods
    ]
    in_param_order = [
        "CD (oracle)",
        "CD (generated queries)",
        "Ours",
        "T2L",
        "Base model w/o context",
    ]
    in_param = [m for m in in_param_order if m in global_methods]

    def make_handles(names):
        return [
            mpl.lines.Line2D(
                [],
                [],
                label=m,
                marker="o",
                linestyle="-",
                linewidth=2,
                markeredgecolor="black",
                markeredgewidth=1,
                markersize=8,
            )
            for m in names
        ]

    ic_handles = make_handles(in_context)
    ip_handles = make_handles(in_param)

    # Adjust bottom for legends
    plt.subplots_adjust(right=0.75, bottom=0.22)

    # Place legends below figure, centered in two groups
    fig.legend(
        ic_handles,
        [h.get_label() for h in ic_handles],
        title="In-context knowledge",
        loc="upper center",
        bbox_to_anchor=(0.25, -0.05),
        ncol=len(ic_handles),
        frameon=True,
        fancybox=True,
        fontsize=10,
        title_fontsize=11,
    )
    fig.legend(
        ip_handles,
        [h.get_label() for h in ip_handles],
        title="In-parameter knowledge",
        loc="upper center",
        bbox_to_anchor=(0.72, -0.05),
        ncol=len(ip_handles),
        frameon=True,
        fancybox=True,
        fontsize=10,
        title_fontsize=11,
    )
    return fig


# Create the plot
fig = plot_performance_comparison(datasets)
plt.show()

# Optionally save the figure
plt.savefig(
    "/home/tan/research/ctx-to-lora/tmp/performance_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
