from dataclasses import dataclass  # added

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Styling (reused)
latte_style = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"
plt.style.use(["ggplot", latte_style])
plt.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#FFB83D",
        "#5571EA",
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
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16


@dataclass
class Point:
    context_length: int
    latency: float | None = None
    latency_std: float | None = None
    peak_mem: float | None = None


def combine_std_for_sum(std_list):
    return np.sqrt(sum(std**2 for std in std_list))


# Unified data structure (merged context + latency via Point)
datasets = {
    "NIAH": {
        "Base model w/ ICL": [
            Point(
                context_length=1024,
                latency=375.689 * 1e-3,
                latency_std=19.718 * 1e-3,
                peak_mem=210.722 / 1024,
            ),
            Point(
                context_length=2**11,
                latency=375.428 * 1e-3,
                latency_std=794.507 * 1e-6,
                peak_mem=235.718 / 1024,
            ),
            Point(
                context_length=2**12,
                latency=384.812 * 1e-3,
                latency_std=4.376 * 1e-3,
                peak_mem=429.681 / 1024,
            ),
            Point(
                context_length=2**13,
                latency=438.886 * 1e-3,
                latency_std=17.096 * 1e-3,
                peak_mem=1.165,
            ),
            Point(
                context_length=2**14,
                # latency=13.877,
                # latency_std=394.682,
                peak_mem=1.740,
            ),
            Point(
                context_length=2**15,
                latency=13.985,
                latency_std=66.566 * 1e-3,
                peak_mem=3.223,
            ),
            Point(
                context_length=2**16,
                latency=14.953,
                latency_std=131.882 * 1e-3,
                peak_mem=5.278,
            ),
            Point(
                context_length=2**17,
                latency=18.489,
                latency_std=227.430 * 1e-3,
                peak_mem=12.072,
            ),
        ],
        "Ours": [
            Point(
                context_length=1024,
                latency=300.995 * 1e-3,
                latency_std=24.009 * 1e-3,
                peak_mem=30.988 / 1024,
            ),
            Point(
                context_length=2048,
                latency=296.856 * 1e-3,
                latency_std=568.892 * 1e-6,
                peak_mem=3.207 / 1024,
            ),
            Point(
                context_length=2**12,
                latency=297.037 * 1e-3,
                latency_std=548.641 * 1e-6,
                peak_mem=4.363 / 1024,
            ),
            Point(
                context_length=2**13,
                latency=296.507 * 1e-3,
                latency_std=18.850 * 1e-3,
                peak_mem=33.449 / 1024,
            ),
            Point(
                context_length=2**14,
                latency=297.774 * 1e-3,
                latency_std=788.634 * 1e-6,
                peak_mem=8.792 / 1024,
            ),
            Point(
                context_length=2**15,
                latency=297.553 * 1e-3,
                latency_std=2.016 * 1e-3,
                peak_mem=14.457 / 1024,
            ),
            Point(
                context_length=2**16,
                latency=296.913 * 1e-3,
                latency_std=13.901 * 1e-3,
                peak_mem=25.003 / 1024,
            ),
            Point(
                context_length=2**17,
                latency=274.299 * 1e-3,
                latency_std=53.295 * 1e-3,
                peak_mem=47.696 / 1024,
            ),
        ],
    }
}


def build_color_map(datasets):
    methods = []
    for data in datasets.values():
        for m in data.keys():
            if m not in methods:
                methods.append(m)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def plot_memory_and_latency(datasets):
    """Plot memory and latency vs context length in two subplots."""
    color_map = build_color_map(datasets)

    fig, (ax1) = plt.subplots(1, 1, figsize=(9, 6))

    # Plot 1: Memory vs Context Length
    for dataset_name, methods in datasets.items():
        for method_name, points in methods.items():
            # Extract data with valid memory values
            context_lengths = [
                p.context_length for p in points if p.peak_mem is not None
            ]
            memories = [p.peak_mem for p in points if p.peak_mem is not None]

            if context_lengths and memories:
                ax1.plot(
                    context_lengths,
                    memories,
                    marker="o",
                    linewidth=2,
                    markersize=10,
                    alpha=1.0,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    markerfacecolor=None,
                    label=method_name,
                    color=color_map[method_name],
                )

    ax1.set_xlabel("Haystack Length (tokens)", fontweight="bold", fontsize=20)
    ax1.set_ylabel("Additional Memory (GB)", fontweight="bold", fontsize=20)
    ax1.set_title(
        "Additional Memory Needed for Inference", fontweight="bold", fontsize=24
    )
    ax1.set_xscale("log", base=2)
    # ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    # ax1.legend()

    # # Plot 2: Latency vs Context Length
    # for dataset_name, methods in datasets.items():
    #     for method_name, points in methods.items():
    #         # Extract data with valid latency values
    #         context_lengths = [
    #             p.context_length for p in points if p.latency is not None
    #         ]
    #         latencies = [p.latency for p in points if p.latency is not None]
    #         latency_stds = [
    #             p.latency_std if p.latency_std is not None else 0
    #             for p in points
    #             if p.latency is not None
    #         ]

    #         if context_lengths and latencies:
    #             ax2.errorbar(
    #                 context_lengths,
    #                 latencies,
    #                 yerr=latency_stds,
    #                 marker="o",
    #                 linewidth=2,
    #                 markersize=6,
    #                 capsize=5,
    #                 label=method_name,
    #                 color=color_map[method_name],
    #             )

    # ax2.set_xlabel("Context Length", fontweight="bold")
    # ax2.set_ylabel("Latency (seconds)", fontweight="bold")
    # ax2.set_title("Latency vs Context Length", fontweight="bold", fontsize=16)
    # ax2.set_xscale("log", base=2)
    # ax2.grid(True, alpha=0.3)
    # ax2.legend()

    plt.tight_layout()
    return fig


# Generate the plots
if __name__ == "__main__":
    fig = plot_memory_and_latency(datasets)
    plt.show()

    # Optionally save the figure
    fig.savefig(
        "niah_memory_latency_vs_context_length.pdf", dpi=300, bbox_inches="tight"
    )
