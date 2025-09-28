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
        # "#555555",
        "#9BC750",
        # "#35775A",
        "#5571EA",
        # "#6CACFF",
        # "#E84494",
        # "#E47FB0",
        # "#785EF0",
        # "#FF924E",
        # "#5BC5DB",
        # "#AD6F50",
        # "#A1A9AD",
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


@dataclass
class Point:
    performance: float
    context_ratio: float | None = None
    latency: float | None = None
    latency_std: float | None = None
    peak_mem: float | None = None  # added
    peak_mem_std: float | None = None  # added


def combine_std_for_sum(std_list):
    return np.sqrt(sum(std**2 for std in std_list))


# Unified data structure (merged context + latency via Point)
datasets = {
    "2WikiMultihopQA\n(Long Context)": {
        "Base model w/ context": [
            Point(
                performance=0.3387,
                context_ratio=1.0,
            ),
        ],
        "CD": [
            Point(
                performance=0.2385,
                context_ratio=0.0,
                latency=72.537,
                latency_std=10,
                peak_mem=79.371,
            ),
        ],
        "Ours": [
            Point(
                performance=0.2857,
                context_ratio=0.0,
                latency=0.551,
                latency_std=combine_std_for_sum([43.423 * 1e-3, 29.673 * 1e-6]),
                peak_mem=3.791,
            ),
        ],
    },
    # "SQuAD": {
    #     "Base model w/ context": [
    #         Point(performance=0.8692, context_ratio=1.0),
    #     ],
    #     # "Base model w/o context": [
    #     #     Point(
    #     #         performance=0.1866,
    #     #         context_ratio=0.0,
    #     #         latency=0.0,
    #     #         latency_std=0.0,
    #     #         peak_mem=0.0,
    #     #     ),
    #     # ],
    #     # "CD (oracle)": [
    #     #     Point(
    #     #         performance=0.859,
    #     #         context_ratio=0.0,
    #     #         latency=39.974 + 456.989 * 1e-3,
    #     #         latency_std=combine_std_for_sum([965.326 * 1e-3, 259.130 * 1e-3]),
    #     #         peak_mem=855.762 / 1024,
    #     #     ),
    #     # ],
    #     "CD": [
    #         Point(
    #             performance=0.5986,
    #             context_ratio=0.0,
    #             latency=21.254 * 4 + 49.358 + 7.450,
    #             latency_std=10,
    #             peak_mem=41.098,
    #         ),
    #     ],
    #     # "Ours (batched)": [
    #     #     Point(
    #     #         performance=0.7174,
    #     #         # context_ratio=0.0,
    #     #         latency=85.954 * 1e-3 + 259.920 * 1e-6,
    #     #         latency_std=combine_std_for_sum([31.057 * 1e-3, 740.286 * 1e-6]),
    #     #         peak_mem=1.151,
    #     #     ),
    #     # ],
    #     "Ours": [
    #         Point(
    #             performance=0.715,
    #             context_ratio=0.0,
    #             latency=426.849 * 1e-3 + 260.073 * 1e-6,
    #             latency_std=combine_std_for_sum([43.423 * 1e-3, 29.673 * 1e-6]),
    #             peak_mem=328.134 / 1024,
    #         ),
    #     ],
    #     # "T2L": [
    #     #     Point(
    #     #         performance=0.1761,
    #     #         context_ratio=0.0,
    #     #         latency=35.480 * 1e-3,
    #     #         latency_std=22.193 * 1e-3,
    #     #         peak_mem=1.793,
    #     #     ),
    #     # ],
    #     # "LLMLingua-2": [
    #     #     Point(performance=0.8492, context_ratio=0.9),
    #     #     Point(performance=0.8278, context_ratio=0.8),
    #     #     Point(performance=0.7749, context_ratio=0.6),
    #     #     Point(performance=0.6624, context_ratio=0.4),
    #     #     Point(performance=0.4271, context_ratio=0.2),
    #     #     Point(performance=0.3192, context_ratio=0.1),
    #     # ],
    # },
    # "DROP": {
    # "Base model w/ context": [
    #     Point(performance=0.4541, context_ratio=1.0),
    # ],
    # "Base model w/o context": [
    #     Point(
    #         performance=0.1417,
    #         context_ratio=0.0,
    #         latency=0.0,
    #         latency_std=0.0,
    #         peak_mem=0.0,
    #     ),
    # ],
    # "CD (oracle)": [
    #     Point(
    #         performance=0.443,
    #         context_ratio=0.0,
    #         latency=39.974 + 422.820 * 1e-3,
    #         latency_std=combine_std_for_sum([965.326 * 1e-3, 241.419 * 1e-3]),
    #         peak_mem=1.381,
    #     ),
    # ],
    # "CD": [
    #     Point(
    #         performance=0.2289,
    #         context_ratio=0.0,
    #         latency=23.883 * 4 + 9.755 + 41.301,
    #         latency_std=14.8852904910,
    #         peak_mem=44.305,
    #     ),
    # ],
    # "Ours (batched)": [
    #     Point(
    #         performance=0.2972,
    #         # context_ratio=0.0,
    #         latency=84.132 * 1e-3 + 251.876 * 1e-6,
    #         latency_std=combine_std_for_sum([30.503 * 1e-3, 22.829 * 1e-6]),
    #         peak_mem=1.976,
    #     ),
    # ],
    # "Ours": [
    #     Point(
    #         performance=0.3002,
    #         context_ratio=0.0,
    #         latency=419.462 * 1e-3 + 254.899 * 1e-6,
    #         latency_std=combine_std_for_sum([40.211 * 1e-3, 29.784 * 1e-6]),
    #         peak_mem=429.881 / 1024,
    #     ),
    # ],
    # "T2L": [
    #     Point(
    #         performance=0.1468,
    #         context_ratio=0.0,
    #         latency=35.036 * 1e-3,
    #         latency_std=18.144 * 1e-3,
    #         peak_mem=3.004,
    #     ),
    # ],
    # "LLMLingua-2": [  #
    #     Point(performance=0.4513, context_ratio=0.9),
    #     Point(performance=0.4488, context_ratio=0.8),
    #     Point(performance=0.4076, context_ratio=0.6),
    #     Point(performance=0.3488, context_ratio=0.4),
    #     Point(performance=0.2617, context_ratio=0.2),
    #     Point(performance=0.1712, context_ratio=0.1),
    # ],
    # },
    # "ROPES": {
    # "Base model w/ context": [
    #     Point(performance=0.7457, context_ratio=1.0),
    # ],
    # "Base model w/o context": [
    #     Point(
    #         performance=0.4545,
    #         context_ratio=0.0,
    #         latency=0.0,
    #         latency_std=0.0,
    #         peak_mem=0.0,
    #     ),
    # ],
    # "CD (oracle)": [
    #     Point(
    #         performance=0.7349,
    #         context_ratio=0.0,
    #         latency=39.633 + 285.467 * 1e-3,
    #         latency_std=combine_std_for_sum([1.752, 72.791 * 1e-3]),
    #         peak_mem=527.312 / 1024,
    #     ),
    # ],
    # "CD": [
    #     Point(
    #         performance=0.5786,
    #         context_ratio=0.0,
    #         latency=23.714 * 4 + 11.473 + 59.660,
    #         latency_std=11.7329040736,
    #         peak_mem=43.271,
    #     ),
    # ],
    # "Ours (batched)": [
    #     Point(
    #         performance=0.6759,
    #         # context_ratio=0.0,
    #         latency=85.652 * 1e-3 + 254.029 * 1e-6,
    #         latency_std=combine_std_for_sum([30.852 * 1e-3, 51.127 * 1e-6]),
    #         peak_mem=496.898 / 1024,
    #     ),
    # ],
    # "Ours": [
    #     Point(
    #         performance=0.6786,
    #         context_ratio=0.0,
    #         latency=417.860 * 1e-3 + 257.422 * 1e-6,
    #         latency_std=combine_std_for_sum([45.514 * 1e-3, 31.419 * 1e-6]),
    #         peak_mem=273.867 / 1024,
    #     ),
    # ],
    # "T2L": [
    #     Point(
    #         performance=0.4661,
    #         context_ratio=0.0,
    #         latency=37.913 * 1e-3,
    #         latency_std=17.313 * 1e-3,
    #         peak_mem=697.217 / 1024,
    #     ),
    # ],
    # "LLMLingua-2": [
    #     Point(performance=0.708, context_ratio=0.9),
    #     Point(performance=0.7126, context_ratio=0.8),
    #     Point(performance=0.7291, context_ratio=0.6),
    #     Point(performance=0.677, context_ratio=0.4),
    #     Point(performance=0.6188, context_ratio=0.2),
    #     Point(performance=0.4894, context_ratio=0.1),
    # ],
    # },
}


def build_color_map(datasets):
    methods = []
    for data in datasets.values():
        for m in data.keys():
            if m not in methods:
                methods.append(m)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def plot_bar_methods(datasets):
    # Use first dataset found
    dataset_name, methods = next(iter(datasets.items()))
    color_map = build_color_map(datasets)

    ref_perf = methods["Base model w/ context"][0].performance
    # Aggregate metrics per method
    stats_by_method = {}
    for method, points in methods.items():
        if method == "Base model w/ context":
            continue
        if not points:
            continue

        perf_vals = [p.performance for p in points if p.performance is not None]
        lat_vals = [p.latency for p in points if p.latency is not None]
        mem_vals = [p.peak_mem for p in points if p.peak_mem is not None]
        perf = max(perf_vals) / ref_perf if perf_vals else np.nan
        lat = float(np.mean(lat_vals)) if lat_vals else np.nan
        mem = float(np.mean(mem_vals)) if mem_vals else np.nan
        stats_by_method[method] = {"perf": perf, "latency": lat, "memory": mem}

    # Ordering by performance (descending), NaNs last
    with_perf = [m for m, s in stats_by_method.items() if not np.isnan(s["perf"])]
    without_perf = [m for m, s in stats_by_method.items() if np.isnan(s["perf"])]
    order = (
        sorted(with_perf, key=lambda m: stats_by_method[m]["perf"], reverse=True)
        + without_perf
    )

    # Build 3 horizontal bar plots: perf, latency, memory
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=False)
    metric_specs = [
        ("perf", "Normalized\nPerformance"),
        ("latency", "Latency\n(s)"),
        ("memory", "Memory\n(GB)"),
    ]
    for ax, (metric_key, y_label) in zip(axes, metric_specs):
        names = [m for m in order if not np.isnan(stats_by_method[m][metric_key])]
        values = [stats_by_method[m][metric_key] for m in names]
        if not names:
            ax.axis("off")
            continue
        y = np.arange(len(names))
        colors = [color_map.get(m, "#888888") for m in names]
        ax.barh(y, values, color=colors, edgecolor="black", alpha=0.9)
        ax.set_yticks(y, labels=names)
        ax.invert_yaxis()  # highest at top to match ordering
        ax.set_ylabel(y_label, fontweight="bold", fontsize=12)
        ax.grid(True, axis="x", alpha=0.3)
        # Make latency and memory log-scale
        if metric_key in ("latency", "memory"):
            ax.set_xscale("log")

    fig.suptitle(dataset_name, fontweight="bold", fontsize=20, y=0.91)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


if __name__ == "__main__":
    # fig = plot_ctx_and_latency(datasets)
    fig = plot_bar_methods(datasets)
    plt.show()
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_ctx_latency_mem_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_ctx_latency_mem_bar.pdf",
        dpi=300,
        bbox_inches="tight",
    )
