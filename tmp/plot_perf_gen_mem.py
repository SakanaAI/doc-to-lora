from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Styling (mirroring the other plot for visual consistency)
latte_style = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"
plt.style.use(["ggplot", latte_style])
plt.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#FFB83D",
        "#EEE684",
        "#555555",
        "#9BC750",
        "#35775A",
        "#5571EA",
        "#6CACFF",
        # "#E84494",
        # "#E47FB0",
        # "#785EF0",
        # "#FF924E",
        # "#5BC5DB",
        "#AD6F50",
        "#A1A9AD",
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
# plt.rcParams["legend.fontsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20


@dataclass
class Point:
    performance: float
    peak_mem: float | None = None  # in GB


# Dataset focused only on memory vs performance (relative performance computed later)
datasets = {
    "2WikiMultihopQA": {
        # We keep Base model w/ context only as a performance reference (no point plotted if memory absent)
        "Base model w/ context": [
            Point(performance=0.242, peak_mem=2.099 * 1024),
        ],
        "Base model w/ truncated context": [
            Point(performance=0.3387, peak_mem=1.194 * 1024),
        ],
        "Base model w/o context": [
            Point(performance=0.2138, peak_mem=65.162),
        ],
        "CD (oracle)": [
            Point(
                performance=0.305,
                peak_mem=18.196,
            ),
        ],
        "CD (5 generated queries)": [
            Point(
                performance=0.2385,
                peak_mem=32.501,
            ),
        ],
        "Ours": [
            Point(
                performance=0.2901,
                peak_mem=35.777,
            ),
        ],
        "Ours + truncated context": [
            Point(
                performance=0.3585,
                peak_mem=1.367 * 1024,
            )
        ],
        # "Ours (8k chunks)": [
        #     Point(
        #         performance=0.2901,
        #         peak_mem=35.777,
        #     ),
        # ],
        "T2L": [
            Point(
                performance=0.2474,
                peak_mem=37.776,
            ),
        ],
        "LLMLingua-2": [
            # Point(performance=0.3141, peak_mem=1.111 * 1024),
            Point(performance=0.3169, peak_mem=1.018 * 1024),
            Point(performance=0.3558, peak_mem=807.366),
            Point(performance=0.3591, peak_mem=592.584),
            Point(performance=0.2767, peak_mem=305.298),
            Point(performance=0.2343, peak_mem=158.760),
        ],
    },
    "MultiFieldQA": {
        "Base model w/ context": [
            Point(performance=0.303, peak_mem=1.211 * 1024),
        ],
        "Base model w/ truncated context": [
            Point(performance=0.3938, peak_mem=1.084 * 1024),
        ],
        "Base model w/o context": [
            Point(performance=0.0948, peak_mem=65.853),
        ],
        "CD (oracle)": [
            Point(
                performance=0.4101,
                peak_mem=7.836,
            ),
        ],
        "CD (5 generated queries)": [
            Point(
                performance=0.165,
                peak_mem=31.587,
            ),
        ],
        "Ours": [
            Point(
                performance=0.1895,
                peak_mem=35.769,
            ),
        ],
        "Ours + truncated context": [
            Point(
                performance=0.4293,
                peak_mem=1.233 * 1024,
            ),
        ],
        "T2L": [
            Point(
                performance=0.083,
                peak_mem=38.907,
            ),
        ],
        "LLMLingua-2": [
            Point(performance=0.4222, peak_mem=909.663),
            Point(performance=0.3898, peak_mem=716.981),
            Point(performance=0.3773, peak_mem=490.478),
            Point(performance=0.2885, peak_mem=256.169),
            Point(performance=0.2298, peak_mem=146.394),
        ],
    },
    "QASPER": {
        # We keep Base model w/ context only as a performance reference (no point plotted if memory absent)
        "Base model w/ context": [
            Point(performance=0.346, peak_mem=2.217 * 1024),
        ],
        "Base model w/ truncated context": [
            Point(performance=0.3839, peak_mem=788.697),
        ],
        "Base model w/o context": [
            Point(performance=0.0859, peak_mem=64.989),
        ],
        "CD (oracle)": [
            Point(
                performance=0.3627,
                peak_mem=19.611,
            ),
        ],
        "CD (5 generated queries)": [
            Point(
                performance=0.1712,
                peak_mem=36.767,
            ),
        ],
        "Ours": [
            Point(
                performance=0.2051,
                peak_mem=46.130,
            ),
        ],
        "Ours + truncated context": [
            Point(
                performance=0.3751,
                peak_mem=873.099,
            ),
        ],
        "T2L": [
            Point(
                performance=0.1259,
                peak_mem=54.176,
            ),
        ],
        "LLMLingua-2": [
            # Point(performance=0.3734, peak_mem=710.176),
            Point(performance=0.3849, peak_mem=647.385),
            Point(performance=0.3838, peak_mem=487.781),
            Point(performance=0.3328, peak_mem=386.447),
            Point(performance=0.3329, peak_mem=219.062),
            Point(performance=0.2614, peak_mem=132.576),
        ],
    },
}


def build_color_map(datasets):
    # Collect all unique methods in a deterministic order
    methods = []
    for data in datasets.values():
        for m in data.keys():
            if m not in methods:
                methods.append(m)

    # Sort methods to ensure consistent ordering across runs
    # methods = sorted(methods)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def plot_memory_vs_performance(datasets):
    n = len(datasets)
    # If multiple datasets, arrange horizontally; otherwise single standard figure
    if n == 1:
        fig, single_ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = [single_ax]
    else:
        fig, axes_arr = plt.subplots(1, n, figsize=(7 * n, 5))
        # Ensure axes is a simple list
        axes = list(np.atleast_1d(axes_arr))

    color_map = build_color_map(datasets)
    global_methods = set()
    ours_methods = {"Ours", "Ours + truncated context"}  # added
    annot_name = {  # added
        "Ours": "Ours",
        "Ours + truncated context": "Ours+ctx",
    }

    highlight_label = "Efficient and Effective\nInternalization"

    for idx, (dataset_name, methods) in enumerate(datasets.items()):
        ax = axes[idx]
        ref_perf = methods["Base model w/ truncated context"][0].performance
        base_wo_ctx_perf_rel = (
            methods["Base model w/o context"][0].performance / ref_perf
            if "Base model w/o context" in methods
            else None
        )

        all_mems = []
        all_rel_perfs = []
        ours_present = [  # added: detect both ours points for staggering
            m
            for m in ours_methods
            if m in methods and any(p.peak_mem is not None for p in methods[m])
        ]
        for method, points in methods.items():
            mem_points = [p for p in points if p.peak_mem is not None]
            if not mem_points:
                continue  # skip reference method if no memory value
            xs = [p.peak_mem for p in mem_points]
            ys = [p.performance / ref_perf for p in mem_points]
            # marker mapping (oracle=diamond, ours=star, others=circle)
            marker = (
                "D"
                if method == "CD (oracle)"
                else ("*" if method in ours_methods else "o")
            )
            size = 20 if "Ours" in method else 12
            ax.plot(
                xs,
                ys,
                linestyle="-",
                marker=marker,
                linewidth=2,
                markersize=size,
                label=method,
                color=color_map[method],
                markeredgecolor="black",
                markeredgewidth=1.5,
                alpha=0.9,
            )
            # Add annotation for Ours, stagger if both are present
            if method in ours_methods:
                yoff = (
                    (10 if method == "Ours" else -10) if len(ours_present) == 2 else 6
                )
                place_left = method == "Ours + truncated context" and dataset_name in {
                    "2WikiMultihopQA",
                    "MultiFieldQA",
                }
                if place_left:
                    yoff = 10  # move annotation above the marker for these two plots
                xoff = -5 if place_left else 8
                ha = "right" if place_left else "left"
                ax.annotate(
                    annot_name[method],
                    (xs[-1], ys[-1]),
                    xytext=(xoff, yoff),
                    textcoords="offset points",
                    fontsize=12,
                    color=color_map[method],
                    fontweight="bold",
                    ha=ha,
                    bbox=None,
                )

            all_mems.extend(xs)
            all_rel_perfs.extend(ys)
            global_methods.add(method)

        if all_mems:
            max_mem = max(all_mems)
            min_mem = min(all_mems)
            span = (
                (max_mem - min_mem)
                if max_mem > min_mem
                else (max_mem if max_mem != 0 else 1)
            )
            ax.set_xlim(5, max_mem * 1.25)

        # Set individual y-axis limits for each subplot
        if all_rel_perfs:
            min_y = min(all_rel_perfs)
            max_y = max(all_rel_perfs)
            y_range = max_y - min_y
            margin = y_range * 0.1 if y_range > 0 else 0.1
            ax.set_ylim(min_y - margin, max_y + margin)

        ax.set_xscale("log")
        ax.set_xlabel(
            "Additional Memory Needed\nfor Generation (MB)",
            fontweight="bold",
            fontsize=22,
        )
        if idx == 0:  # Only show y-label on first plot
            ax.set_ylabel("Normalized Performance", fontweight="bold", fontsize=22)
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, zorder=-1)
        if base_wo_ctx_perf_rel is not None:
            ax.axhline(
                base_wo_ctx_perf_rel, color="gray", linestyle="--", alpha=0.5, zorder=-1
            )

        ax.set_title(dataset_name, fontweight="bold", fontsize=30)

    # Build grouped legends
    in_context = [
        m
        for m in [
            "Base model w/ context",
            "Base model w/ truncated context",
            "LLMLingua-2",
        ]
        if m in global_methods
    ]
    in_param_order = [
        "CD (oracle)",
        "CD (5 generated queries)",
        "Ours",
        "Ours + truncated context",
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
                marker=(
                    "D"
                    if m == "CD (oracle)"
                    else ("*" if m in {"Ours", "Ours + truncated context"} else "o")
                ),
                linestyle="-",
                linewidth=2,
                color=color_map[m],
                markeredgecolor="black",
                markeredgewidth=1.5,
                markersize=22 if "Ours" in m else 14,
            )
            for m in names
        ]

    ic_handles = make_handles(in_context)
    ip_handles = make_handles(in_param)

    # Extra bottom margin for legends
    # (reduced wspace & hspace to tighten gaps)
    # Increased hspace to add more vertical separation between dataset rows
    plt.subplots_adjust(right=0.88, wspace=0.12, hspace=1.0, bottom=0.25)

    # Set legend title fontweight before creating legends
    # plt.rcParams["legend.title_fontsize"] = 14

    fig.legend(
        ic_handles,
        [h.get_label() for h in ic_handles],
        title="In-context knowledge",
        loc="upper center",
        bbox_to_anchor=(0.2, -0.05),
        ncol=2,
        frameon=True,
        fancybox=True,
        fontsize=18,
        title_fontproperties={"weight": "bold", "size": 20},
    )
    fig.legend(
        ip_handles,
        [h.get_label() for h in ip_handles],
        title="In-parameter knowledge",
        loc="upper center",
        bbox_to_anchor=(0.7, -0.05),
        ncol=min(len(ip_handles), 3),
        frameon=True,
        fancybox=True,
        fontsize=18,
        title_fontproperties={"weight": "bold", "size": 20},
    )

    # Adjust spacing: horizontal layout uses wspace, single layout unaffected
    if n == 1:
        plt.subplots_adjust(bottom=0.18, left=0.12, right=0.98)
    else:
        # Tighter horizontal spacing and margins
        plt.subplots_adjust(bottom=0.18, left=0.06, right=0.99, wspace=0.15)
    return fig


if __name__ == "__main__":
    fig = plot_memory_vs_performance(datasets)
    plt.show()
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_gen_mem.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_gen_mem.pdf",
        dpi=300,
        bbox_inches="tight",
    )
