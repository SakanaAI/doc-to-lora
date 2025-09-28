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
    "SQuAD": {
        "Base model w/ context": [
            Point(performance=0.8692, context_ratio=1.0),
        ],
        "Base model w/o context": [
            Point(
                performance=0.1866,
                context_ratio=0.0,
                latency=0.0,
                latency_std=0.0,
                peak_mem=0.0,
            ),
        ],
        "CD (oracle)": [
            Point(
                performance=0.859,
                context_ratio=0.0,
                latency=39.974 + 456.989 * 1e-3,
                latency_std=combine_std_for_sum([965.326 * 1e-3, 259.130 * 1e-3]),
                peak_mem=855.762 / 1024,
            ),
        ],
        "CD (generated queries)": [
            Point(
                performance=0.5986,
                context_ratio=0.0,
                latency=21.254 * 4 + 49.358 + 7.450,
                latency_std=10,
                peak_mem=41.098,
            ),
        ],
        "Ours (batched)": [
            Point(
                # performance=0.7174,
                performance=0.7077,
                # context_ratio=0.0,
                latency=85.954 * 1e-3 + 259.920 * 1e-6,
                latency_std=combine_std_for_sum([31.057 * 1e-3, 740.286 * 1e-6]),
                peak_mem=1.151,
            ),
        ],
        "Ours (iterative)": [
            Point(
                performance=0.715,
                context_ratio=0.0,
                latency=426.849 * 1e-3 + 260.073 * 1e-6,
                latency_std=combine_std_for_sum([43.423 * 1e-3, 29.673 * 1e-6]),
                peak_mem=328.134 / 1024,
            ),
        ],
        "T2L": [
            Point(
                performance=0.1761,
                context_ratio=0.0,
                latency=35.480 * 1e-3,
                latency_std=22.193 * 1e-3,
                peak_mem=1.793,
            ),
        ],
        "LLMLingua-2": [
            Point(performance=0.8492, context_ratio=0.9),
            Point(performance=0.8278, context_ratio=0.8),
            Point(performance=0.7749, context_ratio=0.6),
            Point(performance=0.6624, context_ratio=0.4),
            Point(performance=0.4271, context_ratio=0.2),
            Point(performance=0.3192, context_ratio=0.1),
        ],
    },
    "DROP": {
        "Base model w/ context": [
            Point(performance=0.4541, context_ratio=1.0),
        ],
        "Base model w/o context": [
            Point(
                performance=0.1417,
                context_ratio=0.0,
                latency=0.0,
                latency_std=0.0,
                peak_mem=0.0,
            ),
        ],
        "CD (oracle)": [
            Point(
                performance=0.443,
                context_ratio=0.0,
                latency=39.974 + 422.820 * 1e-3,
                latency_std=combine_std_for_sum([965.326 * 1e-3, 241.419 * 1e-3]),
                peak_mem=1.381,
            ),
        ],
        "CD (generated queries)": [
            Point(
                performance=0.2289,
                context_ratio=0.0,
                latency=23.883 * 4 + 9.755 + 41.301,
                latency_std=14.8852904910,
                peak_mem=44.305,
            ),
        ],
        "Ours (batched)": [
            Point(
                performance=0.2972,
                # context_ratio=0.0,
                latency=84.132 * 1e-3 + 251.876 * 1e-6,
                latency_std=combine_std_for_sum([30.503 * 1e-3, 22.829 * 1e-6]),
                peak_mem=1.976,
            ),
        ],
        "Ours (iterative)": [
            Point(
                # performance=0.3002,
                performance=0.3387,
                context_ratio=0.0,
                latency=419.462 * 1e-3 + 254.899 * 1e-6,
                latency_std=combine_std_for_sum([40.211 * 1e-3, 29.784 * 1e-6]),
                peak_mem=429.881 / 1024,
            ),
        ],
        "T2L": [
            Point(
                performance=0.1468,
                context_ratio=0.0,
                latency=35.036 * 1e-3,
                latency_std=18.144 * 1e-3,
                peak_mem=3.004,
            ),
        ],
        "LLMLingua-2": [  #
            Point(performance=0.4513, context_ratio=0.9),
            Point(performance=0.4488, context_ratio=0.8),
            Point(performance=0.4076, context_ratio=0.6),
            Point(performance=0.3488, context_ratio=0.4),
            Point(performance=0.2617, context_ratio=0.2),
            Point(performance=0.1712, context_ratio=0.1),
        ],
    },
    "ROPES": {
        "Base model w/ context": [
            Point(performance=0.7457, context_ratio=1.0),
        ],
        "Base model w/o context": [
            Point(
                performance=0.4545,
                context_ratio=0.0,
                latency=0.0,
                latency_std=0.0,
                peak_mem=0.0,
            ),
        ],
        "CD (oracle)": [
            Point(
                performance=0.7349,
                context_ratio=0.0,
                latency=39.633 + 285.467 * 1e-3,
                latency_std=combine_std_for_sum([1.752, 72.791 * 1e-3]),
                peak_mem=527.312 / 1024,
            ),
        ],
        "CD (generated queries)": [
            Point(
                performance=0.5786,
                context_ratio=0.0,
                latency=23.714 * 4 + 11.473 + 59.660,
                latency_std=11.7329040736,
                peak_mem=43.271,
            ),
        ],
        "Ours (batched)": [
            Point(
                performance=0.6759,
                # context_ratio=0.0,
                latency=85.652 * 1e-3 + 254.029 * 1e-6,
                latency_std=combine_std_for_sum([30.852 * 1e-3, 51.127 * 1e-6]),
                peak_mem=496.898 / 1024,
            ),
        ],
        "Ours (iterative)": [
            Point(
                # performance=0.6786,
                performance=0.6088,
                context_ratio=0.0,
                latency=417.860 * 1e-3 + 257.422 * 1e-6,
                latency_std=combine_std_for_sum([45.514 * 1e-3, 31.419 * 1e-6]),
                peak_mem=273.867 / 1024,
            ),
        ],
        "T2L": [
            Point(
                performance=0.4661,
                context_ratio=0.0,
                latency=37.913 * 1e-3,
                latency_std=17.313 * 1e-3,
                peak_mem=697.217 / 1024,
            ),
        ],
        "LLMLingua-2": [
            Point(performance=0.708, context_ratio=0.9),
            Point(performance=0.7126, context_ratio=0.8),
            Point(performance=0.7291, context_ratio=0.6),
            Point(performance=0.677, context_ratio=0.4),
            Point(performance=0.6188, context_ratio=0.2),
            Point(performance=0.4894, context_ratio=0.1),
        ],
    },
}


def build_color_map(datasets):
    methods = []
    for data in datasets.values():
        for m in data.keys():
            if m not in methods:
                methods.append(m)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def plot_ctx_and_latency(datasets):
    n = len(datasets)
    # detect memory usage presence
    memory_present = any(
        p.peak_mem is not None
        for methods in datasets.values()
        for plist in methods.values()
        for p in plist
    )
    num_cols = 3 if memory_present else 2
    # Share x-axes across rows only when there are multiple datasets (n > 1)
    if n > 1:
        fig, axes = plt.subplots(
            n,
            num_cols,
            figsize=(6 * num_cols, 5 * n),
            sharey=True,
            sharex="col",
        )
    else:
        fig, axes = plt.subplots(
            n, num_cols, figsize=(6 * num_cols, 5 * n), sharey=True
        )
    if n == 1:
        axes = np.array([axes])

    color_map = build_color_map(datasets)
    global_methods = set()

    label_text = "Efficient and Effective\nInternalization"
    ours_methods = {"Ours (batched)", "Ours (iterative)"}  # added
    annot_name = {
        "Ours (batched)": "Ours (b)",
        "Ours (iterative)": "Ours (i)",
    }

    # Keep track of axes groups per row for later separator lines (always store as list)
    row_axes_groups = []
    for row_idx, (dataset_name, methods) in enumerate(datasets.items()):
        row_axes = axes[row_idx]
        if memory_present:
            ctx_ax, lat_ax, mem_ax = row_axes
        else:
            ctx_ax, lat_ax = row_axes
        # Ensure each row group is a plain Python list of Axes objects
        if isinstance(row_axes, (list, tuple)):
            row_axes_groups.append(list(row_axes))
        elif isinstance(row_axes, np.ndarray):
            row_axes_groups.append(list(row_axes.ravel()))
        else:  # single axis fallback (should not occur with current layout)
            row_axes_groups.append([row_axes])
        # Row subtitle (dataset name) centered across the row using the middle axis
        title_ax = lat_ax  # use latency axis (middle) for centering
        title_ax.set_title(dataset_name, fontweight="bold", fontsize=24, pad=20)

        # Reference performance from ICL
        ref_perf = methods["Base model w/ context"][0].performance
        base_wo_ctx_perf_rel = (
            next(
                p.performance
                for p in methods["Base model w/o context"]
                if p.context_ratio is not None
            )
            / ref_perf
        )

        # --- Context subplot ---
        ours_ctx_present = [  # added
            m
            for m in ours_methods
            if any(p.context_ratio is not None for p in methods.get(m, []))
        ]
        for method, points in methods.items():
            ctx_points = [p for p in points if p.context_ratio is not None]
            if not ctx_points:
                continue
            xs = [p.context_ratio for p in ctx_points]
            ys = [p.performance / ref_perf for p in ctx_points]
            marker = (
                "D"
                if method == "CD (oracle)"
                else ("*" if method in {"Ours (batched)", "Ours (iterative)"} else "o")
            )
            markersize = 17 if "Ours" in method else 12

            if len(xs) == 1:
                ctx_ax.scatter(
                    xs,
                    ys,
                    marker=marker,
                    s=markersize**2,  # scatter uses area, so square the size
                    label=method,
                    color=color_map[method],
                    edgecolors="black",
                    linewidths=1.5,
                    alpha=0.9,
                )
            else:
                ctx_ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    linewidth=2,
                    markersize=markersize,
                    label=method,
                    color=color_map[method],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    alpha=0.9,
                )
            if method in {"Ours (batched)", "Ours (iterative)"}:
                yoff = (
                    (10 if method == "Ours (batched)" else -10)
                    if len(ours_ctx_present) == 2
                    else 6
                )

                ctx_ax.annotate(  # fixed to ctx_ax
                    annot_name[method],
                    (xs[-1], ys[-1]),
                    xytext=(8, yoff),
                    textcoords="offset points",
                    fontsize=12,
                    color=color_map[method],
                    fontweight="bold",
                    bbox=None,
                )
            global_methods.add(method)

        # Add x-axis label on every context subplot (reverted per-user preference)
        ctx_ax.set_xlabel("Context Length Ratio", fontweight="bold", fontsize=16)
        # if row_idx == 0:
        ctx_ax.set_ylabel(
            "Normalized Performance",
            fontweight="bold",
            fontsize=16,
        )
        ctx_ax.grid(True, alpha=0.3)
        ctx_ax.set_xlim(-0.05, 1.05)
        ctx_ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, zorder=-1)
        ctx_ax.axhline(
            base_wo_ctx_perf_rel, color="gray", linestyle="--", alpha=0.5, zorder=-1
        )
        # Removed highlight rectangle and label for context subplot
        # --- Latency subplot ---
        ours_lat_present = [  # added
            m
            for m in ours_methods
            if any(p.latency is not None for p in methods.get(m, []))
        ]
        all_latencies = []
        for method, points in methods.items():
            lat_points = [p for p in points if p.latency is not None]
            if not lat_points:
                continue
            lat_x = [p.latency for p in lat_points]
            perf_rel = [p.performance / ref_perf for p in lat_points]
            xerr = [p.latency_std for p in lat_points]
            marker = (
                "D"
                if method == "CD (oracle)"
                else ("*" if method in {"Ours (batched)", "Ours (iterative)"} else "o")
            )
            markersize = 17 if "Ours" in method else 12

            if len(lat_x) == 1 and all(err == 0 for err in xerr):
                lat_ax.scatter(
                    lat_x,
                    perf_rel,
                    marker=marker,
                    s=markersize**2,
                    label=method,
                    color=color_map[method],
                    edgecolors="black",
                    linewidths=1.5,
                    alpha=0.9,
                )
            else:
                lat_ax.errorbar(
                    lat_x,
                    perf_rel,
                    xerr=xerr,
                    fmt=marker + "-",
                    linewidth=2,
                    markersize=markersize,
                    capsize=4,
                    capthick=1,
                    label=method,
                    color=color_map[method],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    alpha=0.9,
                )
            # Add floating annotation for Ours
            if method in {"Ours (batched)", "Ours (iterative)"}:
                yoff = (
                    (-20 if method == "Ours (batched)" else 10)
                    if len(ours_lat_present) == 2
                    else 6
                )
                lat_ax.annotate(
                    annot_name[method],
                    (lat_x[-1], perf_rel[-1]),
                    xytext=(-25, yoff),
                    textcoords="offset points",
                    fontsize=12,
                    color=color_map[method],
                    fontweight="bold",
                    bbox=None,
                )

            all_latencies.extend(lat_x)
            global_methods.add(method)
        lat_ax.set_xscale("log")
        if all_latencies:
            positive = [x for x in all_latencies if x > 0]
            if positive:
                min_pos = min(positive)
                max_x = max(all_latencies)
                lat_ax.set_xlim(min_pos * 0.8, max_x * 1.2)
        lat_ax.set_xlabel("Update Latency (seconds)", fontweight="bold", fontsize=16)
        lat_ax.grid(True, which="both", alpha=0.3)
        lat_ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, zorder=-1)
        lat_ax.axhline(
            base_wo_ctx_perf_rel, color="gray", linestyle="--", alpha=0.5, zorder=-1
        )
        # Add vertical line at 1 second
        lat_ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, zorder=-1)
        lat_ax.text(
            1.1,
            0.5,
            "Sub-second internalization",
            transform=lat_ax.get_xaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color="gray",
            fontstyle="italic",
            rotation=90,
        )
        # # Highlight rectangle
        #     mpl.patches.Rectangle(
        #         (0.0, 0.5),
        #         0.5,
        #         0.5,
        #         transform=lat_ax.transAxes,
        #         facecolor="green",
        #         edgecolor="none",
        #         alpha=0.15,
        #         zorder=-2,
        #     )
        # )
        # lat_ax.text(
        #     0.02,
        #     0.62,
        #     label_text,
        #     transform=lat_ax.transAxes,
        #     fontsize=8.5,
        #     ha="left",
        #     va="bottom",
        #     color="green",
        #     fontweight="bold",
        # )

        # --- Memory subplot (new) ---
        if memory_present:
            ours_mem_present = [  # added
                m
                for m in ours_methods
                if any(p.peak_mem is not None for p in methods.get(m, []))
            ]
            all_mems = []
            for method, points in methods.items():
                mem_points = [p for p in points if p.peak_mem is not None]
                if not mem_points:
                    continue
                mem_x = [p.peak_mem for p in mem_points]
                perf_rel = [p.performance / ref_perf for p in mem_points]
                xerr = [
                    p.peak_mem_std if p.peak_mem_std is not None else 0
                    for p in mem_points
                ]
                marker = (
                    "D"
                    if method == "CD (oracle)"
                    else (
                        "*" if method in {"Ours (batched)", "Ours (iterative)"} else "o"
                    )
                )
                markersize = 17 if "Ours" in method else 12

                if len(mem_x) == 1 and all(err == 0 for err in xerr):
                    mem_ax.scatter(
                        mem_x,
                        perf_rel,
                        marker=marker,
                        s=markersize**2,
                        label=method,
                        color=color_map[method],
                        edgecolors="black",
                        linewidths=1.5,
                        alpha=0.9,
                    )
                else:
                    mem_ax.errorbar(
                        mem_x,
                        perf_rel,
                        xerr=xerr,
                        fmt=marker + "-",
                        linewidth=2,
                        markersize=markersize,
                        capsize=4,
                        capthick=1,
                        label=method,
                        color=color_map[method],
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        alpha=0.9,
                    )
                # Add floating annotation for Ours
                if method in {"Ours (batched)", "Ours (iterative)"}:
                    yoff = (
                        (-20 if method == "Ours (batched)" else 15)
                        if len(ours_mem_present) == 2
                        else 6
                    )
                    mem_ax.annotate(
                        annot_name[method],
                        (mem_x[-1], perf_rel[-1]),
                        xytext=(-15, yoff),
                        textcoords="offset points",
                        fontsize=12,
                        color=color_map[method],
                        fontweight="bold",
                        bbox=None,
                    )

                all_mems.extend(mem_x)
            if all_mems:
                max_mem = max(all_mems)
                min_mem = min(all_mems)
                span = (
                    max_mem - min_mem
                    if max_mem > min_mem
                    else max_mem
                    if max_mem != 0
                    else 1
                )
                mem_ax.set_xlim(min_mem - 0.05 * span, max_mem * 1.05)
            mem_ax.set_xlabel(
                "Additional Memory Needed\nfor Model Updates (GB)",
                fontweight="bold",
                fontsize=16,
            )
            mem_ax.grid(True, alpha=0.3)
            mem_ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, zorder=-1)
            mem_ax.axhline(
                base_wo_ctx_perf_rel, color="gray", linestyle="--", alpha=0.5, zorder=-1
            )
            # # Highlight rectangle (memory) fixed 20% width in axes coords
            # mem_ax.add_patch(
            #     mpl.patches.Rectangle(
            #         (0.0, 0.6),
            #         0.4,
            #         0.4,
            #         transform=mem_ax.transAxes,
            #         facecolor="green",
            #         edgecolor="none",
            #         alpha=0.15,
            #         zorder=-2,
            #     )
            # )
            # mem_ax.text(
            #     0.02,
            #     0.62,
            #     label_text,
            #     transform=mem_ax.transAxes,
            #     fontsize=8.5,
            #     ha="left",
            #     va="bottom",
            #     color="green",
            #     fontweight="bold",
            # )

    # REMOVE old single legend block
    # (deleted fig.legend(...) that was on the right)

    # Build grouped legends
    in_context = [
        m for m in ["Base model w/ context", "LLMLingua-2"] if m in global_methods
    ]
    in_param_order = [
        "CD (oracle)",
        "CD (generated queries)",
        "Ours (iterative)",
        "Ours (batched)",
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
                    else ("*" if m in {"Ours (batched)", "Ours (iterative)"} else "o")
                ),
                linestyle="-",
                linewidth=2,
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][
                    list(build_color_map(datasets).keys()).index(m)
                    % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
                ],
                markeredgecolor="black",
                markeredgewidth=1.5,
                markersize=20 if "Ours" in m else 10,
            )
            for m in names
        ]

    ic_handles = make_handles(in_context)
    ip_handles = make_handles(in_param)

    # Extra bottom margin for legends
    # (reduced wspace & hspace to tighten gaps)
    # Increased hspace to add more vertical separation between dataset rows
    plt.subplots_adjust(right=0.88, wspace=0.12, hspace=0.8, bottom=0.25)

    # Re-enable x tick labels for all rows (matplotlib hides them for shared axes not on bottom)
    if n > 1:
        for ax in axes.reshape(-1):
            ax.tick_params(labelbottom=True)

    # Set legend title fontweight before creating legends
    plt.rcParams["legend.title_fontsize"] = 14

    fig.legend(
        ic_handles,
        [h.get_label() for h in ic_handles],
        title="In-context knowledge",
        loc="upper center",
        bbox_to_anchor=(0.25, 0.1),
        ncol=len(ic_handles),
        frameon=True,
        fancybox=True,
        fontsize=12,
        title_fontproperties={"weight": "bold", "size": 14},
    )
    fig.legend(
        ip_handles,
        [h.get_label() for h in ip_handles],
        title="In-parameter knowledge",
        loc="upper center",
        bbox_to_anchor=(0.6, 0.1),
        ncol=min(len(ip_handles), 3),
        frameon=True,
        fancybox=True,
        fontsize=12,
        title_fontproperties={"weight": "bold", "size": 14},
    )

    # Removed global suptitle referencing only the last dataset; per-row titles now provided.

    # Add horizontal separator lines between dataset rows (if more than one dataset)
    if len(row_axes_groups) > 1:
        # Use figure coordinate system
        for upper_group, lower_group in zip(row_axes_groups[:-1], row_axes_groups[1:]):
            # Reference axes (first axis in each group)
            upper_ax = upper_group[0]
            lower_ax = lower_group[0]
            upper_bottom = upper_ax.get_position().y0
            lower_top = lower_ax.get_position().y1
            line_y = (upper_bottom + lower_top) / 2.0
            combined = upper_group + lower_group
            x0 = min(ax.get_position().x0 for ax in combined)
            x1 = max(ax.get_position().x1 for ax in combined)
            fig.add_artist(
                mpl.lines.Line2D(
                    [x0, x1],
                    [line_y, line_y],
                    transform=fig.transFigure,
                    color="#cccccc",
                    linewidth=1.2,
                    alpha=0.8,
                )
            )
    return fig


if __name__ == "__main__":
    fig = plot_ctx_and_latency(datasets)
    plt.show()
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_ctx_latency_mem.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        "/home/tan/research/ctx-to-lora/tmp/performance_ctx_latency_mem.pdf",
        dpi=300,
        bbox_inches="tight",
    )
