#!/usr/bin/env python3
"""
Test script to create a line plot showing needle-in-a-haystack task accuracy vs input length.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

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


def main():
    # Token bins from the user's specification
    tok_bins = [1024 * (i + 1) for i in range(16)]
    tok_bins += [2**14 + 2**12 * (i + 1) for i in range(4)]
    tok_bins += [2**15 + 2**13 * (i + 1) for i in range(12)]
    x_values = tok_bins
    print(x_values)

    y_values = {
        "Base model w/ context": [
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            0.99,
            0.80,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
        ],
        "Ours": [
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            0.95,
            0.87,
            0.80,
            0.65,
            0.53,
            0.43,
            0.37,
            0.28,
            0.24,
            0.17,
            0.15,
        ],
    }

    # Create the plot
    cleaner_x_vals = [
        2**10,
        2**11,
        2**12,
        2**13,
        2**13 + 2**10,
        2**13 + 2**11,
        2**14,
        2**15,
        2**15 + 2**13,
    ]
    plt.figure(figsize=(9, 6))
    for name, ys in y_values.items():
        xys = [
            (x, y)
            for x, y in zip(x_values, ys)
            if x in cleaner_x_vals or x > 2**15 + 2**13
        ]
        plt.plot(
            [xy[0] for xy in xys],
            [xy[1] for xy in xys],
            marker="o",
            linewidth=2,
            markersize=8,
            alpha=1.0,
            label=name,
            markeredgecolor="black",
            markeredgewidth=1,
            markerfacecolor=None,
        )

    # Customize the plot
    plt.xlabel("Haystack Length (tokens)", fontsize=20)
    plt.ylabel("ROUGE-L F1 Score", fontsize=20)
    plt.title(
        "NIAH Retrieval Performance",
        fontsize=24,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Set log scale for x-axis
    plt.xscale("log", base=2)

    # Format x-axis labels with K suffix (representing 1024 multiples)
    # x_labels = [
    #     f"{x // 1024}K" if x >= 1024 else str(x)
    #     for x in x_values
    #     if x in cleaner_x_vals or x > 2**15 + 2**13
    # ]
    # plt.xticks(
    #     [x for x in x_values if x in cleaner_x_vals or x > 2**15 + 2**13],
    #     x_labels,
    #     rotation=90,
    #     fontsize=8,
    # )

    # Remove minor ticks for cleaner appearance
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    # Add some styling
    plt.ylim(-0.05, 1.05)
    plt.xlim(min(x_values) * 0.8, max(x_values) * 1.2)

    # Add horizontal lines for reference

    # Add legend
    plt.legend(loc="lower left")

    # Tight layout for better appearance
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        "/home/tan/research/ctx-to-lora/tmp/needle_haystack_accuracy_plot.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # Show statistics
    print(f"Generated plot with {len(x_values)} data points")
    print(f"Input length range: {min(x_values):.0f} - {max(x_values):.0f} tokens")
    print(
        f"Plot saved to: /home/tan/research/ctx-to-lora/tmp/needle_haystack_accuracy_plot.pdf"
    )

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
