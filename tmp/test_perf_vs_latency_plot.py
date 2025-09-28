import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# --- Data Points ---
# Each tuple represents (Update Latency, Performance)
data = {"CD": (27, 92), "TTT": (28, 65), "T2L": (4, 25), "P2L": (6, 80)}

labels = list(data.keys())
x_coords = [val[0] for val in data.values()]
y_coords = [val[1] for val in data.values()]

# --- Plotting Setup ---
# Using a modern and clean style for the plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

# --- Create the Scatter Plot ---
# Using a single color for points and making them large for visibility
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
scatter = ax.scatter(
    x_coords, y_coords, s=200, c=colors, edgecolors="black", alpha=0.8, zorder=5
)

# --- Annotate Each Point ---
# Add text labels next to each point for clarity
for i, label in enumerate(labels):
    # Adjust text position slightly for better aesthetics
    ax.annotate(
        label,
        (x_coords[i], y_coords[i]),
        textcoords="offset points",
        xytext=(15, 0),  # Offset text to the right of the point
        ha="left",
        fontsize=12,
        fontweight="bold",
        zorder=6,
    )

# --- Add the "Efficient Internalization" Zone ---
# Create a transparent rectangle in the top-left corner
efficient_zone = patches.Rectangle(
    (0, 70),  # (x, y) coordinate of the bottom-left corner
    12,  # Width of the rectangle
    30,  # Height of the rectangle
    edgecolor="green",
    facecolor="green",
    alpha=0.15,  # Transparency level
    linewidth=1.5,
    linestyle="--",
)
ax.add_patch(efficient_zone)

# Add text label for the zone
ax.text(
    1,
    85,  # (x, y) position for the text
    "Efficient Internalization",
    fontsize=14,
    fontweight="bold",
    color="darkgreen",
    rotation=0,  # Can be rotated if needed, e.g., rotation=90
)


# --- Final Touches & Labels ---
# Set plot title and axis labels
ax.set_title(
    "Performance vs. Update Latency Analysis", fontsize=18, fontweight="bold", pad=20
)
ax.set_xlabel("Update Latency (seconds)", fontsize=14, labelpad=15)
ax.set_ylabel("Performance Score", fontsize=14, labelpad=15)

# Set axis limits to provide some padding
ax.set_xlim(0, 32)
ax.set_ylim(0, 105)

# Customize tick parameters for a cleaner look
ax.tick_params(axis="both", which="major", labelsize=12)

# Add a subtle grid
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Ensure the layout is tight and clean
plt.tight_layout()

# --- Display the Plot ---
plt.show()
plt.savefig("perf_vs_latency_plot.png", dpi=300)  # Save as high-res PNG
