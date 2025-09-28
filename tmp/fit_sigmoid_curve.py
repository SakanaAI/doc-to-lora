import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def sigmoid(x, k, x0):
    """
    Sigmoid function with L fixed at 1.

    Parameters:
    x (array-like): Input data (independent variable).
    k (float): The steepness or growth rate of the curve.
    x0 (float): The x-value of the sigmoid's midpoint.
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


# --- User Input ---
# Replace these with your actual x and y data points.
# You can have 3 or more points.
x_data = np.array([10, 20, 30])
y_data = np.array([0.503, 0.554, 0.590])
# --------------------

# Check if the data is suitable for a sigmoid curve (monotonically increasing)
if not np.all(np.diff(y_data) > 0):
    print("Warning: The y-values of your data are not strictly increasing.")
    print("A sigmoid fit may not be appropriate for this data.")

try:
    # Use curve_fit to find the best parameters
    # p0 is an initial guess for the parameters [k, x0]
    # We can estimate x0 as the mean of x_data as a starting point.
    initial_guess = [1, np.mean(x_data)]

    # The 'bounds' argument constrains the parameters. We'll keep k positive.
    popt, pcov = curve_fit(
        sigmoid, x_data, y_data, p0=initial_guess, bounds=(0, np.inf)
    )

    # Extract the optimal parameters
    k_opt, x0_opt = popt
    print("Optimal Parameters (L=1):")
    print(f"k = {k_opt:.4f}")
    print(f"x0 = {x0_opt:.4f}")

    # --- Visualization ---
    # Generate a smooth curve for plotting
    x_curve = np.linspace(0, 300, 200)
    y_curve = sigmoid(x_curve, k_opt, x0_opt)

    # Plot the data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, "o", label="Data Points", markersize=8, color="red")
    plt.plot(
        x_curve,
        y_curve,
        label=f"Fitted Sigmoid Curve\n(k={k_opt:.2f}, x₀={x0_opt:.2f})",
        color="blue",
    )
    plt.plot([0, 300], [0.823, 0.823], label="Ours")
    plt.plot([0, 300], [0.988, 0.988], label="CD (oracle)")
    plt.plot([40, 40], [0, 1.2], "--", label="OOM")

    plt.title("Overdetermined Sigmoid Curve Fit (L=1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)  # Set y-axis limits since L=1
    plt.show()
    plt.savefig("sigmoid_fit.png")

except RuntimeError as e:
    print(f"Error: Could not find optimal parameters. {e}")
    print(
        "This can happen if the data is not sigmoid-shaped or if the initial guess is poor."
    )
