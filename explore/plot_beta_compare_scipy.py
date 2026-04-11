import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from boring_math.probability_distributions.distributions.beta import Beta

a = 0.25
b = 5

beta_bm = Beta(a, b)

# Setup SciPy datastructures
scipy_xs = np.linspace(0, 1, 1001)
scipy_ys = beta.cdf(scipy_xs, a, b)

scipy_scatter_xs = np.array((0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0))
scipy_scatter_ys = beta.cdf(scipy_scatter_xs, a, b)

# Setup Boring Math datastructures
steps = 1000
xs = [n/steps for n in range(steps+1)]
ys = [beta_bm.cdf(x) for x in xs]

scatter_xs = [0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0]
scatter_ys = [beta_bm.cdf(x) for x in scatter_xs]

# Plot Boring Math and SciPy versions together
fig, ax = plt.subplots()

ax.plot(scipy_xs, scipy_ys, label=f'SciPy Beta({a}, {b})', color='tomato', linewidth=2)
ax.plot(xs, ys, label=f'Beta({a}, {b})', color='steelblue', linewidth=2)

# Formatting

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # x-axis
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")  # y-axis
ax.set_title("Comparison of BM vs SciPy")
ax.set_xlabel("x")
ax.set_ylabel("cdf")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

# Show plot

plt.tight_layout()
plt.show()
