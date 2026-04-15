#!/usr/bin/env python

import matplotlib.pyplot as plt

# Generate x values using a plain Python list and range
x = [i * 0.1 for i in range(-100, 101)]  # -10.0 to 10.0 in steps of 0.1

# Define two functions
def f(x: float) -> float:
    return x ** 2 - 4

def g(x: float) -> float:
    return 2 * x + 3

# Compute y values using list comprehensions
y_f = [f(val) for val in x]
y_g = [g(val) for val in x]

# Plot both functions
fig, ax = plt.subplots()

ax.plot(x, y_f, label=r"$f(x) = x^2 - 4$", color="steelblue", linewidth=2)
ax.plot(x, y_g, label=r"$g(x) = 2x + 3$", color="tomato", linewidth=2)

# Formatting
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # x-axis
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")  # y-axis
ax.set_title("Plot of Two Functions")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
