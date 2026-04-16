#!/usr/bin/env python

import matplotlib.pyplot as plt
from boring_math.probability_distributions.distributions.beta import Beta

a = 1/3
b = 1/3

beta = Beta(a, b)

steps = 1000
xs = [n/steps for n in range(steps+1)]
ys = [beta.cdf(x) for x in xs]

scatter_xs = [0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0]
scatter_ys = [beta.cdf(x) for x in scatter_xs]

plt.figure(figsize=(8,5))
plt.plot(xs, ys, lw=2)
plt.scatter(scatter_xs, scatter_ys, color='red', zorder=5)
for n in range(len(scatter_xs)):
    x, y = scatter_xs[n], scatter_ys[n]
    plt.annotate(
        f'{scatter_ys[n]:.4f}',
        (x, y),
        textcoords="offset points",
        xytext=(5,-10),
    )

plt.title('CDF of Beta({{a}, {b})')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.ylim(-0.02, 1.02)
plt.grid(alpha=0.3)
plt.show()

for val in ys:
    print(val)
