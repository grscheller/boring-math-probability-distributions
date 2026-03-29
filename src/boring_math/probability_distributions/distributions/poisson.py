# Copyright 2026 Geoffrey R. Scheller
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This module contains software derived from Udacity® exercises.
# Udacity® (https://www.udacity.com/)
#

"""
Poisson Distribution
--------------------

A Poisson distribution class.

"""

from typing import final, Self
from math import floor, ceil, factorial as fac, sqrt
import matplotlib.pyplot as plt
from boring_math.special_functions.exponential import exp
from ..datasets import DataSet
from ..distribution import DiscreteDist

__all__ = ['Poisson']


@final
class Poisson(DiscreteDist):
    """Class for visualizing Poisson distributed data.

    A Poisson distribution that expresses the probability of a given number of events
    occurring in a fixed interval of time if these events occur with a known constant
    mean rate and independently of the time since the last event.

    Attributes (some inherited):

    - ``mean`` (float) representing the mean value of the distribution
    - ``stdev`` (float) representing the standard deviation of the distribution
    - ``data``  extracted from a data file (taken to be a population)
    - ``λ`` (float) number of events in an interval

    """

    def __init__(self, λ: float = 1.0):
        if (λ <= 0.0):
            msg = 'For a Poisson distribution, λ is assumed positive'
            raise ValueError(msg)

        self.λ: float = λ
        self.mean: float = λ
        self.stdev: float = λ

        super().__init__()

    def pdf(self, kf: float) -> float:
        """Poisson probability distribution function."""
        k = floor(kf)
        λ = self.λ
        if k < 0:
            return 0.0
        return λ**k * exp(-λ) / fac(k)

    def cdf(self, kf: float) -> float:
        """Binomial cumulative probability distribution function."""
        return sum((self.pdf(ii) for ii in range(0, ceil(kf))))

    def calculate_mean(self) -> float:
        """Calculate the mean from λ."""
        self.mean = mean = self.λ
        return mean

    def calculate_stdev(self) -> float:
        """Calculate the standard deviation using ``λ``."""
        self.stdev = stdev = sqrt(self.λ)
        return stdev

    def replace_stats_from_dataset(self, dset: DataSet) -> float:
        """Function to calculate p and n from a data set.

        Where the read in data set is taken as the population.
        """
        if dset:
            self.n = n = dset._size
            if (mean := sum(dset._data)/n) <= 0.0:
                msg = 'Inconsistent dataset, mean <= 0'
                raise ValueError(msg)
            self.mean = self.λ = mean = sum(dset._data) / n
            self.stdev = sqrt(mean)
        return self.λ

    def plot_bar_data(self) -> None:
        """Produce a bar-graph of the data using the matplotlib pyplot library."""
        n = self.n
        p = self.λ

        fig, axis = plt.subplots()
        axis.bar(('0', '1'), (n * (1 - p), n * p), color='maroon', width=0.6)
        axis.set_title('Failures and Successes for a sample of {}'.format(n))
        axis.set_xlabel('prob = {}, n = {}'.format(p, n))
        axis.set_ylabel('Sample Count')
        plt.show()

    def plot_bar_pdf(self) -> tuple[list[int], list[float]]:
        """Function to plot the pdf of the binomial distribution.

        :return:
            A tuple containing

            - list[int]: x values used for the pdf plot
            - list[float]: y values used for the pdf plot

        """

        def pdf(ii: int) -> float:
            return self.pdf(float(ii))

        xs: list[int] = list(range(self.n + 1))
        ys: list[float] = list(map(pdf, range(self.n + 1)))

        plt.bar(list(str(x) for x in xs), ys, color='maroon', width=0.4)
        plt.title('Probability Density of Success')
        plt.xlabel('Successes for {} trials'.format(self.n))
        plt.ylabel('Probability')
        plt.show()

        return xs, ys

    def __add__(self, other: Self) -> Self:
        """Add together two Poisson distributions.

        Poisson distributions are closed but not stable, thus if two independent random
        variables ``X₁`` and ``X₂`` are Poisson distributed then ``X₁ + X₂`` is Poisson
        distributed with ``λ = λ₁ + λ₂``. Unlike the Normal distribution, Poisson
        distributed random variables do not scale. That is ``aX₁ + bX₂`` is not Poisson
        unless both ``a`` and ``b`` are 1, or one is ``1`` and the other is ``0``.


        """
        if type(other) is not Poisson:
            msg = 'A Poisson distribution cannot be added to a {}'
            msg = msg.format(type(other))
            raise TypeError(msg)

        return Poisson(self.λ + other.λ)

    def __repr__(self) -> str:
        repr_str = 'Poisson({})'
        return repr_str.format(self.λ)

    def __str__(self) -> str:
        user_str = 'mean {}, standard deviation {}, λ {}'
        return user_str.format(self.mean, self.stdev, self.λ)
