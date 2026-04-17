# Copyright 2024 Geoffrey R. Scheller
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

"""Module boring_math.probability_distributions.distribution

Providing base classes to visualize probability distributions.

- *class* ContDist: base class to visualize continuous pd
- *class* DiscreteDist: base class to visualize discrete pd

"""

from abc import ABC, abstractmethod
from math import isinf, floor
from typing import Self
from pythonic_fp.fptools.maybe import MayBe
from pythonic_fp.iterables.folding import accumulate, fold_left
from .datasets import DataSet

__all__ = ['ContDist', 'DiscreteDist']

tolerance14 = 1.0E-14
tolerance13 = 1.0E-13
tolerance12 = 1.0E-12
tolerance11 = 1.0E-11
tolerance10 = 1.0E-10
tolerance09 = 1.0E-9
tolerance08 = 1.0E-8
tolerance07 = 1.0E-7
tolerance06 = 1.0E-6
tolerance05 = 1.0E-5
tolerance04 = 1.0E-4
tolerance03 = 1.0E-3
tolerance02 = 1.0E-2

class ContDist(ABC):
    """Base class to visualize continuous probability distributions."""

    def __init__(self) -> None:
        self.population: MayBe[DataSet] = MayBe()
        self.samples: list[DataSet] = []

        # Numerically integrated CDF
        self._numerical_cdf_data: MayBe[tuple[float, ...]] = MayBe()
        self._numerical_cdf_steps: MayBe[int] = MayBe()

    @abstractmethod
    def pdf(self, kf: float) -> float:
        """Probability distribution function."""
        ...

    @abstractmethod
    def cdf(self, kf: float) -> float:
        """Cumulative distribution function."""
        ...

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """Add together two compatible distributions."""
        ...

    def _compute_cdf_jump(
        self,
        start: float,
        delta: float,
        /,
        steps: int = 32,
    ) -> float:
        return 42.0

    def _compute_numerical_cdf(self,
        start: float,
        stop: float,
        /,
        steps: int = 2048,
    ) -> None:
        self._numerical_cdf_steps = MayBe(steps)
        delta = (stop-start)/steps
        initial_cdf = 0.0
        initial_step = 0

        if isinf(self.pdf(start)):
            dt = delta/2.0
            while self.pdf(start + dt)*dt > tolerance03:
                dt /= 2

            initial_steps = floor(delta/dt)

            initial_cdf = fold_left(
                ((self.pdf(start + (n-1)*dt) + self.pdf(start + n*dt))*dt/2.0 for n in range (2, initial_steps+1)),
                lambda u, v: u + v,
                self.pdf(start+dt/2)*dt
            )

            initial_step = 2

        self._numerical_cdf_data = MayBe(tuple(accumulate(
            ((self.pdf(start + (n-1)*delta) + self.pdf(start + n*delta))*delta/2.0 for n in range(initial_step, steps+1)),
            lambda u, v: u + v,
            initial_cdf,
        )))


class DiscreteDist(ABC):
    """Base class to visualize discrete probability distributions."""

    def __init__(self) -> None:
        self.population: MayBe[DataSet] = MayBe()
        self.samples: list[DataSet] = []

    @abstractmethod
    def pdf(self, kf: float) -> float: ...

    @abstractmethod
    def cdf(self, kf: float) -> float: ...

    @abstractmethod
    def __add__(self, other: Self) -> Self: ...

    @abstractmethod
    def plot_pdf_bar_graph(
        self,
        /,
        test: bool = False,
        lowercap: int | None = None,
        uppercap: int | None = None,
    ) -> tuple[list[int], list[float]]: ...
