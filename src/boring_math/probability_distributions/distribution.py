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
from typing import Self
from pythonic_fp.fptools.maybe import MayBe
from .datasets import DataSet

__all__ = ['ContDist', 'DiscreteDist']


class ContDist(ABC):
    """Base class to visualize continuous probability distributions."""

    def __init__(self) -> None:
        self.population: MayBe[DataSet] = MayBe()
        self.samples: list[DataSet] = []

        # Numerically integrated CDF
        self._cdf: MayBe[tuple[float, ...]] = MayBe()

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
