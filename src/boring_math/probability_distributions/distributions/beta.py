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

from math import floor, inf
from typing import final, Self

# from ..datasets import DataSet
from boring_math.special_functions.gamma_family.beta import beta_real
from ..distribution import ContDist

__all__ = ['Beta']


@final
class Beta(ContDist):
    """
    .. admonition:: Class for visualizing Beta distributions.

        The Beta distribution is a continuous probability distribution
        defined on the sample space ``[0, 1]`` with probability density
        function

        ``f(x) = xᵅ⁻¹(1-x)ᵝ⁻¹/B(α,β)`` for α,β > 0

        where

        - ``μ = α/(α+β)``
        - ``σ² =  αβ/((α+β)²(α+β+1))``
        - ``mode = (α-1)/(α+β-2)`` for ``α, β > 1``
        - ``B(α,β) = Γ(α)Γ(β)/Γ(α+β)`` is the normalization factor.

        .. note::

            There is no simple closed form formula the CDF valid
            for all ``α, β > 0``. To provide a CDF the PDF is
            numerically integrated by the parent class.

            .. admonition:: TODO

                Implement a CDF using using the Incomplete Beta Function
                once one is implemented in boring-math-special-functions.

    """

    def __init__(self, α: float, β: float):
        super().__init__()

        if α <= 0 or β <= 0:
            msg = 'For a Beta distribution, α, β > 0'
            raise ValueError(msg)

        self.α = α
        self.β = β
        self.μ = α / (α+β)
        self.σ = α*β / ((α+β)**2 * (α+β+1))

    def pdf(self, x: float) -> float:
        """
        .. admonition:: Beta PDF

            Beta probability distribution function.

        :param x: ``x ∈ [0, 1]``
        :returns: Value of the PDF at ``x``, ``0,0`` if outside domain.

        """
        if x < 0 or x > 1:
            return 0.0

        α = self.α
        β = self.β
        try:
            val: float = x**(α-1) * (1-x)**(β-1) / beta_real(α, β)
        except ZeroDivisionError:
            if x == 0 or x == 1:
                return inf
            raise ZeroDivisionError
        else:
            return val

    def cdf(self, x: float) -> float:
        """
        .. admonition:: Beta CDF

            Beta cumulative probability distribution function defined
            on the probability sample space ``[0, 1]``.

            .. note::

                For all ``α, β > 0`` there is no single closed form for
                a beta distribution's CDF. To provide a CDf, the PDF is
                numerically integrated.

        :param x: Where ``x`` is an element of the sample space.
        :returns: CDF at ``x`` obtained by numerically integrated the PDF.

        """
        steps = 2048

        if not self._numerical_cdf_data:
            self._compute_numerical_cdf(0.0, 1.0, steps = steps)

        if x <= 0:
            return 0.0
        elif x >= 1.0:
            return 1.0

        return self._numerical_cdf_data.get()[floor(x * self._numerical_cdf_steps.get())]

    def __add__(self, other: Self) -> Self:
        """
        .. admonition:: Fail if two Beta distributions are added.

            Beta distributions are not stable, thus the sum of two
            random beta distributed variables is not Beta distributed.

        :param other: Another Beta distribution class instance.
        :returns: Never returns, Beta distributions are not stable.
        :raises ValueError: If Beta distributions are added.
        :raises TypeError: If a Beta distributions is added to another
                           type of probability distribution class.

        """
        if type(other) is Beta:
            msg1 = 'The Beta distribution is not a stable distribution,\n'
            msg2 = 'the sum of two is not a Beta distribution.'
            msg = (msg1 + msg2).format()
            raise ValueError(msg)
        msg1 = 'A Beta distribution added to a {} distribution\n'
        msg2 = 'is not a Beta distribution.'
        msg = msg.format(type(other))
        raise TypeError(msg)

    def __repr__(self) -> str:
        """
        :returns: The string ``Beta(α, β)`` where ``α, β > 0``.

        """
        repr_str = 'Beta({}, {})'
        return repr_str.format(self.α, self.β)

    def __str__(self) -> str:
        """
        :returns: The string ``Beta(a=α, b=β)`` where ``α, β > 0``.

        """
        repr_str = 'Beta(a={}, b={})'
        return repr_str.format(self.α, self.β)

    def plot_pdf_bar_graph(
        self,
        /,
        show: bool = True,
        lower_cap: int | None = None,
        upper_cap: int | None = None,
    ) -> tuple[list[int], list[float]]:
        raise NotImplementedError
