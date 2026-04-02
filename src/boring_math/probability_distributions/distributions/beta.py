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

from typing import final, Self
from boring_math.special_functions.gamma_family.beta import beta_real
# import matplotlib.pyplot as plt
# from ..datasets import DataSet
from ..distribution import ContDist

__all__ = ['Beta']


@final
class Beta(ContDist):
    """Class for visualizing Beta distributions.

    .. note::

        The Beta distribution is a continuous probability distribution
        defined on ``(0, 1)`` with probability density function

        ``f(x) = xᵅ⁻¹(1-x)ᵝ⁻¹/B(α,β)`` for α,β > 0

        where

        ``μ = α/(α+β)``

        ``σ² =  αβ/((α+β)²(α+β+1))``

        ``mode = (α-1)/(α+β-2)`` for ``α, β > 1``

        and ``B(α,β) = Γ(α)Γ(β)/Γ(α+β)`` is the normalization factor.

    """

    def __init__(self, α: float, β: float):
        if α <= 0 or β <= 0:
            msg = 'For a Beta distribution, α, β > 0'
            raise ValueError(msg)

        self.α = α
        self.β = β
        self.mu = α/(α+β)
        self.sigma = α*β/((α+β)**2 * (α+β+1))

        super().__init__()

    def __repr__(self) -> str:
        repr_str = 'Beta({}, {})'
        return repr_str.format(self.α, self.β)

    def pdf(self, x: float) -> float:
        """Beta probability distribution function."""
        if x < 0 or x > 1:
            return 0.0

        α = self.α
        β = self.β
        val: float = x**(α-1) * (1-x)**(β-1) / beta_real(α, β)
        return val

    def cdf(self, x: float) -> float:
        """Beta cumulative probability distribution function."""
        raise NotImplementedError("This function not yet implemented")

    def __add__(self, other: Self) -> Self:
        """Fail if two Beta distributions are added.

        Beta distributions are not stable, thus the sum of two random
        beta distributed variables is not Beta distributed.

        """
        if type(other) is Beta:
            msg1 = 'The Beta distribution is not a stable distribution,\n'
            msg2 = 'the sum of two is not a Beta distribution.'
            msg = (msg1 + msg2).format()
            raise ValueError(msg)
        else:
            msg = '\n'
            msg1 = 'The Beta distribution is not a stable distribution,\n'
            msg2 = 'two added together do not procuce another Beta distribution.'
            msg = msg.format(type(other))
            raise TypeError(msg)
