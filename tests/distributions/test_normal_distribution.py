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

from math import sqrt, pi
from boring_math.probability_distributions.distributions.normal import Normal

tolerance14 = 1e-14
tolerance13 = 1e-14
tolerance05 = 1e-5

class Test_Normal:
    def test_normal01(self) -> None:
        norm = Normal(0, 1)
        assert norm.pdf(0) == 1.0/sqrt(2*pi)
        assert norm.cdf(0) == 0.5

    def test_normal11(self) -> None:
        norm = Normal(1, 1)
        assert norm.pdf(1) == 1.0/sqrt(2*pi)
        assert norm.cdf(1) == 0.5

    def test_normal_arbitrary(self) -> None:
        norm = Normal(3.143, 0.756)
        assert abs(norm.pdf(3) - 0.518345) < tolerance05
        assert abs(norm.cdf(3) - 0.424986) < tolerance05
