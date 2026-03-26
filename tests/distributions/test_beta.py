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

from boring_math.probability_distributions.distributions.beta import Beta

tolerance14 = 1e-14
tolerance13 = 1e-14

class Test_Beta:
    def test_beta_2_2(self) -> None:
        beta = Beta(2, 2)
        assert beta.pdf(-0.5) == beta.pdf(1.5) == 0.0
        assert abs(beta.pdf(0.00) - 0) < tolerance14
        assert abs(beta.pdf(0.25) - 9/8) < tolerance14
        assert abs(beta.pdf(0.50) - 3/2) < tolerance14
        assert abs(beta.pdf(0.75) - 9/8) < tolerance14
        assert abs(beta.pdf(1.00) - 0) < tolerance14

    def test_beta_1_3(self) -> None:
        beta = Beta(1, 3)
        assert beta.pdf(-0.5) == beta.pdf(1.5) == 0.0
        assert abs(beta.pdf(0.00) - 3.0000) < tolerance14
        assert abs(beta.pdf(0.25) - 1.6875) < tolerance14
        assert abs(beta.pdf(0.50) - 0.7500) < tolerance14
        assert abs(beta.pdf(0.75) - 0.1875) < tolerance14
        assert abs(beta.pdf(1.00) - 0.0000) < tolerance14

    def test_beta_3_1(self) -> None:
        beta = Beta(3, 1)
        assert beta.pdf(-0.5) == beta.pdf(1.5) == 0.0
        assert abs(beta.pdf(0.00) - 0.0000) < tolerance14
        assert abs(beta.pdf(0.25) - 0.1875) < tolerance14
        assert abs(beta.pdf(0.50) - 0.7500) < tolerance14
        assert abs(beta.pdf(0.75) - 1.6875) < tolerance14
        assert abs(beta.pdf(1.00) - 3.0000) < tolerance14

    def test_beta_1_1(self) -> None:
        beta = Beta(1, 1)
        assert beta.pdf(-0.5) == beta.pdf(1.5) == 0.0
        assert abs(beta.pdf(0.00) - 1.0) < tolerance14
        assert abs(beta.pdf(0.25) - 1.0) < tolerance14
        assert abs(beta.pdf(0.50) - 1.0) < tolerance14
        assert abs(beta.pdf(0.75) - 1.0) < tolerance14
        assert abs(beta.pdf(1.00) - 1.0) < tolerance14
