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

from boring_math.probability_distributions.distributions.poisson import Poisson

tolerance06 = 1.0e-6
tolerance05 = 1.0e-5

class Test_Poisson:
    def test_poisson(self) -> None:
        pois0_5 = Poisson(0.5)

        assert abs(pois0_5.pdf(0) - 0.606531) < tolerance06
        assert abs(pois0_5.pdf(1) - 0.303265) < tolerance06
        assert abs(pois0_5.pdf(2) - 0.075816) < tolerance06
        assert abs(pois0_5.pdf(3) - 0.012636) < tolerance06
        assert abs(pois0_5.pdf(4) - 0.001580) < tolerance06

        assert abs(pois0_5.cdf(0) - 0.606531) < tolerance06
        assert abs(pois0_5.cdf(1) - 0.909796) < tolerance06
        assert abs(pois0_5.cdf(2) - 0.985612) < tolerance06
        assert abs(pois0_5.cdf(3) - 0.998248) < tolerance06
        assert abs(pois0_5.cdf(4) - 0.999828) < tolerance06

        pois0_1 = Poisson(0.1)

        assert abs(pois0_1.pdf(0) - 0.904837) < tolerance06
        assert abs(pois0_1.pdf(1) - 0.090484) < tolerance06
        assert abs(pois0_1.pdf(2) - 0.004524) < tolerance06
        assert abs(pois0_1.pdf(3) - 0.000151) < tolerance06

        assert abs(pois0_1.cdf(0) - 0.904837) < tolerance06
        assert abs(pois0_1.cdf(1) - 0.995321) < tolerance06
        assert abs(pois0_1.cdf(2) - 0.999845) < tolerance06
        assert abs(pois0_1.cdf(3) - 0.999996) < tolerance06

        pois2_0 = Poisson(2.0)
        assert abs(pois2_0.pdf(0) - 0.135335) < tolerance06
        assert abs(pois2_0.pdf(1) - 0.270671) < tolerance06
        assert abs(pois2_0.pdf(2) - 0.270671) < tolerance06
        assert abs(pois2_0.pdf(3) - 0.180447) < tolerance06
        assert abs(pois2_0.pdf(4) - 0.090224) < tolerance06
        assert abs(pois2_0.pdf(5) - 0.036089) < tolerance06
        assert abs(pois2_0.pdf(6) - 0.012030) < tolerance06

        assert abs(pois2_0.cdf(0) - 0.135335) < tolerance06
        assert abs(pois2_0.cdf(1) - 0.406006) < tolerance06
        assert abs(pois2_0.cdf(2) - 0.676676) < tolerance06
        assert abs(pois2_0.cdf(3) - 0.857123) < tolerance06
        assert abs(pois2_0.cdf(4) - 0.947347) < tolerance06
        assert abs(pois2_0.cdf(5) - 0.983436) < tolerance06
        assert abs(pois2_0.cdf(6) - 0.995466) < tolerance06
