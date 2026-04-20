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

from math import pi, sqrt
from cmath import isinf
from boring_math.probability_distributions.distributions.beta import Beta

tolerance14 = 1e-14
tolerance13 = 1e-13
tolerance12 = 1e-12
tolerance11 = 1e-11
tolerance10 = 1e-10
tolerance09 = 1e-9
tolerance08 = 1e-8
tolerance07 = 1e-7
tolerance06 = 1e-6
tolerance05 = 1e-5
tolerance04 = 1e-4
tolerance03 = 1e-3
tolerance02 = 1e-2

class Test_Beta:
    def test_beta_2_2(self) -> None:
        beta = Beta(2, 2)
        assert beta.pdf(-0.5) == 0.0
        assert beta.pdf(0.0) == 0.0
        assert abs(beta.pdf(1/5) - 24/25) < tolerance14
        assert abs(beta.pdf(1/4) - 9/8) < tolerance14
        assert abs(beta.pdf(1/3) - 4/3) < tolerance14
        assert abs(beta.pdf(1/2) - 3/2) < tolerance14
        assert abs(beta.pdf(2/3) - 4/3) < tolerance14
        assert abs(beta.pdf(3/4) - 9/8) < tolerance14
        assert abs(beta.pdf(4/5) - 24/25) < tolerance14
        assert abs(beta.pdf(1.0) - 0.0) < tolerance14
        assert beta.pdf(1.0) == 0.0
        assert beta.pdf(1.5) == 0.0

        assert beta.cdf(-0.5) == 0.0
        assert beta.cdf(0.0) == 0.0
        assert abs(beta.cdf(0.0) - 0.0) < tolerance14
        assert abs(beta.cdf(1/5) - 13/125) < tolerance03
        assert abs(beta.cdf(1/4) - 5/32) < tolerance03
        assert abs(beta.cdf(1/3) - 7/27) < tolerance02
        assert abs(beta.cdf(1/2) - 1/2) < tolerance03
        assert abs(beta.cdf(2/3) - 20/27) < tolerance03
        assert abs(beta.cdf(3/4) - 27/32) < tolerance03
        assert abs(beta.cdf(4/5) - 112/125) < tolerance03
        assert abs(beta.cdf(1.0) - 1.0) < tolerance14
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(1.5) == 1.0

    def test_beta_half_half(self) -> None:
        beta = Beta(1/2, 1/2)
        assert beta.pdf(-42.0) == 0.0
        assert isinf(beta.pdf(0.0))
        assert abs(beta.pdf(1/4) - 4.0/(pi*sqrt(3))) < tolerance14
        assert abs(beta.pdf(1/3) - 3.0/(pi*sqrt(2))) < tolerance14
        assert abs(beta.pdf(1/2) - 2/pi) < tolerance14
        assert abs(beta.pdf(2/3) - 3.0/(pi*sqrt(2))) < tolerance14
        assert abs(beta.pdf(3/4) - 4.0/(pi*sqrt(3))) < tolerance14
        assert isinf(beta.pdf(1.0))
        assert beta.pdf(42.0) == 0.0

        assert beta.cdf(-42.0) == 0.0
        assert beta.cdf(0.0) == 0.0
        assert abs(beta.cdf(1/4) - 1/3) < tolerance03
        assert abs(beta.cdf(1/3) - 0.39182655203060727) < tolerance03
        assert abs(beta.cdf(1/2) - 1/2) < tolerance03
        assert abs(beta.cdf(2/3) - 0.60817344796939273) < tolerance03
        assert abs(beta.cdf(3/4) - 2/3) < tolerance03
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(42.0) == 1.0

    def test_beta_third_third(self) -> None:
        beta = Beta(1/3, 1/3)
        assert beta.pdf(-0.001) == 0.0
        assert isinf(beta.pdf(0.0))
        assert abs(beta.pdf(1/4) - 0.575966) < tolerance07
        assert abs(beta.pdf(1/3) - 0.51428754) < tolerance08
        assert abs(beta.pdf(1/2) - 0.475449418541) < tolerance12
        assert abs(beta.pdf(2/3) - 0.51428754) < tolerance08
        assert abs(beta.pdf(3/4) - 0.575966) < tolerance07
        assert isinf(beta.pdf(1.0))
        assert beta.pdf(1.001) == 0.0

        assert beta.cdf(-0.001) == 0.0
        assert beta.cdf(0.0) == 0.0
        assert abs(beta.cdf(0.0) - 0.0) < tolerance14
        assert abs(beta.cdf(1/4) - 0.3735487913342305) < tolerance03
        assert abs(beta.cdf(1/3) - 0.418684817) < tolerance03
        assert abs(beta.cdf(1/2) - 1/2) < tolerance03
        assert abs(beta.cdf(2/3) - 0.581315183) < tolerance03
        assert abs(beta.cdf(3/4) - 0.626451208) < tolerance03
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(1.001) == 1.0

    def test_beta_1_3(self) -> None:
        beta = Beta(1, 3)
        assert beta.pdf(-0.005) == 0.0
        assert abs(beta.pdf(0.0) - 3.0) < tolerance14
        assert abs(beta.pdf(1/5) - 48/25) < tolerance14
        assert abs(beta.pdf(1/4) - 27/16) < tolerance14
        assert abs(beta.pdf(1/3) - 4/3) < tolerance14
        assert abs(beta.pdf(1/2) - 3/4) < tolerance14
        assert abs(beta.pdf(2/3) - 1/3) < tolerance14
        assert abs(beta.pdf(3/4) - 3/16) < tolerance14
        assert abs(beta.pdf(4/5) - 3/25) < tolerance14
        assert beta.pdf(1.0) == 0.0
        assert beta.pdf(1.005) == 0.0

        assert beta.cdf(-0.005) == 0.0
        assert beta.cdf(0.0) == 0.0
        assert abs(beta.cdf(1/5) - 61/125) < tolerance03
        assert abs(beta.cdf(1/4) - 37/64) < tolerance03
        assert abs(beta.cdf(1/3) - 19/27) < tolerance03
        assert abs(beta.cdf(1/2) - 7/8) < tolerance03
        assert abs(beta.cdf(2/3) - 26/27) < tolerance03
        assert abs(beta.cdf(3/4) - 63/64) < tolerance03
        assert abs(beta.cdf(4/5) - 124/125) < tolerance03
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(1.005) == 1.0

    def test_beta_3_1(self) -> None:
        beta = Beta(3, 1)
        assert beta.pdf(-0.9) == 0.0
        assert abs(beta.pdf(0.0) - 0.0000) < tolerance14
        assert abs(beta.pdf(1/5) - 3/25) < tolerance14
        assert abs(beta.pdf(1/4) - 3/16) < tolerance14
        assert abs(beta.pdf(1/3) - 1/3) < tolerance14
        assert abs(beta.pdf(1/2) - 3/4) < tolerance14
        assert abs(beta.pdf(2/3) - 4/3) < tolerance14
        assert abs(beta.pdf(3/4) - 27/16) < tolerance14
        assert abs(beta.pdf(4/5) - 48/25) < tolerance14
        assert abs(beta.pdf(1.0) - 3.0) < tolerance14
        assert beta.pdf(1.9) == 0.0

        assert beta.cdf(-0.9) == 0.0
        assert abs(beta.cdf(0.0) - 0.0) < tolerance14
        assert beta.cdf(0.0) == 0.0
        assert abs(beta.cdf(1/5) - 1/125) < tolerance03
        assert abs(beta.cdf(1/4) - 1/64) < tolerance03
        assert abs(beta.cdf(1/3) - 1/27) < tolerance03
        assert abs(beta.cdf(1/2) - 1/8) < tolerance03
        assert abs(beta.cdf(2/3) - 8/27) < tolerance03
        assert abs(beta.cdf(3/4) - 27/64) < tolerance03
        assert abs(beta.cdf(4/5) - 64/125) < tolerance02
        assert abs(beta.cdf(1.0) - 1.00) < tolerance03
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(1.9) == 1.0

    def test_beta_1_1(self) -> None:
        beta = Beta(1, 1)
        assert beta.pdf(-0.4) == 0.0
        assert abs(beta.pdf(0.0) - 1.0) < tolerance14
        assert abs(beta.pdf(1/5) - 1.0) < tolerance14
        assert abs(beta.pdf(1/4) - 1.0) < tolerance14
        assert abs(beta.pdf(1/3) - 1.0) < tolerance14
        assert abs(beta.pdf(1/2) - 1.0) < tolerance14
        assert abs(beta.pdf(2/3) - 1.0) < tolerance14
        assert abs(beta.pdf(3/4) - 1.0) < tolerance14
        assert abs(beta.pdf(4/5) - 1.0) < tolerance14
        assert abs(beta.pdf(1.0) - 1.0) < tolerance14
        assert beta.pdf(1.4) == 0.0

        assert beta.cdf(-0.4) == 0.0
        assert abs(beta.cdf(0.0) - 0.0) < tolerance14
        assert abs(beta.cdf(1/5) - 1/5) < tolerance03
        assert abs(beta.cdf(1/4) - 1/4) < tolerance03
        assert abs(beta.cdf(1/3) - 1/3) < tolerance03
        assert abs(beta.cdf(1/2) - 1/2) < tolerance03
        assert abs(beta.cdf(2/3) - 2/3) < tolerance03
        assert abs(beta.cdf(3/4) - 3/4) < tolerance03
        assert abs(beta.cdf(4/5) - 4/5) < tolerance03
        assert abs(beta.cdf(1.0) - 1.0) < tolerance14
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(1.4) == 1.0

    def test_beta_2_3(self) -> None:
        beta = Beta(2, 3)
        assert beta.pdf(-1.5) == 0.0
        assert abs(beta.pdf(0.0) - 0.0) < tolerance14
        assert abs(beta.pdf(1/5) - 192/125) < tolerance14
        assert abs(beta.pdf(1/4) - 27/16) < tolerance14
        assert abs(beta.pdf(1/3) - 16/9) < tolerance14
        assert abs(beta.pdf(1/2) - 1.5) < tolerance14
        assert abs(beta.pdf(2/3) - 8/9) < tolerance14
        assert abs(beta.pdf(3/4) - 9/16) < tolerance14
        assert abs(beta.pdf(4/5) - 48/125) < tolerance14
        assert abs(beta.pdf(1.0) - 0.0) < tolerance14
        assert beta.pdf(2.5) == 0.0

        assert beta.cdf(-1.5) == 0.0
        assert abs(beta.cdf(0.0) - 0.00) < tolerance14
        assert beta.cdf(0.0) == 0.00
        assert abs(beta.cdf(1/5) - 113/625) < tolerance02
        assert abs(beta.cdf(1/4) - 67/256) < tolerance03
        assert abs(beta.cdf(1/3) - 11/27) < tolerance02
        assert abs(beta.cdf(1/2) - 11/16) < tolerance03
        assert abs(beta.cdf(2/3) - 8/9) < tolerance03
        assert abs(beta.cdf(3/4) - 243/256) < tolerance03
        assert abs(beta.cdf(4/5) - 608/625) < tolerance03
        assert abs(beta.cdf(1.0) - 1.00) < tolerance14
        assert beta.cdf(1.0) == 1.0
        assert beta.cdf(2.5) == 1.0

class Test_Beta_CDF_Alpha_lt_1:
    def test_beta_cdf_beta_2(self) -> None:
        beta_11 = Beta(1.1, 2)
        assert abs(beta_11.cdf(0.134) - 0.2140072878589) < tolerance02
        assert abs(beta_11.cdf(0.527) - 0.7514861350347) < tolerance03
        assert abs(beta_11.cdf(0.954) - 0.9975636340196) < tolerance03 

        beta_10 = Beta(1.0, 2)
        assert abs(beta_10.cdf(0.134) - 0.250044) < tolerance03
        assert abs(beta_10.cdf(0.527) - 0.776271) < tolerance03
        assert abs(beta_10.cdf(0.954) - 0.997884) < tolerance03 

        beta_09 = Beta(0.9, 2)
        assert abs(beta_09.cdf(0.134) - 0.2915196975805) < tolerance03
        assert abs(beta_09.cdf(0.527) - 0.8010464558987) < tolerance03
        assert abs(beta_09.cdf(0.954) - 0.9981851637766) < tolerance03 

        beta_08 = Beta(0.8, 2)
        assert abs(beta_08.cdf(0.134) - 0.3390700958548) < tolerance03
        assert abs(beta_08.cdf(0.527) - 0.8257027986635) < tolerance03
        assert abs(beta_08.cdf(0.954) - 0.9984669369324) < tolerance03 

        beta_07 = Beta(0.7, 2)
        assert abs(beta_07.cdf(0.134) - 0.3933444080413) < tolerance03
        assert abs(beta_07.cdf(0.527) - 0.8501158512153) < tolerance03
        assert abs(beta_07.cdf(0.954) - 0.9987291296993) < tolerance03 

        beta_06 = Beta(0.6, 2)
        assert abs(beta_06.cdf(0.134) - 0.4549798507456) < tolerance03
        assert abs(beta_06.cdf(0.527) - 0.8741455287076) < tolerance03
        assert abs(beta_06.cdf(0.954) - 0.9989715509499) < tolerance03 

        beta_05 = Beta(0.5, 2)
        assert abs(beta_05.cdf(0.134) - 0.5245641295399) < tolerance03
        assert abs(beta_05.cdf(0.527) - 0.8976342772811) < tolerance03
        assert abs(beta_05.cdf(0.954) - 0.9991940081886) < tolerance03 

        beta_04 = Beta(0.4, 2)
        assert abs(beta_04.cdf(0.134) - 0.6025818165590) < tolerance03
        assert abs(beta_04.cdf(0.527) - 0.9204053758755) < tolerance03
        assert abs(beta_04.cdf(0.954) - 0.9993963075432) < tolerance03 

    #   Too slow.
    #   beta_03 = Beta(0.3, 2)
    #   assert abs(beta_03.cdf(0.134) - 0.6893393074791) < tolerance02
    #   assert abs(beta_03.cdf(0.527) - 0.9422610898089) < tolerance03
    #   assert abs(beta_03.cdf(0.954) - 0.9995782537559) < tolerance03 

    #   beta_02 = Beta(0.2, 2)
    #   assert abs(beta_02.cdf(0.134) - 0.7848614781823) < tolerance03
    #   assert abs(beta_02.cdf(0.527) - 0.9629806639472) < tolerance03
    #   assert abs(beta_02.cdf(0.954) - 0.9997396501749) < tolerance03 

    #   beta_01 = Beta(0.1, 2)
    #   assert abs(beta_01.cdf(0.134) - 0.8887511599056) < tolerance03
    #   assert abs(beta_01.cdf(0.527) - 0.9823181423195) < tolerance03
    #   assert abs(beta_01.cdf(0.954) - 0.9998802987452) < tolerance03 


# Note 1:
#
#   Using WolframAlpha Functions: https://functions.wolfram.com/
#   as my source of truth for above test values.
#
#     for Beta(2, 3).pdf(1/3) type: pdf(1/3) for Beta(2, 3) distribution
#     for Beta(2, 3).cdf(1/3) type: Beta(1/3, 2, 3)/Beta(2, 3)
#
#   where Beta(1/3, 2, 3) is the incomplete beta function.
#
# Note 2:
#
#   As of 2026-04-18, I am numerically integrating the Beta PDF
#   to get the CDF. Since the incomplete beta function is just the
#   unnormalized CDF, I will not have to do this after I implement
#   the incomplete beta function in boring-math-special-functions.
