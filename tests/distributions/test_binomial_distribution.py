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

from boring_math.probability_distributions.distributions.binomial import Binomial

tolerance06 = 1.0e-6
tolerance05 = 1.0e-5

class Test_Binomial:
    def test_binomial0_5(self) -> None:
        bin3 = Binomial(0.5, 3)

        assert bin3.pdf(0) == 1/8
        assert bin3.pdf(1) == 3/8
        assert bin3.pdf(2) == 3/8
        assert bin3.pdf(3) == 1/8
        assert bin3.pdf(4) == 0.0

        assert bin3.cdf(0) == 1/8
        assert bin3.cdf(1) == 4/8
        assert bin3.cdf(2) == 7/8
        assert bin3.cdf(3) == 1.0
        assert bin3.cdf(4) == 1.0

        bin2 = Binomial(0.5, 2)

        assert bin2.pdf(0) == 1/4
        assert bin2.pdf(1) == 1/2
        assert bin2.pdf(2) == 1/4
        assert bin2.pdf(3) == 0.0

        assert bin2.cdf(0) == 1/4
        assert bin2.cdf(1) == 3/4
        assert bin2.cdf(2) == 1.0
        assert bin2.cdf(3) == 1.0

        bin1 = Binomial(0.5, 1)

        assert bin1.pdf(0) == 1/2
        assert bin1.pdf(1) == 1/2
        assert bin1.pdf(2) == 0.0

        assert bin1.cdf(0) == 1/2
        assert bin1.cdf(1) == 1.0
        assert bin1.cdf(2) == 1.0

        bin0 = Binomial(0.5, 0)
        assert bin0.pdf(0) == 1.0
        assert bin0.pdf(1) == 0.0
        assert bin0.pdf(2) == 0.0

        assert bin0.cdf(0) == 1.0
        assert bin0.cdf(1) == 1.0
        assert bin0.cdf(2) == 1.0


    def test_binomial0_75(self) -> None:
        bin3 = Binomial(0.75, 3)

        assert bin3.pdf(0) == 1/64
        assert bin3.pdf(1) == 9/64
        assert bin3.pdf(2) == 27/64
        assert bin3.pdf(3) == 27/64
        assert bin3.pdf(4) == 0.0

        assert bin3.cdf(0) == 1/64
        assert bin3.cdf(1) == 5/32
        assert bin3.cdf(2) == 37/64
        assert bin3.cdf(3) == 1.0
        assert bin3.cdf(4) == 1.0


class Test_Binomial_Error:
    def test_binomial_error(self) -> None:
        bin_error = Binomial(0.5, 20)
        assert abs(bin_error.pdf(10) - 0.176197) < tolerance06

        try:
            bin_error = Binomial(0.5, -3)
        except ValueError as e:
            assert str(e) == 'Binomial: For a Binomial distribution, the number of trials n must be non-negative.'
        else:
            assert False

        try:
            bin_error = Binomial(-0.5, 30)
        except ValueError as e:
            emsg = 'Binomial: For a Binomial distribution, 0 <= p <= 1.'
            assert str(e) == emsg
        else:
            assert False

        try:
            bin_error = Binomial(1.5, 30)
        except ValueError as e:
            emsg = 'Binomial: For a Binomial distribution, 0 <= p <= 1.'
            assert str(e) == emsg
        else:
            assert False

        try:
            bin_error = Binomial(1.5, -30)
        except ValueError as e:
            emsg1 = 'Binomial: For a Binomial distribution, '
            emsg2 = '0 <= p <= 1'
            emsg3 = 'the number of trials n must be non-negative'
            emsg = emsg1 + emsg2 + ', and ' + emsg3 + '.'
            assert str(e) == emsg
        else:
            assert False
