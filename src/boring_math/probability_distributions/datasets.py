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

"""
Managing Data
-------------

.. admonition:: Managing sample/population data

    - Class **DataSet**

      - contains a single sample or population data

    - Class **DataSets**

      - base class for managing multiple data sets

"""

import math
import sys
from collections.abc import Iterator
from typing import final, Never, Self
from pythonic_fp.fptools.maybe import MayBe

__all__ = ['DataSet', 'DataSets']


@final
class DataSet:
    """
    .. admonition:: Class containing a finite set of data

        Data can be a sample or the population.

        - all data internally stored as floats (even integer data)
        - data sorted smallest to largest
        - methods provided to

            - read in data from a file
            - computing data statistics
            - add or remove data

    """

    @classmethod
    def read_data_from_file(cls, file_name: str, sample: bool = False) -> Self:
        """
        .. admonition:: Create data set from data file

            Read in data from a text file, calculate some statistics,
            and return a DataSet object. Fail fast if there is a problem
            with the data file.

            The text file should

            - have one number (float) per line

                - if sample is true, calculate sample stats
                - if sample is false (default), calculate population stats

            - blank lines and lines beginning with '#' are ignored

        :param file_name: Path to file from which to read in data.
        :param sample: If true treat data as a sample.
        :returns: A ``DataSet`` object containing the data from the file.

        """
        data: list[float] = []
        try:
            with open(file_name) as file:
                line = file.readline()
                while line:
                    if line[0] != '#' and line[0] != chr(ord('\n')):
                        data.append(float(line))
                        line = file.readline()
                    else:
                        line = file.readline()
        except FileNotFoundError:
            sys.exit(f'Error: Cannot find data file "{file_name}"')
        except PermissionError:
            sys.exit(f'Error: No read permissiona for data file "{file_name}"')
        except ValueError:
            sys.exit(f'Error: Problems parsing data file "{file_name}"')
        else:
            return DataSet(*data, sample=sample)

    def __init__(self, *data: int | float, sample: bool = False) -> None:
        self._sample: bool = sample
        self._data = list(map(lambda d: float(d), data))
        self._size = len(self._data)
        self._data.sort()
        self._calculate_stats()

    def _calculate_stats(self) -> None:
        """Calculate data statistics"""
        self._mean = self._calculate_mean()
        self._stdev = self._calculate_stdev()
        self._quartiles = self._calculate_quartiles()
        self._median = self._quartiles.map(lambda t: t[1])

    def _calculate_mean(self) -> MayBe[float]:
        """Calculate the mean of the data set, if it exists."""
        if self:
            return MayBe(sum(self._data) / self._size)
        else:
            return MayBe()

    def _calculate_stdev(self) -> MayBe[float]:
        """From the data set, calculate & return the stdev if it exists.

        - If sample is True, calculate a sample standard deviation
        - If sample is False, calculate a population standard deviation

        """
        data = self._data
        n = self._size
        if self._mean:
            mean = self._mean.get()
            if self._sample:
                if n > 1:
                    return MayBe(
                        math.sqrt(sum(((x - mean) ** 2 for x in data)) / (n - 1))
                    )
            else:
                if n > 0:
                    return MayBe(math.sqrt(sum(((x - mean) ** 2 for x in data)) / n))
        return MayBe()

    def _calculate_quartiles(self) -> MayBe[tuple[float, float, float]]:
        """Calculate first, second (median), and third quartiles

        Using the "trimmed mid-range" of the data.

        """
        n = self._size
        data = self._data
        if data:
            q = n // 2
            r = q // 2

            if n % 2 == 0:
                second = (data[q - 1] + data[q]) / 2.0
                lower_half = data[:q]
                upper_half = data[q:]
                if q % 2 == 0:
                    first = (lower_half[r - 1] + lower_half[r]) / 2.0
                    third = (upper_half[r - 1] + upper_half[r]) / 2.0
                else:
                    first = lower_half[(r)]
                    third = upper_half[(r)]
            else:
                second = data[q]
                lower_half = data[:q]
                upper_half = data[q:]
                if q % 2 == 0:
                    first = (lower_half[r - 1] + lower_half[r]) / 2.0
                    third = (upper_half[r - 1] + upper_half[r]) / 2.0
                else:
                    first = lower_half[(r)]
                    third = upper_half[(r)]

            return MayBe((first, second, third))
        else:
            return MayBe()

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def is_sample(self) -> bool:
        return self._sample

    def is_population(self) -> bool:
        return not self._sample

    def has_mean(self) -> bool:
        return bool(self._mean)

    def has_median(self) -> bool:
        return bool(self._median)

    def has_stdev(self) -> bool:
        return bool(self._stdev)

    def has_quartiles(self) -> bool:
        return bool(self._quartiles)

    @property
    def mean(self) -> float | Never:
        return self._mean.get()

    @property
    def median(self) -> float | Never:
        return self._median.get()

    @property
    def stdev(self) -> float | Never:
        return self._stdev.get()

    @property
    def quartiles(self) -> tuple[float, float, float] | Never:
        return self._quartiles.get()


class DataSets:
    """
    .. admonition:: Class to manage data sets.

        Base class for managing ``DataSet`` objects.

        - data sets can be samples or populations
        - child class should provide methods to

          - add or remove data sets
          - plot data sets

        - how data sets relate to each other is up to the child class

    """

    def __init__(self) -> None:
        self.data_sets: list[DataSet] = []
