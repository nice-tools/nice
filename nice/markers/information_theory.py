# NICE
# Copyright (C) 2017 - Authors of NICE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# You can be released from the requirements of the license by purchasing a
# commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities as mentioned in the GNU Affero General Public
# License version 3 without disclosing the source code of your own
# applications.
#
from collections import OrderedDict
from .base import BaseMarker
from ..algorithms.information_theory import (epochs_compute_komplexity,
                                             epochs_compute_pe)


class KolmogorovComplexity(BaseMarker):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, backend="python", nbins=32,
                 method_params=None, comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        if method_params is None:
            method_params = {}

        self.nbins = nbins
        self.backend = backend
        self.method_params = method_params

    def _fit(self, epochs):
        komp = epochs_compute_komplexity(
            epochs, nbins=self.nbins, tmin=self.tmin,
            tmax=self.tmax, backend=self.backend,
            method_params=self.method_params)
        self.data_ = komp

    @property
    def _axis_map(self):
        return OrderedDict([
            ('channels', 0),
            ('epochs', 1)
        ])


def read_komplexity(fname, comment='default'):
    return KolmogorovComplexity._read(fname, comment=comment)


class PermutationEntropy(BaseMarker):
    """docstring for PermutationEntropy"""

    def __init__(self, tmin=None, tmax=None, kernel=3, tau=8, backend="python",
                 comment='default', method_params=None):
        BaseMarker.__init__(self, tmin, tmax, comment)
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.method_params = method_params

    def _fit(self, epochs):
        pe, _ = epochs_compute_pe(
            epochs, tmin=self.tmin, tmax=self.tmax,
            kernel=self.kernel, tau=self.tau, backend=self.backend,
            method_params=self.method_params)
        self.data_ = pe

    @property
    def _axis_map(self):
        return OrderedDict([
            ('channels', 0),
            ('epochs', 1)
        ])


def read_pe(fname, comment='default'):
    return PermutationEntropy._read(fname, comment=comment)
