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
import numpy as np

from .base import BaseMarker
from ..algorithms.connectivity import epochs_compute_wsmi


class SymbolicMutualInformation(BaseMarker):
    """docstring for SymbolicMutualInformation"""

    def __init__(self, tmin=None, tmax=None, kernel=3, tau=8, backend="python",
                 method_params=None, method='weighted', comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        if method_params is None:
            method_params = {}
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.method_params = method_params
        self.method = method

    def _fit(self, epochs):
        wsmi, smi, _, _ = epochs_compute_wsmi(
            epochs, kernel=self.kernel, tau=self.tau, tmin=self.tmin,
            tmax=self.tmax, backend=self.backend,
            method_params=self.method_params)
        data = wsmi if self.method == 'weighted' else smi
        data += np.transpose(data, [1, 0, 2])
        self.data_ = data

    @property
    def _axis_map(self):
        return OrderedDict([
            ('channels', 0),
            ('channels_y', 1),
            ('epochs', 2)
        ])


def read_smi(fname, comment='default'):
    return SymbolicMutualInformation._read(fname, comment=comment)
