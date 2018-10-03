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
import zlib
import numpy as np

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

import mne

from nice import utils
from nice.algorithms.information_theory.komplexity import (_komplexity_python,
                                                           _symb_python)
from nice.algorithms.optimizations import ompk

n_epochs = 3

raw = utils.create_mock_data_egi(6, n_epochs * 386, stim=True)

triggers = np.arange(50, n_epochs * 386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * n_epochs

events = mne.find_events(raw)
event_id = {
    'HSTD': 10,
    # 'HDVT': 20,
    # 'LSGS': 30,
    # 'LSGD': 40,
    # 'LDGS': 60,
    # 'LDGD': 50,  # yes XXX for now correct
}
epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                    preload=True, reject=None, picks=None,
                    baseline=(None, 0), verbose=False)
epochs.drop_channels(['STI 014'])

n_channels = 6
n_bins = 3

test_data = np.reshape(np.array([
    [10.0, 11.0, 12.0],
    [10.0, 12.0, 11.0],
    [11.0, 10.0, 12.0],
    [11.0, 12.0, 10.0],
    [12.0, 10.0, 11.0],
    [12.0, 11.0, 10.0],
]), [1, 6, 3])

test_data_symb = np.reshape(np.array([
    ['ABC'],
    ['ACB'],
    ['BAC'],
    ['BCA'],
    ['CAB'],
    ['CBA'],
]), [6, 1, 1])


test_data_symb_4bin = np.reshape(np.array([
    ['ACD'],
    ['ADC'],
    ['CAD'],
    ['CDA'],
    ['DAC'],
    ['DCA'],
]), [6, 1, 1])


v = np.double(len(zlib.compress(bytes('ABC', 'ascii')))) / 3
test_data_k = np.reshape(np.array([
    [v],
    [v],
    [v],
    [v],
    [v],
    [v],
]), [6, 1])


def test_komplexity_python():
    """ Test simple symbolic transformation """
    for chan in range(n_channels):
        symb = _symb_python(test_data[0, chan, :], n_bins)
        assert_equal(symb.decode('ascii'), test_data_symb[chan])

    # Test simple symbolic transformation with more bins
    for chan in range(n_channels):
        symb = _symb_python(test_data[0, chan, :], 4)
        assert_equal(symb.decode('ascii'), test_data_symb_4bin[chan])

    # Test compressions
    komp = _komplexity_python(test_data, n_bins)
    assert_array_equal(komp, test_data_k)


def test_komplexity_omp():
    """ Test compressions """
    komp = ompk.komplexity(test_data, n_bins, 1)
    assert_array_equal(komp, test_data_k)

    komp = ompk.komplexity(test_data, n_bins, 4)
    assert_array_equal(komp, test_data_k)

    komp = ompk.komplexity(test_data, n_bins + 1, 1)
    assert_array_equal(komp, test_data_k)

    komp = ompk.komplexity(test_data, n_bins + 1, 4)
    assert_array_equal(komp, test_data_k)


# def test_komplexity():
#     """ test komplexity metric """
#     k1 = epochs_compute_komplexity(epochs, nbins=32, backend='python')
#     k2 = epochs_compute_komplexity(epochs, nbins=32, backend='openmp')
#     assert_array_equal(k1, k2)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
