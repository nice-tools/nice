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
import numpy as np

from numpy.testing import (assert_array_equal, assert_almost_equal)
from nose.tools import assert_raises

import mne

from nice import utils
from nice.algorithms.information_theory import epochs_compute_pe
from nice.algorithms.information_theory.permutation_entropy import _symb_python
from nice.algorithms.optimizations.jivaro import pe as jpe

n_epochs = 3
raw = utils.create_mock_data_egi(6, n_epochs * 386, stim=True)
triggers = np.arange(50, n_epochs * 386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * n_epochs

events = mne.find_events(raw)
event_id = {
    'HSTD': 10,
}
epochs = mne.epochs.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                           preload=True, reject=None, picks=None,
                           baseline=(None, 0),
                           verbose=False)
epochs.drop_channels(['STI 014'])
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

n_symbols = 6
n_channels = 6

test_data_t1 = np.reshape(np.array([
    [10.0, 11.0, 12.0],
    [10.0, 12.0, 11.0],
    [11.0, 10.0, 12.0],
    [11.0, 12.0, 10.0],
    [12.0, 10.0, 11.0],
    [12.0, 11.0, 10.0],
]), [6, 3, 1])

test_data_t8 = np.reshape(np.array([
    [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
    [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
    [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
    [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
    [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
    [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
]), [6, 17, 1])

test_data_symb = np.reshape(np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
]), [6, 1, 1])

test_data_count = np.reshape(np.eye(n_symbols), [n_channels, n_symbols, 1])
test_data_pe = np.zeros((n_channels, 1))

test_data_t8_2 = np.reshape(np.array([
    [10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0,
     12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 11.0],
    [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0,
     10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0],
    [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0,
     10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 12.0],
    [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0,
     12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 10.0],
    [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0,
     12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 10.0],
    [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0,
     10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0],
]), [6, 18, 1])

test_data_symb_2 = np.reshape(np.array([
    [0, 1],
    [1, 2],
    [0, 2],
    [1, 3],
    [0, 3],
    [1, 2],
]), [n_channels, 2, 1])

p = 1.0 / 2

test_data_count_2 = np.reshape(np.array([
    [p, p, 0, 0, 0, 0],
    [0, p, p, 0, 0, 0],
    [p, 0, p, 0, 0, 0],
    [0, p, 0, p, 0, 0],
    [p, 0, 0, p, 0, 0],
    [0, p, p, 0, 0, 0],
]), [6, 6, 1])

val = 2 * (-p * np.log(p))

test_data_pe_2 = np.array([
    [val],
    [val],
    [val],
    [val],
    [val],
    [val]
])


def test_pe():
    """Test permutation entropy"""
    symb, count = _symb_python(test_data_t1, kernel=3, tau=1)
    assert_array_equal(test_data_symb, symb)
    assert_array_equal(test_data_count, count)

    pe, symb = jpe(test_data_t1, 3, 1)
    assert_array_equal(test_data_pe, pe)
    assert_array_equal(test_data_symb, symb)

    # Test error with not enough data
    with assert_raises(ValueError):
        symb, count = _symb_python(test_data_t1, kernel=3, tau=8)

    # Test simple symbolic transformation for tau 8
    symb, count = _symb_python(test_data_t8, kernel=3, tau=8)
    assert_array_equal(test_data_symb, symb)
    assert_array_equal(test_data_count, count)

    pe, symb = jpe(test_data_t8, 3, 8)
    assert_array_equal(test_data_symb, symb)
    assert_array_equal(test_data_pe, pe)

    symb, count = _symb_python(test_data_t8_2, kernel=3, tau=8)
    assert_array_equal(test_data_symb_2, symb)
    assert_array_equal(test_data_count_2, count)

    pe, symb = jpe(test_data_t8_2, 3, 8)
    assert_array_equal(test_data_symb_2, symb)
    assert_array_equal(test_data_pe_2, pe)

    # test across backends
    pe_1, symb_1 = epochs_compute_pe(epochs, kernel=3, tau=8, backend='python')
    pe_2, symb_2 = epochs_compute_pe(epochs, kernel=3, tau=8, backend='c')
    assert_almost_equal(pe_1, pe_2)
    assert_almost_equal(pe_1, pe_2)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
