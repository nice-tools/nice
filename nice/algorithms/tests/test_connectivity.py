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

from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal)

import mne

from nice import utils
from nice.algorithms.connectivity import (epochs_compute_wsmi, _wsmi_python,
                                          _get_weights_matrix)
from nice.algorithms.optimizations.jivaro import wsmi as jwsmi

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

wts = _get_weights_matrix(n_symbols)

test_wsmi_result = np.zeros((6, 6, 1))

w = p * np.log(p / p / p) / np.log(n_symbols)
s = 2 * w
test_smi_2_result = np.reshape(np.array([
    [0, s, s, s, s, s],
    [0, 0, s, s, s, s],
    [0, 0, 0, s, s, s],
    [0, 0, 0, 0, s, s],
    [0, 0, 0, 0, 0, s],
    [0, 0, 0, 0, 0, 0],
]), [6, 6, 1])

test_wsmi_2_result = np.reshape(np.array([
    [0, s, w, s, w, s],
    [0, 0, w, 0, w, 0],
    [0, 0, 0, w, 0, w],
    [0, 0, 0, 0, w, 0],
    [0, 0, 0, 0, 0, w],
    [0, 0, 0, 0, 0, 0],
]), [6, 6, 1])


def test_wsmi():
    """Test wsmi"""

    # Test wsmi in python
    wsmi_data, smi_data = _wsmi_python(test_data_symb, test_data_count, wts)
    assert_array_equal(test_wsmi_result, smi_data)
    assert_array_equal(test_wsmi_result, wsmi_data)

    wsmi_data, smi_data = _wsmi_python(test_data_symb_2,
                                       test_data_count_2, wts)
    assert_array_almost_equal(test_smi_2_result, smi_data)
    assert_array_almost_equal(test_wsmi_2_result, wsmi_data)

    # Test wsmi in c
    wsmi_data, smi_data, symb, count = jwsmi(test_data_t8, 3, 8, wts, 4)
    assert_array_equal(test_data_symb, symb)
    assert_array_equal(test_data_count, count)
    assert_array_equal(test_wsmi_result, wsmi_data)
    assert_array_equal(test_wsmi_result, smi_data)

    wsmi_data, smi_data, symb, count = jwsmi(test_data_t8_2, 3, 8, wts, 4)
    assert_array_equal(test_data_symb_2, symb)
    assert_array_equal(test_data_count_2, count)
    assert_array_equal(test_smi_2_result, smi_data)
    assert_array_equal(test_wsmi_2_result, wsmi_data)

    pmp = {'bypass_csd': True}
    # Test wsmi across backends
    wsmi_1, smi_1, sym_1, count_1 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='python', method_params=pmp)

    mp = {'nthreads': 1, 'bypass_csd': True}
    wsmi_2, smi_2, sym_2, count_2 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='openmp', method_params=mp)

    assert_array_equal(sym_1, sym_2)
    assert_array_equal(count_1, count_2)
    assert_almost_equal(wsmi_1, wsmi_2)
    assert_almost_equal(smi_1, smi_2)

    epochs.drop([0])
    wsmi_3, smi_3, sym_3, count_3 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='python', method_params=pmp)
    mp.update(nthreads=1)
    wsmi_4, smi_4, sym_4, count_4 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='openmp', method_params=mp)

    assert_array_equal(sym_3, sym_4)
    assert_array_equal(count_3, count_4)
    assert_almost_equal(wsmi_3, wsmi_4)
    assert_almost_equal(smi_3, smi_4)

    # Test with more threads
    wsmi_1, smi_1, sym_1, count_1 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='python', method_params=pmp)

    mp.update(nthreads=14)

    wsmi_2, smi_2, sym_2, count_2 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='openmp', method_params=mp)

    assert_array_equal(sym_1, sym_2)
    assert_array_equal(count_1, count_2)
    assert_almost_equal(wsmi_1, wsmi_2)
    assert_almost_equal(smi_1, smi_2)

    epochs.drop([0])
    wsmi_3, smi_3, sym_3, count_3 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='python', method_params=pmp)

    mp.update(nthreads=3)
    wsmi_4, smi_4, sym_4, count_4 = epochs_compute_wsmi(
        epochs, kernel=3, tau=8, backend='openmp', method_params=mp)
    assert_array_equal(sym_3, sym_4)
    assert_array_equal(count_3, count_4)
    assert_almost_equal(wsmi_3, wsmi_4)
    assert_almost_equal(smi_3, smi_4)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
