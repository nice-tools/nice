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

import os.path as op

from nose.tools import assert_equal, assert_true

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import mne
from mne.utils import _TempDir

# our imports
from nice.markers import PowerSpectralDensity
from nice.markers import ContingentNegativeVariation
from nice.markers import PermutationEntropy
from nice.markers import TimeLockedTopography
from nice.markers import TimeLockedContrast
from nice.markers import PowerSpectralDensityEstimator

from nice import Markers, read_markers


matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = {'a': 1, 'b': 2}
preload = True


def _get_data():
    raw = mne.io.Raw(raw_fname)
    raw.info['lowpass'] = 70.  # To avoid warning
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    return epochs


def _compare_instance(inst1, inst2):
    for k, v in vars(inst1).items():
        v2 = getattr(inst2, k)
        if k == 'ch_info_' and v2 is None:
            continue
        if isinstance(v, np.ndarray):
            assert_array_equal(v, v2)
        elif isinstance(v, mne.io.meas_info.Info):
            pass
        else:
            assert_equal(v, v2)


def test_collecting_feature():
    """Test computation of spectral markers"""
    epochs = _get_data()[:2]
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    markers_list = [
        PowerSpectralDensity(estimator=estimator, fmin=1, fmax=4),
        ContingentNegativeVariation(),
        TimeLockedTopography(tmin=0.1, tmax=0.2),
        TimeLockedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                           condition_b='b'),
        TimeLockedContrast(tmin=0.1, tmax=0.3, condition_a='a',
                           condition_b='b', comment='another_erp')
    ]

    markers = Markers(markers_list)
    # check states and names
    for name, marker in markers.items():
        assert_true(not any(k.endswith('_') for k in vars(marker)))
        assert_equal(name, marker._get_title())

    # check order
    assert_equal(list(markers.values()), markers_list)

    # check fit
    markers.fit(epochs)
    for t_marker in markers_list:
        assert_true(any(k.endswith('_') for k in vars(t_marker)))

    tmp = _TempDir()
    tmp_fname = tmp + '/test_markers.hdf5'
    markers.save(tmp_fname)
    markers2 = read_markers(tmp_fname)
    for ((k1, v1), (k2, v2)) in zip(markers.items(), markers2.items()):
        assert_equal(k1, k2)
        assert_equal(
            {k: v for k, v in vars(v1).items() if not k.endswith('_') and
             not k == 'estimator'},
            {k: v for k, v in vars(v2).items() if not k.endswith('_') and
             not k == 'estimator'})
    pe = PermutationEntropy().fit(epochs)
    markers._add_marker(pe)

    tmp = _TempDir()
    tmp_fname = tmp + '/test_markers.hdf5'
    markers.save(tmp_fname)
    markers3 = read_markers(tmp_fname)
    assert_true(pe._get_title() in markers3)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
