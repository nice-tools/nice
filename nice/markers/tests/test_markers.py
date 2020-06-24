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

from nose.tools import assert_equal, assert_true, assert_raises

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import functools

import mne
import h5py
from mne.utils import _TempDir

# our imports
from nice.markers import PowerSpectralDensity, read_psd
from nice.markers import ContingentNegativeVariation, read_cnv
from nice.markers import KolmogorovComplexity, read_komplexity
from nice.markers import PermutationEntropy, read_pe
from nice.markers import SymbolicMutualInformation, read_smi

from nice.markers import TimeLockedTopography, read_ert

from nice.markers import TimeLockedContrast, read_erc

from nice.markers import WindowDecoding, read_wd
from nice.markers import TimeDecoding, read_td
from nice.markers import GeneralizationDecoding, read_gd
from nice.markers import PowerSpectralDensityEstimator

matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

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


def _get_decoding_data():
    raw = mne.io.Raw(raw_fname)
    raw.info['lowpass'] = 70.  # To avoid warning
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id_2, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    return epochs


def _compare_values(v, v2):
    if isinstance(v, np.ndarray):
        assert_array_equal(v, v2)
    elif isinstance(v, mne.io.meas_info.Info):
        pass
    elif isinstance(v, dict):
        for key, value in v.items():
            _compare_values(v[key], v2[key])
    elif isinstance(v, PowerSpectralDensityEstimator):
        _compare_instance(v, v2)
    else:
        assert_equal(v, v2)


def _compare_instance(inst1, inst2):
    for k, v in vars(inst1).items():
        v2 = getattr(inst2, k)
        if k == 'ch_info_' and v2 is None:
            continue
        _compare_values(v, v2)


def _base_io_test(inst, epochs, read_fun):
    tmp = _TempDir()
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5', overwrite='update')
    inst2 = read_fun(tmp + '/test.hdf5')
    _compare_instance(inst, inst2)


def _erfp_io_test(tmp, inst, epochs, read_fun, comment='default'):
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5', overwrite='update')
    inst2 = read_fun(tmp + '/test.hdf5', epochs, comment=comment)
    assert_array_equal(inst.epochs_.get_data(), inst2.epochs_.get_data())
    _compare_instance(inst, inst2)


def _base_reduction_test(inst, epochs):
    sc = inst.reduce_to_scalar(None)
    if inst.data_.ndim == 3:
        sc2 = np.mean(np.mean(np.mean(inst.data_, axis=0), axis=0), axis=0)
    else:
        sc2 = np.mean(np.mean(inst.data_, axis=0), axis=0)
    assert_equal(sc, sc2)
    topo = inst.reduce_to_topo(None)
    topo_chans = len(mne.io.pick.pick_types(epochs.info, meg=True, eeg=True))
    assert_equal(topo.shape, (topo_chans,))


def _base_compression_test(inst, epochs):
    orig_shape = inst.data_.shape
    inst.compress(np.mean)
    axis = inst._axis_map['epochs']
    new_shape = np.array(orig_shape)
    new_shape[axis] = 1
    assert_array_equal(inst.data_.shape, new_shape)


def test_spectral():
    """Test computation of spectral markers"""
    epochs = _get_data()[:2]
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    psd = PowerSpectralDensity(estimator, fmin=1., fmax=4.)
    _base_io_test(psd, epochs,
                  functools.partial(read_psd,
                                    estimators={'default': estimator}))

    reduction_func = [
        {'axis': 'frequency', 'function': np.sum},
        {'axis': 'channels', 'function': np.mean},
        {'axis': 'epochs', 'function': np.mean}]
    out = psd._prepare_data(picks=None, target='scalar')
    out = np.sum(out, axis=-1)
    out = 10 * np.log10(out)
    out = np.mean(out, axis=-1)
    out = np.mean(out, axis=-1)
    scalar = psd.reduce_to_scalar(reduction_func)
    assert_equal(scalar, out)
    # TODO: Fix this test
    # _base_reduction_test(psd, epochs)
    # _base_compression_test(psd, epochs)


def test_time_locked():
    """Test computation of time locked markers"""

    raw = mne.io.Raw(raw_fname)
    raw.info['lowpass'] = 70.  # To avoid warning
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id_2, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    cnv = ContingentNegativeVariation()
    _base_io_test(cnv, epochs, read_cnv)
    _base_reduction_test(cnv, epochs)

    tmp = _TempDir()
    # with h5py.File(tmp + '/test.hdf5', 'r') as fid:
    #     assert_true('nice/data/epochs' not in fid)
    ert = TimeLockedTopography(tmin=0.1, tmax=0.2)
    _erfp_io_test(tmp, ert, epochs, read_ert)
    with h5py.File(tmp + '/test.hdf5', 'r') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])

    tmp = _TempDir()
    # with h5py.File(tmp + '/test.hdf5', 'r') as fid:
    #     assert_true('nice/data/epochs' not in fid)
    erc = TimeLockedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                             condition_b='b')
    _erfp_io_test(tmp, erc, epochs, read_erc)
    with h5py.File(tmp + '/test.hdf5', 'r') as fid:
        assert_true('nice/data/epochs' in fid)
    erc = TimeLockedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                             condition_b='b', comment='another_erp')
    _erfp_io_test(tmp, erc, epochs, read_erc, comment='another_erp')
    with h5py.File(tmp + '/test.hdf5', 'r') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])


def test_komplexity():
    """Test computation of komplexity marker"""
    epochs = _get_data()[:2]
    komp = KolmogorovComplexity()
    _base_io_test(komp, epochs, read_komplexity)
    _base_reduction_test(komp, epochs)
    _base_compression_test(komp, epochs)


def test_pe():
    """Test computation of permutation entropy marker"""
    epochs = _get_data()[:2]
    pe = PermutationEntropy()
    _base_io_test(pe, epochs, read_pe)
    _base_reduction_test(pe, epochs)
    _base_compression_test(pe, epochs)


def test_wsmi():
    """Test computation of wsmi marker"""
    epochs = _get_data()[:2]
    method_params = {'bypass_csd': True}
    wsmi = SymbolicMutualInformation(method_params=method_params)
    _base_io_test(wsmi, epochs, read_smi)
    _base_reduction_test(wsmi, epochs)
    _base_compression_test(wsmi, epochs)


def test_window_decoding():
    """Test computation of window decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        sample_weight='auto',
        clf=None,
        cv=None,
        n_jobs=1,
        random_state=42,
        labels=None
    )

    wd = WindowDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                        condition_b='b', decoding_params=decoding_params)
    _base_io_test(wd, epochs, read_wd)


def test_time_decoding():
    """Test computation of time decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        clf=None,
        cv=2,
        n_jobs=1
    )

    td = TimeDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                      condition_b='b', decoding_params=decoding_params)
    _base_io_test(td, epochs, read_td)


def test_generalization_decoding():
    """Test computation of time generalization decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        clf=None,
        cv=2,
        n_jobs=1
    )

    gd = GeneralizationDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                                condition_b='b',
                                decoding_params=decoding_params)
    _base_io_test(gd, epochs, read_gd)


picking_data = np.zeros((3, 4, 5), dtype=np.float)
for i in range(4):
    picking_data[:, i, :] = np.array([
        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
    ]) * pow(10, i)

# 3 Epochs - 4 channels, 5 samples
# array([[
#     [1., 2., 3., 4., 5.],
#     [10., 20., 30., 40., 50.],
#     [100., 200., 300., 400., 500.],
#     [1000., 2000., 3000., 4000., 5000.]
# ], [
#     [6., 7., 8., 9., 10.],
#     [60., 70., 80., 90., 100.],
#     [600., 700., 800., 900., 1000.],
#     [6000., 7000., 8000., 9000., 10000.]
# ], [
#     [11., 12., 13., 14., 15.],
#     [110., 120., 130., 140., 150.],
#     [1100., 1200., 1300., 1400., 1500.],
#     [11000., 12000., 13000., 14000., 15000.]
# ])


def test_picking():
    """Test picking axis when reducing"""
    epochs = _get_data()[:3]
    epochs = epochs.pick_channels(epochs.ch_names[:4])
    epochs.crop(0, 0.004 * 5)
    epochs._data = picking_data

    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=5., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    psd = PowerSpectralDensity(estimator, fmin=1., fmax=5., dB=False)

    method_params = {'bypass_csd': True}
    wsmi = SymbolicMutualInformation(method_params=method_params)
    wsmi.data_ = picking_data.transpose(1, 2, 0)

    ert = TimeLockedTopography(tmin=0, tmax=0.004 * 5)
    ert.fit(epochs)

    psd.data_ = picking_data
    psd.estimator.data_ = picking_data
    psd.estimator.data_norm_ = picking_data
    psd.estimator.freqs_ = np.array([1., 2., 3., 4., 5.])

    extreme_picks = {
        'channels': np.array([0, 3]),
        'times': np.array([0, 4]),
        'epochs': np.array([0, 2])
    }

    red_fun = [
        {'axis': 'epochs', 'function': np.sum},
        {'axis': 'times', 'function': np.sum},
        {'axis': 'channels', 'function': np.sum},
    ]

    psd_extreme_picks = {
        'channels': np.array([0, 3]),
        'epochs': np.array([0, 2])
    }

    psd_red_fun = [
        {'axis': 'epochs', 'function': np.sum},
        {'axis': 'frequency', 'function': np.sum},
        {'axis': 'channels', 'function': np.sum},
    ]

    wsmi_red_fun = [
        {'axis': 'epochs', 'function': np.sum},
        {'axis': 'channels_y', 'function': np.sum},
        {'axis': 'channels', 'function': np.sum},
    ]

    wsmi_extreme_picks = {
        'channels': np.array([0, 3]),
        'channels_y': np.array([0, 4]),
        'epochs': np.array([0, 2])
    }

    extreme_topo_expected = np.array(
        [12 + 20., 120 + 200., 1200 + 2000., 12000 + 20000.])
    extreme_topo_obtained = ert.reduce_to_topo(red_fun, extreme_picks)
    assert_array_equal(extreme_topo_obtained, extreme_topo_expected)

    assert_raises(
        ValueError, wsmi.reduce_to_topo, wsmi_red_fun, extreme_picks)
    extreme_topo_obtained = wsmi.reduce_to_topo(
        wsmi_red_fun, wsmi_extreme_picks)
    assert_array_equal(extreme_topo_obtained, extreme_topo_expected)

    extreme_topo_expected = np.array([80., 800., 8000., 80000.])
    extreme_topo_obtained = psd.reduce_to_topo(psd_red_fun, psd_extreme_picks)
    assert_array_equal(extreme_topo_obtained, extreme_topo_expected)

    extreme_scalar_expected = 12 + 20. + 12000 + 20000.
    extreme_scalar_obtained = ert.reduce_to_scalar(red_fun, extreme_picks)
    assert_equal(extreme_scalar_obtained, extreme_scalar_expected)

    extreme_scalar_obtained = wsmi.reduce_to_scalar(
        wsmi_red_fun, wsmi_extreme_picks)
    assert_equal(extreme_scalar_obtained, extreme_scalar_expected)

    extreme_scalar_expected = 80. + 80000.
    extreme_scalar_obtained = psd.reduce_to_scalar(
        psd_red_fun, psd_extreme_picks)
    assert_equal(extreme_scalar_obtained, extreme_scalar_expected)

    extreme_time_expected = np.array(
        [12 + 12000., 14 + 14000., 16 + 16000., 18 + 18000., 20 + 20000.])
    extreme_time_obtained = ert._reduce_to(red_fun, 'times', extreme_picks)
    assert_array_equal(extreme_time_obtained, extreme_time_expected)

    assert_raises(
        ValueError, psd._reduce_to, psd_red_fun, 'times', extreme_picks)

    picks = {
        'channels': np.array([1, 3]),
        'times': np.array([1, 3, 4]),
        'epochs': np.array([0, 1])
    }

    psd_picks = {
        'channels': np.array([1, 3]),
        'epochs': np.array([0, 1])
    }

    wsmi_picks = {
        'channels': np.array([1, 3]),
        'channels_y': np.array([1, 3, 4]),
        'epochs': np.array([0, 1])
    }

    topo_expected = np.array(
        [9 + 13 + 15., 90 + 130 + 150., 900 + 1300 + 1500,
         9000 + 13000 + 15000.])
    topo_obtained = ert.reduce_to_topo(red_fun, picks)
    assert_array_equal(topo_obtained, topo_expected)
    topo_obtained = wsmi.reduce_to_topo(wsmi_red_fun, wsmi_picks)
    assert_array_equal(topo_obtained, topo_expected)

    topo_expected = np.array(
        [7 + 9 + 11 + 13 + 15., 70 + 90 + 110 + 130 + 150.,
         700 + 900 + 1100 + 1300 + 1500,
         7000 + 9000 + 11000 + 13000 + 15000.])
    topo_obtained = psd.reduce_to_topo(psd_red_fun, psd_picks)
    assert_array_equal(topo_obtained, topo_expected)

    scalar_expected = 90 + 130 + 150. + 9000 + 13000 + 15000.
    scalar_obtained = ert.reduce_to_scalar(red_fun, picks)
    assert_equal(scalar_obtained, scalar_expected)
    scalar_obtained = wsmi.reduce_to_scalar(wsmi_red_fun, wsmi_picks)
    assert_equal(scalar_obtained, scalar_expected)

    scalar_expected = 550. + 55000.
    scalar_obtained = psd.reduce_to_scalar(psd_red_fun, psd_picks)
    assert_equal(scalar_obtained, scalar_expected)

    time_expected = np.array(
        [70 + 7000., 90 + 9000., 110 + 11000., 130 + 13000., 150 + 15000.])
    time_obtained = ert._reduce_to(red_fun, 'times', picks)
    assert_array_equal(time_obtained, time_expected)

    channel_pick = {
        'channels': np.array([1]),
        'times': np.array([2, 3]),
        'epochs': None
    }

    psd_channel_pick = {
        'channels': np.array([1]),
        'epochs': None
    }

    wsmi_channel_pick = {
        'channels': np.array([1]),
        'channels_y': np.array([2, 3]),
        'epochs': None
    }

    topo_expected = np.array([
        3 + 8 + 13. + 4 + 9 + 14.,
        30 + 80 + 130. + 40 + 90 + 140.,
        300 + 800 + 1300 + 400 + 900 + 1400.,
        3000 + 8000 + 13000 + 4000 + 9000 + 14000.])
    topo_obtained = ert.reduce_to_topo(red_fun, channel_pick)
    assert_array_equal(topo_obtained, topo_expected)
    topo_obtained = wsmi.reduce_to_topo(wsmi_red_fun, wsmi_channel_pick)
    assert_array_equal(topo_obtained, topo_expected)

    topo_expected = np.array([
        1 + 6 + 11. + 2 + 7 + 12 + 3 + 8 + 13. + 4 + 9 + 14. + 5 + 10 + 15,
        10 + 60 + 110 + 20 + 70 + 120 + 30 + 80 +  # noqa
        130. + 40 + 90 + 140. + 50 + 100 + 150,
        100 + 600 + 1100 + 200 + 700 + 1200 +  # noqa
        300 + 800 + 1300 + 400 + 900 + 1400. + 500 + 1000 + 1500,
        1000 + 6000 + 11000 + 2000 + 7000 + 12000 +  # noqa
        3000 + 8000 + 13000 + 4000 + 9000 + 14000. + 5000 + 10000 + 15000])
    topo_obtained = psd.reduce_to_topo(psd_red_fun, psd_channel_pick)
    assert_array_equal(topo_obtained, topo_expected)

    scalar_expected = 30 + 80 + 130. + 40 + 90 + 140
    scalar_obtained = ert.reduce_to_scalar(red_fun, channel_pick)
    assert_equal(scalar_obtained, scalar_expected)
    scalar_obtained = wsmi.reduce_to_scalar(wsmi_red_fun, wsmi_channel_pick)
    assert_equal(scalar_obtained, scalar_expected)

    scalar_expected = (10 + 60 + 110 + 20 + 70 + 120 +  # noqa
                       30 + 80 + 130. + 40 + 90 + 140. + 50 + 100 + 150)
    scalar_obtained = psd.reduce_to_scalar(psd_red_fun, psd_channel_pick)
    assert_equal(scalar_obtained, scalar_expected)

    time_expected = np.array(
        [10 + 60 + 110., 20 + 70 + 120., 30 + 80 + 130., 40 + 90 + 140.,
         50 + 100 + 150.])
    time_obtained = ert._reduce_to(red_fun, 'times', channel_pick)
    assert_array_equal(time_obtained, time_expected)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
