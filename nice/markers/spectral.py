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
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import h5py

import numpy as np

from mne.utils import logger
from mne.externals.h5io import write_hdf5

from .base import (BaseMarker, BaseContainer, _get_title, _read_container)
from ..algorithms.spectral import psd_welch


class BasePowerSpectralDensity(BaseMarker):
    def __init__(self, tmin, tmax, fmin, fmax, estimator=None,
                 comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        self.fmin = fmin
        self.fmax = fmax
        self.estimator = estimator

    def _fit(self, epochs):
        if self.estimator is None:
            raise ValueError('Need an estimator to be able to fit')
        self._check_freq_time_range(epochs)
        self.estimator_comment_ = self.estimator.comment
        if not hasattr(self.estimator, 'data_'):
            self.estimator.fit(epochs)
        self.data_ = self.estimator.data_

    def _check_freq_time_range(self, epochs):
        this_min = self.fmin
        if this_min is None:
            this_min = 0.
        this_max = self.fmax
        if this_max is None:
            this_max = epochs.info['sfreq'] / 2

        in_range = (self.estimator.fmin <= this_min and  # noqa
                    self.estimator.fmax >= this_max)
        if not in_range:
            raise ValueError('Spectral frequencies do not match')

        in_range = (self.estimator.tmin == self.tmin and  # noqa
                    self.estimator.tmax == self.tmax)

    def save(self, fname, overwrite=False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(
            exclude=['ch_info_', 'estimator', 'data_'])

        has_estimator = False
        estimator_name = self.estimator._get_title()
        with h5py.File(fname, 'r') as h5fid:
            if estimator_name in h5fid:
                has_estimator = True
                logger.info('PSDS Estimator already present in HDF5 file, '
                            'will not be overwritten')

        if not has_estimator:
            logger.info('Writing PSDS Estimator to HDF5 file')
            self.estimator.save(fname, overwrite=overwrite)
        write_hdf5(
            fname, save_vars, overwrite=overwrite,
            title=_get_title(self.__class__, self.comment), slash='replace')

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    @classmethod
    def _read(cls, fname, estimators=None, comment='default'):
        return _read_power_spectral(cls, fname=fname, estimators=estimators,
                                    comment=comment)


def _read_power_spectral(cls, fname, estimators=None, comment='default'):
    out = _read_container(cls, fname, comment=comment)
    if estimators is None:
        this_estimator = PowerSpectralDensityEstimator._read(
            fname, out.estimator_comment_)
    else:
        this_estimator = estimators[out.estimator_comment_]
    out.estimator = this_estimator
    out.data_ = this_estimator.data_
    return out


class PowerSpectralDensityEstimator(BaseContainer):
    def __init__(self, tmin, tmax, fmin, fmax, psd_method, psd_params,
                 comment):
        BaseContainer.__init__(self, comment=comment)
        self.psd_method = psd_method
        self.psd_params = deepcopy(psd_params)
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax

    def fit(self, epochs):
        if self.psd_method == 'welch':
            function = psd_welch
        self.psd_params.update(
            tmin=self.tmin, tmax=self.tmax, fmin=self.fmin, fmax=self.fmax)
        self.data_, self.freqs_ = function(epochs, **self.psd_params)
        self.data_norm_ = self.data_ / self.data_.sum(axis=-1)[..., None]

    def save(self, fname, overwrite=False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(exclude=['ch_info_', 'data_norm_'])
        write_hdf5(
            fname,
            save_vars,
            title=_get_title(self.__class__, self.comment),
            overwrite=overwrite, slash='replace')

    @classmethod
    def _read(cls, fname, comment='default'):
        psde = _read_container(cls, fname, comment=comment)
        psde.data_norm_ = psde.data_ / psde.data_.sum(axis=-1)[..., None]
        return psde


def read_psd_estimator(fname, comment='default'):
    # Estimators is either None or a list of estimators
    return PowerSpectralDensityEstimator._read(fname, comment=comment)


class PowerSpectralDensity(BasePowerSpectralDensity):
    """docstring for PSD"""

    def __init__(self, estimator=None, tmin=None, tmax=None, fmin=0,
                 fmax=np.inf, normalize=False, dB=True, comment='default'):
        BasePowerSpectralDensity.__init__(
            self, tmin=None, tmax=None, fmin=fmin, fmax=fmax,
            estimator=estimator, comment=comment)
        self.normalize = normalize
        self.dB = dB

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('frequency', 2)
        ])

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    def _prepare_data(self, picks, target):
        this_picks = {k: None for k in ['channels', 'epochs']}
        if picks is not None:
            if any([x not in this_picks.keys() for x in picks.keys()]):
                raise ValueError('Picking is not compatible for {}'.format(
                    self._get_title()))
        if picks is None:
            picks = {}
        if 'frequency' in picks:
            logger.warning('Picking in frequency axis is currently not '
                           'supported. This will not have effect.')
        this_picks.update(picks)
        to_preserve = self._get_preserve_axis(target)
        if len(to_preserve) > 0:
            for axis in to_preserve:
                this_picks[axis] = None

        freqs = self.estimator.freqs_
        start = np.searchsorted(freqs, self.fmin, 'left')
        end = np.searchsorted(freqs, self.fmax, 'right')

        ch_picks = this_picks['channels']
        if ch_picks is None:
            ch_picks = Ellipsis

        epochs_picks = this_picks['epochs']
        if epochs_picks is None:
            epochs_picks = Ellipsis

        if self.normalize:
            this_psds = self.estimator.data_norm_[
                ..., start:end][:, ch_picks][epochs_picks]
        else:
            this_psds = self.estimator.data_[
                ..., start:end][:, ch_picks][epochs_picks]
        return this_psds

    def _reduce_to(self, reduction_func, target, picks):
        if not hasattr(self, 'data_'):
            raise ValueError('You did not fit me. Do it again after fitting '
                             'some data!')
        out, funcs, axis = self._prepare_reduction(
            reduction_func, target, picks, return_axis=True)
        for func, ax in zip(funcs, axis):
            out = func(out, axis=0)
            if (self.dB is True and self.normalize is False and  # noqa
                    ax == 'frequency'):
                out = 10 * np.log10(out)
        return out

    @classmethod
    def _read(cls, fname, estimators=None, comment='default'):
        out = _read_power_spectral(
            cls, fname=fname, estimators=estimators, comment=comment)
        out.dB = bool(out.dB)
        out.normalize = bool(out.normalize)
        return out


def read_psd(fname, estimators=None, comment='default'):
    # Estimators is either None or a list of estimators
    out = PowerSpectralDensity._read(
        fname, estimators=estimators, comment=comment)
    return out


class PowerSpectralDensitySummary(BasePowerSpectralDensity):
    """docstring for PSD"""

    def __init__(self, percentile, estimator=None, tmin=None, tmax=None,
                 fmin=0, fmax=np.inf, comment='default'):
        BasePowerSpectralDensity.__init__(
            self, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
            estimator=estimator, comment=comment)
        self.percentile = percentile

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1)
        ])

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    def _prepare_data(self, picks, target):
        this_picks = {k: None for k in ['channels', 'epochs']}
        if picks is not None:
            if any([x not in this_picks.keys() for x in picks.keys()]):
                raise ValueError('Picking is not compatible for {}'.format(
                    self._get_title()))
        if picks is None:
            picks = {}
        if 'frequency' in picks:
            logger.warning('Picking in frequency axis is currently not '
                           'supported. This will not have effect.')
        this_picks.update(picks)
        to_preserve = self._get_preserve_axis(target)
        if len(to_preserve) > 0:
            for axis in to_preserve:
                this_picks[axis] = None

        freqs = self.estimator.freqs_
        start = np.searchsorted(freqs, self.fmin, 'left')
        end = np.searchsorted(freqs, self.fmax, 'right')

        ch_picks = this_picks['channels']
        if ch_picks is None:
            ch_picks = Ellipsis

        epochs_picks = this_picks['epochs']
        if epochs_picks is None:
            epochs_picks = Ellipsis

        this_psds = self.estimator.data_norm_[
            ..., start:end][:, ch_picks][epochs_picks]
        this_freqs = freqs[start:end]

        cumulative_spectra = np.cumsum(this_psds, axis=-1)
        idx = np.argmin((cumulative_spectra - self.percentile) ** 2, axis=-1)

        if this_psds.ndim > 2:
            data = np.zeros_like(idx, dtype=np.float)
            for iepoch in range(cumulative_spectra.shape[0]):
                data[iepoch] = freqs[idx[iepoch]]
        else:
            data = this_freqs[idx]
        return data


def read_psds(fname, estimators=None, comment='default'):
    return PowerSpectralDensitySummary._read(
        fname, estimators=estimators, comment=comment)
