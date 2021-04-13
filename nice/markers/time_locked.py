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
from collections import Counter, OrderedDict

import numpy as np

from mne.utils import _time_mask
from mne.io.pick import pick_types

from .base import BaseMarker, BaseTimeLocked

from ..recipes.time_locked import (epochs_compute_cnv, cv_decode_sliding,
                                   decode_sliding, cv_decode_generalization,
                                   decode_generalization)
from ..utils import mne_epochs_key_to_index, epochs_has_event
from ..algorithms.decoding import decode_window


class ContingentNegativeVariation(BaseMarker):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)

    def _fit(self, epochs):
        slopes, intercepts = epochs_compute_cnv(epochs, self.tmin, self.tmax)
        self.data_ = slopes
        self.intercepts_ = intercepts

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1)
        ])


def read_cnv(fname, comment='default'):
    return ContingentNegativeVariation._read(fname, comment=comment)


class TimeLockedTopography(BaseTimeLocked):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, subset=None, missing_nan=False,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.subset = subset
        self.missing_nan = missing_nan

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('times', 2)
        ])

    def _prepare_data(self, picks, target):
        this_picks = {k: None for k in ['times', 'channels', 'epochs']}
        if picks is not None:
            if any([x not in this_picks.keys() for x in picks.keys()]):
                raise ValueError('Picking is not compatible for {}'.format(
                    self._get_title()))
        if picks is None:
            picks = {}
        this_picks.update(picks)
        to_preserve = self._get_preserve_axis(target)
        if len(to_preserve) > 0:
            for axis in to_preserve:
                this_picks[axis] = None

        # Pick Times based on original times
        time_picks = this_picks['times']
        time_mask = _time_mask(self.epochs_.times, self.tmin, self.tmax)
        if time_picks is not None:
            picks_mask = np.zeros(len(time_mask), dtype=np.bool)
            picks_mask[time_picks] = True
            time_mask = np.logical_and(time_mask, picks_mask)

        # Pick epochs based on original indices
        epochs_picks = this_picks['epochs']
        this_epochs = self.epochs_
        if epochs_picks is not None:
            this_epochs = this_epochs[epochs_picks]

        # Pick channels based on original indices
        ch_picks = this_picks['channels']
        if ch_picks is None:
            ch_picks = pick_types(this_epochs.info, eeg=True, meg=True)

        if (self.subset and self.missing_nan and not
                epochs_has_event(this_epochs, self.subset)):
            data = np.array([[[np.nan]]])
        else:
            if self.subset:
                this_epochs = this_epochs[self.subset]
            data = this_epochs.get_data()[:, ch_picks][..., time_mask]

        return data


def read_ert(fname, epochs, comment='default'):
    return TimeLockedTopography._read(fname, epochs=epochs, comment=comment)


class TimeLockedContrast(BaseTimeLocked):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, condition_a, condition_b, missing_nan=False,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.missing_nan = missing_nan

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('times', 2)
        ])

    def _reduce_to(self, reduction_func, target, picks):
        cont_list = list()
        for cond in [self.condition_a, self.condition_b]:
            ert = TimeLockedTopography(self.tmin, self.tmax, subset=cond,
                                       missing_nan=self.missing_nan)
            ert.fit(self.epochs_)
            cont_list.append(ert._reduce_to(reduction_func, target, picks))
        return cont_list[0] - cont_list[1]


def read_erc(fname, epochs, comment='default'):
    return TimeLockedContrast._read(fname, epochs=epochs, comment=comment)


class WindowDecoding(BaseMarker):
    def __init__(self, tmin, tmax, condition_a, condition_b,
                 decoding_params=None, comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        if decoding_params is None:
            decoding_params = dict(
                sample_weight='auto',
                n_jobs='auto',
                cv=None,
                clf=None,
                labels=None,
                random_state=None,
            )
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = self.decoding_params

        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(self.tmin, self.tmax)

        X, y, sample_weight = _prepare_decoding(
            epochs, self.condition_a, self.condition_b)

        X.reshape(len(y), -1)

        if dp['sample_weight'] not in ('auto', None):
            sample_weight = dp['sample_weight']

        probas, predictions, scores = decode_window(
            X, y, clf=dp['clf'], cv=dp['cv'],
            sample_weight=sample_weight, n_jobs=dp['n_jobs'],
            random_state=dp['random_state'], labels=dp['labels'])
        self.data_ = scores
        self.other_outputs_ = {'probas': probas, 'predictions': predictions}
        self.shape_ = self.data_.shape

    @property
    def _axis_map(self):
        return OrderedDict([
            ('folds', 0),
        ])


def read_wd(fname, comment='default'):
    return WindowDecoding._read(fname, comment=comment)


class TimeDecoding(BaseMarker):
    def __init__(self, tmin, tmax, condition_a, condition_b,
                 decoding_params=None, comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = self.decoding_params
        if dp is None:
            dp = {}
        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(self.tmin, self.tmax)

        if 'train_condition' in dp:
            train_cond = dp['train_condition']
            test_cond = dp['test_condition']
            X_train, y_train, _ = _prepare_decoding(
                epochs[train_cond], self.condition_a, self.condition_b
            )
            X_test, y_test, _ = _prepare_decoding(
                epochs[test_cond], self.condition_a, self.condition_b
            )
            dp = {k: v for k, v in dp.items()
                  if k not in ['train_condition', 'test_condition']}
            scores = decode_sliding(X_train, y_train, X_test, y_test, **dp)
        else:
            # Normal CV decoding
            X, y, sample_weight = _prepare_decoding(
                epochs, self.condition_a, self.condition_b)

            scores = cv_decode_sliding(X, y, **dp)

        self.data_ = np.array(scores)
        self.shape_ = self.data_.shape
        del epochs

    @property
    def _axis_map(self):
        return OrderedDict([
            ('folds', 0),
            ('times', 1)
        ])


def read_td(fname, comment='default'):
    return TimeDecoding._read(fname, comment=comment)


class GeneralizationDecoding(BaseMarker):
    def __init__(self, tmin, tmax, condition_a, condition_b,
                 decoding_params=None, comment='default'):
        BaseMarker.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = self.decoding_params
        if dp is None:
            dp = {}
        if self.tmin is not None or self.tmax is not None:
            epochs = epochs.copy().crop(self.tmin, self.tmax)

        if 'train_condition' in dp:
            train_cond = dp['train_condition']
            test_cond = dp['test_condition']
            X_train, y_train, _ = _prepare_decoding(
                epochs[train_cond], self.condition_a, self.condition_b
            )
            X_test, y_test, _ = _prepare_decoding(
                epochs[test_cond], self.condition_a, self.condition_b
            )
            dp = {k: v for k, v in dp.items()
                  if k not in ['train_condition', 'test_condition']}
            scores = decode_generalization(
                X_train, y_train,
                X_test, y_test, **dp)
        else:
            # Normal CV decoding
            X, y, sample_weight = _prepare_decoding(
                epochs, self.condition_a, self.condition_b)
            scores = cv_decode_generalization(X, y, **dp)

        self.data_ = np.array(scores)
        self.shape_ = self.data_.shape
        del epochs

    @property
    def _axis_map(self):
        return OrderedDict([
            ('folds', 0),
            ('train_times', 1),
            ('test_times', 2)
        ])


def read_gd(fname, comment='default'):
    return GeneralizationDecoding._read(fname, comment=comment)


def _prepare_sample_weights(epochs, condition_a_mask, condition_b_mask):
    count = Counter(epochs.events[:, 2])
    id_event = {v: k for k, v in epochs.event_id.items()}
    class_weights = {id_event[k]: 1. / v for k, v in count.items()}
    sample_weight = np.zeros(len(epochs.events), dtype=np.float)
    for k, v in epochs.event_id.items():
        this_index = epochs.events[:, 2] == v
        sample_weight[this_index] = class_weights[k]

    sample_weight_a = sample_weight[condition_a_mask]
    sample_weight_b = sample_weight[condition_b_mask]
    sample_weight = np.r_[sample_weight_b, sample_weight_a]
    return sample_weight


def _prepare_y(epochs, condition_a, condition_b):
    condition_a_mask = mne_epochs_key_to_index(epochs, condition_a)
    condition_b_mask = mne_epochs_key_to_index(epochs, condition_b)

    y = np.r_[np.zeros(condition_b_mask.shape[0]),
              np.ones(condition_a_mask.shape[0])]

    return y, condition_a_mask, condition_b_mask


def _prepare_decoding(epochs, condition_a, condition_b):
    y, condition_a_mask, condition_b_mask = _prepare_y(
        epochs, condition_a, condition_b)
    sample_weight = _prepare_sample_weights(
        epochs, condition_a_mask, condition_b_mask)
    X = np.concatenate([
        epochs.get_data()[condition_b_mask],
        epochs.get_data()[condition_a_mask]
    ])

    return X, y, sample_weight
