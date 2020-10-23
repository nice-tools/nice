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
from ..utils import write_hdf5_mne_epochs, info_to_dict
import numpy as np

from mne.utils import logger
from mne.epochs import _compare_epochs_infos
from mne.io.meas_info import Info
from mne.externals.h5io import write_hdf5, read_hdf5
import h5py


class BaseContainer(object):
    def __init__(self, comment):
        self.comment = comment

    def _save_info(self, fname, overwrite=False):
        has_ch_info = False
        if fname.exists():
            with h5py.File(fname, 'r') as h5fid:
                if 'nice/data/ch_info' in h5fid:
                    has_ch_info = True
                    logger.info('Channel info already present in HDF5 file, '
                                'will not be overwritten')

        if not has_ch_info:
            logger.info('Writing channel info to HDF5 file')
            write_hdf5(fname, info_to_dict(self.ch_info_),
                       title='nice/data/ch_info',
                       overwrite=overwrite, slash='replace')

    def _get_save_vars(self, exclude):
        return {k: v for k, v in vars(self).items() if
                k not in exclude}

    def save(self, fname, overwrite=False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(exclude=['ch_info_'])
        write_hdf5(
            fname,
            save_vars,
            title=_get_title(self.__class__, self.comment),
            overwrite=overwrite, slash='replace')

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    @classmethod
    def _read(cls, fname, comment='default'):
        return _read_container(cls, fname, comment=comment)


class BaseMarker(BaseContainer):
    """Base class for M/EEG markers"""

    def __init__(self, tmin, tmax, comment):
        BaseContainer.__init__(self, comment=comment)
        self.tmin = tmin
        self.tmax = tmax

    @property
    def _axis_map(self):
        raise NotImplementedError('This should be in every marker')

    def fit(self, epochs):
        self._fit(epochs)
        self.ch_info_ = epochs.info
        return self

    def transform(self, epochs):
        self._transform(epochs)
        return self

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    def _get_preserve_axis(self, targets):
        to_preserve = []
        if not isinstance(targets, list):
            targets = [targets]
        for elem in targets:
            if 'topography' == elem or 'channels' == elem:
                to_preserve.append('channels')
            elif 'times' == elem:
                to_preserve.append('times')
            elif 'epochs' == elem:
                to_preserve.append('epochs')
        if any(x not in self._axis_map.keys() for x in to_preserve):
            raise ValueError('Cannot reduce {} to {}'.format(
                self._get_title(), targets))
        return to_preserve

    def _reduce_to(self, reduction_func, target, picks):
        if not hasattr(self, 'data_'):
            raise ValueError('You did not fit me. Do it again after fitting '
                             'some data!')
        out, funcs = self._prepare_reduction(reduction_func, target, picks)
        for func in funcs:
            out = func(out, axis=0)
        return out

    def reduce_to_epochs(self, reduction_func, picks=None):
        """Reduce  marker to a single value per epoch.

        Parameters
        ----------
        reduction_func : list of dictionaries.
            Each dictionary should have two keys: 'axis' and 'function'.
            The marker is going to be reduced following the order of the list.
            Selecting the corresponding axis and applying the corresponding
            function.
        picks : dictionary of axis to array.
            Before appling the reduction function, the corresponding axis will
            be subselected by picks. A value of None indicates all the
            elements.

        Example:
            reduction_func = [
                {'axis': 'frequency', 'function': np.sum},
                {'axis': 'channels', 'function': np.mean},
                {'axis': 'epochs', 'function': np.mean}]
            picks = {'epochs': None, 'channels': np.arange(224)}

        Returns
        -------
        out : np.ndarray of float, shape(n_epochs,)
            The value of the marker for each epoch.
        """
        return self._reduce_to(
            reduction_func, target='epochs', picks=picks)

    def reduce_to_topo(self, reduction_func, picks=None):
        return self._reduce_to(
            reduction_func, target='topography', picks=picks)

    def reduce_to_scalar(self, reduction_func, picks=None):
        return self._reduce_to(reduction_func, target='scalar', picks=picks)

    def compress(self, reduction_func):
        if not hasattr(self, 'data_'):
            raise ValueError('You did not fit me. Do it again after fitting '
                             'some data!')
        if 'epochs' in self._axis_map:
            axis = self._axis_map['epochs']
            logger.info(
                'Compressing {} on axis {} (epochs)'.format(
                    self._get_title(), axis)
            )
            data = reduction_func(self.data_, axis=axis)
            # Keep dimension
            self.data_ = np.expand_dims(data, axis=axis)

    def _prepare_data(self, picks, target):
        data = self.data_
        to_preserve = self._get_preserve_axis(target)
        if picks is not None:
            if any([x not in self._axis_map for x in picks.keys()]):
                raise ValueError('Picking is not compatible for {}'.format(
                    self._get_title()))
            for axis, ax_picks in picks.items():
                if axis in to_preserve:
                    continue
                if ax_picks is not None:
                    this_axis = self._axis_map[axis]
                    data = (data.swapaxes(this_axis, 0)[ax_picks, ...]
                                .swapaxes(0, this_axis))
        return data

    def _prepare_reduction(self, reduction_func, target, picks,
                           return_axis=False):
        data = self._prepare_data(picks, target)
        _axis_map = self._axis_map
        funcs = list()
        axis_to_preserve = self._get_preserve_axis(target)
        if len(axis_to_preserve) > 0:
            removed_axis = []
            for this_axis_to_preserve in axis_to_preserve:
                removed_axis.append(_axis_map.pop(this_axis_to_preserve))
            if reduction_func is not None:
                reduction_func = [i for i in reduction_func
                                  if i['axis'] not in axis_to_preserve]
        permutation_list = list()
        permutation_axes = list()
        if reduction_func is None:
            for ax_name, remaining_axis in _axis_map.items():
                permutation_list.append(remaining_axis)
                permutation_axes.append(ax_name)
                funcs.append(np.mean)
        elif len(reduction_func) == len(_axis_map):
            for rec in reduction_func:
                this_axis = _axis_map.pop(rec['axis'])
                permutation_axes.append(rec['axis'])
                permutation_list.append(this_axis)
                funcs.append(rec['function'])
        else:
            raise ValueError('Run `python -c "import this"` to see '
                             'why we will not tolerate these things')
        if len(axis_to_preserve) > 0:
            permutation_list += removed_axis
        logger.info('Reduction order for {}: {}'.format(
            self._get_title(), permutation_axes))
        data = np.transpose(data, permutation_list)
        if return_axis is False:
            out = data, funcs
        else:
            out = data, funcs, permutation_axes
        return out


class BaseTimeLocked(BaseMarker):

    def __init__(self, tmin, tmax, comment):
        BaseMarker.__init__(self, tmin, tmax, comment)

    def fit(self, epochs):
        self.ch_info_ = epochs.info
        self.shape_ = epochs.get_data().shape
        self.epochs_ = epochs
        self.data_ = epochs.get_data()
        return self

    def compress(self, reduction_func):
        logger.info(
            'TimeLocked markers cannot be compressed '
            'epoch-wise ({})'.format(self._get_title()))

    def save(self, fname, overwrite=False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(
            exclude=['ch_info_', 'data_', 'epochs_'])

        has_epochs = False
        with h5py.File(fname, 'r') as h5fid:
            if 'nice/data/epochs' in h5fid:
                has_epochs = True
                logger.info('Epochs already present in HDF5 file, '
                            'will not be overwritten')

        if not has_epochs:
            epochs = self.epochs_
            logger.info('Writing epochs to HDF5 file')
            write_hdf5_mne_epochs(fname, epochs, overwrite=overwrite)
        write_hdf5(
            fname, save_vars, overwrite=overwrite,
            title=_get_title(self.__class__, self.comment), slash='replace')

    @classmethod
    def _read(cls, fname, epochs, comment='default'):
        return _read_time_locked(cls, fname=fname, epochs=epochs,
                                 comment=comment)

    def _get_title(self):
        return _get_title(self.__class__, self.comment)


class BaseDecoding(BaseMarker):
    def __init__(self, tmin, tmax, comment):
        BaseMarker.__init__(self, tmin, tmax, comment)

    def fit(self, epochs):
        self._fit(epochs)
        return self

    def _get_title(self):
        return _get_title(self.__class__, self.comment)


def _get_title(klass, comment):
    if issubclass(klass, BaseMarker):
        kind = 'marker'
    elif issubclass(klass, BaseContainer):
        kind = 'container'
    else:
        raise NotImplementedError('Oh no-- what is this?')
    return '/'.join([
        'nice', kind, klass.__name__, comment])


def _read_container(klass, fname, comment='default'):
    data = read_hdf5(fname,  _get_title(klass, comment), slash='replace')
    init_params = {k: v for k, v in data.items() if not k.endswith('_')}
    attrs = {k: v for k, v in data.items() if k.endswith('_')}
    file_info = read_hdf5(fname, title='nice/data/ch_info', slash='replace')
    if 'filename' in file_info:
        del(file_info['filename'])
    attrs['ch_info_'] = Info(file_info)
    out = klass(**init_params)
    for k, v in attrs.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out


def _check_epochs_consistency(info1, info2, shape1, shape2):
    _compare_epochs_infos(info1, info2, 2)
    np.testing.assert_array_equal(shape1, shape2)


def _read_time_locked(cls, fname, epochs, comment='default'):
    out = _read_container(cls, fname, comment=comment)
    shape1 = epochs.get_data().shape
    shape2 = out.shape_
    _check_epochs_consistency(out.ch_info_, epochs.info, shape1, shape2)
    out.epochs_ = epochs
    out.data_ = epochs.get_data()
    return out
