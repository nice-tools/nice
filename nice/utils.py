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

import numpy as np
from scipy.io import loadmat
import h5py

from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.externals.h5io import write_hdf5

from six import string_types

def create_mock_data_egi(n_channels, n_samples, stim=True):
    """Load and configure testing data
    Parameters
    ----------
    n_channels : int
        The number of EEG channels.
    n_samples : int
        The number of time samples.
    stim : bool
        Whether to add a stim channel or not. Defaults to True.
    Returns
    -------
    raw : instance of mne.RawArry
        The testing data.
    """
    mat_contents = loadmat(
        op.join(op.realpath(op.dirname(__file__)),
                'tests', 'data', 'test-eeg.mat'))

    data = mat_contents['data'][:n_channels, :n_samples] * 1e-7
    sfreq = 250.
    if stim is True:
        ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
        ch_names += ['STI 014']
        ch_types = ['eeg'] * n_channels
        ch_types += ['stim']
        data = np.r_[data, data[-1:]]
        data[-1].fill(0)
    else:
        ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
        ch_types = ['eeg'] * n_channels

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=data, info=info)
    montage = make_standard_montage('GSN-HydroCel-257')
    raw.set_montage(montage)
    info['description'] = 'egi/256'
    return raw


def h5_listdir(fname, max_depth=4):

    datasets = list()

    def _get_dataset(name, member):
        if isinstance(member, h5py.Dataset):
            name_ = '/'.join(name.split('/', max_depth)[:-1])
            if name_ not in datasets:
                datasets.append(name_)

    with h5py.File(fname, 'r') as fid:
        fid.visititems(_get_dataset)

    return datasets


def info_to_dict(info):
    info_dict = {k: v for k, v in info.items()}
    info_dict['ch_names'] = list(info_dict['ch_names'])
    return info_dict


def write_hdf5_mne_epochs(fname, epochs, overwrite=False):
    epochs_vars = {k: v for k, v in vars(epochs).items() if
                   not k.startswith('_') or k == '_data'}
    epochs_vars['tmin'] = epochs.tmin
    epochs_vars['tmax'] = epochs.tmax
    epochs_vars['info'] = info_to_dict(epochs_vars['info'])
    write_hdf5(fname, epochs_vars, overwrite=overwrite,
               title='nice/data/epochs', slash='replace')


def mne_epochs_key_to_index(epochs, key):
    if isinstance(key, string_types):
            key = [key]

    if isinstance(key, (list, tuple)) and isinstance(key[0], string_types):
        if any('/' in k_i for k_i in epochs.event_id.keys()):
            if any(k_e not in epochs.event_id for k_e in key):
                # Select a given key if the requested set of
                # '/'-separated types are a subset of the types in that key
                key = [k for k in epochs.event_id.keys()
                       if any(set(k_i.split('/')).issubset(k.split('/'))
                              for k_i in key)]
                if len(key) == 0:
                    raise KeyError('Attempting selection of events via '
                                   'multiple/partial matching, but no '
                                   'event matches all criteria.')
        select = epochs._keys_to_idx(key)
    else:
        select = key if isinstance(key, slice) else np.atleast_1d(key)

    return select


def epochs_has_event(epochs, keys):
    event_id = epochs.event_id
    keys = [keys] if not isinstance(keys, (list, tuple)) else keys
    matches = []
    for key in keys:
        matches.extend(k for k in event_id.keys()
                       if set(key.split('/')).issubset(k.split('/')))
    return len(matches) > 0
