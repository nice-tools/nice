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
from scipy import linalg

import mne
from mne.io.pick import _picks_by_type, pick_info
from mne.defaults import _handle_default
from mne.utils import _time_mask


def epochs_compute_cnv(epochs, tmin=None, tmax=None):
    """Compute contingent negative variation (CNV)

    Parameters
    ----------
    epochs : instance of Epochs
        The input data.
    tmin : float | None
        The first time point to include, if None, all samples form the first
        sample of the epoch will be used. Defaults to None.
    tmax : float | None
        The last time point to include, if None, all samples up to the last
        sample of the epoch wi  ll be used. Defaults to None.
    return_epochs : bool
        Whether to compute an average or not. If False, data will be
        averaged and put in an Evoked object. Defaults to False.

    Returns
    -------
    cnv : ndarray of float (n_channels, n_epochs) | instance of Evoked
        The regression slopes (betas) represewnting contingent negative
        variation.
    """
    picks = mne.pick_types(epochs.info, meg=True, eeg=True)
    n_epochs = len(epochs.events)
    n_channels = len(picks)
    # we reduce over time samples
    slopes = np.zeros((n_epochs, n_channels))
    intercepts = np.zeros((n_epochs, n_channels))
    if tmax is None:
        tmax = epochs.times[-1]
    if tmin is None:
        tmin = epochs.times[0]

    fit_range = np.where(_time_mask(epochs.times, tmin, tmax))[0]

    # design: intercept + increasing time
    design_matrix = np.c_[np.ones(len(fit_range)),
                          epochs.times[fit_range] - tmin]

    # estimate single trial regression over time samples
    scales = np.zeros(n_channels)
    info_ = pick_info(epochs.info, picks)
    for this_type, this_picks in _picks_by_type(info_):
        scales[this_picks] = _handle_default('scalings')[this_type]

    for ii, epoch in enumerate(epochs):
        y = epoch[picks][:, fit_range].T  # time is samples
        betas, _, _, _ = linalg.lstsq(a=design_matrix, b=y * scales)
        intercepts[ii] = betas[0]
        slopes[ii] = betas[1]

    return slopes, intercepts
