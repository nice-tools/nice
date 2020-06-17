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

# XXX: Copy of MNE welch method, since nperseg cannot be passed to scipy.signal
from mne.time_frequency.psd import _check_psd_data, _check_nfft
from mne.utils import logger
from mne.parallel import parallel_func


def _pwelch(epoch, nperseg, noverlap, nfft, fs, freq_mask, welch_fun):
    return [welch_fun(channel, nperseg=nperseg, noverlap=noverlap,
                      nfft=nfft, fs=fs, window='hamming')[1][..., freq_mask]
            for channel in epoch]


def _psd_welch(x, sfreq, fmin=0, fmax=np.inf, nperseg=256, n_fft=256,
               n_overlap=0, n_jobs=1):
    """Compute power spectral density (PSD) using Welch's method.

    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
    n_jobs : int
        Number of CPUs to use in the computation.

    Returns
    -------
    psds : ndarray, shape (..., n_freqs) or
        The power spectral densities. All dimensions up to the last will
        be the same as input.
    freqs : ndarray, shape (n_freqs,)
        The frequencies.
    """
    from scipy.signal import welch
    # import pdb; pdb.set_trace()
    dshape = x.shape[:-1]
    n_times = x.shape[-1]
    x = x.reshape(-1, n_times)

    if n_jobs == 'auto':
        try:
            import multiprocessing as mp
            mp.set_start_method('forkserver')
            n_jobs = mp.cpu_count()
            logger.info(
                'Autodetected number of jobs {}'.format(n_jobs))
        except Exception:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1

    # Prep the PSD
    # XXX: Dont use _check_nfft with n_fft but nperseg
    n_fft, nperseg, n_overlap = _check_nfft(n_times, n_fft, nperseg, n_overlap)
    win_size = n_fft / float(sfreq)
    logger.info("Effective window size : %0.3f (s)" % win_size)
    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Parallelize across first N-1 dimensions
    parallel, my_pwelch, n_jobs = parallel_func(_pwelch, n_jobs=n_jobs)
    x_splits = np.array_split(x, n_jobs)
    f_psd = parallel(my_pwelch(d, noverlap=n_overlap, nperseg=nperseg,
                     nfft=n_fft, fs=sfreq, freq_mask=freq_mask,
                     welch_fun=welch)
                     for d in x_splits)

    # Combining/reshaping to original data shape
    f_psd = [x for x in f_psd if len(x) != 0]
    psds = np.concatenate(f_psd, axis=0)
    psds = psds.reshape(np.hstack([dshape, -1]))
    return psds, freqs


def psd_welch(inst, fmin=0, fmax=np.inf, tmin=None, tmax=None, n_fft=256,
              n_overlap=0, nperseg=256, picks=None, proj=False, n_jobs='auto'):
    """Compute the power spectral density (PSD) using Welch's method.

    Calculates periodigrams for a sliding window over the
    time dimension, then averages them together for each channel/epoch.

    Parameters
    ----------
    inst : instance of Epochs or Raw or Evoked
        The data for PSD calculation
    fmin : float
        Min frequency of interest
    fmax : float
        Max frequency of interest
    tmin : float | None
        Min time of interest
    tmax : float | None
        Max time of interest
    n_fft : int
        The length of the tapers ie. the windows. The smaller
        it is the smoother are the PSDs. The default value is 256.
        If ``n_fft > len(inst.times)``, it will be adjusted down to
        ``len(inst.times)``.
    n_overlap : int
        The number of points of overlap between blocks. Will be adjusted
        to be <= n_fft. The default value is 0.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    picks : array-like of int | None
        The selection of channels to include in the computation.
        If None, take all channels.
    proj : bool
        Apply SSP projection vectors. If inst is ndarray this is not used.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    psds : ndarray, shape (..., n_freqs)
        The power spectral densities. If input is of type Raw,
        then psds will be shape (n_channels, n_freqs), if input is type Epochs
        then psds will be shape (n_epochs, n_channels, n_freqs).
    freqs : ndarray, shape (n_freqs,)
        The frequencies.

    See Also
    --------
    mne.io.Raw.plot_psd, mne.Epochs.plot_psd, psd_multitaper

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Prep data
    data, sfreq = _check_psd_data(inst, tmin, tmax, picks, proj)
    return _psd_welch(data, sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft,
                      n_overlap=n_overlap, nperseg=nperseg, n_jobs=n_jobs)
