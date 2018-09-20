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
import time
import numpy as np
import zlib

from mne import pick_types
from mne.utils import logger, _time_mask


def epochs_compute_komplexity(epochs, nbins, tmin=None, tmax=None,
                              backend='python', method_params=None):
    """Compute complexity (K)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    nbins : int
        Number of bins to use for symbolic transformation
    method_params : dictionary.
        Overrides default parameters for the backend used.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'python'.
    """
    picks = pick_types(epochs.info, meg=True, eeg=True)

    if method_params is None:
        method_params = {}

    data = epochs.get_data()[:, picks if picks is not None else Ellipsis]
    time_mask = _time_mask(epochs.times, tmin, tmax)
    data = data[:, :, time_mask]
    logger.info("Running KolmogorovComplexity")

    if backend == 'python':
        start_time = time.time()
        komp = _komplexity_python(data, nbins)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    elif backend == 'openmp':
        from ..optimizations.ompk import komplexity as _ompk_k
        nthreads = (method_params['nthreads']
                    if 'nthreads' in method_params else 1)
        if nthreads == 'auto':
            try:
                import mkl
                nthreads = mkl.get_max_threads()
                logger.info(
                    'Autodetected number of threads {}'.format(nthreads))
            except:
                logger.info('Cannot autodetect number of threads')
                nthreads = 1
        start_time = time.time()
        komp = _ompk_k(data, nbins, nthreads)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    else:
        raise ValueError('backend %s not supported for KolmogorovComplexity'
                         % backend)
    return komp


def _symb_python(signal, nbins):
    """Compute symbolic transform"""
    ssignal = np.sort(signal)
    items = signal.shape[0]
    first = int(items / 10)
    last = items - first if first > 1 else items - 1
    lower = ssignal[first]
    upper = ssignal[last]
    bsize = (upper - lower) / nbins

    osignal = np.zeros(signal.shape, dtype=np.uint8)
    maxbin = nbins - 1

    for i in range(items):
        tbin = int((signal[i] - lower) / bsize)
        osignal[i] = ((0 if tbin < 0 else maxbin
                       if tbin > maxbin else tbin) + ord('A'))

    return osignal.tostring()


def _komplexity_python(data, nbins):
    """Compute komplexity (K)"""
    ntrials, nchannels, nsamples = data.shape
    k = np.zeros((nchannels, ntrials), dtype=np.float64)
    for trial in range(ntrials):
        for channel in range(nchannels):
            string = _symb_python(data[trial, channel, :], nbins)
            cstring = zlib.compress(string)
            k[channel, trial] = float(len(cstring)) / float(len(string))

    return k
