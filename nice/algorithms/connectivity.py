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

import mne
from mne.utils import logger, _time_mask
from scipy.signal import butter, filtfilt


def _get_weights_matrix(nsym):
    """Aux function"""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


def epochs_compute_wsmi(epochs, kernel, tau, tmin=None, tmax=None,
                        backend='python', method_params=None, n_jobs='auto'):
    """Compute weighted mutual symbolic information (wSMI)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    method_params : dictionary.
        Overrides default parameters.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'pytho'.
    """
    if method_params is None:
        method_params = {}

    if n_jobs == 'auto':
        try:
            import multiprocessing as mp
            mp.set_start_method('forkserver')
            import mkl
            n_jobs = int(mp.cpu_count() / mkl.get_max_threads())
            logger.info(
                'Autodetected number of jobs {}'.format(n_jobs))
        except Exception:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1

    if 'bypass_csd' in method_params and method_params['bypass_csd'] is True:
        logger.info('Bypassing CSD')
        csd_epochs = epochs
        picks = mne.io.pick.pick_types(csd_epochs.info, meg=True, eeg=True)
    else:
        logger.info('Computing CSD')
        # try:
        #     from pycsd import epochs_compute_csd
        # except Exception:
        #     raise ValueError('PyCSD not available. '
        #                      'Please install this dependency.')
        # csd_epochs = epochs_compute_csd(epochs, n_jobs=n_jobs)
        csd_epochs = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=1e-5)
        picks = mne.io.pick.pick_types(csd_epochs.info, csd=True)

    freq = csd_epochs.info['sfreq']

    data = csd_epochs.get_data()[:, picks, ...]
    n_epochs = len(data)

    if 'filter_freq' in method_params:
        filter_freq = method_params['filter_freq']
    else:
        filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')
    data = np.hstack(data)

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]
    if backend == 'python':
        from .information_theory.permutation_entropy import _symb_python
        logger.info("Performing symbolic transformation")
        sym, count = _symb_python(fdata, kernel, tau)
        nsym = count.shape[1]
        wts = _get_weights_matrix(nsym)
        logger.info("Running wsmi with python...")
        wsmi, smi = _wsmi_python(sym, count, wts)
    elif backend == 'openmp':
        from .optimizations.jivaro import wsmi as jwsmi
        nsym = np.math.factorial(kernel)
        wts = _get_weights_matrix(nsym)
        nthreads = (method_params['nthreads'] if 'nthreads' in
                    method_params else 1)
        if nthreads == 'auto':
            try:
                import mkl
                nthreads = mkl.get_max_threads()
                logger.info(
                    'Autodetected number of threads {}'.format(nthreads))
            except Exception:
                logger.info('Cannot autodetect number of threads')
                nthreads = 1
        wsmi, smi, sym, count = jwsmi(fdata, kernel, tau, wts, nthreads)
    else:
        raise ValueError('backend %s not supported for wSMI'
                         % backend)

    return wsmi, smi, sym, count


def _wsmi_python(data, count, wts):
    """Compute wsmi"""
    nchannels, nsamples, ntrials = data.shape
    nsymbols = count.shape[1]
    smi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    for trial in range(ntrials):
        for channel1 in range(nchannels):
            for channel2 in range(channel1 + 1, nchannels):
                pxy = np.zeros((nsymbols, nsymbols))
                for sample in range(nsamples):
                    pxy[data[channel1, sample, trial],
                        data[channel2, sample, trial]] += 1
                pxy = pxy / nsamples
                for sc1 in range(nsymbols):
                    for sc2 in range(nsymbols):
                        if pxy[sc1, sc2] > 0:
                            aux = pxy[sc1, sc2] * np.log(
                                pxy[sc1, sc2] /  # noqa
                                count[channel1, sc1, trial] /  # noqa
                                count[channel2, sc2, trial])
                            smi[channel1, channel2, trial] += aux
                            wsmi[channel1, channel2, trial] += \
                                (wts[sc1, sc2] * aux)
    wsmi = wsmi / np.log(nsymbols)
    smi = smi / np.log(nsymbols)
    return wsmi, smi
