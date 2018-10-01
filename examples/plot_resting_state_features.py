import mne

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from nice import Features
from nice.measures import (PowerSpectralDensity, ContingentNegativeVariation,
                           KolmogorovComplexity, PermutationEntropy,
                           SymbolicMutualInformation, TimeLockedTopography,
                           TimeLockedContrast, PowerSpectralDensitySummary,
                           WindowDecoding, PowerSpectralDensityEstimator)


# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = mne.Epochs(
    raw, events, event_id, tmin, tmax, proj=True, picks=picks,
    baseline=None, preload=True)

psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)

base_psd = PowerSpectralDensityEstimator(
    psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,
    psd_params=psds_params, comment='default')

backend = 'python'

features = Features([
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=False, comment='delta'),
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=True, comment='deltan'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=False, comment='theta'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=True, comment='thetan'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=False, comment='alpha'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=True, comment='alphan'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=False, comment='beta'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=True, comment='betan'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=False, comment='gamma'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=True, comment='gamman'),

    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                         normalize=True, comment='summary_se'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.5, comment='summary_msf'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.9, comment='summary_sef90'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.95, comment='summary_sef95'),

    PermutationEntropy(tmin=None, tmax=0.6, backend=backend),

    # csd needs to be skipped
    SymbolicMutualInformation(
        tmin=None, tmax=0.6, method='weighted', backend=backend,
        method_params={'nthreads': 'auto', 'bypass_csd': True},
        comment='weighted'),

    KolmogorovComplexity(tmin=None, tmax=0.6, backend=backend,
                         method_params={'nthreads': 'auto'}),
])

features.fit(epochs)


