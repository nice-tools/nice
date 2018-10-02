import numpy as np
import mne

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

from nice import Features
from nice.measures import (PowerSpectralDensity,
                           KolmogorovComplexity,
                           PermutationEntropy,
                           SymbolicMutualInformation,
                           PowerSpectralDensitySummary,
                           PowerSpectralDensityEstimator)


# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 0, 2
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)
raw.filter(1, 45)

mne.set_eeg_reference(raw, copy=False)

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

features = [
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
                         normalize=False, comment='summary_se'),
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
]

# prepare reductions
channels_fun = np.mean
epochs_fun = np.mean
reduction_params = {
    'PowerSpectralDensity': {
        'reduction_func': [
            {'axis': 'frequency', 'function': np.sum},
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'PowerSpectralDensitySummary': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'SymbolicMutualInformation': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun},
            {'axis': 'channels_y', 'function': channels_fun}]
    },
    'PermutationEntropy': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'KolmogorovComplexity': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    }
}

X = np.empty((len(epochs), len(features)))
for ii in range(len(epochs)):
    my_features = Features(features)
    my_features.fit(epochs[ii])
    X[ii, :] = my_features.reduce_to_scalar(measure_params=reduction_params)
    for measure in my_features.values():
        delattr(measure, 'data_')
    delattr(base_psd, 'data_')

y = epochs.events[:, 2] - 2

##############################################################################
# Original DOC-Forest recipe

n_estimators = 2000  # this will take some time.
doc_forest = make_pipeline(
    RobustScaler(),
    ExtraTreesClassifier(
        n_estimators=n_estimators, max_features=1, criterion='entropy',
        max_depth=4, random_state=42, class_weight='balanced'))

cv = GroupShuffleSplit(n_splits=50, train_size=0.8, test_size=0.2,
                       random_state=42)

aucs = cross_val_score(
    X=X, y=y, estimator=doc_forest,
    scoring='roc_auc', cv=cv, groups=np.arange(len(epochs)))


# Let's inspect variable importances, for convenience we fit in-sample.
# In the paper we sometimes looked at the distributions across CV-folds.

doc_forest.fit(X, y)
variable_importance = doc_forest.steps[-1][-1].feature_importances_
sorter = variable_importance.argsort()

# shorten the names a bit.
var_names = list(my_features.keys())
var_names = [var_names[ii].lstrip('nice/measure/') for ii in sorter]

# let's plot it
plt.figure(figsize=(8, 6))
plt.scatter(
    doc_forest.steps[-1][-1].feature_importances_[sorter],
    np.arange(17))
plt.yticks(np.arange(17), var_names)
plt.subplots_adjust(left=.46)
plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
plt.show()
