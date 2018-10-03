"""

==================================================
Compute features used for DOC-Forest recipe
==================================================

Here we compute the features from previously computed markers as published [1].

For simplicity, we only compute scalars using a trimmed mean (80%) accross
epochs and the mean across channels.

References
----------
[1] Engemann D.A., Raimondo F., King JR., Rohaut B., Louppe G., Faugeras F.,
    Annen J., Cassol H., Gosseries O., Fernandez-Slezak D., Laureys S.,
    Naccache L., Dehaene S. and Sitt J.D. (in press).
    Robust EEG-based cross-site and cross-protocol classification of
    states of consciousness (in press). Brain.
"""

# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Federico Raimondo <federaimondo@gmail.com>

import numpy as np
from scipy.stats import trim_mean

from nice.features import read_features


def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)


def entropy(a, axis=0):
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])


fc = read_features('JSXXX-features.hdf5')

reduction_params = {}
scalp_roi = np.arange(224)
cnv_roi = np.array([5,  6, 13, 14, 15, 21, 22])
mmn_roi = np.array([5,   6,   8,  13,  14,  15,  21,  22,  44,  80, 131, 185])
p3b_roi = np.array([8,  44,  80,  99, 100, 109, 118, 127, 128, 131, 185])
p3a_roi = np.array([5,   6,   8,  13,  14,  15,  21,  22,  44,  80, 131, 185])

channels_fun = np.mean
epochs_fun = trim_mean80

reduction_params['PowerSpectralDensity'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'frequency', 'function': np.sum}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi}}

reduction_params['PowerSpectralDensity/summary_se'] = {
    'reduction_func':
        [{'axis': 'frequency', 'function': entropy},
         {'axis': 'epochs', 'function': np.mean},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi}}

reduction_params['PowerSpectralDensitySummary'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi}}

reduction_params['PermutationEntropy'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi}}

reduction_params['SymbolicMutualInformation'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels_y', 'function': np.median},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels_y': scalp_roi,
        'channels': scalp_roi}}

reduction_params['KolmogorovComplexity'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi}}

reduction_params['ContingentNegativeVariation'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': cnv_roi}}

reduction_params['TimeLockedTopography'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'times', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi,
        'times': None}}

reduction_params['TimeLockedContrast'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'times', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': scalp_roi,
        'times': None}}

reduction_params['TimeLockedContrast/mmn'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'times', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': mmn_roi,
        'times': None}}

reduction_params['TimeLockedContrast/p3b'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'times', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': p3b_roi,
        'times': None}}

reduction_params['TimeLockedContrast/p3a'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'times', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': p3a_roi,
        'times': None}}

scalars = fc.reduce_to_scalar(reduction_params)
