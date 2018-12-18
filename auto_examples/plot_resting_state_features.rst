.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_resting_state_features.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_resting_state_features.py:


==================================================
Apply DOC-Forest recipe to single subject BCI data
==================================================

Here we use the resting state DOC-Forest [1] recipe to analyze BCI data.
Compared to the original reference, 2 major modifications are done.

1) For simplicity, we only compute 1 feature per marker, not 4
2) For speed, we use 200 trees, not 2000.

Compared to the common spatial patterns example form the MNE website,
the result is not particularly impressive. This is because a
global statistic like the mean or the standard deviation are a good
abstraction for severly brain injured patients but not for different
conditions in a BCI experiment conducted with healthy participants.

References
----------
.. [1] Engemann D.A.*, Raimondo F.*, King JR., Rohaut B., Louppe G.,
       Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
       Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
       Robust EEG-based cross-site and cross-protocol classification of
       states of consciousness. Brain. doi:10.1093/brain/awy251



.. code-block:: python


    # Authors: Denis A. Engemann <denis.engemann@gmail.com>
    #          Federico Raimondo <federaimondo@gmail.com>

    import numpy as np
    import mne

    from mne.io import concatenate_raws, read_raw_edf
    from mne.datasets import eegbci

    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import cross_val_score, GroupShuffleSplit
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline

    import matplotlib.pyplot as plt

    from nice import Markers
    from nice.markers import (PowerSpectralDensity,
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






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Extracting EDF parameters from /Users/dengeman/mne_data/MNE-eegbci-data/physiobank/database/eegmmidb/S001/S001R06.edf...
    EDF file detected
    EDF annotations detected (consider using raw.find_edf_events() to extract them)
    Setting channel info structure...
    Creating raw.info structure...
    Reading 0 ... 19999  =      0.000 ...   124.994 secs...
    Used Annotations descriptions: ['T0', 'T2', 'T1']
    Extracting EDF parameters from /Users/dengeman/mne_data/MNE-eegbci-data/physiobank/database/eegmmidb/S001/S001R10.edf...
    EDF file detected
    EDF annotations detected (consider using raw.find_edf_events() to extract them)
    Setting channel info structure...
    Creating raw.info structure...
    Reading 0 ... 19999  =      0.000 ...   124.994 secs...
    Used Annotations descriptions: ['T0', 'T1', 'T2']
    Extracting EDF parameters from /Users/dengeman/mne_data/MNE-eegbci-data/physiobank/database/eegmmidb/S001/S001R14.edf...
    EDF file detected
    EDF annotations detected (consider using raw.find_edf_events() to extract them)
    Setting channel info structure...
    Creating raw.info structure...
    Reading 0 ... 19999  =      0.000 ...   124.994 secs...
    Used Annotations descriptions: ['T0', 'T2', 'T1']
    Setting up band-pass filter from 1 - 45 Hz
    l_trans_bandwidth chosen to be 1.0 Hz
    h_trans_bandwidth chosen to be 11.2 Hz
    Filter length of 529 samples (3.306 sec) selected
    Setting up band-pass filter from 1 - 45 Hz
    l_trans_bandwidth chosen to be 1.0 Hz
    h_trans_bandwidth chosen to be 11.2 Hz
    Filter length of 529 samples (3.306 sec) selected
    Setting up band-pass filter from 1 - 45 Hz
    l_trans_bandwidth chosen to be 1.0 Hz
    h_trans_bandwidth chosen to be 11.2 Hz
    Filter length of 529 samples (3.306 sec) selected
    Applying average reference.
    Applying a custom EEG reference.
    Trigger channel has a non-zero initial value of 1 (consider using initial_event=True to detect this event)
    Removing orphaned offset at the beginning of the file.
    71 events found
    Event IDs: [1 2 3]
    45 matching events found
    No baseline correction applied
    Not setting metadata
    0 projection items activated
    Loading data for 45 events and 321 original time points ...
    0 bad epochs dropped


Prepare markers



.. code-block:: python


    backend = 'python'  # This gives maximum compatibility across platforms.
    # For improved speed, checkout the optimization options using C extensions.

    # We define one base estimator to avoid recomputation when looking up markers.
    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,
        psd_params=psds_params, comment='default')


    # Here are the resting-state compatible markers we considered in the paper.

    markers = Markers([
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
    ])







Prepare reductions.
Keep in mind that this is BCI, we have some localized effects.
Therefore we will consider the standard deviation acros channels.
Contraty to the paper, this is a single subject analysis. We therefore do
not pefrorm a full reduction but only compute one statistic
per marker and per epoch. In the paper, instead, we computed summaries over
epochs and sensosrs, yielding one value per marker per EEG recoding.



.. code-block:: python


    epochs_fun = np.mean
    channels_fun = np.std
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

    X = np.empty((len(epochs), len(markers)))
    for ii in range(len(epochs)):
        markers.fit(epochs[ii])
        X[ii, :] = markers.reduce_to_scalar(marker_params=reduction_params)
        # XXX hide this inside code
        for marker in markers.values():
            delattr(marker, 'data_')
        delattr(base_psd, 'data_')

    y = epochs.events[:, 2] - 2





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006327152252197266 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006224155426025391 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006200075149536133 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.005933046340942383 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006819009780883789 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007508993148803711 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006106853485107422 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007560014724731445 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006006002426147461 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006104946136474609 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006037235260009766 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006891012191772461 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006190061569213867 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006645917892456055 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.005964994430541992 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.00590205192565918 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006412029266357422 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006489992141723633 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006025075912475586 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006042003631591797 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006777048110961914 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006400108337402344 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.00613713264465332 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006067991256713867 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.005914926528930664 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.0064046382904052734 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006426095962524414 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006092071533203125 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006269931793212891 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007688045501708984 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006402015686035156 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.0059130191802978516 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007200956344604492 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006131887435913086 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006292819976806641 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006123065948486328 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006599903106689453 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006311893463134766 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.0068302154541015625 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006311893463134766 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007031917572021484 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006860256195068359 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006250858306884766 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.007226228713989258 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']
    Fitting nice/marker/PowerSpectralDensity/delta
    Autodetected number of jobs 8
    Effective window size : 25.600 (s)
    Fitting nice/marker/PowerSpectralDensity/deltan
    Fitting nice/marker/PowerSpectralDensity/theta
    Fitting nice/marker/PowerSpectralDensity/thetan
    Fitting nice/marker/PowerSpectralDensity/alpha
    Fitting nice/marker/PowerSpectralDensity/alphan
    Fitting nice/marker/PowerSpectralDensity/beta
    Fitting nice/marker/PowerSpectralDensity/betan
    Fitting nice/marker/PowerSpectralDensity/gamma
    Fitting nice/marker/PowerSpectralDensity/gamman
    Fitting nice/marker/PowerSpectralDensity/summary_se
    Fitting nice/marker/PowerSpectralDensitySummary/summary_msf
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef90
    Fitting nice/marker/PowerSpectralDensitySummary/summary_sef95
    Fitting nice/marker/PermutationEntropy/default
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Fitting nice/marker/SymbolicMutualInformation/weighted
    Autodetected number of jobs 2
    Bypassing CSD
    Filtering  at 6.67 Hz
    Performing symbolic transformation
    Running wsmi with python...
    Fitting nice/marker/KolmogorovComplexity/default
    Running KolmogorovComplexity
    Elapsed time 0.006193876266479492 sec
    Reducing to scalars
    Reducing nice/marker/PowerSpectralDensity/delta
    Reduction order for nice/marker/PowerSpectralDensity/delta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/deltan
    Reduction order for nice/marker/PowerSpectralDensity/deltan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/theta
    Reduction order for nice/marker/PowerSpectralDensity/theta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/thetan
    Reduction order for nice/marker/PowerSpectralDensity/thetan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alpha
    Reduction order for nice/marker/PowerSpectralDensity/alpha: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/alphan
    Reduction order for nice/marker/PowerSpectralDensity/alphan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/beta
    Reduction order for nice/marker/PowerSpectralDensity/beta: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/betan
    Reduction order for nice/marker/PowerSpectralDensity/betan: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamma
    Reduction order for nice/marker/PowerSpectralDensity/gamma: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/gamman
    Reduction order for nice/marker/PowerSpectralDensity/gamman: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensity/summary_se
    Reduction order for nice/marker/PowerSpectralDensity/summary_se: ['frequency', 'epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_msf
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_msf: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef90
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef90: ['epochs', 'channels']
    Reducing nice/marker/PowerSpectralDensitySummary/summary_sef95
    Reduction order for nice/marker/PowerSpectralDensitySummary/summary_sef95: ['epochs', 'channels']
    Reducing nice/marker/PermutationEntropy/default
    Reduction order for nice/marker/PermutationEntropy/default: ['epochs', 'channels']
    Reducing nice/marker/SymbolicMutualInformation/weighted
    Reduction order for nice/marker/SymbolicMutualInformation/weighted: ['epochs', 'channels', 'channels_y']
    Reducing nice/marker/KolmogorovComplexity/default
    Reduction order for nice/marker/KolmogorovComplexity/default: ['epochs', 'channels']


Original DOC-Forest recipe



.. code-block:: python


    # NOTE: It was 2000 in the paper. Bnut we want to save time.
    n_estimators = 200
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







Inspect variable importances
We will use, for convenience, the in-sample fit.
In the paper we sometimes looked at the distributions across CV-folds.



.. code-block:: python



    doc_forest.fit(X, y)
    variable_importance = doc_forest.steps[-1][-1].feature_importances_
    sorter = variable_importance.argsort()

    # shorten the names a bit.
    var_names = list(markers.keys())
    var_names = [var_names[ii].lstrip('nice/marker/') for ii in sorter]

    # let's plot it
    plt.figure(figsize=(8, 6))
    plt.scatter(
        doc_forest.steps[-1][-1].feature_importances_[sorter],
        np.arange(17))
    plt.yticks(np.arange(17), var_names)
    plt.subplots_adjust(left=.46)
    plt.title('AUC={:0.3f}'.format(np.mean(aucs)))
    plt.show()



.. image:: /auto_examples/images/sphx_glr_plot_resting_state_features_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  33.522 seconds)


.. _sphx_glr_download_auto_examples_plot_resting_state_features.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_resting_state_features.py <plot_resting_state_features.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_resting_state_features.ipynb <plot_resting_state_features.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
