import numpy as np

from nice.utils import create_mock_data_egi
from nice.algorithms.decoding import decode_window

import mne

n_epochs = 30
raw = create_mock_data_egi(6, n_epochs * 386, stim=True)

triggers = np.arange(50, n_epochs * 386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * int(n_epochs / 2) + [20] * int(n_epochs / 2)

events = mne.find_events(raw)
event_id = {
    'foo': 10,
    'bar': 20,
}
epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                    preload=True, reject=None, picks=None,
                    baseline=(None, 0), verbose=False)
epochs.drop_channels(['STI 014'])
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')


def test_decode_window():
    X = epochs.get_data()
    y = epochs.events[:, 2]
    probas, predictions, scores = decode_window(X, y)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
