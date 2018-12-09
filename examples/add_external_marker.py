from collections import OrderedDict

from mne.externals.h5io import read_hdf5, write_hdf5
from mne.io.meas_info import Info
from mne.utils import _TempDir

from nice.markers.base import BaseMarker
from nice.collection import register_marker_class

from nice.markers import ContingentNegativeVariation
from nice.markers import PermutationEntropy

from nice import Markers, read_markers

from nice.tests.test_collection import _get_data


class MyCustomMarker(BaseMarker):
    def __init__(self, tmin=None, tmax=None, param1=None, param2=None,
                 method_params=None, comment='default'):
        # Call super constructor
        BaseMarker.__init__(self, tmin=None, tmax=None, comment=comment)

        # Custom marker parameters
        if method_params is None:
            method_params = {}
       
        self.method_params = method_params
        self.param1 = param1
        self.param2 = param2

    
    # MANDATORY: Axis map
    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0)
            ('channels', 1),
            ('times', 2)
            
        ])

    # MANDATORY: Override _get_title method to use the custom name
    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    # MANDATORY: Override _fit method to compute the marker
    def _fit(self, epochs):
        data = epochs.get_data()
        # Compute something
        self.data_ = data
    
    # MANDATORY: Save method should be overriden to use the custom title param
    def save(self, fname, overwrite=False):
        self._save_info(fname, overwrite=overwrite)
        save_vars = self._get_save_vars(exclude=['ch_info_'])
        write_hdf5(
            fname,
            save_vars,
            title=_get_title(self.__class__, self.comment),
            overwrite=overwrite, slash='replace')
    
    # MANDATORY: Read method should be implemented
    @classmethod
    def _read(cls, fname, comment='default'):
        return _read_my_marker(cls, fname=fname, comment=comment)


def _get_title(klass, comment):
    if issubclass(klass, BaseMarker):
        kind = 'marker'
    else:
        raise NotImplementedError('Oh no-- what is this?')
    # MANDATORY: Change the package of the title from nice to something else
    _title = '/'.join([
        'nice_custom_marker', kind, klass.__name__, comment])
    return _title


# MANDATORY: This method should work for any marker as it is now.
def _read_my_marker(klass, fname, comment='default'):
    data = read_hdf5(fname,  _get_title(klass, comment), slash='replace')
    init_params = {k: v for k, v in data.items() if not k.endswith('_')}
    attrs = {k: v for k, v in data.items() if k.endswith('_')}
    file_info = read_hdf5(fname, title='nice/data/ch_info', slash='replace')
    if 'filename' in file_info:
        del(file_info['filename'])
    attrs['ch_info_'] = Info(file_info)
    out = klass(**init_params)
    for k, v in attrs.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out


# Resgister the marker to NICE
register_marker_class(MyCustomMarker)

# Now you can create a collection with nice markers and the custom marker

markers_list = [
    PermutationEntropy(),
    ContingentNegativeVariation(),
    MyCustomMarker()
]

markers = Markers(markers_list)

# Fit on test data
epochs = _get_data()[:2]
markers.fit(epochs)

# Save to a file
tmp = _TempDir()
tmp_fname = tmp + '/test-markers.hdf5'
markers.save(tmp_fname)

# Read from file
markers2 = read_markers(tmp_fname)