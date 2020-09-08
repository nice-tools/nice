.. -*- mode: rst -*-

`NICE Tools`
=======================================================

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git://github.com/nice-tools/nice.git

If you don't have git installed, you can download a zip or tarball
of the latest code: https://github.com/nice-tools/nice/archives/master

Install nice
^^^^^^^^^^^^^^^^^^

As any Python packages, to install NICE, go in the nice source
code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

You can also install the latest latest development version with pip::

    pip install -e git+https://github.com/nice-tools/nice#egg=nice-dev --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are:
* python >= 2.7 | python >= 3.4
* scipy >= 0.18.1
* numpy >= 1.11.1
* h5py >= 2.6.0

And principally, mne-python >= 0.20:
http://mne-tools.github.io/stable/index.html


Some functions require pandas >= 0.7.3.

To run the tests you will also need nose >= 0.10.

Optimizations (optional)
^^^^^^^^^^^^^^^^^^^^^^^^

Aditionally, we ship optimized versions of some algorithms.
They can be compiled using the conda CLANG and openmp.
Both can be installed as follows::

    conda install clang
    conda install openmp


To build, go to the nice soure code directory and do::

    CC=clang make

Then set backend='c' or 'openmp' instead of the defalut backend='python' in markers functions

If running on osx 10.15, then you might face this error::

    /anaconda3/include/python3.7m/Python.h:25:10: fatal error: 'stdio.h' file not found

Running this command will fix it::

    sudo ln -s /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/* /usr/local/include/


Running the test suite
^^^^^^^^^^^^^^^^^^^^^^

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.

Cite
^^^^

If you use this code in your project, please cite::

    Engemann D.A.*, Raimondo F.*, King JR., Rohaut B., Louppe G.,
    Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
    Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
    Robust EEG-based cross-site and cross-protocol classification of
    states of consciousness. Brain. Vol 141 (11), 3160–3178, doi:10.1093/brain/awy251

Licensing
^^^^^^^^^

NICE is licensed under the GNU Affero General Public License version 3:

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2017, authors of NICE - All rights reserved.

    * This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

    * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

    * You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

    * You can be released from the requirements of the license by purchasing a commercial license. Buying such a license is mandatory as soon as you develop commercial activities as mentioned in the GNU Affero General Public License version 3 without disclosing the source code of your own applications.
