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

And principally, mne-python >= 0.13:
http://mne-tools.github.io/stable/index.html


Some functions require pandas >= 0.7.3.

To run the tests you will also need nose >= 0.10.

To use wSMI with CSD you need pycsd: https://github.com/nice-tools/pycsd/

Optimizations
^^^^^^^^^^^^^

Aditionally, we ship optimized versions of some algorithms. To build, just
go to the nice soure code directory and do::

    make

Running the test suite
^^^^^^^^^^^^^^^^^^^^^^

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.

Cite
^^^^

If you use this code in your project, please cite::

    *Denis Engemann, *Federico Raimondo, Jean-Remi King, Mainak Jas, Alexandre Gramfort, Stanislas Dehaene, Lionel Naccache, Jacobo Sitt
    "Automated Measurement and Prediction of Consciousness in Vegetative and Minimally Conscious Patients"
    in ICML Workshop on Statistics, Machine Learning and Neuroscience (Stamlins 2015)

Licensing
^^^^^^^^^

NICE is licensed under the GNU Affero General Public License version 3:

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2017, authors of NICE - All rights reserved.

    * This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    * This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    * You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    * You can be released from the requirements of the license by purchasing a
    commercial license. Buying such a license is mandatory as soon as you
    develop commercial activities as mentioned in the GNU Affero General Public
    License version 3 without disclosing the source code of your own
    applications.
