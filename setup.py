#! /usr/bin/env python

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
# License version 3 without disclosing the source code of your own applications.
#

import os
from os import path as op

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup
from setuptools import find_packages

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('nice', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """NICE python project for MEG and EEG data analysis."""

DISTNAME = 'nice'
DESCRIPTION = descr
MAINTAINER = '@dengemann and @fraimondo'
MAINTAINER_EMAIL = 'federaimondo@gmail.com;denis.engemann@gmail.com'
URL = 'http://github.com/nice/nice'
LICENSE = 'GNU GPLv3'
DOWNLOAD_URL = 'http://github.com/nice-tools/nice'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=find_packages(
              exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
          # XXX put stuff here that's not gonna be shipped
          package_data={'nice': []},
          scripts=[])
