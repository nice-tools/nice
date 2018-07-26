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
.PHONY: clean

MODULES=jivaro.so
PYTHON?=/usr/bin/env python
SOURCES=blocktrie.h blocktrie.c smi.h smi.c symb_transf.h symb_transf.c helpers.h types.h Makefile

all: $(MODULES)

%.so: %.c $(SOURCES)
	$(PYTHON) setup.py build
	cp build/lib.*/*.so $@
	cp $@ ../

clean:
	rm -rf build
	rm -f $(MODULES)

MKLROOT=/opt/intel/mkl
PLATFORM = $(shell gcc -dumpmachine)
UNAME = $(shell uname)
MACHINE = $(shell uname -m)

ifeq "$(UNAME)" "Darwin"
	CC?=icc
	CFLAGS = -Wl,-no_pie
	LDFLAGS= $(MKLROOT)/lib/libmkl_intel_lp64.a $(MKLROOT)/lib/libmkl_core.a $(MKLROOT)/lib/libmkl_sequential.a  -openmp-link static
endif
ifeq "$(UNAME)" "Linux"
	CFLAGS = -fPIC
	LDFLAGS= $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_sequential.a  -openmp-link static
endif

OBJS=symb_transf.o blocktrie.o smi.o
INCLUDE=-I./ -I $(MKLROOT)/include -I ../
CFLAGS +=-openmp -Wall -pedantic -O3 -m64 $(INCLUDE) -std=c99

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: test_perm.c $(OBJS)
	$(CC) $(INCLUDE) $(LDFLAGS) $(CFLAGS) $^ -o $@
