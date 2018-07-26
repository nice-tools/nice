/* NICE
 * Copyright (C) 2017 - Authors of NICE
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you
 * develop commercial activities as mentioned in the GNU Affero General Public
 * License version 3 without disclosing the source code of your own applications.
 */

#ifndef __SMI_H__
#define __SMI_H__

#include <types.h>

/*
 * Compute SMI and wSMI on symbolic transformed data.
 *
 * Uses OpenMP to parallelize across trials.
 *
 * data: input data, must be symbolic with values in the range [0, nsymbols-1]
 *		 nchannels by nsamples by ntrials C ordered matrix.
 *
 * count: the probability of each symbol.
 *        nchannels by nsymbols by ntrials C ordered matrix.
 *
 * wts: the weight matrix to use in the wSMI computation
 *		nsymbols by nsymbols C ordered matrix.
 *
 * mi: Symbolic Mutual Information result.
 *	   nchannels by nchannels by ntrials upper diagonal C ordered matrix.
 *
 * wmi: Weighted Symbolic Mutual Information result.
 *	   nchannels by nchannels by ntrials upper diagonal C ordered matrix.
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 * nsymbols: number of symbols in the data.
 * nthreads: amount of threads to use in the computation.
 */
int smi(
		int * data,
		double * count,
		double * wts,
		double * mi,
		double * wmi,
		int nchannels,
		int nsamples,
		int ntrials,
		int nsymbols,
		int nthreads
	);

#endif
