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
 
#ifndef __KOMPLEXITY_HH__
#define __KOMPLEXITY_HH__
/*
 * Calculate the complexity for all the channels and trials.
 *
 * data: array of double with size nchannels by nsamples by ntrials
 *       it should be stored in COLUMN MAJOR ORDER. So every sample in
 *       each channel is stored contigously.
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 *
 * results: array used to store the results.
 *          Size: ntrials * nchannels * sizeof(double)
 *
 * nbins: number of bins used to compute the symbolic transformation.
 */
int do_process_all_trials(
	double * data,
	long nsamples,
	long nchannels,
	long ntrials,
	double * results,
	long nbins,
	int nthreads
	);

#endif
