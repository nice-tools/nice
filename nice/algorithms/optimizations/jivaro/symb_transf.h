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

#ifndef __SYMB_TRANSF_H__
#define __SYMB_TRANSF_H__

#include <types.h>
#include <blocktrie.h>

void definePermutations(unsigned char *str, int len, trie_t * symbols);

extern int nsymbols;
extern int n_total_symbols;

/*
 * Generate a symbolic transformation on the data .
 *
 * data: input data, must be filtered with a lowpass at sampling_freq/kernel/tau.
 *		 nchannels by nsamples by ntrials C ordered matrix.
 *
 * kernel: number of samples to transform into a single symbol.
 *
 * tau: number of samples between each sample to account in the transformation.
 *
 * signal_symb_transf: the symbolic transformation returned.
 *			  nchannels by n_symbol_samples by ntrials C ordered matrix.
 *			  n_symbol_samples = nsamples - tau * (kernel - 1);
 *
 * count: the probability of each symbol.
 *        nchannels by nsymbols by ntrials C ordered matrix.
 *        nsymbols = kernel!
 *
 * nchannels: number of channels in the data.
 * nsamples: number of samples in the data.
 * ntrials: number of trials in the data.
 */
int symb_transf(
		double * data,
		int kernel,
		int tau,
		int * signal_symb_transf,
		double * count,
		int nchannels,
		int nsamples,
		int ntrials
	) ;

#endif
