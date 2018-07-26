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

#include <smi.h>

#include <helpers.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define SECTIMER_START(name) \
		struct timeval name##_tstart; \
		struct timeval name##_tend; \
		time_t name##_msecs; \
		gettimeofday(&name##_tstart, NULL);

#define SECTIMER_END(name) \
		gettimeofday(&name##_tend, NULL); \
		{ time_t msecs = (((name##_tend.tv_usec < name##_tstart.tv_usec) ? 1000000 : 0 ) + name##_tend.tv_usec - name##_tstart.tv_usec)/1000;\
		msecs += (((name##_tend.tv_usec < name##_tstart.tv_usec) ? -1 : 0 ) +  name##_tend.tv_sec - name##_tstart.tv_sec) * 1000; \
		name##_msecs = msecs; }; \
		printf("Elapsed ms in %s = %lu\n", str(name), name##_msecs);

#define str(x) #x
#define xtr(x) str(x)

#define DPRINTSTRUCT(x) \
	printf("%s shape: ", str(x)); \
	{ \
	int i; \
	for (i = 0; i < x##_ndims; i++) { \
		printf("%ld (%ld) ", x##_dims[i], x##_strides[i]); \
	} \
	} \
	printf("\n"); \

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
	) {

	printf("SymbolicMutualInformation:: Running @(%p, %p) dims (%d, %d, %d) nsym %d and storing in %p and %p\n",
	(void *)data, (void *)count, nchannels, nsamples, ntrials, nsymbols, (void *)mi, (void *)wmi);

	/* Temporal working matrix for each thread*/
	double ** pxy = malloc(nthreads * sizeof(double *));
	int thread;
	for (thread = 0; thread < nthreads; thread++) {
		pxy[thread] = (double *)malloc(nsymbols * nsymbols * sizeof(double));
	}
	int sc1;
	int sc2;
	double w;
	double aux = 1.0;
	int channel1;
	int channel2;
	int symbol_ch1;
	int symbol_ch2;
	printf("Using %d threads to compute\n", nthreads);

	SECTIMER_START(smi)
	int trial = 0;

	/* for each trial (in parallel) */
	#pragma omp parallel for private(sc1, sc2, w, aux, channel1, channel2, symbol_ch1, symbol_ch2)
	for (trial = 0; trial < ntrials; trial++) {

		int thread_id = omp_get_thread_num();
		double * tpxy = pxy[thread_id];
		/* for each pair of channels */
		for (channel1 = 0; channel1 < nchannels; channel1++) {
			for (channel2 = channel1+1; channel2 < nchannels; channel2++) {
				/* erase the working variable */
				memset(tpxy, 0, nsymbols * nsymbols * sizeof(double));
				/* count the number of times each pair of symbol appears */
				for (int sample = 0; sample < nsamples; sample++) {
					sc1 = MAT3D(data, channel1, sample, trial, nchannels, nsamples);
					sc2 = MAT3D(data, channel2, sample, trial, nchannels, nsamples);
					tpxy[sc1 + sc2 * nsymbols]++;
				}
				/* for each pair of possible symbols */
				for (symbol_ch1 = 0; symbol_ch1 < nsymbols; symbol_ch1++) {
					for (symbol_ch2 = 0; symbol_ch2 < nsymbols; symbol_ch2++) {
						tpxy[symbol_ch1 + symbol_ch2 * nsymbols] = tpxy[symbol_ch1 + symbol_ch2 * nsymbols]/nsamples;
						w = MAT2D(wts, symbol_ch1, symbol_ch2, nsymbols);
						if (tpxy[symbol_ch1 + symbol_ch2 * nsymbols] > 0) {
							/* compute the value = jointprob(s1,s2) / prob(s1) / prob(s2) */
							aux = tpxy[symbol_ch1 + symbol_ch2 * nsymbols] *
								log(tpxy[symbol_ch1 + symbol_ch2 * nsymbols]/
									MAT3D(count, channel1, symbol_ch1, trial, nchannels, nsymbols)/
									MAT3D(count, channel2, symbol_ch2, trial, nchannels, nsymbols)
									);
							/* that value is smi*/
							MAT3D(mi, channel1, channel2, trial, nchannels, nchannels) += aux;
							/* the same value, but weighted, is wsmi */
							MAT3D(wmi, channel1, channel2, trial, nchannels, nchannels) += w*aux;
						}
					}
				}
				/* divide by log(nsymbols) after the pair of channels has been computed */
				MAT3D(mi, channel1, channel2, trial, nchannels, nchannels) /= log(nsymbols);
				MAT3D(wmi, channel1, channel2, trial, nchannels, nchannels) /= log(nsymbols);

			}
		}
	}
	SECTIMER_END(smi)
	for (int thread = 0; thread < nthreads; thread++) {
		free(pxy[thread]);
	}
	free(pxy);
	return 0;
}
