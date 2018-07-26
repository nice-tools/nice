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
 
#include <komplexity.h>
#include <zlib.h>
#include <helpers.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SET_BINARY_MODE(file)


/* Following function is needed for library function qsort(). Refer
   http://www.cplusplus.com/reference/clibrary/cstdlib/qsort/ */
int compare (const void *a, const void * b) {
	if (*(double*)a > *(double*)b) return 1;
	else if (*(double*)a < *(double*)b) return -1;
	else return 0;
}


/*
 * Generate a symbolic transformation on the data and compute the
 * compress ratio.
 *
 * data: array of double. It lenght should be nsampels.
 *
 * nsamples: number of samples in the array.
 *
 * nbins: number of bins used to compute the symbolic transformation
 *
 * sortbuffer: array used to sort the data.
 * 			   It should be of the same size as data: nsamples * sizeof(double)
 *
 * sortbuffer: array used to store the symbolic transformation.
 * 			   Size: nsamples * sizeof(unsigned char)
 *
 * cmpbuffer: memory used to compress. It size depends on the zlib function
 * 			  compressBound() and the number of samples.
 * 			  It should be compressBound(nsamples) * sizeof(unsigned char)
 *
 * result: pointer to double, used to store the result.
 */
int calc_komplexity(
		double * data,
		long nsamples,
		long nbins,
		double * sortbuffer,
		unsigned char * strbuffer,
		unsigned char * cmpbuffer,
		double * result
	) {

	memcpy(sortbuffer, data, nsamples * sizeof(double));

	// Sort the data in increasing order
    qsort(sortbuffer, nsamples, sizeof(double), compare);

    long first = nsamples / 10;
    long last = nsamples - first;

    double lower = sortbuffer[first];
    double upper = sortbuffer[last];
    double bsize = (upper-lower)/nbins;

    // printf(" [ %f (%li) - %f - %f (%li)]\n", lower, first, bsize, upper, last);

    long maxbin = nbins - 1;
    long tbin;

    for (long i = 0; i < nsamples; i ++) {
	tbin = (int)floor((data[i] - lower) / bsize);
	// printf("%f => %li => ", data[i], tbin);
	strbuffer[i] = (tbin < 0 ? 0 : (tbin > maxbin ? maxbin : tbin)) + 'A';
	// printf("%c", strbuffer[i]);
    }
    // printf("\n");

    unsigned long cmplength = compressBound(nsamples);

    int cres = compress(cmpbuffer, &cmplength, strbuffer, nsamples);
    if (cres != Z_OK) {
	printf("Error compressing: %d\n", cres);
    } else {
	*result = (double)cmplength/(double)nsamples;
	// printf("%li => %.16f \n", cmplength, *result);
    }
    return cres;

}

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
	) {

	printf("K:: Using %d threads to compute\n", nthreads);
	omp_set_num_threads(nthreads);
	printf("K:: Buffer size: %lu\n", compressBound(nsamples));
	unsigned char ** cmpbuffer = (unsigned char**) malloc(nthreads * sizeof(unsigned char *));
	unsigned char ** strbuffer = (unsigned char**)malloc(nthreads * sizeof(unsigned char *));
	double ** sortbuffer = (double **)malloc(nthreads * sizeof(double *));

	for (int thread = 0; thread < nthreads; thread++) {
		cmpbuffer[thread] = malloc(compressBound(nsamples) * sizeof(unsigned char));
		sortbuffer[thread] = malloc(nsamples * sizeof(double));
		strbuffer[thread] = malloc(nsamples * sizeof(unsigned char));
	}

	#pragma omp parallel for
	for (long t = 0; t < ntrials; t++) {
		int thread_id = omp_get_thread_num();
		for (long c = 0; c < nchannels; c++) {
			calc_komplexity(
				&data[c * nsamples + t * nsamples * nchannels],
				nsamples,
				nbins,
				sortbuffer[thread_id],
				strbuffer[thread_id],
				cmpbuffer[thread_id], &results[t * nchannels + c]);
		}
	}

	for (int thread = 0; thread < nthreads; thread++) {
		free(cmpbuffer[thread]);
		free(sortbuffer[thread]);
		free(strbuffer[thread]);
	}

	free(cmpbuffer);
	free(sortbuffer);
	free(strbuffer);

	return 0;

}
