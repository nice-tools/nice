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

#include <symb_transf.h>
#include <blocktrie.h>
#include <helpers.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int nsymbols = 0;
int n_total_symbols = 0;

/* Comparte two pairs by its value (not their index) */
int pair_compare( const void* va, const void* vb) {
	t_pair *a = (t_pair *)va;
	t_pair *b = (t_pair *)vb;
	if ( a->value == b->value ) return 0;
	else if ( a->value < b->value ) return -1;
	else return 1;
}

void swap(unsigned char *a, unsigned char *b) { unsigned char t = *a; *a = *b; *b = t; }

/* Following function is needed for library function qsort(). Refer
   http://www.cplusplus.com/reference/clibrary/cstdlib/qsort/ */
int compare (const void *a, const void * b) { return ( *(unsigned char *)a - *(unsigned char *)b ); }



// This function finds the index of the smallest character
// which is greater than 'first' and is present in str[l..h]
int findCeil (unsigned char str[], unsigned char first, int l, int h) {
    // initialize index of ceiling element
    int ceilIndex = l;

    // Now iterate through rest of the elements and find
    // the smallest character greater than 'first'
    int i;
    for (i = l+1; i <= h; i++) {
	if (str[i] > first && str[i] < str[ceilIndex]) {
		ceilIndex = i;
	}
    }
    return ceilIndex;
}

/* Define all the permutations of str into symbols trie */
void definePermutations(unsigned char *str, int len, trie_t * symbols) {
    // Get size of string
    int size = len;

    // Sort the string in increasing order
    qsort( str, size, sizeof( str[0] ), compare );

    int isFinished = 0;
    while (!isFinished) {
	if (trie_defined(symbols, str, 0, len) == -1) {
	        trie_define(symbols, str, 0, len, nsymbols);
	        nsymbols ++;
	        trie_define_reverse(symbols, str, 0, len, n_total_symbols-nsymbols);
	    }
        // Find the rightmost character which is smaller than its next
        // character. Let us call it 'first char'
        int i;
        for ( i = size - 2; i >= 0; --i ) {
			if (str[i] < str[i+1]) {
			break;
			}
         }

        // If there is no such character, all are sorted in decreasing order,
        // means we just made the last permutation and we are done.
        if (i == -1) {
            isFinished = 1;
        } else {
            // Find the ceil of 'first char' in right of first character.
            // Ceil of a character is the smallest character greater than it
            int ceilIndex = findCeil( str, str[i], i + 1, size - 1 );

            // Swap first and second characters
            swap( &str[i], &str[ceilIndex] );

            // Sort the string on right of 'first char'
            qsort( str + i + 1, size - i - 1, sizeof(str[0]), compare );
        }
    }
}

int symb_transf(
		double * data,
		int kernel,
		int tau,
		int * signal_symb_transf,
		double * count,
		int nchannels,
		int nsamples,
		int ntrials
	) {

    printf("Running symb_transf @%p for kernel %d, tau %d, dims (%d, %d, %d) and storing in %p and %p\n",
     (void *)data, kernel, tau, nchannels, nsamples, ntrials, (void *)signal_symb_transf, (void *)count);

    /* Kernel = 4 => nsymbols = 4! = 24
     * kernel! should be less than TRIE_MAX_SYMBOLS, otherwise BOOM...
     */
	if (kernel > 4) {
		printf("This function (symb_transf) maximum kernel value is %d\n", TRIE_MAX_SYMBOLS);
		return -1;
	}

	/*
	 * Define all the posible permutations of the symbols and
	 * asign each one of them a unique value.
	 */
	nsymbols = 0;
	/* SMI and SymbolicMutualInformation parameters */
	n_total_symbols = 1;
    for (int c = 1; c <= kernel; c++) {
		n_total_symbols = n_total_symbols * c;
    }

	unsigned char * symbolstr = malloc(kernel+1 * sizeof(char));
	int i;
	for (i = 0; i < kernel; i++) {
		symbolstr[i] = i;
	}
	trie_t * symbols = trie_create(kernel);
	definePermutations(symbolstr, kernel, symbols);
	free(symbolstr);

	nsymbols = n_total_symbols;

	printf("Total number of symbols %d\n", nsymbols);

	t_pair * elems = malloc(kernel * sizeof(t_pair));

	int value;
	int * sym_count = malloc(nsymbols * sizeof(int));
	int n_symb_samples = nsamples - tau * (kernel - 1);
	/* for each trial */
	int trial;
	int channel;
	int k;
	int s;
	for (trial = 0; trial < ntrials; trial++) {
		/* for each channel */
		for (channel = 0; channel < nchannels; channel++) {
			/* reset the symbol counter */
			memset(sym_count, 0, nsymbols * sizeof(int));
			/* for each posible sample */
			for (k = 0; k < n_symb_samples; k++) {
				/* take kernel samples separated by tau samples */
				for (i = 0; i < kernel; i ++) {
					elems[i].index = i;
					elems[i].value = MAT3D(data, channel, k + i * tau, trial, nchannels, nsamples);
				}
				/* sort them */
				qsort(elems, kernel, sizeof(t_pair), pair_compare);
				/* get the value of the permutation that sorted the elems */
				value = trie_defined_pair(symbols, elems, 0, kernel);
				/* that value is the symbol */
				MAT3D(signal_symb_transf, channel, k, trial, nchannels, n_symb_samples) = value;
				sym_count[value]++;
			}
			/* divide the count by n_symb_samples to get the probability */
			for (s = 0; s < nsymbols; s++) {
				MAT3D(count, channel, s, trial, nchannels, nsymbols) = ((double)sym_count[s])/n_symb_samples;
			}
		}
	}
	free(elems);
	free(sym_count);
	trie_free(&symbols);
	return 0;
}
