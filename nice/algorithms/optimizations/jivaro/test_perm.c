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

#include <blocktrie.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <symb_transf.h>



int main(char argc, char* argv[]) {
	nsymbols = 0;
	int kernel = 3;

	n_total_symbols = 1;
    for (int c = 1; c <= kernel; c++) {
    	n_total_symbols = n_total_symbols * c;
    }


	unsigned char * symbolstr = malloc(kernel+1 * sizeof(char));
	for (int i = 0; i < kernel; i++) {
		symbolstr[i] = i;
	}
	trie_t * symbols = trie_create(kernel);

	definePermutations(symbolstr, kernel, symbols);

	trie_dinfo(symbols);

	free(symbolstr);


	printf("Value for 123 = %d\n", trie_defined(symbols, "\0\1\2", 0, 3));
	printf("Value for 132 = %d\n", trie_defined(symbols, "\0\2\1", 0, 3));
	printf("Value for 213 = %d\n", trie_defined(symbols, "\1\0\2", 0, 3));
	printf("Value for 312 = %d\n", trie_defined(symbols, "\2\0\1", 0, 3));
	printf("Value for 231 = %d\n", trie_defined(symbols, "\1\2\0", 0, 3));
	printf("Value for 321 = %d\n", trie_defined(symbols, "\2\1\0", 0, 3));
}
