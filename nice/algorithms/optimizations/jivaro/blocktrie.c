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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <blocktrie.h>


trie_t * trie_create(int nsymbols) {
	if (nsymbols > TRIE_MAX_SYMBOLS) {
		fprintf(stderr, "This trie is configured with a maximum of %d symbols. "
			"Please recompile with -DTRIE_MAX_SYMBOLS=%d\n", TRIE_MAX_SYMBOLS, nsymbols);
		exit(-1);
	}
    trie_t * retorno = (trie_t *)malloc(sizeof(trie_t));
    retorno->lastfreeslot = 1;
    retorno->nblocks = 1;
    retorno->ndefs = 0;
    retorno->blocks[0] = (nodo_t*)malloc(TRIE_BLOCK_SIZE * sizeof(nodo_t));
    memset(retorno->blocks[0], 0, sizeof(nodo_t));
	retorno->memused = TRIE_BLOCK_SIZE * sizeof(nodo_t);
    return retorno;
}


void trie_dinfo(trie_t * trie) {
	printf("Trie at %p (%lu bytes %lu mbytes %d defs) { last %d nblocks %d }\n", (void *)trie, trie->memused,trie->memused/1024/1024, trie->ndefs, trie->lastfreeslot, trie->nblocks);
}

void trie_free(trie_t **trie) {
	trie_t * ltrie = *trie;
	int i;
    for (i = 0; i < ltrie->nblocks; i++) {
		free(ltrie->blocks[i]);
	}
	free(ltrie);
	*trie = NULL;
}

nodo_t * nextfree(trie_t *t) {
	nodo_t * retorno = NULL;
	if (t->lastfreeslot == TRIE_BLOCK_SIZE) {
		t->lastfreeslot = 0;
		t->blocks[t->nblocks] = (nodo_t*)malloc(TRIE_BLOCK_SIZE * sizeof(nodo_t));
		t->memused += TRIE_BLOCK_SIZE * sizeof(nodo_t);
		t->nblocks++;
	}
	retorno = &(t->blocks[t->nblocks-1][t->lastfreeslot]);
	t->lastfreeslot++;
	memset(retorno, 0, sizeof(nodo_t));
	retorno->value = -1;
	return retorno;
}

void trie_define(trie_t *trie, unsigned char * buf, int st, int end, int value) {
	nodo_t * curnode = trie->blocks[0];
	unsigned char cursym = buf[st];
	int i;
	for (i = st; i < end; i++) {
		cursym = buf[i];
		if (curnode->sons[cursym] == NULL) {
			curnode->sons[cursym] = nextfree(trie);
		}
		curnode = curnode->sons[cursym];
	}
	curnode->value = value;
	trie->ndefs++;
}

void trie_define_reverse(trie_t *trie, unsigned char * buf, int st, int end, int value) {
	nodo_t * curnode = trie->blocks[0];
	unsigned char cursym = buf[st];
	for (int i = end-1; i >= st; i--) {
		cursym = buf[i];
		if (curnode->sons[cursym] == NULL) {
			curnode->sons[cursym] = nextfree(trie);
		}
		curnode = curnode->sons[cursym];
	}
	curnode->value = value;
	trie->ndefs++;
}

int trie_defined_all(trie_t *trie, unsigned char * buf) {
	int st = 0;
	int end = strlen((const char *)buf);
	return trie_defined(trie, buf, st, end);
}

int trie_defined(trie_t *trie, unsigned char * buf, int st, int end) {
	nodo_t * curnode = trie->blocks[0];
	unsigned char cursym = buf[st];
	int i;
	for (i = st; i < end; i++) {
		cursym = buf[i];
		if (curnode->sons[cursym] == NULL) {
			curnode->sons[cursym] = nextfree(trie);	//Keep looking, it will be defined later.
		}
		curnode = curnode->sons[cursym];
	}
	return curnode->value;
}

int trie_defined_pair(trie_t *trie, t_pair * buf, int st, int end) {
	nodo_t * curnode = trie->blocks[0];
	unsigned char cursym = buf[st].index;
	int i;
	for (i = st; i < end; i++) {
		cursym = buf[i].index;
		if (curnode->sons[cursym] == NULL) {
			curnode->sons[cursym] = nextfree(trie);	//Keep looking, it will be defined later.
		}
		curnode = curnode->sons[cursym];
	}
	return curnode->value;
}
