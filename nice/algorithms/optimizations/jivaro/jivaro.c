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

#ifndef __JIVARO_MAIN__
#define __JIVARO_MAIN__

#include <Python.h>
#include <symb_transf.h>
#include <smi.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL jivaro_main_array_symbol

#ifndef __JIVARO_MAIN__
#define NO_IMPORT_ARRAY
#endif

#include <types.h>

#include <numpy/arrayobject.h>
#include <helpers.h>
#include <omp.h>
static PyObject *jivaroError;


static PyObject * jivaro_wsmi(PyObject *self, PyObject *args) {
	PyArrayObject *data;
	int kernel;
	int tau;
	PyArrayObject *wts;
	int nthreads;
	if (!PyArg_ParseTuple(args, "OiiOi", &data, &kernel, &tau, &wts, &nthreads)) {
		PyErr_SetString(jivaroError, "Invalid parameters.");
	}
	// printf ("Calling pe with %p, kernel %d and tau %d\n", data, kernel, tau);
	// PyArrayObject * signal_symb = NULL;
	// PyArrayObject * count = NULL;


	npy_intp * dims = PyArray_DIMS(data);

	int nchannels = dims[0];
	int nsamples = dims[1];
	int ntrials = dims[2];

	double * c_data = malloc(nchannels * nsamples * ntrials * sizeof(double));

	int trial;
	int channel;
	int sample;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples; sample++) {
				MAT3D(c_data, channel, sample, trial, nchannels, nsamples) =
					*(double *)PyArray_GETPTR3(data, channel, sample, trial);
			}
		}
	}

	int nsymbols = 1;
	int c;
    for (c = 1; c <= kernel; c++) {
	nsymbols = nsymbols * c;
    }

    double * c_wts = calloc(nsymbols * nsymbols, sizeof(double));
    int r;
    for (c = 0; c < nsymbols; c++) {
	for (r = 0; r < nsymbols; r++) {
			c_wts[r + nsymbols * c] =  *(double *)PyArray_GETPTR2(wts, c, r);
		}
    }

    int nsamples_symbolic = nsamples - tau * (kernel - 1);

    int * signal_symb_transf = malloc(nchannels * nsamples_symbolic * ntrials * sizeof(int));
	double * c_count = malloc(nchannels * nsymbols * ntrials * sizeof(double));

	int symb_transf_result = symb_transf(
	        c_data,
	        kernel,
	        tau,
	        signal_symb_transf,
	        c_count,
	        nchannels,
	        nsamples,
	        ntrials
	);

	if (symb_transf_result != 0) {
		PyErr_SetString(jivaroError, "Unable to compute Symbolic Transformation ");
	}

    omp_set_num_threads(nthreads);

    free(c_data);

    /* Initialize working variables */
    double * smi_datas = calloc(nchannels * nchannels * ntrials, sizeof(double));
    double * wsmi_datas = calloc(nchannels * nchannels * ntrials, sizeof(double));

	// /* Output variables */
	// double * smi_results = calloc(nchannels * nchannels, sizeof(double));
	// double * wsmi_results = calloc(nchannels * nchannels, sizeof(double));

	int smi_result = smi(
	        signal_symb_transf,
	        c_count,
	        c_wts,
	        smi_datas,
	        wsmi_datas,
	        nchannels,
	        nsamples_symbolic,
	        ntrials,
	        nsymbols,
	        nthreads
	    );
	if (smi_result != 0) {
		PyErr_SetString(jivaroError, "Unable to compute wSMI ");
	}

	int mi_ndims = 3;
	npy_intp * mi_dims = malloc(mi_ndims * sizeof(npy_intp));

	mi_dims[0] = nchannels;
	mi_dims[1] = nchannels;
	mi_dims[2] = ntrials;

	PyArrayObject * smi = (PyArrayObject *) PyArray_ZEROS(mi_ndims, mi_dims, NPY_DOUBLE, 0); //CTYPE Zeros array
	PyArrayObject * wsmi = (PyArrayObject *) PyArray_ZEROS(mi_ndims, mi_dims, NPY_DOUBLE, 0); //CTYPE Zeros array

	int channel1;
	int channel2;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel1 = 0; channel1 < nchannels; channel1++) {
			for (channel2 = 0; channel2 < nchannels; channel2++) {
				*(double *)PyArray_GETPTR3(smi, channel1, channel2, trial) =
					MAT3D(smi_datas, channel1, channel2, trial, nchannels, nchannels);
				*(double *)PyArray_GETPTR3(wsmi, channel1, channel2, trial) =
					MAT3D(wsmi_datas, channel1, channel2, trial, nchannels, nchannels);
			}
		}
	}

	mi_dims[0] = nchannels;
	mi_dims[1] = nsamples_symbolic;
	mi_dims[2] = ntrials;
	PyArrayObject *symb = (PyArrayObject *) PyArray_ZEROS(mi_ndims, mi_dims, NPY_INT32, 0); //CTYPE Zeros array

	mi_dims[0] = nchannels;
	mi_dims[1] = nsymbols;
	mi_dims[2] = ntrials;
	PyArrayObject *count = (PyArrayObject *) PyArray_ZEROS(mi_ndims, mi_dims, NPY_DOUBLE, 0); //CTYPE Zeros array

	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples_symbolic; sample++) {
				*(int *)PyArray_GETPTR3(symb, channel, sample, trial) =
					MAT3D(signal_symb_transf, channel, sample, trial, nchannels, nsamples_symbolic);
			}
		}
	}

	int symbol;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (symbol = 0; symbol < nsymbols; symbol++) {
				*(double *)PyArray_GETPTR3(count, channel, symbol, trial) =
					MAT3D(c_count, channel, symbol, trial, nchannels, nsymbols);
			}
		}
	}

	// mi_dims[0] = nsymbols;
	// mi_dims[1] = nsymbols;
	// mi_dims[2] = ntrials;
	// PyArrayObject *wts_2 = (PyArrayObject *) PyArray_ZEROS(2, mi_dims, NPY_DOUBLE, 0); //CTYPE Zeros array
	// int symbol1;
	// int symbol2;
	// for (symbol1 = 0; symbol1 < nsymbols; symbol1++) {
	// 	for (symbol2 = 0; symbol2 < nsymbols; symbol2++) {
	// 		*(double *)PyArray_GETPTR2(wts_2, symbol1, symbol2) =
	// 			MAT2D(c_wts, symbol1, symbol2, nsymbols);
	// 	}
	// }
	free(c_wts);
	free(signal_symb_transf);
	free(c_count);
	free(smi_datas);
	free(wsmi_datas);
	free(mi_dims);

	PyObject * retorno = PyTuple_New(4);
	if (PyTuple_SetItem(retorno, 0, (PyObject*) wsmi) != 0) {
		PyErr_SetString(jivaroError, "Error setting wsmi output");
	}
	if (PyTuple_SetItem(retorno, 1, (PyObject*) smi) != 0) {
		PyErr_SetString(jivaroError, "Error setting smi output");
	}
	if (PyTuple_SetItem(retorno, 2, (PyObject*) symb) != 0) {
		PyErr_SetString(jivaroError, "Error setting symb output");
	}
	if (PyTuple_SetItem(retorno, 3, (PyObject*) count) != 0) {
		PyErr_SetString(jivaroError, "Error setting count output");
	}
	return retorno;
}

static PyObject * jivaro_pe(PyObject *self, PyObject *args) {
	PyArrayObject *data;
	int kernel;
	int tau;
	if (!PyArg_ParseTuple(args, "Oii", &data, &kernel, &tau)) {
		PyErr_SetString(jivaroError, "Invalid parameters.");
	}
	// printf ("Calling pe with %p, kernel %d and tau %d\n", data, kernel, tau);
	// PyArrayObject * signal_symb = NULL;
	// PyArrayObject * count = NULL;


	npy_intp * dims = PyArray_DIMS(data);

	int nchannels = dims[0];
	int nsamples = dims[1];
	int ntrials = dims[2];

	double * c_data = malloc(nchannels * nsamples * ntrials * sizeof(double));

	int trial;
	int channel;
	int sample;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples; sample++) {
				MAT3D(c_data, channel, sample, trial, nchannels, nsamples) =
					*(double *)PyArray_GETPTR3(data, channel, sample, trial);
			}
		}
	}

	int nsymbols = 1;
	int c;
    for (c = 1; c <= kernel; c++) {
	nsymbols = nsymbols * c;
    }

    int nsamples_symbolic = nsamples - tau * (kernel - 1);

    int * signal_symb_transf = malloc(nchannels * nsamples_symbolic * ntrials * sizeof(int));
	double * c_count = malloc(nchannels * nsymbols * ntrials * sizeof(double));

	int symb_transf_result = symb_transf(
	        c_data,
	        kernel,
	        tau,
	        signal_symb_transf,
	        c_count,
	        nchannels,
	        nsamples,
	        ntrials
	);

	if (symb_transf_result != 0) {
		PyErr_SetString(jivaroError, "Unable to compute Symbolic Transformation ");
	}

	int symb_ndims = 3;
	npy_intp * symb_dims = malloc(symb_ndims * sizeof(npy_intp));

	symb_dims[0] = nchannels;
	symb_dims[1] = nsamples_symbolic;
	symb_dims[2] = ntrials;
	PyArrayObject *symb = (PyArrayObject *) PyArray_ZEROS(symb_ndims, symb_dims, NPY_INT32, 0); //CTYPE Zeros array

	int pe_ndims = 2;
	symb_dims[0] = nchannels;
	symb_dims[1] = ntrials;
	PyArrayObject *count = (PyArrayObject *) PyArray_ZEROS(pe_ndims, symb_dims, NPY_DOUBLE, 0); //CTYPE Zeros array

	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples_symbolic; sample++) {
				*(int *)PyArray_GETPTR3(symb, channel, sample, trial) =
					MAT3D(signal_symb_transf, channel, sample, trial, nchannels, nsamples_symbolic);
			}
		}
	}

	int symbol;
	double sum;
	double value = 0;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			sum = 0;
			for (symbol = 0; symbol < nsymbols; symbol++) {
				value = MAT3D(c_count, channel, symbol, trial, nchannels, nsymbols);
				if (value > 0) {
					sum += value * log(value);
				}
			}
			*(double *)PyArray_GETPTR2(count, channel, trial) = -sum;
		}
	}
	free(signal_symb_transf);
	free(c_count);
	free(symb_dims);

	PyObject * retorno = PyTuple_New(2);
	if (PyTuple_SetItem(retorno, 0, (PyObject*) count) != 0) {
		PyErr_SetString(jivaroError, "Error setting PE output");
	}
	if (PyTuple_SetItem(retorno, 1, (PyObject*) symb) != 0) {
		PyErr_SetString(jivaroError, "Error setting count output");
	}
	return retorno;
}

static PyMethodDef JivaroMethods[] = {
	{"wsmi",  jivaro_wsmi, METH_VARARGS, "The mutual information and weighted mutual information of the given matrix across first dimension"},
	{"pe",  jivaro_pe, METH_VARARGS, "The permutation entropy of the given matrix across first dimension"},
	{NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef jivaromodule = {
   PyModuleDef_HEAD_INIT,
   "jivaro",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   JivaroMethods
};
#endif

#if PY_MAJOR_VERSION == 3
PyMODINIT_FUNC PyInit_jivaro(void) {
	PyObject *m;
	m = PyModule_Create(&jivaromodule);
	if (m == NULL)
		return NULL;
#else
PyMODINIT_FUNC initjivaro(void) {
	PyObject *m;
	m = Py_InitModule("jivaro", JivaroMethods);
	if (m == NULL)
		return;
#endif

	jivaroError = PyErr_NewException("jivaro.error", NULL, NULL);
	Py_INCREF(jivaroError);
	PyModule_AddObject(m, "error", jivaroError);
	import_array();
#if PY_MAJOR_VERSION == 3
	return m;
#endif

}

#endif
