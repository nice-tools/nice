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
#ifndef __OMP_KOMPLEXITY_MAIN__
#define __OMP_KOMPLEXITY_MAIN__

#include <Python.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL jivaro_main_array_symbol

#ifndef __OMP_KOMPLEXITY_MAIN__
#define NO_IMPORT_ARRAY
#endif

#include <types.h>

#include <numpy/arrayobject.h>
#include <komplexity.h>
#include <helpers.h>

static PyObject *ompkError;


static PyObject * ompk_komplexity(PyObject *self, PyObject *args) {
	PyArrayObject *data;
	int nbins;
	int nthreads;
	if (!PyArg_ParseTuple(args, "Oii", &data, &nbins, &nthreads)) {
		PyErr_SetString(ompkError, "Invalid parameters.");
	}
	// printf ("Calling pe with %p, kernel %d and tau %d\n", data, kernel, tau);
	// PyArrayObject * signal_symb = NULL;
	// PyArrayObject * count = NULL;


	npy_intp * dims = PyArray_DIMS(data);

	int ntrials = dims[0];
	int nchannels = dims[1];
	int nsamples = dims[2];

	double * c_data = malloc(nchannels * nsamples * ntrials * sizeof(double));

	int trial;
	int channel;
	int sample;
	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			for (sample = 0; sample < nsamples; sample++) {
				MAT3D(c_data, sample, channel, trial, nsamples, nchannels) =
					*(double *)PyArray_GETPTR3(data, trial, channel, sample);
			}
		}
	}

	double * c_results = malloc(nchannels * ntrials * sizeof(double));

	int all_trial_result = do_process_all_trials(c_data, nsamples, nchannels, ntrials, c_results, nbins, nthreads);

	if (all_trial_result != 0) {
		PyErr_SetString(ompkError, "Unable to compute KolmogorovComplexity ");
	}





	int result_ndims = 2;
	npy_intp * result_dims = malloc(result_ndims * sizeof(npy_intp));

	result_dims[0] = nchannels;
	result_dims[1] = ntrials;
	PyArrayObject *py_result = (PyArrayObject *) PyArray_ZEROS(result_ndims, result_dims, NPY_DOUBLE, 0); //CTYPE Zeros array

	for (trial = 0; trial < ntrials; trial++) {
		for (channel = 0; channel < nchannels; channel++) {
			*(double *)PyArray_GETPTR2(py_result, channel, trial) =
				MAT2D(c_results, channel, trial, nchannels);
		}
	}
	free(c_data);
	free(c_results);
	free(result_dims);

	PyObject * retorno = (PyObject *) py_result;
	return retorno;
}

static PyMethodDef OmpkMethods[] = {
	{"komplexity",  ompk_komplexity, METH_VARARGS, "The KolmogorovComplexity of the given matrix across second dimension"},
	{NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef ompkmodule = {
   PyModuleDef_HEAD_INIT,
   "ompk",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   OmpkMethods
};
#endif

#if PY_MAJOR_VERSION == 3
PyMODINIT_FUNC PyInit_ompk(void) {
	PyObject *m;
	m = PyModule_Create(&ompkmodule);
	if (m == NULL)
		return NULL;
#else
PyMODINIT_FUNC initompk(void) {
	PyObject *m;
	m = Py_InitModule("ompk", OmpkMethods);
	if (m == NULL)
		return;
#endif

	ompkError = PyErr_NewException("ompk.error", NULL, NULL);
	Py_INCREF(ompkError);
	PyModule_AddObject(m, "error", ompkError);
	import_array();
#if PY_MAJOR_VERSION == 3
	return m;
#endif

}

#endif
