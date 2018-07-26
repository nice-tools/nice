# NICE
# Copyright (C) 2017 - Authors of NICE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# You can be released from the requirements of the license by purchasing a
# commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities as mentioned in the GNU Affero General Public
# License version 3 without disclosing the source code of your own
# applications.
#
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import mne
from mne.utils import logger


def _check_clf(clf, cv, class_weight, random_state):
    if clf is None:
        logger.info(
            'Using default Pipeline (Standard Scaler + Linear SVC C=1)')
        scaler = StandardScaler()
        svc = SVC(C=1, kernel='linear', probability=True,
                  class_weight=class_weight,
                  random_state=random_state)
        clf = Pipeline([('scaler', scaler), ('svc', svc)])

    if cv is None:
        cv = 5

    if isinstance(cv, int):
        logger.info('Using K ({}) Stratified Folds'.format(cv))
        cv = StratifiedKFold(cv, random_state=random_state)

    return clf, cv


def cv_decode_sliding(X, y, clf=None, cv=None, class_weight=None,
                      scoring='roc_auc', random_state=None,
                      picks=None, n_jobs=-1):
    all_scores = []
    if not isinstance(random_state, list):
        random_state = [random_state]

    for t_random in random_state:
        logger.info('Using random state {}'.format(t_random))
        clf, cv = _check_clf(clf, cv, class_weight, t_random)

        se = mne.decoding.SlidingEstimator(clf, scoring=scoring)
        if picks is not None:
            logger.info('Picking channels')
            X = X[:, picks, :]
        scores = mne.decoding.cross_val_multiscore(
            se, X, y, cv=cv, n_jobs=n_jobs)
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)


def decode_sliding(X_train, y_train, X_test, y_test, clf=None,
                   class_weight=None, scoring='roc_auc', random_state=None,
                   picks=None, n_jobs=-1):

    clf, _ = _check_clf(clf, None, class_weight, random_state)

    se = mne.decoding.SlidingEstimator(clf, scoring=scoring)
    if picks is not None:
        logger.info('Picking channels')
        X_train = X_train[:, picks, :]
        X_test = X_test[:, picks, :]

    se.fit(X_train, y_train)
    scores = se.score(X_test, y_test)
    return scores[None, :]


def cv_decode_generalization(X, y, clf=None, cv=None, class_weight=None,
                             scoring='roc_auc', random_state=None, picks=None,
                             n_jobs=-1):

    all_scores = []
    if not isinstance(random_state, list):
        random_state = [random_state]

    for t_random in random_state:
        logger.info('Using random state {}'.format(t_random))
        clf, cv = _check_clf(clf, cv, class_weight, t_random)

        ge = mne.decoding.GeneralizingEstimator(
            clf, scoring=scoring, n_jobs=n_jobs)
        if picks is not None:
            logger.info('Picking channels')
            X = X[:, picks, :]
        scores = mne.decoding.cross_val_multiscore(ge, X, y, cv=cv, n_jobs=1)
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)


def decode_generalization(X_train, y_train, X_test, y_test, clf=None,
                          class_weight=None, scoring='roc_auc',
                          random_state=None, picks=None, n_jobs=-1):

    clf, _ = _check_clf(clf, None, class_weight, random_state)

    ge = mne.decoding.GeneralizingEstimator(
        clf, scoring=scoring, n_jobs=n_jobs)

    if picks is not None:
        logger.info('Picking channels')
        X_train = X_train[:, picks, :]
        X_test = X_test[:, picks, :]

    ge.fit(X_train, y_train)
    scores = ge.score(X_test, y_test)

    return scores[None, :]
