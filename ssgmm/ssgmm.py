"""Semi-supervised Gaussian Mixture Models."""

from __future__ import print_function
from time import time

import numpy as np

from sklearn.utils import check_array
from sklearn.utils.extmath import logsumexp
from sklearn.mixture.gmm import (
    GMM,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type,)
from sklearn.cluster import KMeans


class SSGMM(GMM):
    """Semi-supervised Gaussian Mixture Models.

    Representation of a Gaussian mixture model probability distribution,
    where some class labels are provided.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Read more in the scikit-learn GMM documentation.
    """
    def __init__(self, n_components=1, covariance_type='diag',
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmc', init_params='wmc',
                 verbose=0):
        super(SSGMM, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            random_state=random_state, tol=tol, min_covar=min_covar,
            n_iter=n_iter, n_init=n_init, params=params,
            init_params=init_params, verbose=verbose)

    def _score_samples(self, X, y):
        """
        Calculate the log-probability for each datapoint for each class.

        In the case where labels are provided, log-probabilities for other class
        labels are set to -infinity so that they end up contributing zero to the
        posterior.
        """

        # Firstly, calculate old unsupervised likelihood.
        lpr = (log_multivariate_normal_density(X, self.means_, self.covars_,
                                               self.covariance_type) +
               np.log(self.weights_))

        # Remove contributions of classes not matching labelled data.
        for label in self.label_classes_:
            label_indices = np.where(y == label)[0]
            label_mask = np.delete(np.arange(self.n_components), label)
            # @@@ probably better numpythonic way to do this...
            for m in label_mask:
                lpr[label_indices, m] = -np.inf

        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities

    def score_samples(self, X):
        return self._score_samples(X, y=np.array([]))

    def _init_params(self, X, y):
        """
        Initialise parameters for each run of the EM algorithm.

        For classes with labels provided, means are initialised as the centroid
        of the labelled data. Remaining class means are initialised via K-Means.
        """
        if 'm' in self.init_params or not hasattr(self, 'means_'):
            # First include means of labelled data.
            self.means_ = np.zeros((self.n_components, X.shape[1]))
            for label in self.label_classes_:
                self.means_[label] = np.mean(
                    X[np.where(y == label)[0], :], axis=0)

            # Initialise remaining means (ones with no labels) using K-Means.
            blanks = np.delete(np.arange(self.n_components), self.label_classes_)
            if len(blanks) != 0:
                X_unlabelled = X[y == -1, :]

                means = KMeans(
                    n_clusters=len(blanks), random_state=self.random_state).fit(
                        X_unlabelled).cluster_centers_

                self.means_[blanks, :] = means

            if self.verbose > 1:
                print('\tMeans have been initialized.')
                print(self.means_)

        if 'w' in self.init_params or not hasattr(self, 'weights_'):
            self.weights_ = np.tile(1.0 / self.n_components,
                                    self.n_components)
            if self.verbose > 1:
                print('\tWeights have been initialized.')
                print(self.weights_)

        if 'c' in self.init_params or not hasattr(self, 'covars_'):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components)
            if self.verbose > 1:
                print('\tCovariance matrices have been initialized.')
                print(self.covars_)

    def _fit(self, X, y, do_prediction=False):
        # initialization step
        X = check_array(X, dtype=np.float64, ensure_min_samples=2,
                        estimator=self)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        if y is None:
            self.label_classes_ = []
        else:
            labels = np.unique(y)
            self.label_classes_ = labels[labels != -1].astype(int)
            if max(labels) > self.n_components:
                raise ValueError(
                    'More labels provided than number of components.')

        if self.verbose > 0:
            print('Expectation-maximization algorithm started.')

        for init in range(self.n_init):
            # Initialize prior distribution.
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()
            self._init_params(X, y)

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            for i in range(self.n_iter):
                if self.verbose > 0:
                    print('\tEM iteration ' + str(i + 1))
                    start_iter_time = time()
                prev_log_likelihood = current_log_likelihood

                # Expectation step
                log_likelihoods, responsibilities = self._score_samples(X, y)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if self.verbose > 1:
                        print('\t\tChange: ' + str(change))
                    if change < self.tol:
                        self.converged_ = True
                        if self.verbose > 0:
                            print('\t\tEM algorithm converged.')
                        break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)
                if self.verbose > 1:
                    print('\t\tEM iteration ' + str(i + 1) + ' took {0:.5f}s'.format(
                        time() - start_iter_time))

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
                    if self.verbose > 1:
                        print('\tBetter parameters were found.')

            if self.verbose > 1:
                print('\tInitialization ' + str(init + 1) + ' took {0:.5f}s'.format(
                    time() - start_init_time))

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:  # self.n_iter == 0 occurs when using GMM within HMM
            # Need to make sure that there are responsibilities to output
            # Output zeros because it was just a quick initialization
            responsibilities = np.zeros((X.shape[0], self.n_components))

        return responsibilities

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : array_like, shape (n_samples)
            List of labels for data points in X. Labels should be non-negative
            integers, no greater than n_components. Unlabelled data should
            be given label -1. If None, defaults to standard GMM.

        Returns
        -------
        self
        """
        self._fit(X, y)
        return self

