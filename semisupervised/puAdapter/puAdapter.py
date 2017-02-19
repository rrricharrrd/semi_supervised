"""Positive-Unlabelled learning based on a supervised classfier"""


import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import CalibratedClassifierCV


class PUAdapter(BaseEstimator):
    """
    Convert a supervised binary classifier into one suitable for
    positive-unlabelled learning.

    Parameters
    ----------
    estimator : classifier
        A classifier used to learn :math:`p(s=1 \mid x)`. If this classfier
        does not implement its own predict_proba, a calibrated version is used.

    c_estimator : {e1, e2, e3}
        Method used to estimate :math:`p(s=1 \mid y=1)`.

        e1 :
            Use mean of the underlying estimator probabilities on positive
            validation set.
        e2 :
            Use ratio of positive vs all the underlying estimator probabilites
            on validation set.
        e3 :
            Use maximum of underlying estimator on validation set.

    c_estimator_size : float
        The ratio of training examples that should be held out of theset of
        examples to estimate :math:`p(s=1 \mid y=1)`.
    
    Attributes
    ----------
    c : float
        Estimate of :math:`p(s=1 \mid y=1)`.

    References
    ----------
    Charles Elkan, Keith Noto.
        Learning classifiers from only positive and unlabeled data.
        Proceeding of the 14th ACM SIGKDD international conference on Knowledge
        discovery and data mining. ACM, 2008.
    """

    def __init__(self, estimator, c_estimator='e1', c_estimator_size=0.2):
        self.c_estimator = c_estimator
        self.c_estimator_size = c_estimator_size
        
        # May need to calibrate first to yield probabilities
        if hasattr(estimator, 'predict_proba'):
            self.clf = estimator
        else:
            self.clf = CalibratedClassifierCV(estimator)

    def __repr__(self):
        return 'Positive-Unlabelled adaptation of ' + repr(self.clf)

    def fit(self, X, y, sample_weight=None):
        """
        Train estimator on positive vs. unlablled examples and use that to
        estimator :math:`p(s=1 \mid x)`, and thence :math:`p(s=1 \mid y=1,x)`.

        Parameters
        ----------
        X : array
            List of feature vectors.
        y : array
            Observed labels for each feature vector in X.

        Returns
        -------
        self : object
            Returns self.
        """
        positives = np.where(y == 1)[0]
        n = int(np.ceil(len(positives) * self.c_estimator_size))
        if len(positives) <= n:
            raise ValueError("Not enough seeds provided")

        np.random.shuffle(positives)
        mask = positives[:n]

        # First train the classfier on positive vs unlabelled data.
        X_red = np.delete(X, mask, axis=0)
        y_red = np.delete(y, mask)
        if sample_weight is not None:
            self.clf.fit(X_red, y_red, sample_weight)
        else:
            self.clf.fit(X_red, y_red)

        # Now estimate c.
        V = X[mask]
        self._calculate_c(V)
        
        return self

    def _calculate_c(self, V):
        """
        Estimate c=p(s=1|y=1,x), the scale factor to apply to base classifier.
        """
        c_estimates = self.clf.predict_proba(V)[:, 1]

        if self.c_estimator == 'e1':
            c = np.mean(c_estimates)
        elif self.c_estimator == 'e2':
            raise NotImplementedError
        elif self.c_estimator == 'e3':
            c = max(c_estimates)
        else:
            raise ValueError('Invalid c estimator provided.')

        self.c = c

    def predict(self, X, threshold=0.5):
        """
        Predict label for data.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Data to predict labels for.

        threshold : float
            The decision threshold between positive and negative predictions.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted label per sample.
        """
        check_is_fitted(self, 'c')

        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict :math:`p(y=1 \mid x)` for data.

        Parameters
        ---------
        X : array, shape = [n_samples, n_features]
            Data to predict probabilities for.

        Returns
        -------
        T : array, shape = [n_samples, 2]
            Probability estimate of negative/positive label per sample.
        """
        check_is_fitted(self, 'c')
        
        pos_probs = self.clf.predict_proba(X)[:, 1] / self.c
        probs = np.vstack([1 - pos_probs, pos_probs]).T
        
        return probs


    def predict_positive_count(self, X, y):
        """
        Use estimated weights to predict overall positive count.

        Parameters
        ----------
        X : array
            Array of features.

        Returns
        -------
        p : int
            Estimated number of positive examples in X.
        """
        negatives = np.where(y == 0)[0]
        positives = np.where(y == 1)[0]
        neg_preds = self.clf.predict_proba(X[negatives])[:, 1]
        return (len(positives) +
                sum((1 - self.c)/self.c * neg_preds / (1 - neg_preds)))

    def _get_weights(self, X, y):
        negatives = np.where(y == 0.)[0]
        weights = np.ones(y.shape)
        prob_unlabelled_pos = self.clf.predict_proba(X[negatives])[:, 1]
        pos_weights = 1 - (1 - self.c)/self.c * \
            prob_unlabelled_pos / (1 - prob_unlabelled_pos)
        weights[negatives] = pos_weights
        weights = np.hstack((weights, 1 - pos_weights))
        return weights

    def _fit2(self, X, y):
        """
        Alternative approach to fitting, where unlabelled examples are given
        weights according to estmated probabilities of being positive.
        @@@
        """
        self.fit(X, y)
        negatives = np.where(y == 0.)[0]
        X2 = np.vstack((X, X[negatives, :]))
        y2 = np.hstack((np.ones(y.shape), np.zeros(negatives.shape)))
        weights = self._get_weights(X, y)
        self.clf.fit(X2, y2, sample_weight=weights)

