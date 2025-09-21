import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):
        """Fit training dat.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        # Initialize weights (including threshold)
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        print(self.w_)
        #For each iteration, train the perceptron with all samples
        for _ in range(self.n_iter):
            #For each sample, update the weights
            for i in range(len(X)):
                sample = X[i]
                print(sample)
                prediction = self.predict(sample)
                error = y[i] - prediction
                self.w_[1:] += self.eta * error * sample # Update feature weights
                self.w_[0] += self.eta * error  # Update threshold


    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """
        # Prediction is the dot product between X and weights
        predicts = np.dot(X, self.w_[1:]) + self.w_[0]
        #Returning class labels after applying the step function
        return np.where(predicts >= 0.0, 1, -1)
