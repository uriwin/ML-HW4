import math

import numpy as np


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    ###########################################################################
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_minus_mean_x = np.subtract(x, mean_x)
    y_minus_mean_y = np.subtract(y, mean_y)
    numerator = np.sum(x_minus_mean_x @ y_minus_mean_y)
    denominator = np.sqrt(np.sum(np.square(x_minus_mean_x)) * np.sum(np.square(y_minus_mean_y)))
    r = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    X_temp = X.select_dtypes(include=['number'])
    y_numpy = y.to_numpy()
    features_correlation = {}
    for column_name, series in X_temp.items():
        corr = pearson_correlation(series.to_numpy(), y_numpy)
        features_correlation[column_name] = corr

    best_features = [k for k, v in
                     sorted(features_correlation.items(), key=lambda item: item[1], reverse=True)[:n_features]]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = apply_bias_trick(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            current_gradiant = self.calculate_gradient(X, y)
            self.theta = self.theta - self.eta * current_gradiant
            cost = self.calculate_cost(X, y)
            self.Js.append(cost)
            self.thetas.append(self.theta)
            if i > 0:
                if abs(self.Js[i - 1] - cost) < self.eps:
                    break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calculate_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_cost(self, X, y):
        z = np.dot(X, self.theta)
        h = self.calculate_sigmoid(z)
        eps=1e-5  # small value to avoid log(0).
        J = (-1.0 / len(y)) * (np.dot(y.T, np.log(h + eps)) + np.dot((1 - y).T, np.log(1 - h + eps)))
        return J

    def calculate_gradient(self, X, y):
        error = self.calculate_sigmoid(np.dot(X, self.theta)) - y
        return np.dot(X.T, error) / len(X)


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = apply_bias_trick(X)
        z = np.dot(X, self.theta)
        h = self.calculate_sigmoid(z)
        for sigmoid_result in h:
            if sigmoid_result > 0.5:
                preds.append(1)
            else:
                preds.append(0)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    return np.insert(X, 0, 1, axis=1)


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    complete_set = np.column_stack((X, y))
    np.random.shuffle(complete_set)
    split_set = np.array_split(complete_set, folds, axis=0)
    accuracies = []
    # bla
    for fold in split_set[:-1]:
        fold_data = fold[:, :-1]
        fold_labels = fold[:, -1]
        algo.fit(fold_data, fold_labels)
        predictions_for_fold = algo.predict(fold_data)
        comparison = predictions_for_fold == fold_labels
        successful_predicts = np.sum(comparison)
        accuracies.append(successful_predicts / fold.shape[0])
    cv_accuracy = np.average(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # p = (1 / np.sqrt(2 * np.pi * (sigma ** 2))) * math.e ** ((-((data - mu) ** 2)) / (2 * (sigma ** 2)))
    p = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((data - mu) ** 2) / (2 * sigma ** 2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.responsibilities = {}
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []
        for gaussian in range(self.k):
            self.weights.append(1/self.k)
            self.mus.append(0.5)
            self.sigmas.append(1)


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        weighted_pdfs = np.array([])
        for gaussian_index in range(self.k):
            pdfs = self.weights[gaussian_index] * norm_pdf(data, self.mus[gaussian_index], self.sigmas[gaussian_index])
            weighted_pdfs = np.append(weighted_pdfs, pdfs)
        denominator = np.sum(weighted_pdfs)

        for gaussian_index, gaussian in enumerate(weighted_pdfs):
            self.responsibilities[gaussian_index] = (gaussian/denominator)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.expectation(data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
