import numpy as np
from matplotlib import pyplot as plt


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
        eps = 1e-5  # small value to avoid log(0).
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

    for test_data_index, test_data in enumerate(split_set):
        test_data_without_labels = test_data[:, :-1]
        test_data_labels = test_data[:, -1]

        train_data = np.concatenate([split_set[i] for i in range(len(split_set)) if i != test_data_index])
        train_data_without_labels = train_data[:, :-1]
        train_data_labels = train_data[:, -1]

        algo.fit(train_data_without_labels, train_data_labels)

        predictions_for_train = algo.predict(test_data_without_labels)
        comparison = predictions_for_train == test_data_labels
        successful_predicts = np.sum(comparison)
        accuracies.append(successful_predicts / test_data.shape[0])

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
    p = (1 / np.sqrt(2 * np.pi * np.square(sigma))) * np.exp(-(np.square(data - mu)) / (2 * np.square(sigma)))

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
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes].reshape(self.k)
        self.sigmas = []
        self.costs = []

        for gaussian in range(self.k):
            self.weights.append(1 / self.k)
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
        weighted_pdfs = np.empty((0, len(data)))
        for gaussian_index in range(self.k):
            pdfs = self.weights[gaussian_index] * norm_pdf(data, self.mus[gaussian_index], self.sigmas[gaussian_index])
            pdfs = pdfs.reshape(1, -1)
            weighted_pdfs = np.append(weighted_pdfs, pdfs, axis=0)

        denominator = np.sum(weighted_pdfs, axis=0)

        for gaussian_index, gaussian in enumerate(weighted_pdfs):
            self.responsibilities[gaussian_index] = gaussian / denominator

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
        for gaussian_index in range(self.k):
            gaussian_responsibility = self.responsibilities[gaussian_index]
            self.weights[gaussian_index] = (1 / len(data)) * np.sum(gaussian_responsibility)
            left_side = (1 / (self.weights[gaussian_index] * len(data)))

            self.mus[gaussian_index] = left_side * np.sum(gaussian_responsibility[:, np.newaxis] * data)
            self.sigmas[gaussian_index] = np.sqrt(
                left_side * np.sum(gaussian_responsibility[:, np.newaxis] * np.square(data - self.mus[gaussian_index])))
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

        for iteration in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            current_cost = self.calculate_cost(data)
            self.costs.append(current_cost)
            if iteration > 0:
                if abs(self.costs[iteration - 1] - current_cost) < self.eps:
                    break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calculate_cost(self, data):
        cost = 0
        for gaussian_index in range(self.k):
            weighted_pdf = np.log(
                self.weights[gaussian_index] * norm_pdf(data, self.mus[gaussian_index], self.sigmas[gaussian_index]))
            cost += weighted_pdf.sum()
        return -cost

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
    pdf = np.sum(weights * norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)
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
        self.gaussians_details = {}
        self.classes_prior = {}

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
        instances_count = len(X)
        unique_classes = np.unique(y)
        em = EM(k=self.k)
        for unique_class in unique_classes:
            class_instances = X[y == unique_class]
            self.gaussians_details[unique_class] = {}
            for feature_index in range(class_instances.shape[1]):
                feature_data = class_instances[:, feature_index]
                em.fit(feature_data.reshape(-1, 1))
                self.gaussians_details[unique_class][feature_index] = em.get_dist_params()

            self.classes_prior[unique_class] = len(class_instances) / instances_count

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calculate_likelihood(self, data, class_label):
        result = 1.0
        for feature_index in self.gaussians_details[class_label].keys():
            weights, mus, sigmas = self.gaussians_details[class_label][feature_index]
            result *= gmm_pdf(data[feature_index], weights, mus, sigmas)
        return result

    def calculate_posterior(self, data, class_label):
        return self.classes_prior[class_label] * self.calculate_likelihood(data, class_label)

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
        preds = []
        for instance in X:
            classes_posterior = {}
            for unique_class in self.gaussians_details.keys():
                class_posterior = self.calculate_posterior(instance, unique_class)
                classes_posterior[unique_class] = class_posterior
            instance_prediction = max(classes_posterior, key=classes_posterior.get)
            preds.append(instance_prediction)
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
    logistic_regression_classifier = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression_classifier.fit(x_train, y_train)
    naive_bayes_gaussian_classifier = NaiveBayesGaussian(k=k)
    naive_bayes_gaussian_classifier.fit(x_train, y_train)

    logistic_regression_predictions = logistic_regression_classifier.predict(x_train)
    lor_train_acc = np.sum(y_train == logistic_regression_predictions) / len(y_train)

    logistic_regression_predictions = logistic_regression_classifier.predict(x_test)
    lor_test_acc = np.sum(y_test == logistic_regression_predictions) / len(y_test)

    naive_bayes_gaussian_predictions = naive_bayes_gaussian_classifier.predict(x_train)
    bayes_train_acc = np.sum(y_train == naive_bayes_gaussian_predictions) / len(y_train)

    naive_bayes_gaussian_predictions = naive_bayes_gaussian_classifier.predict(x_test)
    bayes_test_acc = np.sum(y_test == naive_bayes_gaussian_predictions) / len(y_test)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


# Seed for reproducibility
np.random.seed(42)

def generate_datasets():
    '''
    This function generates two datasets:
    - dataset_a: Where Naive Bayes performs better
    - dataset_b: Where Logistic Regression performs better

    It also visualizes the datasets with 2D plots for pairwise feature relationships.
    '''
    def generate_dataset_a():
        n_samples = 500

        # Parameters for class 0
        mean_0 = [1, 1, 1]
        cov_0 = np.diag([1, 2, 1.5])

        # Parameters for class 1
        mean_1 = [4, 4, 4]
        cov_1 = np.diag([1.5, 1, 2])

        # Generate samples
        X0 = np.random.multivariate_normal(mean_0, cov_0, n_samples // 2)
        X1 = np.random.multivariate_normal(mean_1, cov_1, n_samples // 2)

        # Combine samples and labels
        X = np.vstack((X0, X1))
        y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

        return X, y

    def generate_dataset_b():
        n_samples = 500

        # Parameters for class 0
        mean_0 = [1, 1, 1]
        cov_0 = [[1, 0.8, 0.6], [0.8, 1, 0.6], [0.6, 0.6, 1]]

        # Parameters for class 1
        mean_1 = [4, 4, 4]
        cov_1 = [[1, 0.8, 0.6], [0.8, 1, 0.6], [0.6, 0.6, 1]]

        # Generate samples
        X0 = np.random.multivariate_normal(mean_0, cov_0, n_samples // 2)
        X1 = np.random.multivariate_normal(mean_1, cov_1, n_samples // 2)

        # Combine samples and labels
        X = np.vstack((X0, X1))
        y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

        return X, y

    # Generate the datasets
    dataset_a_features, dataset_a_labels = generate_dataset_a()
    dataset_b_features, dataset_b_labels = generate_dataset_b()

    # Function to plot the datasets
    def plot_datasets(X_a, y_a, X_b, y_b):
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Plot dataset_a
        axs[0, 0].scatter(X_a[:, 0], X_a[:, 1], c=y_a, cmap='bwr', alpha=0.5)
        axs[0, 0].set_title('**Naive Bayes using EM** - dataset_a: Feature 1 vs Feature 2')
        axs[1, 0].scatter(X_a[:, 0], X_a[:, 2], c=y_a, cmap='bwr', alpha=0.5)
        axs[1, 0].set_title('**Naive Bayes using EM** - dataset_a: Feature 1 vs Feature 3')
        axs[2, 0].scatter(X_a[:, 1], X_a[:, 2], c=y_a, cmap='bwr', alpha=0.5)
        axs[2, 0].set_title('**Naive Bayes using EM** - dataset_a: Feature 2 vs Feature 3')

        # Plot dataset_b
        axs[0, 1].scatter(X_b[:, 0], X_b[:, 1], c=y_b, cmap='bwr', alpha=0.5)
        axs[0, 1].set_title('**Logistic regression** - dataset_b: Feature 1 vs Feature 2')
        axs[1, 1].scatter(X_b[:, 0], X_b[:, 2], c=y_b, cmap='bwr', alpha=0.5)
        axs[1, 1].set_title('**Logistic regression** - dataset_b: Feature 1 vs Feature 3')
        axs[2, 1].scatter(X_b[:, 1], X_b[:, 2], c=y_b, cmap='bwr', alpha=0.5)
        axs[2, 1].set_title('**Logistic regression** - dataset_b: Feature 2 vs Feature 3')

        plt.show()

    # Plot the datasets
    plot_datasets(dataset_a_features, dataset_a_labels, dataset_b_features, dataset_b_labels)

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }
