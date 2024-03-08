#%%
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

class Condiitonal_gmms():
    def __init__(self, n_components=3):
        """
        Initialize the Conditional Gaussian Mixture Model.

        Parameters:
        n_components (int): Number of mixture components.
        """
        self.n_components = n_components
        self.gmm = None
        self.x_length = None

    def fit(self, X, Y):
        """
        Fit the Conditional Gaussian Mixture Model.

        Parameters:
        X (array-like): The input features.
        Y (array-like): The target variables.
        """
        # Combining X and Y for joint modeling
        XY = np.hstack([X, Y])
        self.x_length = X.shape[1]
        
        # Fit a Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=0).fit(XY)


    def conditional_distribution(self, given):
        """
        Compute the conditional distribution given part of the data.

        Parameters:
        given (array-like): The given part of the data.

        Returns:
        Tuple: Means and covariances of the conditional distribution.
        """
        conditional_means = []
        conditional_covariances = []
        
        for i in range(self.n_components):
            # Extract the full covariance and mean
            full_covariance = self.gmm.covariances_[i]
            full_mean = self.gmm.means_[i]

            # Split into sub-components
            cov_XX = full_covariance[:self.x_length, :self.x_length]
            cov_YY = full_covariance[self.x_length:, self.x_length:]
            cov_XY = full_covariance[:self.x_length, self.x_length:]
            cov_YX = full_covariance[self.x_length:, :self.x_length]
            mean_X = full_mean[:self.x_length]
            mean_Y = full_mean[self.x_length:]

            # Compute conditional mean and covariance
            conditional_mean = mean_Y + cov_YX @ np.linalg.inv(cov_XX) @ (given - mean_X)
            conditional_covariance = cov_YY - cov_YX @ np.linalg.inv(cov_XX) @ cov_XY

            conditional_means.append(conditional_mean)
            conditional_covariances.append(conditional_covariance)
        
        return np.array(conditional_means), np.array(conditional_covariances)

    def sample(self, given_X, n_samples=1):
        """
        Sample from the conditional distribution of the Gaussian Mixture Model.

        Parameters:
        given_X (array-like): The given part of the data.
        n_samples (int): Number of samples to generate.

        Returns:
        array: Samples from the conditional distribution.
        """
        means, covariances = self.conditional_distribution(given_X)
        samples = []
        for _ in range(n_samples):
            # Choose a component based on mixing proportions
            component = np.random.choice(len(self.gmm.weights_), p=self.gmm.weights_)
            # Sample from the chosen component
            sample = np.random.multivariate_normal(means[component], covariances[component])
            samples.append(sample)

        return np.array(samples)
 