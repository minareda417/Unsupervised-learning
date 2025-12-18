import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, x:np.array(np.array(np.float32))):
        self.x = x
        self.N = self.x.shape[0]
        self.D = self.x.shape[1]
        self.principal_components, self.eigenvalues = self._get_principal_components()

    def _get_principal_components(self):
        covariance_matrix = (self.x.T @ self.x) / self.N
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sort = np.argsort(eigenvalues)[::-1]
        principal_components = eigenvectors[:, sort]
        return principal_components, np.sort(eigenvalues)[::-1]

    def _plot_explained_variance(self, explained_variance: np.array(np.float32), cum_explained_variance: np.array(np.float32)):
        x = np.linspace(1, self.D, self.D)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(x, explained_variance, marker='o')
        ax[0].set_title('Explained Variance by Principal Component')
        ax[0].set_xlabel('Principal Component')
        ax[0].set_ylabel('Explained Variance (%)')

        ax[1].plot(x, cum_explained_variance, marker='o', color='orange')
        ax[1].set_title('Cumulative Explained Variance')
        ax[1].set_xlabel('Principal Component')
        ax[1].set_ylabel('Cumulative Explained Variance (%)')
        ax[1].axhline(y=80, color="grey", linestyle='--')
        ax[1].axhline(y=90, color="grey", linestyle='--')
        ax[1].axhline(y=95, color="grey", linestyle='--')
        ax[1].axhline(y=97, color="grey", linestyle='--')
        ax[1].axhline(y=99, color="grey", linestyle='--')
        ax[1].axhline(y=100, color="grey", linestyle='--')

        plt.show()


    def explained_variance_calculation(self):
        explained_variance = self.eigenvalues * 100 / np.sum(self.eigenvalues)
        cum_explained_variance = np.cumsum(explained_variance)
        for i in range(self.D):
            print(f"Principal component {i+1}:")
            print(f"\teigenvalue = {self.eigenvalues[i]}")
            print(f"\tvariance ratio = {explained_variance[i]}")
            print(f"\tcumulative variance ratio = {cum_explained_variance[i]}")
            print(f"\treconstruction error = {cum_explained_variance[-1]-cum_explained_variance[i]}")
            print("="*30)
            print()
        self._plot_explained_variance(explained_variance, cum_explained_variance)


    def fit(self, n: int):
        assert self.D >= n > 0
        n_principal_components = self.principal_components[:, :n]
        return n_principal_components

    def fit_transform(self, n: int):
        n_principal_components = self.fit(n)
        return self.x @ n_principal_components

    def inverse_transform(self, z: np.array(np.array(np.float32))):
        n = z.shape[1]
        n_principal_components = self.principal_components[:, :n]
        return z @ n_principal_components.T


