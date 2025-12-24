import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PCA:
    def __init__(self, x:np.ndarray):
        self.x = x
        self.N = self.x.shape[0]
        self.D = self.x.shape[1]
        self.principal_components, self.eigenvalues, self.covariance_matrix = self._get_principal_components()

    def _get_principal_components(self):
        """
            Time complexity -> O(N*D^2+ D^3), Space complexity -> O(D^2)
        """
        covariance_matrix = (self.x.T @ self.x) / self.N # O(D*D*N)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) # O(D^3)
        sort = np.argsort(eigenvalues)[::-1] # O(DlogD)
        principal_components = eigenvectors[:, sort]
        return principal_components, np.sort(eigenvalues)[::-1], covariance_matrix

    def plot_covariance_matrix(self):
        plt.matshow(self.covariance_matrix, cmap='Blues', aspect='auto')
        plt.title('Covariance Matrix', pad=20, fontsize=14, fontweight='bold')
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Feature Index', fontsize=12)
        plt.show()

    def plot2d(self, y:pd.DataFrame):
        z = self.fit_transform(2)
        unique_labels = y.unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(z[mask, 0], z[mask, 1], c=[colors[i]], label=label, alpha=0.7)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA - 2D Projection')
        plt.legend()
        plt.show()

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
        ax[1].axhline(y=50, color="grey", linestyle='--')
        ax[1].axhline(y=80, color="grey", linestyle='--')
        ax[1].axhline(y=80, color="grey", linestyle='--')
        ax[1].axhline(y=90, color="grey", linestyle='--')
        ax[1].axhline(y=95, color="grey", linestyle='--')
        ax[1].axhline(y=97, color="grey", linestyle='--')
        ax[1].axhline(y=80, color="grey", linestyle='--')
        ax[1].axhline(y=99, color="grey", linestyle='--')
        ax[1].axhline(y=100, color="grey", linestyle='--')

        plt.show()


    def explained_variance_calculation(self):
        explained_variance = self.eigenvalues * 100 / np.sum(self.eigenvalues)
        cum_explained_variance = np.cumsum(explained_variance)
        thresholds = [50, 70, 80, 90, 95, 97, 98, 99, 100]
        for i in range(self.D):
            print(f"Principal component {i+1}:")
            print(f"\teigenvalue = {self.eigenvalues[i]}")
            print(f"\tvariance ratio = {explained_variance[i]}")
            print(f"\tcumulative variance ratio = {cum_explained_variance[i]}")
            print(f"\treconstruction error = {cum_explained_variance[-1]-cum_explained_variance[i]}")
            print("-"*50)
            print()
        print("=" * 50)
        print("Threshold Analysis:")
        for threshold in thresholds:
            idx = np.where(cum_explained_variance >= threshold)[0]
            if len(idx) > 0:
                first_component = idx[0] + 1  # +1 for 1-based indexing
                print(f"First component to exceed {threshold}%: PC{first_component} ")
        print("=" * 50)
        print()
        self._plot_explained_variance(explained_variance, cum_explained_variance)


    def _get_n_components(self, n: int):
        assert self.D >= n > 0
        return self.principal_components[:, :n]

    def fit_transform(self, n: int):
        n_principal_components = self._get_n_components(n)
        return self.x @ n_principal_components

    def inverse_transform(self, z: np.ndarray):
        n = z.shape[1]
        n_principal_components = self.principal_components[:, :n]
        return z @ n_principal_components.T

    def mse(self, z: np.ndarray):
        reconstructed_x = self.inverse_transform(z)
        reconstruction_error = np.mean((self.x - reconstructed_x) ** 2)
        print(reconstruction_error * 100)



