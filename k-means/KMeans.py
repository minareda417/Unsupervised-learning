import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, x: np.array(np.array(np.float32))):
        self.x = x
        self.N = self.x.shape[0]
        self.D = self.x.shape[1]

    def _smart_initialization(self, k: int) -> np.array(np.array(np.float32)):
        centroids = [self.x[np.random.randint(self.N)]]
        for i in range(k-1):
            distances = []
            for point in self.x:
                distances.append(min([np.linalg.norm(point - centroid) for centroid in centroids]))
            distances = np.array(distances)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            new_centroid = self.x[np.random.choice(len(self.x), p=probabilities)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def _random_initialization(self, k: int) -> np.array(np.array(np.float32)):
        return np.random.default_rng().choice(self.x, k, replace=False)

    def _cluster(self, centroids: np.array(np.array(np.float32))) -> np.array(np.array(np.float32)):
        distances =  np.linalg.norm((self.x[:, np.newaxis] - centroids), axis = 2)
        inertia = np.sum(np.min(distances, axis = 1))
        return np.argmin(distances, axis = 1), inertia

    def _calculate_new_centroids(self, k: int, clusters: np.array(np.array(np.float32))) -> np.array(np.array(np.float32)):
        new_centroids = np.zeros((k, self.D))
        for i in range(k):
            new_centroids[i] = self.x[clusters == i].mean(axis=0)
        return new_centroids

    def fit(self, k: int, smart_initialization: bool = True, tol: float = 1e-4, max_iter: int = 300, plot_inertia: bool = False):
        assert self.N >= k > 0
        centroids = self._smart_initialization(k) if smart_initialization else self._random_initialization(k)
        inertias = []
        for _ in range(max_iter):
            clusters, inertia = self._cluster(centroids)
            new_centroids = self._calculate_new_centroids(k, clusters)
            inertias.append(inertia)
            if np.linalg.norm((new_centroids - centroids), axis = -1) <= tol:
                break
            centroids = new_centroids
        if plot_inertia:
            plt.plot(np.linspace(1, len(inertias), len(inertias)),inertias)
            plt.title("Inertia over iterations")
            plt.ylabel("inertia")
            plt.xlabel("iteration")
            plt.tight_layout()
            plt.show()

    def find_best_k(self):
        pass






