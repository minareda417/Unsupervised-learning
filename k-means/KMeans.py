import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, x: np.ndarray, seed: int = 42):
        self.x = x
        self.N = self.x.shape[0]
        self.D = self.x.shape[1]
        self.rng = np.random.RandomState(42)
        np.random.seed(seed)

    def _smart_initialization(self, k: int) -> np.ndarray:
        centroids = [self.x[self.rng.randint(self.N)]]
        for i in range(k-1):
            distances = []
            for point in self.x:
                distances.append(min([np.linalg.norm(point - centroid) for centroid in centroids]))
            distances = np.array(distances)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            new_centroid = self.x[self.rng.choice(len(self.x), p=probabilities)]
            centroids.append(new_centroid)
        return np.array(centroids)

    def _random_initialization(self, k: int) -> np.ndarray:
        return self.x[self.rng.choice(len(self.x), k, replace=False)]

    def _cluster(self, centroids: np.ndarray):
        distances =  np.linalg.norm((self.x[:, np.newaxis] - centroids), axis = 2)
        inertia = np.sum(np.min(distances, axis = 1))
        return np.argmin(distances, axis = 1), inertia

    def _calculate_new_centroids(self, k: int, clusters: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros((k, self.D))
        for i in range(k):
            new_centroids[i] = self.x[clusters == i].mean(axis=0)
        return new_centroids

    def fit(self, k: int, smart_initialization: bool = True, tol: float = 1e-4, max_iter: int = 300, plot_metrics: bool = False):
        assert self.N >= k > 0
        centroids = self._smart_initialization(k) if smart_initialization else self._random_initialization(k)
        inertias = []
        for _ in range(max_iter):
            clusters, inertia = self._cluster(centroids)
            new_centroids = self._calculate_new_centroids(k, clusters)
            inertias.append(inertia)
            if np.all(np.linalg.norm(new_centroids - centroids, axis=-1) <= tol):
                break
            centroids = new_centroids
        results = {"inertia": inertia, "silhouette score": self._calculate_silhouette(clusters, k), "gap score": self._calculate_gap(k, inertia, tol=tol, max_iter=max_iter)}
        if plot_metrics:
            print(f"inertia = {results['inertia']}")
            print(f"Silhouette score = {results['silhouette score']}")
            print(f"Gap score = {results['gap score']}")
            plt.plot(np.linspace(1, len(inertias), len(inertias)),inertias)
            plt.title("Inertia over iterations")
            plt.ylabel("inertia")
            plt.xlabel("iteration")
            plt.tight_layout()
            plt.show()
        return clusters, centroids, results

    def _calculate_silhouette(self, clusters: np.ndarray, k: int):
        silhouette_scores = np.zeros(self.N)
        for i in range(self.N):
            cluster = clusters[i]
            same_custer = self.x[clusters == cluster]
            if len(same_custer) > 1:
                a_i = np.mean([np.linalg.norm(self.x[i]-point) for point in same_custer if not np.array_equal(point, self.x[i])])
            else:
                a_i = 0
            b_i = np.inf
            for j in range(k):
                if j != cluster:
                    other_cluster = self.x[clusters == j]
                    if len(other_cluster) > 0:
                        mean_distance = np.mean([np.linalg.norm(self.x[i] - point) for point in other_cluster])
                        b_i = min(b_i, mean_distance)
            if max(a_i, b_i) > 0:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        return np.mean(silhouette_scores)

    def _calculate_gap(self, k: int, inertia: float, n_refs: int = 10, tol: float = 1e-4, max_iter: int = 100) -> float:
        mins = self.x.min(axis=0)
        maxs = self.x.max(axis=0)
        ref_inertias = []
        for _ in range(n_refs):
            ref_data = self.rng.uniform(mins, maxs, size=self.x.shape)
            ref_centroids = ref_data[self.rng.choice(len(ref_data), k, replace=False)]
            for _ in range(max_iter):
                ref_distances = np.linalg.norm((ref_data[:, np.newaxis] - ref_centroids), axis=2)
                ref_clusters = np.argmin(ref_distances, axis=1)
                ref_inertia = np.sum(np.min(ref_distances, axis=1))
                new_ref_centroids = np.zeros((k, self.D))
                for i in range(k):
                    if np.any(ref_clusters == i):
                        new_ref_centroids[i] = ref_data[ref_clusters == i].mean(axis=0)
                    else:
                        new_ref_centroids[i] = ref_centroids[i]
                if np.all(np.linalg.norm((new_ref_centroids - ref_centroids), axis=-1)) <= tol:
                    break
                ref_centroids = new_ref_centroids
            ref_inertias.append(np.log(ref_inertia))
        gap = np.mean(ref_inertias) - np.log(inertia)
        return gap

    def find_best_k(self, k_values=None, smart_initialization: bool = True, tol: float = 1e-4, max_iter: int = 300) -> dict[str, int]:
        if k_values is None:
            k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        elbows = []
        silhouette_scores = []
        gap_scores = []
        for k in k_values:
            clusters, centroids, results = self.fit(k, smart_initialization, tol, max_iter)
            elbows.append(results['inertia'])
            silhouette_scores.append(results['silhouette score'])
            gap_scores.append(results['gap score'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(k_values, elbows, 'bo-')
        axes[0].set_xlabel('Number of clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)

        axes[1].plot(k_values, silhouette_scores, 'go-')
        axes[1].set_xlabel('Number of clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        best_sil_k = list(k_values)[np.argmax(silhouette_scores)]
        axes[1].axvline(x=best_sil_k, color='r', linestyle='--', alpha=0.5)
        axes[1].grid(True)

        axes[2].plot(k_values, gap_scores, 'ro-')
        axes[2].set_xlabel('Number of clusters (k)')
        axes[2].set_ylabel('Gap Statistic')
        axes[2].set_title('Gap Statistic')
        best_gap_k = list(k_values)[np.argmax(gap_scores)]
        axes[2].axvline(x=best_gap_k, color='r', linestyle='--', alpha=0.5)
        axes[2].grid(True)
        plt.tight_layout()
        plt.show()

        best_k_silhouette = list(k_values)[np.argmax(silhouette_scores)]
        best_k_gap = list(k_values)[np.argmax(gap_scores)]

        print(f"Best k by Silhouette Score: {best_k_silhouette} (score: {max(silhouette_scores):.4f})")
        print(f"Best k by Gap Statistic: {best_k_gap} (gap: {max(gap_scores):.4f})")
        print("Check the Elbow plot for visual inflection point")

        return {
                'k_values': list(k_values),
                'inertias': elbows,
                'silhouette_scores': silhouette_scores,
                'gap_stats': gap_scores,
                'best_k_silhouette': best_k_silhouette,
                'best_k_gap': best_k_gap
            }