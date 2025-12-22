import numpy as np

def silhouette_score(X, labels):
    n = X.shape[0]
    s = np.zeros(n)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0.0 
    
    for i in range(n):
        same = labels == labels[i]
        other_clusters = [l for l in unique_labels if l != labels[i]]
        a = np.mean(np.linalg.norm(X[i] - X[same], axis=1))
        if other_clusters:
            b = np.min([np.mean(np.linalg.norm(X[i] - X[labels == l], axis=1)) for l in other_clusters])
        else:
            b = 0.0
        s[i] = (b - a) / max(a, b)
    return np.mean(s)

def davies_bouldin(X, labels):
    clusters = np.unique(labels)
    if len(clusters) == 1:
        return 0.0 
    
    centroids = np.array([X[labels==c].mean(axis=0) for c in clusters])
    s = np.array([np.mean(np.linalg.norm(X[labels==c] - centroids[i], axis=1)) for i,c in enumerate(clusters)])
    
    db = []
    for i in range(len(clusters)):
        R = []
        for j in range(len(clusters)):
            if j != i:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist > 0:
                    R.append((s[i] + s[j]) / dist)
        if R:
            db.append(max(R))
    if db:
        return np.mean(db)
    else:
        return 0.0

def calinski_harabasz(X, labels):
    n, d = X.shape
    clusters = np.unique(labels)
    k = len(clusters)
    if k == 1:
        return 0.0  
    overall_mean = X.mean(axis=0)
    
    W = 0
    B = 0
    for c in clusters:
        cluster_points = X[labels==c]
        centroid = cluster_points.mean(axis=0)
        W += np.sum((cluster_points - centroid)**2)
        B += len(cluster_points) * np.sum((centroid - overall_mean)**2)
    
    return (B / (k-1)) / (W / (n-k))

def wcss(X, labels):
    clusters = np.unique(labels)
    total = 0
    for c in clusters:
        cluster_points = X[labels==c]
        centroid = cluster_points.mean(axis=0)
        total += np.sum((cluster_points - centroid)**2)
    return total
