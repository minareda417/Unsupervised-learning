import numpy as np

def adjusted_rand_index(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n = len(labels_true)

    def comb2(n):
        return n*(n-1)/2

    cm = confusion_matrix(labels_true, labels_pred)

    sum_comb_c = np.sum(cm * (cm - 1) / 2)
    sum_comb_t = np.sum(comb2(np.sum(cm, axis=1)))
    sum_comb_p = np.sum(comb2(np.sum(cm, axis=0)))

    expected_index = sum_comb_t * sum_comb_p / comb2(n)
    max_index = (sum_comb_t + sum_comb_p) / 2
    
    # handle where max_index == expected_index
    if max_index == expected_index:
        return 1.0 if sum_comb_c == expected_index else 0.0
    
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari

def normalized_mutual_info(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    n = np.sum(cm)
    
    # joint probability
    pij = cm / n
    
    # marginal probab
    pi = np.sum(pij, axis=1, keepdims=True)
    pj = np.sum(pij, axis=0, keepdims=True) 
    
    # outer product
    pi_pj = pi * pj 
    
    # only compute where pij > 0
    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / pi_pj[nz]))
    
    # Entropy
    pi = pi.flatten()
    pj = pj.flatten()
    h_true = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    h_pred = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    
    # normalized mutual information
    if h_true == 0 or h_pred == 0:
        return 0.0
    
    nmi = 2 * mi / (h_true + h_pred)
    return nmi

def purity_score(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    pred_clusters = np.unique(labels_pred)
    
    total = 0
    for c in pred_clusters:
        cluster_mask = labels_pred == c
        cluster_labels = labels_true[cluster_mask]
        if len(cluster_labels) > 0:
            unique, counts = np.unique(cluster_labels, return_counts=True)
            total += np.max(counts)
    
    return total / len(labels_true)

def confusion_matrix(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    cm = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, t in enumerate(classes):
        for j, c in enumerate(clusters):
            cm[i,j] = np.sum((labels_true==t) & (labels_pred==c))
    return cm

def assign_clusters_to_classes(labels_true, labels_pred):

    cm = confusion_matrix(labels_true, labels_pred)
    true_classes = np.unique(labels_true)
    pred_clusters = np.unique(labels_pred)
    
    # assign each cluster to the true class it has most samples from
    cluster_to_class = {}
    for j, cluster in enumerate(pred_clusters):
        # find which true class has the most samples in this cluster
        best_class_idx = np.argmax(cm[:, j])
        cluster_to_class[cluster] = true_classes[best_class_idx]
    
    return cluster_to_class

def map_clusters_to_labels(labels_pred, cluster_to_class):
    mapped_labels = np.array([cluster_to_class[c] for c in labels_pred])
    return mapped_labels

def compute_binary_metrics(labels_true, labels_pred, positive_class='M'):

    # map clusters to true classes
    cluster_mapping = assign_clusters_to_classes(labels_true, labels_pred)
    mapped_pred = map_clusters_to_labels(labels_pred, cluster_mapping)
    
    # convert to positive and negative
    y_true_binary = (np.array(labels_true) == positive_class).astype(int)
    y_pred_binary = (mapped_pred == positive_class).astype(int)
    
    # confusion matrix components
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }