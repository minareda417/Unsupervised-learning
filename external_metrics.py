import numpy as np

def adjusted_rand_index(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n = len(labels_true)

    def comb2(n):
        return n*(n-1)/2

    # unique labels
    true_classes = np.unique(labels_true)
    pred_classes = np.unique(labels_pred)

    # contingency table
    cont = np.zeros((len(true_classes), len(pred_classes)))
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            cont[i,j] = np.sum((labels_true==t) & (labels_pred==p))

    sum_comb_c = np.sum(comb2(cont))
    sum_comb_t = np.sum(comb2(np.sum(cont, axis=1)))
    sum_comb_p = np.sum(comb2(np.sum(cont, axis=0)))

    expected_index = sum_comb_t * sum_comb_p / comb2(n)
    max_index = (sum_comb_t + sum_comb_p) / 2
    ari = (sum_comb_c - expected_index) / (max_index - expected_index)
    return ari

def normalized_mutual_info(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n = len(labels_true)

    true_classes = np.unique(labels_true)
    pred_classes = np.unique(labels_pred)

    cont = np.zeros((len(true_classes), len(pred_classes)))
    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            cont[i,j] = np.sum((labels_true==t) & (labels_pred==p))

    pij = cont / n
    pi = np.sum(pij, axis=1)
    pj = np.sum(pij, axis=0)

    # mutual info
    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / (pi[:,None][nz] * pj[None,:][nz])))

    # entropy
    h_true = -np.sum(pi * np.log(pi + 1e-12))
    h_pred = -np.sum(pj * np.log(pj + 1e-12))

    return 2*mi / (h_true + h_pred)

def purity_score(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    clusters = np.unique(labels_pred)
    total = 0
    for c in clusters:
        indices = np.where(labels_pred == c)[0]
        counts = np.zeros(np.max(labels_true)+1)
        for idx in indices:
            counts[labels_true[idx]] += 1
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

