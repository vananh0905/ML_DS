from collections import defaultdict
import random
import numpy as np

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf

        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('./20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    ids = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        labels.append(label)
        data.append(r_d)
        ids.append(doc_id)

    return data, labels, ids

# use scikit-learn
def clustering_with_KMeans():
    data, labels, ids = load_data('./20news-bydate/data_tf_idf.txt')

    #use csr_matrix to create a sparse matrix with efficient row slicing
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix

    X = csr_matrix(data)
    kmeans = KMeans(
        n_clusters=20,      # number of clusters
        init = 'random',   # method of initialization (default = kmean++)
        n_init=5,          # number of time the Kmeans algorithm will be run with different centroid seeds
        tol=1e-3,          # relative tolerance with regards to inertia to declare convergence
        random_state=2020  # set to get deterministic results
    ).fit(X)
    clusters = kmeans.labels_
    with open('./20news-bydate/result_kmeans_by_scikit-learn.txt','w') as f:
        f.write('\n'.join([str(labels[i]) + '<fff>' + str(ids[i]) + '<fff>' + str(clusters[i])
                          for i in range(len(data))]))

class Member:
    def __init__(self, r_d, label, doc_id):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

class Cluster:
    def __init__(self, centroid=None):
        self._centroid = centroid # centroid of Cluster
        self._members = [] # list member in Cluster

    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)

class Kmeans:
    #initialization
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters # number of clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)] # list of Clusters
        self._E = [] # list of centroids
        self._S = 0  # overall similarity

    #load data from data_path
    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                    index = int(index_tfidf.split(':')[0])
                    tfidf = float(index_tfidf.split(':')[1])
                    r_d[index] = tfidf

            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('./20news-bydate/words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())

        self._data = []  # list of Member
        self._label_count = defaultdict(int) # the number of members in label[i]
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    # choose centroid randomly from data for cluster
    def random_init(self, seed_value):
        for i in range(seed_value):
            x = random.choice(self._data)
            self._E.append(x)
            self._clusters[i]._centroid = x

    def compute_similarity(self, member, centroid):
        #cosine_similarity
        cosine = np.abs(np.sum(member._r_d * centroid._r_d)) / (np.sqrt(np.sum(member._r_d ** 2)) * np.sqrt(np.sum(centroid._r_d ** 2)))
        return cosine

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1

        # choose the best cluster for member
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity

        best_fit_cluster.add_member(member) # add member for cluster
        return max_similarity

    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])

        cluster._centroid._r_d = new_centroid #update new centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria

        if criterion == 'max_iters': # the number of iteration > threshold -> return
            if self._iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid': # the number of new centroid < threshold -> return
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new
                             if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else: # similarity. The increase of similarity < threshold -> return
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

    # run algorithm
    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)

        #continually update clusters util convergence
        self._iteration = 0
        while True:
            #reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0

            #choose member for cluster
            for member in self._data:
                max_s = self.select_cluster_for(member=member)
                self._new_S += max_s

            #update centroid of cluster
            for cluster in self._clusters:
                self.update_centroid_of(cluster=cluster)

            self._iteration += 1
            #check stopping condition
            if self.stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1./len(self._data)

    # normalized mutual information
    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += - wk / N * np.log10(wk/N)
            member_labels = [member._label for member in cluster._members]

            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj =self._label_count[label]
            H_C += - cj / N * np.log10(cj / N + 1e-12)
        return I_value * 2 / (H_C + H_omega)

def print_result(seed_value, result):
    with open('./20news-bydate/result_kmeans.txt','w') as f:
        f.write('\n'.join([str(member._label) + '<fff>' + str(member._doc_id) + '<fff>' + str(i)
                          for i in range(seed_value) for member in result._clusters[i]._members]))


if __name__ == "__main__":
    max_Kmeans = -1
    result = Kmeans(num_clusters=20) # store the best result

    #choose the best result
    for _ in range(5):
        kmeans = Kmeans(num_clusters=20)
        kmeans.load_data('./20news-bydate/data_tf_idf.txt')
        kmeans.run(seed_value=20, criterion='similarity', threshold=1e5)
        compute_NMI = kmeans.compute_NMI()
        if max_Kmeans < compute_NMI:
            max_Kmeans = compute_NMI
            result = kmeans

    print_result(seed_value=20, result=result)
    clustering_with_KMeans()
