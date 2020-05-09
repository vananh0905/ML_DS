import numpy as np

#load data from data_path
def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size=14410):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf

        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2])

        labels.append(label)
        data.append(r_d)

    return data, labels


def compute_accuracy(predicted_Y, expected_Y):
    matches = np.equal(predicted_Y, expected_Y)
    accuracy = np.sum(matches.astype(float) / len(expected_Y))
    return accuracy

def classifying_with_linear_SVMs():
    train_X, train_Y = load_data('./20news-bydate/20news_train_tf_idf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,       #  penalty coeff
        tol=0.001,    #  Tolerance for stopping criteria.
        verbose=True  #  whether prints out logs or not
    )
    classifier.fit(train_X, train_Y)

    test_X, test_Y = load_data('./20news-bydate/20news_test_tf_idf.txt')

    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_Y=predicted_Y, expected_Y=test_Y)
    print('Accuracy: ' + str(accuracy))

if __name__ == '__main__':
    classifying_with_linear_SVMs()