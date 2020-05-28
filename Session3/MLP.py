import random

import tensorflow as tf
import numpy as np

print(tf.__version__)
NUM_CLASSES = 20
tf.compat.v1.disable_eager_execution()


class MLP:
    def __init__(self, vocab_size, hidden_size):
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

    def build_graph(self):
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, self._vocab_size])
        self.real_Y = tf.compat.v1.placeholder(tf.int32, shape=[None, ])

        # weights, bias
        weights_1 = tf.compat.v1.get_variable(
            name='weights_input_hidden',
            shape=(self._vocab_size, self._hidden_size),
            initializer=tf.random_normal_initializer(seed=2018),
        )

        biases_1 = tf.compat.v1.get_variable(
            name='biases_input_hidden',
            shape=self._hidden_size,
            initialiser=tf.random_normal_initializer(seed=2018)
        )

        weights_2 = tf.compat.v1.get_variable(
            name='weights_hidden_ouput',
            shape=(self._hidden_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2018),
        )

        biases_2 = tf.compat.v1.get_variable(
            name='biases_hidden_output',
            shape=NUM_CLASSES,
            initialiser=tf.random_normal_initializer(seed=2018)
        )

        hidden = tf.matmul(self._X, weights_1) + biases_1
        hidden = tf.sigmoid(hidden)
        logits = tf.matmul(hidden, weights_2) + biases_2

        labels_one_hot = tf.one_hot(indices=self._real_Y, depth=NUM_CLASSES, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)

        prob = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(prob, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    @staticmethod
    def trainer(loss, learning_rate):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op

    @property
    def X(self):
        return self._X

    @property
    def real_Y(self):
        return self._real_Y

    @X.setter
    def X(self, value):
        self._X = value

    @real_Y.setter
    def real_Y(self, value):
        self._real_Y = value


class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        self._batch_size = batch_size
        self._data = []
        self._labels = []
        
        with open(data_path) as f:
            d_lines = f.read().splitlines()
            
        for data_id, line in enumerate(d_lines):
            vector = [0.0 for _ in range(vocab_size)]
            features = line.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            tokens = features[2].split()
            for token in tokens:
                index, value = int(token.split(':')[0]), float(token.split(':')[1])
                vector[index] = value
            self._data.append(vector)
            self._labels.append(label)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        # Load data from batch and shuffle
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0

            indices = list(range(len(self._data)))
            random.seed(2018)
            random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        return self._data[start:end], self._labels[start:end]

    @property
    def batch_id(self):
        return self._batch_id


# Save parameters of model
def save_parameters(name, value, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1:  # is a vector
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[row]]) for row in range(value.shape[0])])
    with open('./saved_paras/saved_paras' + filename, 'w') as f:
        f.write(string_form)
    return epoch


# Restore parameters
def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open('./saved_paras/saved_paras' + filename) as f:
        lines = f.read().splitlines()
        if len(lines) == 1:  # is a vector
            value = [float(number) for number in lines[0].split(',')]
        else:  # is a matrix
            value = [[float(number) for number in lines[row].split(',')] for row in range(len(lines))]
    return value


def main():
    def get_vocab_size():
        with open('../20news-bydate/words_idfs.txt') as f:
            lines = f.read().splitlines()
        return len(lines)

    vocab_size = get_vocab_size()
    print(vocab_size)
    mlp = MLP(vocab_size=vocab_size, hidden_size=50)
    predicted_labels, loss = mlp.build_graph()
    train_op = mlp.trainer(loss=loss, learning_rate=0.01)

    with tf.compat.v1.Session() as sess:
        train_data_reader = DataReader(data_path='../20news-bydate/20news_train_tf_idf.txt', batch_size=50, vocab_size=vocab_size)
        step, MAX_STEP = 0, 5000
        sess.run(tf.compat.v1.global_variables_initializer())
        while step < MAX_STEP:
            train_data, train_labels = train_data_reader.next_batch()
            predicted_label_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    mlp.X: train_data,
                    mlp.real_Y: train_labels
                }
            )
            step += 1
            print("step: {}, loss: {}".format(step, loss_eval))
            if loss_eval < 1e-5:
                break

        # Save parameters
        trainable_variables = tf.compat.v1.trainable_variables()
        for variable in trainable_variables:
            epo = save_parameters(
                name=variable.name,
                value=variable.eval(),
                epoch=train_data_reader._num_epoch
            )

    test_data_reader = DataReader(data_path='../20news-bydate/20news_test_tf_idf.txt', batch_size=50, vocab_size=vocab_size)

    with tf.compat.v1.Session() as sess:
        epoch = epo

        trainable_variables = tf.compat.v1.trainable_variables()
        for variable in trainable_variables:
            saved_value = restore_parameters(variable.name, epoch)
            assign_op = variable.assign(saved_value)
            sess.run(assign_op)

        num_true_preds = 0
        while True:
            test_data, test_labels = test_data_reader.next_batch()
            test_plabels_eval = sess.run(
                predicted_labels,
                feed_dict={
                    mlp.X: test_data,
                    mlp.real_Y: test_labels
                }
            )
            matches = np.equal(test_plabels_eval, test_labels)
            num_true_preds += np.sum(matches.astype(float))

            if test_data_reader.batch_id == 0:
                break

        print("Epoch: ", epoch)
        print("Accuracy on test data: ", num_true_preds / len(test_data_reader._data))


if __name__ == "__main__":
    main()
