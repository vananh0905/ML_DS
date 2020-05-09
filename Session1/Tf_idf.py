from collections import defaultdict
from os import listdir
from os.path import isfile
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def get_tf_idf(datapath):
    with open('./20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                      for line in f.read().splitlines()]
        word_IDs = dict([(word, index)
                         for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    with open(datapath) as f:
        documents = [
            (int(line.split('<fff>')[0]),
             int(line.split('<fff>')[1]),
             line.split('<fff>')[2])
            for line in f.read().splitlines()]

    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        wordset = list(set(words))
        max_term_freq = max([words.count(word) for word in wordset])

        words_tfidfs = []
        sum_squares = 0.0
        for word in wordset:
            term_freq = words.count(word)
            tf_idf_value = term_freq * 1. / max_term_freq * idfs[word]
            words_tfidfs.append((word_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value ** 2

        words_tfidfs_normalized = [str(index) + ':' + str(tf_idf_value / np.sqrt(sum_squares))
                                   for index, tf_idf_value in words_tfidfs]

        sparse_rep = ' '.join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open('./20news-bydate/data_tf_idf.txt', 'w') as f:
        f.write('\n'.join(
            [str(label) + '<fff>' + str(word_id) + '<fff>' + sparse_rep for label, word_id, sparse_rep in data_tf_idf]))


def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)

    with open(data_path) as f:
        lines = f.read().splitlines()
        doc_count = defaultdict(int)
        corpus_size = len(lines)

        for line in lines:
            features = line.split('<fff>')
            text = features[-1]
            words = list(set(text.split()))
            for word in words:
                doc_count[word] += 1

        words_idfs = [(word,compute_idf(document_freg, corpus_size))
                  for word, document_freg in zip(doc_count.keys(), doc_count.values())
                    if document_freg > 10 and not word.isdigit()]

        words_idfs.sort(key=lambda word: -word[1])

        print('Vocabulary size: {}'.format(len(words_idfs)))
        with open('./20news-bydate/words_idfs.txt', 'w') as f:
            f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))

def gather_20newsgroups_data():
    path = './20news-bydate/'
    dirs = [path + dir_name + '/' for dir_name in listdir(path)
            if not isfile(path + dir_name)]

    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])

    list_newsgroups = [newsgroup for newsgroup in listdir(train_dir)]
    list_newsgroups.sort()

    with open('./20news-bydate/stop_word') as f:
        stop_words = f.read().splitlines()

    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'

            files = [(filename, dir_path + filename)
                    for filename in listdir(dir_path)
                    if isfile(dir_path + filename)]

            files.sort()
            symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                     # remove stop words
                    words = [stemmer.stem(word)
                             for word in re.split('\W+', text)
                            if word not in stop_words and word not in symbols]
                    # combine remaining words
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data

    train_data = collect_data_from(parent_dir=train_dir, newsgroup_list=list_newsgroups)
    test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)

    full_data = train_data + test_data
    with open('./20news-bydate/20news-train-processed.text','w') as f:
        f.write('\n'.join(train_data))

    with open('./20news-bydate/20news-test-processed.text','w') as f:
        f.write('\n'.join(test_data))

    with open('./20news-bydate/20news-full-processed.text','w') as f:
        f.write('\n'.join(full_data))
    return full_data

if __name__ == "__main__":
    #gather_20newsgroups_data()
    generate_vocabulary('./20news-bydate/20news-full-processed.text')
    get_tf_idf('./20news-bydate/20news-full-processed.text')
