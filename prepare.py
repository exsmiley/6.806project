'''Prepares the data for our neural networks'''
import gzip
import numpy as np
import torch
import torch.utils.data as d
from torch import Tensor
import time
import tqdm
import random

start = time.time()

NUM_NEGATIVE_SAMPLES = 20
LEN_EMBEDDING = 200

def load_word_vector():
    embedding_path = 'vector/vectors_pruned.200.txt.gz'
    embeddings = [np.zeros(LEN_EMBEDDING)]
    word_to_index = {}

    print "Embeddings Loading..."

    with gzip.open(embedding_path) as file:
        for i,line in tqdm.tqdm(enumerate(file)):
            line = line.split()
            word = line[0]
            vec = np.array(map(float, line[1:]), dtype='float64')
            embeddings.append(vec)
            word_to_index[word] = i + 1

    print "Embeddings Loaded! \n"

    return np.array(embeddings), word_to_index


embeddings, word_to_index = load_word_vector()

def get_word_index(word):
    if word in word_to_index:
        return word_to_index[word]
    else:
        return 0

def load_tokenized_text(fname, name, body_trim_length=100):
    '''Creates dictionary of {id: (title, body)}
    This is okay to have in memory since it's just like 160MB or so...
    '''
    print name + " Question Text Loading..."

    data = {}
    with gzip.open(fname) as f:
        for line in tqdm.tqdm(f):
            sections = line.strip().split('\t')
            qid, title, body  = sections[0], sections[1], sections[2] if len(sections) > 2 else ''

            body = " ".join(body.split(' ')[:body_trim_length]) # Trims body to 100 words
            body = [get_word_index(word) for word in body]

            title = [get_word_index(word) for word in sections[1]]

            data[qid] = (title, body)

    print name + " Question Text Loaded! \n"
    return data


# use as global variable for reading from

ubuntu_data = load_tokenized_text('ubuntu_data/text_tokenized.txt.gz', 'Ubuntu')
# android_data = load_tokenized_text('android_data/corpus.tsv.gz', 'Android')


def load_ubuntu_examples(file):
    '''yields data in the form of
    (id, query (title, body), example (title, body), +1/-1)'''

    print "Loading Data From " + file + "..."

    if 'dev' in file or 'test' in file:
        with open(file) as f:
            for line in tqdm.tqdm(f):
                query, positive, negative, bm25 = map(lambda x: x.split(), line.split('\t'))
                query = query[0]
                for p in positive:
                    yield (query, p, negative, bm25)

    else:
        with open(file) as f:
            for line in tqdm.tqdm(f):
                query, positive, negative = map(lambda x: x.split(), line.split('\t'))
                query = query[0]
                for p in positive:
                    yield (query, p, negative)

    print "Loaded Data From " + file + "! \n"

# def load_android_examples(dev=False, test=False):
#     '''yields data in the form of
#     (id, query (title, body), example (title, body), +1/-1)'''
#     assert dev or test  # can only be dev or test sets
#     neg_file = 'android_data/dev.neg.txt'
#     pos_file = 'android_data/dev.pos.txt'
#     if test:
#         neg_file = 'android_data/test.neg.txt'
#         pos_file = 'android_data/test.pos.txt'
#     with open(neg_file) as f:
#         for line in f:
#             query, compare = line.split()
#             yield (query, android_data[query], android_data[compare], -1)
#     with open(pos_file) as f:
#         for line in f:
#             query, compare = line.split()
#             yield (query, android_data[query], android_data[compare], 1)


# I DON'T THINK WE NEED THIS BUT YOU DECIDE

# def example_to_word_vector(query, pos, neg):
#     query = ''.join(query).split()
#     pos = ''.join(pos).split()
#     neg = ''.join(neg).split()
#
#     query_vec = np.zeros(200)
#     pos_vec = np.zeros(200)
#     neg_vec = np.zeros(200)
#     count = 0
#     for word in query:
#         if word in embeddings:
#             count += 1
#             query_vec += embeddings[word]
#     if count > 0:
#         query_vec = query_vec/count
#         count = 0
#     for word in pos:
#         if word in embeddings:
#             pos_vec += embeddings[word]
#             count += 1
#     if count > 0:
#         pos_vec = pos_vec/count
#         count = 0
#     for word in neg:
#         if word in embeddings:
#             neg_vec += embeddings[word]
#             count += 1
#     if count > 0:
#         neg_vec = neg_vec/count
#     return (query_vec, pos_vec, neg_vec)
#
#
# class UbuntuDataSet(d.Dataset):
#     '''Loads the training set for the Ubuntu Dataset'''
#
#     def __init__(self):
#         self.data = list(load_ubuntu_examples())
#
#     def __getitem__(self, index):
#         (query_vec, pos_vec, neg_vec) = example_to_word_vector(*self.data[index])
#         return torch.from_numpy(np.vstack((query_vec, pos_vec))).float(), torch.from_numpy(np.vstack((query_vec, neg_vec))).float(), 1.0
#
#     def __len__(self):
#         return len(self.data)


class UbuntuSequentialDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset with sequential word vectors'''

    def __init__(self, file):
        self.data = list(load_ubuntu_examples(file))

    def __getitem__(self, index):
        qid, pid, nids = self.data[index]

        neg_samples = random.sample(nids, NUM_NEGATIVE_SAMPLES)
        candidate_set = [pid] + neg_samples

        q = ubuntu_data[qid]
        c = [ubuntu_data[i] for i in candidate_set]
        l = [1] + [0] * len(nids)

        return q, c, l


    def __len__(self):
        return len(self.data)

# TODO make data set for evaluation of dev/test
class UbuntuEvaluationDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset with sequential word vectors'''

    def __init__(self, file):
        self.data = list(load_ubuntu_examples(file))

    def __getitem__(self, index):
        qid, pid, nids, bm25s = self.data[index]

        nids = zip(nids, bm25s)

        neg_samples = random.sample(nids, NUM_NEGATIVE_SAMPLES)
        bm25s = [s[1] for s in neg_samples]
        neg_samples = [s[0] for s in neg_samples]
        candidate_set = [pid] + neg_samples

        q = ubuntu_data[qid]
        c = [ubuntu_data[i] for i in candidate_set]
        l = [1] + [0] * len(nids)

        return q, c, l, bm25s


    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    s = UbuntuSequentialDataSet()
    print "total time:", time.time()-start
