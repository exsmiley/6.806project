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

def load_tokenized_text(fname, name, return_index=True, body_trim_length=100, title_trim_length=40):
    '''Creates dictionary of {id: (title, body)}
    This is okay to have in memory since it's just like 160MB or so...
    '''
    print name + " Question Text Loading..."

    data = {}
    with gzip.open(fname) as f:
        for line in tqdm.tqdm(f):
            sections = line.strip().split('\t')
            qid, title, body  = sections[0], sections[1], sections[2] if len(sections) > 2 else ''

            b = [0] * body_trim_length
            b_mask = np.zeros(body_trim_length)
            body = " ".join(body.split(' ')[:body_trim_length]) # Trims body to 100 words
            for i, word in enumerate(body.split(' ')):
                if return_index:
                    b[i] = get_word_index(word)
                else:
                    b[i] = word
                b_mask[i] = 1

            t = [0] * title_trim_length
            t_mask = np.zeros(title_trim_length)
            title = " ".join(title.split(' ')[:title_trim_length]) # Trims body to 40 words

            for i, word in enumerate(title.split(' ')):
                if return_index:
                    t[i] = get_word_index(word)
                else:
                    t[i] = word
                t_mask[i] = 1

            data[qid] = (t, t_mask, b, b_mask)

    print name + " Question Text Loaded! \n"
    return data


# use as global variable for reading from
# embeddings, word_to_index = load_word_vector()
ubuntu_data = load_tokenized_text('ubuntu_data/text_tokenized.txt.gz', 'Ubuntu')



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

def load_android_examples(android_data, test=False):
    '''yields data in the form of
    (id, query (title, body), example (title, body), +1/-1)'''

    neg_file = 'android_data/dev.neg.txt'
    pos_file = 'android_data/dev.pos.txt'
    if test:
        neg_file = 'android_data/test.neg.txt'
        pos_file = 'android_data/test.pos.txt'
    d = {}

    with open(pos_file) as f:
        for line in f:
            query, compare = line.split()
            if query not in d:
                d[query] = []
            d[query].append(compare)

    with open(neg_file) as f:
        for line in f:
            query, compare = line.split()
            d[query].append(compare)

    for q, c in d.items():
        yield (q, c[0], c[1:])


class UbuntuSequentialDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset with sequential word vectors'''

    def __init__(self, file):
        self.data = list(load_ubuntu_examples(file))

    def __getitem__(self, index):
        qid, pid, nids = self.data[index]

        neg_samples = random.sample(nids, NUM_NEGATIVE_SAMPLES)
        candidate_set = [pid] + neg_samples

        q = ubuntu_data[qid]
        q_title, q_title_mask, q_body, q_body_mask = q

        c = [ubuntu_data[i] for i in candidate_set]
        c_titles = [i[0] for i in c]
        c_titles_mask = [i[1] for i in c]
        c_bodies = [i[2] for i in c]
        c_bodies_mask = [i[3] for i in c]

        l = [1] + [0] * len(neg_samples)


        return Tensor(q_title), Tensor(q_title_mask), Tensor(q_body), Tensor(q_body_mask), Tensor(c_titles), Tensor(c_titles_mask), Tensor(c_bodies), Tensor(c_bodies_mask), Tensor(l)


    def __len__(self):
        return len(self.data)

# TODO make data set for evaluation of dev/test
class UbuntuEvaluationDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset with sequential word vectors'''

    def __init__(self, file):
        self.data = list(load_ubuntu_examples(file))

    def __getitem__(self, index):
        qid, pid, nids, bm25s = self.data[index]

        neg_samples = random.sample(nids, NUM_NEGATIVE_SAMPLES)
        candidate_set = [pid] + neg_samples

        q = ubuntu_data[qid]
        q_title, q_title_mask, q_body, q_body_mask = q

        c = [ubuntu_data[i] for i in candidate_set]
        c_titles = [i[0] for i in c]
        c_titles_mask = [i[1] for i in c]
        c_bodies = [i[2] for i in c]
        c_bodies_mask = [i[3] for i in c]

        l = [1] + [0] * len(neg_samples)


        return Tensor(q_title), Tensor(q_title_mask), Tensor(q_body), Tensor(q_body_mask), Tensor(c_titles), Tensor(c_titles_mask), Tensor(c_bodies), Tensor(c_bodies_mask), Tensor(l)


    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    s = UbuntuSequentialDataSet()
    print "total time:", time.time()-start
