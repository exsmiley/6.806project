'''Prepares the data for our neural networks'''
from gensim.models.keyedvectors import KeyedVectors
import gzip
import numpy as np
import torch.utils.data as d


def load_word_vector():
    embedding_path = 'vector/vectors_pruned.200.txt.gz'
    embeddings = {}
    with gzip.open(embedding_path) as file:
        for line in file:
            line = line.split()
            word = line[0]
            vec = np.array(map(float, line[1:]), dtype='float64')
            embeddings[word] = vec
    return embeddings


def load_tokenized_text_ubuntu():
    '''Creates dictionary of {id: (title, body)}
    This is okay to have in memory since it's just like 160MB or so...
    '''
    data = {}
    with gzip.open('ubuntu_data/text_tokenized.txt.gz') as f:
        for line in f:
            sections = line.strip().split('\t')
            body = sections[2] if len(sections) > 2 else ''
            data[sections[0]] = (sections[1], body)
    return data


# use as global variable for reading from
embeddings = load_word_vector()
ubuntu_data = load_tokenized_text_ubuntu()


def load_ubuntu_examples(dev=False, test=False):
    '''yields data in the form of
    (id, query (title, body), example (title, body), +1/-1)'''
    file = 'ubuntu_data/train_random.txt'
    if dev:
        file = 'ubuntu_data/dev.txt'
    elif test:
        file = 'ubuntu_data/test.txt'
    with open(file) as f:
        for line in f:
            query, positive, negative = map(lambda x: x.split(), line.split('\t'))
            query = query[0]
            for p in positive:
                yield (query, ubuntu_data[query], ubuntu_data[p], 1)
            for n in negative:
                yield (query, ubuntu_data[query], ubuntu_data[n], -1)


def load_ubuntu_word_vectors(dev=False, test=False):
    '''makes an average word vector for query/examples'''
    data = []
    for (qid, query, example, label) in load_ubuntu_examples():
        query = ''.join(query).split()
        example = ''.join(example).split()

        query_vec = np.zeros(200)
        example_vec = np.zeros(200)

        for word in query:
            if word in embeddings:
                query_vec += embeddings[word]
        for word in example:
            if word in embeddings:
                example_vec += embeddings[word]
        data.append((qid, query_vec, example_vec, label))
    return data


class UbuntuDataSet(d.Dataset):

    def __init__(self):
        self.data = load_ubuntu_word_vectors()

    def __getitem__(self, index):
        (qid, query_vec, example_vec, label) = self.data[index]
        return torch.from_numpy(np.concatenate((query_vec, example_vec))), label

    def __len__(self):
        return len(self.data)

