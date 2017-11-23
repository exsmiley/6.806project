'''Prepares the data for our neural networks'''
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


def load_tokenized_text(fname):
    '''Creates dictionary of {id: (title, body)}
    This is okay to have in memory since it's just like 160MB or so...
    '''
    data = {}
    with gzip.open(fname) as f:
        for line in f:
            sections = line.strip().split('\t')
            body = sections[2] if len(sections) > 2 else ''
            data[sections[0]] = (sections[1], body)
    return data


# use as global variable for reading from
embeddings = load_word_vector()
ubuntu_data = load_tokenized_text('ubuntu_data/text_tokenized.txt.gz')
android_data = load_tokenized_text('android_data/corpus.tsv.gz')


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


def load_android_examples(dev=False, test=False):
    '''yields data in the form of
    (id, query (title, body), example (title, body), +1/-1)'''
    assert dev or test  # can only be dev or test sets
    neg_file = 'android_data/dev.neg.txt'
    pos_file = 'android_data/dev.pos.txt'
    if test:
        neg_file = 'android_data/test.neg.txt'
        pos_file = 'android_data/test.pos.txt'
    with open(neg_file) as f:
        for line in f:
            query, compare = line.split()
            yield (query, android_data[query], android_data[compare], -1)
    with open(pos_file) as f:
        for line in f:
            query, compare = line.split()
            yield (query, android_data[query], android_data[compare], 1)


def load_example_word_vectors(dev=False, test=False, ubuntu=True):
    '''makes an average word vector for query/examples'''
    data = []
    example_func = load_ubuntu_examples
    if not ubuntu:
        example_func = load_android_examples

    for (qid, query, example, label) in example_func(dev=dev, test=test):
        query = ''.join(query).split()
        example = ''.join(example).split()

        query_vec = np.zeros(200)
        example_vec = np.zeros(200)
        count = 0
        for word in query:
            if word in embeddings:
                count += 1
                query_vec += embeddings[word]
        if count > 0:
            query_vec = query_vec/count
            count = 0
        for word in example:
            if word in embeddings:
                example_vec += embeddings[word]
                count += 1
        if count > 0:
            example_vec = example_vec/count
        data.append((qid, query_vec, example_vec, label))
    return data

def load_sequential_word_vectors(dev=False, test=False, ubuntu=True):
    '''makes a list of word vectors for query/examples'''
    data = []
    example_func = load_ubuntu_examples
    if not ubuntu:
        example_func = load_android_examples

    for (qid, query, example, label) in example_func(dev=dev, test=test):
        query = ''.join(query).split()
        example = ''.join(example).split()

        query_embeddings = []
        example_embeddings = []

        for word in query:
            if word in embeddings:
                query_embeddings.append(embeddings[word])
        for word in example:
            if word in embeddings:
                example_embeddings.append(embeddings[word])
        data.append((qid, query_embeddings, example_embeddings, label))
    return data


class UbuntuDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset'''

    def __init__(self):
        self.data = load_example_word_vectors(ubuntu=True)

    def __getitem__(self, index):
        (qid, query_vec, example_vec, label) = self.data[index]
        return torch.from_numpy(np.concatenate((query_vec, example_vec))), label

    def __len__(self):
        return len(self.data)


class UbuntuSequentialDataSet(d.Dataset):
    '''Loads the training set for the Ubuntu Dataset with sequential word vectors'''

    def __init__(self):
        self.data = load_example_word_vectors(ubuntu=True)

    def __getitem__(self, index):
        # query_vecs and example_vecs are lists of length 200 word vectors
        (qid, query_vecs, example_vecs, label) = self.data[index]
        # return torch.from_numpy(np.concatenate((query_vec, example_vec))), label
        # TODO
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

