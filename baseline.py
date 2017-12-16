from sklearn.feature_extraction.text import TfidfVectorizer
from prepare import load_tokenized_text, load_android_examples
import numpy as np
from meter import AUCMeter
import tqdm
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity

android_data = load_tokenized_text('android_data/corpus.tsv.gz', 'Android', return_index=False)
android_dev = list(load_android_examples(android_data, test=False))
android_test = list(load_android_examples(android_data, test=True))

question_ids = {}
contents = []
count = 0

fil = lambda token: token not in string.punctuation and not token.isdigit()

for q, v in android_data.items():
    t, t_mask, b, b_mask = v

    t = list(map(lambda x: x[0] ,filter(lambda x: x[1] == 1 and x[0] and fil(x[0]),zip(t, t_mask))))
    b = list(map(lambda x: x[0] ,filter(lambda x: x[1] == 1 and x[0] and fil(x[0]),zip(b, b_mask))))
    contents.append(' '.join(t) + ' ' + ' '.join(b))
    question_ids[q] = count
    count += 1




stop_words = stopwords.words('english')
stop_words.append('')
meter = AUCMeter()

vectorizer = TfidfVectorizer(lowercase=True, stop_words=stop_words, use_idf=True, ngram_range=(1, 2), tokenizer=lambda x: x.split(' '))
vs = vectorizer.fit_transform(contents)

res = []
for q, p, ns in tqdm.tqdm(android_test):
    sims = []
    question = vs[question_ids[q]]
    for candidate in [p]+ns:
        cos_sim = cosine_similarity(question, vs[question_ids[candidate]])
        sims.append(cos_sim[0][0])
    sims = np.array(sims)
    ind = np.argsort(sims)[::-1]
    labels = np.array([1] + [0] * len(ns))
    labels = labels[ind]
    meter.add(sims[ind], labels[ind])



#
# predicted = np.array(predicted)
# map, mrr, p_at_one, p_at_five = Evaluation(predicted).evaluate()
# print('\n')
# print("MAP: {0}, MRR: {1}, P@1: {2}, P@5: {3}".format(map, mrr, p_at_one, p_at_five))

print(meter.value(0.05))
