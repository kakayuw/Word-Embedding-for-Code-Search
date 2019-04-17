
# coding: utf-8
import gc
import time
import re
import os
import numpy as np
import json


from nltk import PorterStemmer, stem

np.random.seed(13)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   # assign with requirement
sess = tf.Session(config=config)

KTF.set_session(sess)


from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams

from gensim.models import Word2Vec,KeyedVectors


# execute SQL script fetching text

stemmer = stem.PorterStemmer()


def hump2underline(hunp_str):
    # use re to match position between lower letter and upper letter
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sub = re.sub(p, r'\1_\2', hunp_str).lower()
    return sub


def getEmbeddingTrainData():
    dataset = []
    # java platform api
    func_list = []
    size = 0
    with open("trainData/result.json") as jo:
        hjson = json.loads(jo.read())
        for fc in hjson:
            name = fc['name']
            funcs = fc['func']
            for method in funcs:
                meth_name = name + '/' + method['funcName']
                meth_desc = method['funcDes']
                func_list.append((meth_name, meth_desc))
    func_list = func_list[:2000]
    for name, desc in func_list:
        dataset.append(' '.join(re.findall(r"\w+", name + "\t" + desc)))
    print("java platform api count:", len(func_list))
    size = len(dataset)
    # Birt api and jdt api
    with open("./trainData/jdtApi.txt", "r") as jdt:
        for jdtline in jdt.readlines():
            dataset.append(' '.join(re.findall(r"\w+", jdtline)))
        print("jdt api count:", len(dataset) - size)
        size = len(dataset)
    with open("./trainData/birtApi.txt", "r") as birt:
        for birtline in birt.readlines():
            dataset.append(' '.join(re.findall(r"\w+", birtline)))
        print("birt api count:", len(dataset) - size)
        size = len(dataset)
    # top 1000 java projs
    with open("trainData/split0.txt", 'rb') as fo0:
        for idxk, k in enumerate(fo0.readlines()):
            k = str(k)
            dataset.append(' '.join(re.findall(r"\w+", k)))
            if idxk > 1000: break
    with open("trainData/split1.txt", 'rb') as fo1:
        for idxj, j in enumerate(fo1.readlines()):
            j = str(j)
            dataset.append(' '.join(re.findall(r"\w+", j)))
            if idxj > 1000: break
        print("top 1000 java projects count:", len(dataset) - size)
    print("Total count:", len(dataset))
    return dataset


def splitStem(dataset):
    new_data = []
    gc.collect()
    for i in dataset:
        split_hump = hump2underline(i).replace('_', ' ').split()
        stemmed = ' '.join([stemmer.stem(j) for j in split_hump])
        remove_num = str(''.join(filter(lambda x: not x.isdigit(), stemmed)))
        new_data.append(remove_num)
    return new_data


# ### Skip-gram 
rawText = getEmbeddingTrainData()
rawText = splitStem(rawText)

# split words with camel case


time_start=time.time()

# corpus = [sentence for sentence in rawText if sentence.count(' ') >= 2]
corpus = [sentence for sentence in rawText]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
V = len(tokenizer.word_index) + 1
V


dim_embedddings = 100

# inputs
w_inputs = Input(shape=(1, ), dtype='int32')
w = Embedding(V, dim_embedddings)(w_inputs)

# context
c_inputs = Input(shape=(1, ), dtype='int32')
c  = Embedding(V, dim_embedddings)(c_inputs)
o = Dot(axes=2)([w, c])
o = Reshape((1,), input_shape=(1, 1))(o)
o = Activation('sigmoid')(o)

SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
SkipGram.summary()
SkipGram.compile(loss='binary_crossentropy', optimizer='adam')


# In[7]:

#
# for iter in range(50):
#     loss = 0.
#     for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):
#         data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=10, negative_samples=25.)
#         x = [np.array(x) for x in zip(*data)]
#         y = np.array(labels, dtype=np.int32)
#         if x:
#             loss += SkipGram.train_on_batch(x, y)
#     f = open('vectorsEpoch' + str(iter) + '.txt', 'w', encoding='utf-8')
#     f.write('{} {}\n'.format(V - 1, dim_embedddings))
#     vectors = SkipGram.get_weights()[0]
#     for word, i in tokenizer.word_index.items():
#         f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
#     f.close()
#     print(loss)

def skipgrams_generator():
    for  doc in tokenizer.texts_to_sequences(corpus):
        data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=10, negative_samples=25.)
        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        yield x, y


SkipGram.fit_generator(skipgrams_generator(), epochs=50)
f = open('vectorsEpoch70.txt', 'w')
f.write('{} {}\n'.format(V - 1, dim_embedddings))
vectors = SkipGram.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

time_end = time.time()
sum_time = time_end-time_start
print('Train embedding model cost', int(sum_time/3600), 'h', int(sum_time/60 % 60), 'min', int(sum_time % 60), 's')


print("execute model over")