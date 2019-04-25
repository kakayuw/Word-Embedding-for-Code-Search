
# coding: utf-8

# In[1]:
import re
import time
import warnings, math

from nltk import stem

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec,KeyedVectors
from gensim import corpora

from nltk.stem.porter import PorterStemmer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

import MySQLdb


stemmer = stem.PorterStemmer()
# ### Load word2vec model

# In[2]:


w2v = KeyedVectors.load_word2vec_format('./embeddingVectors/vectorsEpoch60.txt', binary=False)


# ### Prepared doc & query for matching

# In[3]:
# fetch docs in javadoc
def split_word(str):
    str = str.replace('_', ' ')
    return re.findall(r"\w+", str)


def load_doc():
    with open("./javaTest/test.rawcode.txt", 'r') as fo:
        codes = fo.readlines()
        codelist = []
        for code in codes:
            codelist.append(code)
        return codelist


def load_qry():
    with open("./javaTest/test.java.desc.txt", "r") as fo:
        qrys = fo.readlines()
        qry_list = []
        for qry in qrys:
            # porter_stemmer = PorterStemmer()
            # rawQuery = porter_stemmer.stem(qry)
            # rawQueryList = rawQuery.split(' ');
            # rawQuery = ' '.join([porter_stemmer.stem(i) for i in rawQueryList])
            rawQuery = qry.replace('_', ' ')
            rawQuery = ' '.join([stemmer.stem(j) for j in rawQuery.split() if len(j) > 1])  # remove single letter
            qry_list.append(rawQuery)
        return qry_list


def load_vocab():
    codes = load_doc()
    qrys = load_qry()
    vocab = dict()
    for i in range(len(codes)):
        vocab[codes[i]] = qrys[i]
    return vocab
# # fetch docs from mysql
# documents = []
# sum_terms_count = 0
#
# # text in aspectj
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="aspectj")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("aspectj terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# # text in birt
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="birt")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("birt terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# # text in eclipse_platform_ui
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="eclipse_platform_ui")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("eclipse_platform_ui terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# # text in jdt
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="jdt")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("jdt terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# # text in tomcat
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="tomcat")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("tomcat terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# # text in swt
# db = MySQLdb.connect(host="localhost", port = 3307, user="root", passwd="mypassword", db="swt")
# cur = db.cursor()
# count = cur.execute("""SELECT summary_stemmed, description_stemmed FROM bug_and_files""")
# print("swt terms count:", count)
# sum_terms_count += count
# for i in cur.fetchmany(count):
#     if len(i[0]) > 0:
#         documents.append(i[0])
#     if len(i[1]) > 0:
#         documents.append(i[1])
#
# print('total terms count:', sum_terms_count)
#

# In[4]:

documents = load_doc()
vocab = load_vocab()
qrys = load_qry()

# print("documents:\n", documents[0:1000])
print("doc size:", len(documents))




# build corpora from documents for computing tfidf

doc_dict = corpora.Dictionary([split_word(i) for i in documents] + [split_word(i) for i in qrys])





# ### Matching & Ranking

# In[13]:


def getIDF(word):
    try:
        return math.log10( (doc_dict.num_docs + 1) / (1 + doc_dict.dfs[doc_dict.token2id[word]]) )
    except ZeroDivisionError:
        return 0


def wwSimilarity(word1, word2):
    try:
        sim = w2v.wv.similarity(str.lower(word1), str.lower(word2))
        return sim
    except KeyError:
        return 0



def wTSimilarity(word, text):
    # print("word", word)
    # print([(i, wwSimilarity(word, i)) for i in split_word(text)])
    # print("value:" ,max([wwSimilarity(word, i) for i in split_word(text)]))
    return max([wwSimilarity(word, i) for i in split_word(text)])


def TSSimilarity(qryStr, docStr):
    # print("query:", (qryStr))
    # print("doc:", (docStr))
    try:
        ts = (sum([(wTSimilarity(i, docStr) * getIDF(i)) for i in split_word(qryStr)]) ) / (sum([getIDF(i) for i in split_word(docStr)]) )
        # print("ts", ts)
        return ts
    except ZeroDivisionError:
        return 0


def getTotalSimilarity(rawQuery, doc):
    # print("query:", split_word(rawQuery))
    # print("doc:", split_word(doc))
    return TSSimilarity(rawQuery, doc) + TSSimilarity(doc, rawQuery)


def getTopNRank(docList, n, query):
    resultList = []
    for i, code in enumerate(docList):
        score = getTotalSimilarity(code, query)
        resultList.append((i, code, score))
    resultList = sorted(resultList, key=lambda x:x[2], reverse=True)
    return resultList[:n]


def eval(topK):
    time_start = time.time()

    def ACC(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1
        return sum / float(len(real))

    def MAP(real, predict):
        sum = 0.0
        for id, val in enumerate(real):
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + (id + 1) / float(index + 1)
        return sum / float(len(real))

    def MRR(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1.0 / float(index + 1)
        return sum / float(len(real))

    def NDCG(real, predict):
        dcg = 0.0
        idcg = IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i + 1
                dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcg)

    def IDCG(n):
        idcg = 0
        itemRelevance = 1
        for i in range(n):
            idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg

    acc, mrr, map, ndcg = 0, 0, 0, 0
    data_len = len(qrys)
    print("Total : " + str(data_len))
    for i in range(data_len):
        print(i)
        desc = qrys[i]  # good desc
        # print("desc:", desc)
        # print("descs:", descs)
        n_results = topK
        predict = getTopNRank(documents, topK, desc)
        # for j in predict:
        #     print (j[0],j[2], j[1])
        predict = [i[0] for i in predict]
        real = [i]
        acc += ACC(real, predict)
        mrr += MRR(real, predict)
        map += MAP(real, predict)
        ndcg += NDCG(real, predict)
    acc = acc / float(data_len)
    mrr = mrr / float(data_len)
    map = map / float(data_len)
    ndcg = ndcg / float(data_len)
    # acc, mrr = self.valid(model, 1000, 10)
    logger.info('ACC={}, MRR={}, MAP={}, nDCG={}'.format(acc, mrr, map, ndcg))
    time_end = time.time()
    cost_s = int(time_end - time_start)
    cost_m = int(cost_s / 60)
    cost_h = int(cost_m / 60)
    print('totally cost :', cost_h, 'h', cost_m % 60, 'min', cost_s % 60, 's')
    return acc, mrr, map, ndcg


# eval(10)


query = qrys[2]
print("query1:",query )
top5test = getTopNRank(documents, 30, query)
for i in top5test:
    print (i[0], i[1][:50], i[2])


# alist = split_word(documents[0])
# for i in alist:
#     print("permiss ", i, " sim:", wwSimilarity("permiss", i))

# print("query 0 :", qrys[0])
# print("docs 0 :", documents[0])
# print(getTotalSimilarity(qrys[2], documents[2]))

# print(getTotalSimilarity(query, query))
# print(wwSimilarity("PERMISSION_CODE", "Listener"))