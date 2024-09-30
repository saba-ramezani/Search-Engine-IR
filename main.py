from __future__ import unicode_literals

import math
import numpy as np
from numpy.linalg import norm
from hazm import *
from hazm.utils import stopwords_list
import json

with open("IR_data_news_12k.json", "r") as read_file:
    data = json.load(read_file)

normalizer = Normalizer()
word_tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
# stemmer = Stemmer()

my_stopwords = stopwords_list()
my_stopwords.append(".")
my_stopwords.append("،")
my_stopwords.append("؛")
my_stopwords.append("!")
my_stopwords.append("؟")
my_stopwords.append(")")
my_stopwords.append("(")
my_stopwords.append("}")
my_stopwords.append("{")
my_stopwords.append("\"")
my_stopwords.append(":")
my_stopwords.append("]")
my_stopwords.append("[")
my_stopwords.append("»")
my_stopwords.append("«")

query_stopwords = stopwords_list()
my_stopwords.append(".")
my_stopwords.append("،")
my_stopwords.append("؛")
my_stopwords.append("؟")
my_stopwords.append(")")
my_stopwords.append("(")
my_stopwords.append("}")
my_stopwords.append("{")
my_stopwords.append(":")
my_stopwords.append("]")
my_stopwords.append("[")

# preprocessing
dataOut = dict()
n = len(data)
# n = 20
k = 5
championListLength = 1000

print("Initializing...")
for i in range(n):
    doc = data[str(i)]
    txt = doc["content"]
    # print("content: " + txt)
    # removing the "انتهای پیام/" string from the content before initializing preprocessing
    txt = str(txt).replace('انتهای پیام/', '')
    # print("content after removing /انتهای پیام: \n" + txt)
    # normalizing the content of the document
    normalized_txt = normalizer.normalize(txt)
    # print("normalized txt: " + normalized_txt)
    # splitting the content to words(tokens)
    words = word_tokenizer.tokenize(normalized_txt)
    # print("tokenized txt: \n" + str(words))
    # removing the stop words from tokens
    filtered_txt = list()
    for word in words:
        if word not in my_stopwords:
            filtered_txt.append(word)
    # print("\nfiltered txt: \n" + str(filtered_txt))
    # stemming filtered tokens
    # stemmed_txt = list()
    # for word in filtered_txt:
    #     stemmed_txt.append(stemmer.stem(word))
    # print("\nstemmed txt: \n" + str(stemmed_txt))
    # lemmatizing filtered tokens
    lemmatized_txt = list()
    for word in filtered_txt:
        lemmatized_txt.append(lemmatizer.lemmatize(word))
    # print("\nlemmatized txt: \n" + str(lemmatized_txt))
    docOut = dict()
    docOut["title"] = doc["title"]
    docOut["content"] = lemmatized_txt
    docOut["full-content"] = txt
    docOut["tags"] = doc["tags"]
    docOut["date"] = doc["date"]
    docOut["url"] = doc["url"]
    docOut["category"] = doc["category"]
    dataOut[str(i)] = docOut

print("Preprocessing finished...")

docs_tokens = {}

# creating positional indexing
positional_index = dict()
print("Indexing started...")
for docID in dataOut.keys():
    visited_tokens = list()
    docTokens = dataOut[docID]["content"]
    docTokens_length = len(docTokens)
    for i in range(0, docTokens_length):
        token_index = dict()
        token_positions = list()
        token_num_in_each_doc = dict()
        token = docTokens[i]
        num = 0
        if token not in visited_tokens:
            token_positions.append(i)
            num += 1
            for j in range(i + 1, docTokens_length):
                token_to_check = docTokens[j]
                if token == token_to_check:
                    token_positions.append(j)
                    num += 1
            visited_tokens.append(token)
            if token not in positional_index.keys():
                dict_to_add = dict()
                temp = dict()
                temp[docID] = token_positions
                dict_to_add["position"] = temp
                temp = dict()
                temp[docID] = num
                dict_to_add["number"] = temp
                dict_to_add["total"] = num
                positional_index[token] = dict_to_add
            else:
                token_index[docID] = token_positions
                token_num_in_each_doc[docID] = num
                dict_to_add = positional_index.get(token)
                temp = dict_to_add["position"]
                temp[docID] = token_positions
                dict_to_add["position"] = temp
                temp = dict_to_add["number"]
                temp[docID] = num
                dict_to_add["number"] = temp
                dict_to_add["total"] += num
                positional_index[token] = dict_to_add
    docs_tokens[docID] = docTokens
sorted_positional_index = dict(sorted(positional_index.items()))
print("Positional index created...")


def calculateTF(token, docID):
    # this line fetches the number of ocuurrences of a token in document d
    f_td = sorted_positional_index[token]['number'][docID]
    tf = 1 + math.log10(f_td)
    return tf


def calculateIDF(token):
    N = n
    # this line calculates the number of documents that contain token
    n_t = len(sorted_positional_index[token]['number'].keys())
    idf = math.log10(N / n_t)
    return idf


vectorSpaceModel = dict()

sumOfScoresPerDoc = {}


def createVectorSpaceModel():
    for token in sorted_positional_index.keys():
        idf = calculateIDF(token)
        for docID in sorted_positional_index[token]['number'].keys():
            tf = calculateTF(token, docID)
            if token in vectorSpaceModel.keys():
                temp = vectorSpaceModel[token]
            else:
                temp = dict()
            tf_idf = tf * idf
            temp[docID] = tf_idf
            vectorSpaceModel[token] = temp
            if docID in sumOfScoresPerDoc.keys():
                sumOfScoresPerDoc[docID] += tf_idf * tf_idf
            else:
                sumOfScoresPerDoc[docID] = tf_idf * tf_idf
        for docID in sumOfScoresPerDoc.keys():
            sumOfScoresPerDoc[docID] = math.sqrt(sumOfScoresPerDoc[docID])


championList = dict()


def createChampionList():
    for token in vectorSpaceModel:
        sorted_docs = sorted(vectorSpaceModel[token].items(), key=lambda x: x[1] / sumOfScoresPerDoc[x[0]],
                             reverse=True)
        temp = {}
        rng = min(championListLength, len(sorted_docs))
        for i in range(0, rng):
            docID = sorted_docs[i][0]
            temp[docID] = vectorSpaceModel[token][docID]
        championList[token] = temp


def getQueryDict(queryTokens):
    queryDict = {}
    for token in queryTokens:
        idf = calculateIDF(token)
        f_td = queryTokens.count(token)
        tf = 1 + math.log10(f_td)
        queryDict[token] = tf * idf

    return queryDict


def fetchAllRelatedDocs(queryTokens):
    docs = []
    for token in queryTokens:
        # Implementing index elimination
        # token_docs = vectorSpaceModel[token].keys()
        token_docs = championList[token].keys()
        for docID in token_docs:
            if docs.count(docID) <= 0:
                docs.append(docID)
    return docs


def getSimilarity(docID, docDict, queryDict, similarityFunction):
    queryVector = []
    docVector = []
    for token in queryDict.keys():
        queryVector.append(queryDict[token])
        docVector.append(docDict[token])
    if similarityFunction == "cosine":
        cosine = np.dot(queryVector, docVector) / (norm(queryVector) * norm(docVector))
        print(queryVector)
        print(docVector)
        print(norm(queryVector))
        print(norm(docVector))
        return cosine
    elif similarityFunction == "jaccard":
        intersection = 0
        for token in queryDict.keys():
            if token in docDict.keys():
                intersection += docs_tokens[docID].count(token)

        union = (len(queryDict.keys()) + len(docs_tokens[docID])) - intersection
        jaccard = float(intersection) / union
        return jaccard


def getDocDict(docID, queryTokens):
    docDict = {}
    for token in queryTokens:
        # postings = vectorSpaceModel[token]
        postings = championList[token]
        tf_idf = 0
        if docID in postings.keys():
            tf_idf = postings[docID]
        docDict[token] = tf_idf
    return docDict


def getResults(queryTokens, similarityFunction):
    queryDict = getQueryDict(queryTokens)
    all_related_docs = fetchAllRelatedDocs(queryTokens)
    doc_scores = {}
    for docID in all_related_docs:
        docDict = getDocDict(docID, queryTokens)
        similarity = getSimilarity(docID, docDict, queryDict, similarityFunction)
        doc_scores[docID] = similarity

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs, doc_scores


def showFinalDocsDetails(sorted_docs, doc_scores):
    if len(sorted_docs) == 0:
        print("*********************************************************************************")
        print("Sorry! No result found for your query..")
    else:
        print("*********************************************************************************")
        print(str(len(sorted_docs)) + " documents found for your query-->")
        if len(sorted_docs) > k:
            n = k
        else:
            n = len(sorted_docs)
        print("The first " + str(n) + " documents are listed below:")
        for i in range(0, n):
            d = sorted_docs[i]
            docID = str(d[0])
            print("*********************************************************************************")
            print('ID: ' + str(docID))
            print('Score:' + str(doc_scores[docID]))
            print('Title: ' + str(dataOut[docID]['title']))
            print('URL: ' + str(dataOut[docID]['url']))
            print('Content: ' + dataOut[docID]['full-content'])


print("Creating vector space model...")
createVectorSpaceModel()
print("Vector space model created...")
print("Creating champion list...")
createChampionList()
print("Champion list created...")
similarityFunction = input("cosine or jaccard?\n")
while True:
    query = input("please enter your query:\n")
    query = normalizer.normalize(query)
    query_words = word_tokenizer.tokenize(query)
    filtered_query = list()
    for word in query_words:
        if word not in query_stopwords:
            filtered_query.append(word)
    lemmatized_query = []
    for word in filtered_query:
        lemmatized_query.append(lemmatizer.lemmatize(word))
    single_words = []
    for word in lemmatized_query:
        if single_words.count(word) <= 0:
            single_words.append(word)
    sorted_docs, doc_scores = getResults(single_words, similarityFunction)
    showFinalDocsDetails(sorted_docs, doc_scores)
