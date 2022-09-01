from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from NewsExtraction import cnn_extract as cnn


# lemmatizes/stems a single document given as a list of words
def code_lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for cnter, k in enumerate(words):
        new = lemmatizer.lemmatize(k)
        words[cnter] = new
        words[cnter] = stemmer.stem(new)
    return words


# takes a preprocessed corpus as a list of documents NOT as object Corpus
# returns a dictionary where the keys are each word in the corpus, and the values are a list containing which documents
# use the term
def dfdictionary(corpus):
    dicti = {}
    for cnt, doc in enumerate(corpus):
        for word in doc:
            # if the term is not in the dictionary, add it
            try:
                dicti[word].add(cnt)
            except:
                dicti[word] = {cnt}
    return dicti


# calculates idf using dfdictionary
def idf(corpus, term):
    dicti = dfdictionary(corpus)
    return np.log(len(corpus) / (len(dicti[term])))


# takes a preproccesed document
# returns a dictionary where the keys are each word in the given doc, and the values are the indices where the term is
def tfdictionary(doc):
    dicti = {}
    for cnt, word in enumerate(doc):
        # if the term is not yet in the dictionary, add it
        try:
            dicti[word].add(cnt)
        except:
            dicti[word] = {cnt}
    return dicti


# takes a preproccesed document and
# returns tf of a given document and term using tfdictionary
def tf(doc, term):
    dicti = tfdictionary(doc)
    return len(dicti[term]) / len(doc)


# calculates tf-idf
def tfidf(corpus, doc, term):
    return tf(doc, term) * idf(corpus, term)


# generalized cosine similarity function for comparing two vectorized tf-df documents
def cossim(veca, vecb):
    return np.dot(veca, vecb) / (np.linalg.norm(veca) * np.linalg.norm(vecb))


class Corpus:
    def __init__(self, texts):
        self.docs = []
        self.vects = []
        self.docs = texts
        self.preprocess()

    def printall(self):
        for i in self.docs:
            print(i)

    def preprocess(self):
        stopword = stopwords.words('english')
        # will get rid of punctuation in document as well
        stopword.extend([',', '.', '``', "''", '--', '?', "n't", "'s", ':', '$', "'ve", "'d", "'", "-", '"', "'"])
        for cnt in range(len(self.docs)):
            self.docs[cnt] = self.docs[cnt].lower()
            self.docs[cnt] = nltk.word_tokenize(self.docs[cnt])
            self.docs[cnt] = [word for word in self.docs[cnt] if word not in stopword]
            self.docs[cnt] = code_lemmatize(self.docs[cnt])


links = ["https://www.cnn.com/2022/08/31/tennis/serena-williams-anett-kontaveit-us-open-spt-intl/index.html"]
articles = []
for i in links:
    articles.append(cnn(i))
test = Corpus(articles)
for i in test.docs:
    print(i)
