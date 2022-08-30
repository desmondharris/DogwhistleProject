from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


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
        self.vectorize()
        self.similarity_matrix = self.simmat()

    def printall(self):
        for i in self.docs:
            print(i)

    def preprocess(self):
        stopword = stopwords.words('english')
        # will get rid of punctuation in document as well
        stopword.extend([',', '.', '``', "''", '--', '?', "n't", "'s", ':', '$', "'ve", "'d"])
        for cnt in range(len(self.docs)):
            self.docs[cnt] = self.docs[cnt].lower()
            self.docs[cnt] = nltk.word_tokenize(self.docs[cnt])
            self.docs[cnt] = [word for word in self.docs[cnt] if word not in stopword]
            self.docs[cnt] = code_lemmatize(self.docs[cnt])

    def vectorize(self):
        # creates 1-d array to later add to
        self.vects = list(np.zeros(len(self.docs)))
        # creates list of unique vocabulary words in a corpus
        # so that we can find the number of words to compare
        wordlist = []
        for i in self.docs:
            wordlist.extend(tfdictionary(i).keys())
        wordlist = set(wordlist)
        for cnt in range(len(self.docs)):
            # converts vects into 2d array, where each row is a vector
            self.vects[cnt] = [0] * len(wordlist)
        for i, doc in enumerate(self.docs):
            for j, word in enumerate(tfdictionary(doc).keys()):
                self.vects[i][j] = tfidf(self.docs, doc, word)
        # completes vectorization, where each row is a vector corresponding to a document
        # containing tf-idf for a specific term in that document. each column represents one term

    # returns cosine similarity matrix of all documents in corpus
    def simmat(self):
        mat = [[0 for x in range(len(self.docs))] for y in range(len(self.docs))]
        for row in range(len(self.docs)):
            for column in range(len(self.docs)):
                # this is s//et to round purely for the purpose of displaying a readable matrix
                mat[row][column] = round(cossim(self.vects[row], self.vects[column]), 2)
        return mat

g = Corpus(["good", "bad"])
