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
def idf_dictionary(corpus):
    dicti = {}
    for cnt, doc in enumerate(corpus):
        for word in doc:
            try:
                dicti[word].add(cnt)
            except KeyError:
                dicti[word] = {cnt}
    condensed_dict = {}
    for key in dicti:
        condensed_dict[key] = np.log(len(corpus)/(len(dicti[key]) + 1)) + 1
    return condensed_dict


# takes a preproccesed document
# returns a dictionary where the keys are each word in the given doc, and the values are the indices where the term is
def tf_dictionary(doc):
    dicti = {}
    for word in doc:
        try:
            dicti[word] += 1
        except KeyError:
            dicti[word] = 1
    n = 0
    for i in dicti.values():
        n += i
    for i in dicti:
        dicti[i] = dicti[i] / n
    return dicti


# generalized cosine similarity function for comparing two vectorized tf-df documents
def cossim(veca, vecb):
    return np.dot(veca, vecb) / (np.linalg.norm(veca) * np.linalg.norm(vecb))


class Corpus:
    def __init__(self, texts):
        self.docs = []
        self.vectors = []
        self.token_list = []
        self.similarity_matrix = []
        self.docs = texts
        self.preprocess()
        self.create_token_list()
        self.vectorize()
        self.create_similarity_matrix()

    def printall(self):
        for i in self.docs:
            print(i)

    def preprocess(self):
        stopword = stopwords.words('english')
        # will get rid of punctuation in document as well
        stopword.extend([',', '.', '``', "''", '--', '?', "n't", "'s", ':', '$', "'ve", "'d", "'", "-", '"', "'", '’',
                         '—', '“', '(', ')', "i'm"])
        for cnt in range(len(self.docs)):
            self.docs[cnt] = self.docs[cnt].lower()
            self.docs[cnt] = nltk.word_tokenize(self.docs[cnt])
            self.docs[cnt] = [word for word in self.docs[cnt] if word not in stopword]
            self.docs[cnt] = code_lemmatize(self.docs[cnt])

    def create_token_list(self):
        for i in self.docs:
            for word in i:
                if word not in self.token_list:
                    self.token_list.append(word)

    def create_similarity_matrix(self):
        self.similarity_matrix = np.zeros((len(self.docs), len(self.docs)))
        for row in range(self.similarity_matrix.shape[0]):
            for column in range(self.similarity_matrix.shape[1]):
                self.similarity_matrix[row, column] = cossim(self.vectors[row], self.vectors[column])

    def vectorize(self):
        idf_dict = idf_dictionary(self.docs)
        self.vectors = np.zeros((len(self.docs), len(self.token_list)))
        for document_index, document in zip(range(len(self.vectors)), self.docs):
            tf_dict = tf_dictionary(document)
            for term_index, term in zip(range(len(self.token_list)), self.token_list):
                try:
                    self.vectors[document_index, term_index] = tf_dict[term]*idf_dict[term]
                except KeyError:
                    self.vectors[document_index, term_index] = 0

