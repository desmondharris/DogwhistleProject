import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def code_lemmatize(words):
    """
    Lemmatize, then stem a tokenized document using NLTK's WordNetLemmatizer and PorterStemmer.

    :param words: Document as a list of words
    :type words: list
    :return: List of lemmatized and stemmed words
    :rtype: list
    """
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for cnter, k in enumerate(words):
        new = lemmatizer.lemmatize(k)
        words[cnter] = new
        words[cnter] = stemmer.stem(new)
    return words


def idf_dictionary(corpus):
    """
    Create inverse document frequency(IDF) dictionary for a corpus using IDF = ln( N / (DF + 1)) + 1

    :param corpus: List of tokenized documents
    :type corpus: list
    :return: Dictionary with each individual word in the corpus, with the IDF value of it attached
    :rtype: dict
    """
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


def tf_dictionary(doc):
    """
    Create a term frequency(TF) dictionary for a document in a corpus

    :param doc: Tokenized document
    :type doc: list
    :return: Dictionary of each word in the document as keys, and each word's TF value as values
    :rtype: dict
    """
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
def cossim(vec_a, vec_b):
    """
    Calculate cosine similarity value for two given vectors

    :param vec_a: First vector
    :type vec_a: list
    :param vec_b: Second vector
    :type vec_b: list
    :return: Cosine similarity value for the two vectors
    :rtype: float
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


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
        """
        Tokenize, convert to lowercase, lemmatize, stem, and remove stop words from a Corpus object.

        :return: None
        :rtype: None
        """
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
        """
        Create list containing each unique vocab word in the Corpus once.

        :return: None
        """
        for i in self.docs:
            for word in i:
                if word not in self.token_list:
                    self.token_list.append(word)

    def create_similarity_matrix(self):
        """
        Create similarity matrix for Corpus object.

        :return: None
        """
        self.similarity_matrix = np.zeros((len(self.docs), len(self.docs)))
        for row in range(self.similarity_matrix.shape[0]):
            for column in range(self.similarity_matrix.shape[1]):
                self.similarity_matrix[row, column] = cossim(self.vectors[row], self.vectors[column])

    def vectorize(self):
        """
        Vectorize a Corpus object's documents with TF-IDF values.

        :return: None
        """
        idf_dict = idf_dictionary(self.docs)
        self.vectors = np.zeros((len(self.docs), len(self.token_list)))
        for document_index, document in zip(range(len(self.vectors)), self.docs):
            tf_dict = tf_dictionary(document)
            for term_index, term in zip(range(len(self.token_list)), self.token_list):
                try:
                    self.vectors[document_index, term_index] = tf_dict[term]*idf_dict[term]
                except KeyError:
                    self.vectors[document_index, term_index] = 0

