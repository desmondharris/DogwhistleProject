import warnings

import nltk
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec as g2w
import urllib.request
import re

from nltk.corpus import stopwords

from bs4 import BeautifulSoup
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PresidentialScraper:
    def __init__(self, link):
        self.pages = []
        self.corpus = ""

        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")
        evens = soup.select(".even")
        odds = soup.select(".odd")
        for i in odds:
            evens.append(i)
        for i in evens:
            g = "https://www.presidency.ucsb.edu" + i.select('a')[1]['href']
            self.pages.append(urllib.request.urlopen(g))

    def create_corpus(self):
        for i in self.pages:
            self.corpus += extract_speech(i)
        self.corpus = self.corpus.lower()

        replace_dict = {
            '[': "",
            ']': "",
            "(": "",
            ")": "",
            ".": "",
            ",": "",
            "!": "",
            "#": "",
            ":": "",
            "@": "",
            "?": "",
            "'": "",
            '"': "",
            "&": ""
        }
        self.corpus = multiple_replace(replace_dict, self.corpus)

        self.corpus = nltk.word_tokenize(self.corpus)
        stop = stopwords.words("english")
        self.corpus = [word for word in self.corpus if word not in stop]
        temp = [self.corpus]
        self.corpus = temp


def extract_speech(page):
    full_text = ""
    soup = BeautifulSoup(page, "html.parser")
    article = soup.select(".field-docs-content")
    for i in article:
        temp = i.getText()
        full_text += temp
    return full_text


# Function used is from ActiveState
# https://code.activestate.com/recipes/81330-single-pass-multiple-replace/
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


if __name__ == '__main__':
    glove_dict = {}
    with open("Pretrained Vectors/glove.6B.50d.txt", 'r', encoding="utf-8") as file:
        for i in file:
            line = i.split()
            key = line[0]
            vector = np.array(line[1:], "float32")
            glove_dict[key] = vector

    keywords = ["urban", "thug", "drugs", "terror"]
    glove_vecs = r"Pretrained Vectors/glove.6B.50d.txt"
    temp = get_tmpfile("test_word2vec.txt")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    _ = g2w(glove_vecs, temp)
    glove = KeyedVectors.load_word2vec_format(temp)

    TrumpList = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                                    "&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2=200301&items_per_page=100")
    TrumpList.create_corpus()
    new_data = TrumpList.corpus

    bare = Word2Vec(vector_size=50, min_count=5)
    print(f"1 bare vocab size: {len(bare.wv.key_to_index.keys())}")
    print(f"1 'make' is at index {bare.wv.key_to_index.get('make', None)}")
    bare.build_vocab(new_data)
    print(f"2 bare vocab size: {len(bare.wv.key_to_index.keys())}")
    print(f"2 'make' is at index {bare.wv.key_to_index.get('make', None)}")
    bare.build_vocab([list(glove.key_to_index.keys())], update=True)
    print(f"3 bare vocab size: {len(bare.wv.key_to_index.keys())}")
    print(f"3 'make' is at index {bare.wv.key_to_index.get('make', None)}")
    total = bare.corpus_count
    bare.train(new_data, total_examples=bare.corpus_count, epochs=bare.epochs)
    vectors = bare.wv
    for i in keywords:
        print(vectors.most_similar(i))
