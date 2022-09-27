import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec as g2w
import urllib.request
import re

from nltk.corpus import stopwords

from bs4 import BeautifulSoup


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
            "!": ""
        }
        self.corpus = multiple_replace(replace_dict, self.corpus)


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
    '''  
    glove_dict = {}
    with open("Pretrained Vectors/glove.6B.50d.txt", 'r', encoding="utf-8") as file:
        for i in file:
            line = i.split()
            key = line[0]
            vector = np.array(line[1:], "float32")
            glove_dict[key] = vector
    keywords = ["urban", "thug", "drugs", "terror"]
    glove_vecs = r"Pretrained Vectors\glove.6B.50d.txt"
    temp = get_tmpfile("test_word2vec.txt")
    _ = g2w(glove_vecs, temp)
    glove = KeyedVectors.load_word2vec_format(temp)
    '''
    test = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                               "&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2=200301&items_per_page=5&"
                               "f%5B0%5D=field_docs_attributes%3A205")
    test.create_corpus()
    print(test.corpus)

