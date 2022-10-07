import nltk
from nltk.corpus import stopwords

import urllib.request
import re

from gensim.test.utils import datapath
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec as g2w
from gensim.models import KeyedVectors, Word2Vec

from bs4 import BeautifulSoup


class PresidentialScraper:
    def __init__(self, link):
        self.pages = []
        self.corpus = ""

        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")

        while page is not None:
            evens = soup.select(".even")
            odds = soup.select(".odd")
            for i in odds:
                evens.append(i)
            for i in evens:
                link = "https://www.presidency.ucsb.edu" + i.select('a')[1]['href']
                self.pages.append(urllib.request.urlopen(link))
            try:
                page = urllib.request.urlopen("https://www.presidency.ucsb.edu" +
                                              soup.find('a', {'title': 'Go to next page'})['href'])
                soup = BeautifulSoup(page, "html.parser")
            except TypeError:
                break

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
    print('loading glove vecs')
    glove_file = datapath(r"C:\Users\dsm84762\PycharmProjects\DogwhistleProject\Pretrained Vectors\glove.6B.50d.txt")
    tmp_file = get_tmpfile("test_word2vec.txt")

    _ = g2w(glove_file, tmp_file)

    glove_vectors = KeyedVectors.load_word2vec_format(tmp_file)


    TrumpList = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                                    "&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2=200301&category2%5B0%5D="
                                    "54&items_per_page=5")
    TrumpList.create_corpus()
    cleaned_sentences = TrumpList.corpus

    base_model = Word2Vec(vector_size=300, min_count=5)
    base_model.build_vocab(cleaned_sentences)
    total_examples = base_model.corpus_count

    base_model.build_vocab([list(glove_vectors.index_to_key)], update=True)

    base_model.train(cleaned_sentences, total_examples=total_examples, epochs=base_model.epochs)
    base_model_wv = base_model.wv

    print(base_model_wv.most_similar("criminal"))
    print(base_model_wv.most_similar("urban"))
