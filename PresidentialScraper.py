import nltk
import gensim.downloader as api
from gensim.models import Word2Vec
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
    glove = api.load('glove-wiki-gigaword-50')
    old_data = [list(glove.key_to_index.keys())]

    TrumpList = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                                    "&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2=200301&category2%5B0%5D="
                                    "54&items_per_page=5")
    TrumpList.create_corpus()
    new_data = TrumpList.corpus

    bare = Word2Vec(vector_size=50, min_count=5)
    bare.build_vocab(new_data)
    old_data = [list(glove.key_to_index.keys())]
    bare.build_vocab(old_data, update=True)
    total = bare.corpus_count
    bare.train(new_data, total_examples=bare.corpus_count, epochs=bare.epochs)
    vectors = bare.wv

    keywords = ["urban", "thug", "drugs", "terror"]
    for i in keywords:
        print(vectors.most_similar(i))
