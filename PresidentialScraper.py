import platform

import urllib.request

import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec as g2w
from gensim.test.utils import datapath
import urllib.request
import warnings
import re
import pandas as pd

from bs4 import BeautifulSoup
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class PresidentialScraper:
    def __init__(self, link):
        self.pages = []
        self.corpus = ""

        page = urllib.request.urlopen(link)
        soup = BeautifulSoup(page, "html.parser")

        pageCount = 0
        while page is not None:
            print(f"Processing query page {pageCount}")
            pageCount += 1
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
        print(f"create corpus from {len(self.pages)} pages")

        for i in self.pages:
            self.corpus += extract_speech(i)
        self.corpus = self.corpus.lower()

        replace_dict = {
            '[': "",
            ']': "",
            "(": "",
            ")": "",
            # ".": "",
            "mr.": "mr",
            "mrs.": "mrs",
            "ms.": "ms",
            ",": "",
            "!": "",
            "#": "",
            ":": "",
            "@": "",
            "?": "",
            "they're": "they are",
            "what's": "what is",
            "let's": "let us",
            "there've": "there have",
            "i've": "i have",
            "don't": "do not",
            "we'd": "we had",
            "he'd": "he had",
            "we're": "we are",
            "they're": "they are",
            "is'nt": "is not",
            "that's": "that is",
            "it's": "it is",
            "he's": "he is",
            "wasn't": "was not",
            "i'd": "i would",
            "you've": "you have",
            "you're": "you are",
            "you'd": "you would",
            "i'm": "i am",
            "we've": "we have",
            '-': " ",
            '"': "",
            "&": ""
        }
        # need to remove periods that are not ending a sentence like "Mr. Trump"
        stop = stopwords.words("english")
        self.corpus = multiple_replace(replace_dict, self.corpus)
        self.corpusSentenceTokens = [[token for token in nltk.word_tokenize(sent)
                                      if token not in stop]
                                     for sent in self.corpus.split('.')]
        self.corpus = [token for token in nltk.word_tokenize(self.corpus) if token != '.']
        originalLength = len(self.corpus)
        self.corpus = [word for word in self.corpus if word not in stop]
        print(f"Removing stop words reduced the {originalLength}-word "
              f"corpus by {originalLength - len(self.corpus)} to {len(self.corpus)} words.")
        temp = [self.corpus]
        self.corpus = temp
        print(f"corpus of {len(self.corpus[0])} tokenized words.")
        print(f"corpus of {len(self.corpusSentenceTokens)} tokenized sentences "
              f"containing {sum(len(sent) for sent in self.corpusSentenceTokens)} words.")
        return


def extract_speech(page):
    full_text = ""
    soup = BeautifulSoup(page, "html.parser")
    if "Tweets" in soup.select_one("title").getText():
        tweets = soup.select("td")
        for num, tweet in enumerate(tweets):
            # go through td elements and add only odd ones due to indexing
            if num % 2 != 0:
                temp = tweet.getText()
                temp = temp.split("Retweets", 1)
                full_text += temp[0] + " "

        full_text = multiple_replace({"RT": "", "@": ""}, full_text)
    else:
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


def showSample(vectors, target, count=2, modelLabel=""):
    if target in vectors.key_to_index.keys():
        sims = vectors.most_similar(target)[:2]
        if len(sims) >= count:
            print(f"{count} similar words for '{target}' using the {modelLabel}: {sims}")
            print(f"with {modelLabel} vocab size {len(vectors.key_to_index)}")
        else:
            print(f"{target} did not produce {count} similar words using {modelLabel}.")
    else:
        print(f"{target} is not available in the vocabulary of {modelLabel}.")
    return


if __name__ == '__main__':
    TrumpList = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                                    "&field-keywords3=&from%5Bdate%5D=&to%5Bdate%5D=&person2=200301&category2%5B%5D="
                                    "83&items_per_page=100")
    TrumpList.create_corpus()
    cleaned_sentences = TrumpList.corpusSentenceTokens # corpus

    dims = 50 # 300
    print(f"loading {dims}-dimensional glove vecs")
    if platform.system() == 'Darwin':
        fName = f"/Users/lance/Documents/GitHub/DogwhistleProject/Pretrained Vectors/glove.6B.{dims}d.txt"
    else:
        fName = f"C:\\Users\\dsm84762\\PycharmProjects\\DogwhistleProject\\Pretrained Vectors\\glove.6B.{dims}d.txt"
    glove_file = datapath(fName)
    tmp_file = get_tmpfile("test_word2vec.txt")
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    _ = g2w(glove_file, tmp_file)

    glove_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    print(f"Vectors loaded from {glove_file} have {glove_vectors.vector_size} element vectors for "
          f"{len(glove_vectors.key_to_index)} words.")


    base_model = Word2Vec(vector_size=dims, min_count=5)
    print(f"Build vocabulary for base_model using {sum(len(sent) for sent in cleaned_sentences)} words "
          f"in {len(cleaned_sentences)} sentences.")
    base_model.build_vocab(cleaned_sentences)
    print(f"base_model vocabulary is now {len(base_model.wv.vectors)}")
    total_examples = base_model.corpus_count

    print("Build vocabulary for base_model with glove vocabulary")
    base_model.build_vocab([list(glove_vectors.index_to_key)], update=True)
    print(f"base_model vocabulary is now {len(base_model.wv.vectors)}")

    base_model.train(cleaned_sentences, total_examples=total_examples, epochs=base_model.epochs)
    base_model_wv = base_model.wv

    showSample(glove_vectors, 'president',  modelLabel="base glove vectors")
    showSample(base_model_wv, 'president', modelLabel="retrained model")

    showSample(glove_vectors, 'crime', modelLabel="base glove vectors")
    showSample(base_model_wv, 'crime', modelLabel="retrained model")

    showSample(glove_vectors, 'terrorism', modelLabel="base glove vectors")
    showSample(base_model_wv, 'terrorism', modelLabel="retrained model")

    showSample(glove_vectors, 'drugs', modelLabel="base glove vectors")
    showSample(base_model_wv, 'drugs', modelLabel="retrained model")



