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
            soup = BeautifulSoup(i, "html.parser")
            title = soup.select_one(".diet-title").getText()
            title += soup.select_one(".field-ds-doc-title h1").getText()
            title += soup.select_one(".date-display-single").getText()
            replace_dict = {" ": ""}
            title = multiple_replace(replace_dict, title)
            title = title.lower()

            # Need to eventually check to see if file exists first
            temp = open(f"C:\\Users\\dsm84762\\PycharmProjects\\DogwhistleProject\\Presidential Corpus\\"
                        f"Raw Files\\{title}.txt", "w")
            speech = extract_speech(urllib.request.urlopen(i.url))
            temp.write(speech)

            self.corpus += speech
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
            "&": "",
            "*": ""
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
        sims = vectors.most_similar(target)[:count]
        if len(sims) >= count:
            print(f"{count} similar words for '{target}' using the {modelLabel}: {sims}")
            print(f"with {modelLabel} vocab size {len(vectors.key_to_index)}")
        else:
            print(f"{target} did not produce {count} similar words using {modelLabel}.")
    else:
        print(f"{target} is not available in the vocabulary of {modelLabel}.")
    return


if __name__ == '__main__':
    scraper = PresidentialScraper("https://www.presidency.ucsb.edu/advanced-search?field-keywords=&field-keywords2="
                                  "&field-keywords3=&from%5Bdate%5D=04-01-1840&to%5Bdate%5D=04-10-1840&person2="
                                  "&items_per_page=100")
    scraper.create_corpus()



