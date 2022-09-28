import numpy as np
from scipy import spatial
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Several methods used came from Sebastian Theiler in Analytics Vidhya.
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db

def loadEmbedding(fName):
    glove_dict = {}
    with open(fName, 'r', encoding="utf-8") as file:
        for i in file:
            line = i.split()
            key = line[0]
            vector = np.array(line[1:], "float32")
            glove_dict[key] = vector
        return glove_dict

def find_closest_embeddings(embedding):
    return sorted(glove_dict.keys(), key=lambda word: spatial.distance.euclidean(glove_dict[word], embedding))

def distance(v0, v1):
    """
    Given two vectors, return the Euclidian distance between them
    :param v0: a vector of floats
    :param v1: a vector of floats
    :return: Euclidian distance as a float
    """
    return sum((v0[i] -v1[i])**2 for i in range(len(v0)))**.5

### Main

glove_dict = loadEmbedding("Pretrained Vectors/glove.6B.50d.txt")

words = ["crime", "immigration", "drug", "thug"]
new_words = []
source = {}
for word in words:
    closeWords = find_closest_embeddings(glove_dict[word])
    new_words.extend(closeWords[1:10])
    source.update({close: word for close in closeWords[0:10]})
new_words.extend(words)
vects = np.array([glove_dict[word] for word in new_words])
tsne = TSNE(n_components=2, random_state=0, perplexity=5)

two_dim = tsne.fit_transform(vects)

plt.scatter(two_dim[:, 0], two_dim[:, 1])
for label, x, y in zip(new_words, two_dim[:, 0], two_dim[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    print(f"{label:15}: {x:6.4}, {y:6.4} {source[label]:12} "
          f"{distance(glove_dict[source[label]], glove_dict[label]):4.4}")
plt.show()
