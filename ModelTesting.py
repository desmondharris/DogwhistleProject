import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Several methods used came from Sebastian Theiler in Analytics Vidhya.
# https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
glove_dict = {}
with open("Pretrained Vectors/glove.6B.50d.txt", 'r', encoding="utf-8") as file:
    for i in file:
        line = i.split()
        key = line[0]
        vector = np.array(line[1:], "float32")
        glove_dict[key] = vector


def find_closest_embeddings(embedding):
    return sorted(glove_dict.keys(), key=lambda word: spatial.distance.euclidean(glove_dict[word], embedding))


words = ["crime", "immigration", "drug", "thug"]
new_words = []
for i in words:
    temp = find_closest_embeddings(glove_dict[i])
    temp = temp[1:10]
    for j in temp:
        new_words.append(j)
for i in words:
    new_words.append(i)
vects = np.array([glove_dict[word] for word in new_words])
tsne = TSNE(n_components=2, random_state=0, perplexity=5)

two_dim = tsne.fit_transform(vects)

plt.scatter(two_dim[:, 0], two_dim[:, 1])
for label, x, y in zip(new_words, two_dim[:, 0], two_dim[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
plt.show()
