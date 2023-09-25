# Import necessary libraries
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random

# Initialize stemmer
stemmer = LancasterStemmer()

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Initialize variables
words = []
labels = []
docs_x = []
docs_y = []

# Tokenize patterns and get unique labels
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each pattern
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    
    # Add unique label
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stemming, removing duplicates and sorting
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Initialize training data
training = []
output = []

# Placeholder for output
out_empty = [0 for _ in range(len(labels))]

# Create bag-of-words representation
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    # Mark the label in output
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert training data and output to numpy arrays
training = numpy.array(training)
output = numpy.array(output)

# Reset TensorFlow graph
tensorflow.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Train and save the model
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
