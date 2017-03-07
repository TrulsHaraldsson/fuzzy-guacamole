import filereader
from NearestNeighbor import *
import numpy as np

# Just some constants
LABELS = 0
DATA = 1
filename = "DSL-StrongPasswordData.csv"

# Read file
result = filereader.readFile(filename)

# Extract labels and data from two subjects
personOne = [0, 400]
personTwo = [400, 800]

labelsP1 = np.array(result[LABELS][personOne[0]:personOne[1]])
dataP1 = np.array(result[DATA][personOne[0]:personOne[1]]) 

labelsP2 = np.array(result[LABELS][personTwo[0]:personTwo[1]])
dataP2 = np.array(result[DATA][personTwo[0]:personTwo[1]]) 

labels = np.concatenate((labelsP1[:300], labelsP2[:300]), axis = 0)
data = np.concatenate((dataP1[:300], dataP2[:300]), axis = 0)

test_labels = np.concatenate((labelsP1[300:400], labelsP2[300:400]), axis = 0)
test_data = np.concatenate((dataP1[300:400], dataP2[300:400]), axis = 0)

# Shuffle the data
rng_state = np.random.get_state()
np.random.shuffle(labels)
np.random.set_state(rng_state)
np.random.shuffle(data)

# Train NN
nn = NearestNeighbor()
#print "data length:", len(data),"\nlabel length:", len(labels)
#print type(data)," ", type(labels)
k = nn.cross_validation(data, labels, 4)
print "the highest value k is equal to :", k, "for testing."

# Time to test out test_data/labels
nn.train(data, labels, k)
prediction = nn.predict(test_data)
predictionAccuracy = '%f' % (np.mean(prediction == test_labels) )
print"Test complete!\nThe accurracy is : ", predictionAccuracy


