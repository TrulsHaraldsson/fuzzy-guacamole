import filereader
from NearestNeighbor import *
import numpy as np
from datahelper import *
import matplotlib.pyplot as plt


# Just some constants
LABELS = 0
DATA = 1
filename = "DSL-StrongPasswordData.csv"

# Read file
result = filereader.readFile(filename)
dm = DataManager()

# Read file
dm.pre_process(filename)

#list all the pair accuracies
accuracyPairs = []

#Loop the subjects and extract labels and data from two subjects.
#for p in range(len(dm.subjects)):
for p in range(3):
    #for pp in range(len(dm.subjects)):
    for pp in range(3):
        if p != pp and (p < pp): # index of p is always less than index of pp.
            p1 = dm.get_subject(p)
            p2 = dm.get_subject(pp)
            
            #Person 1 label and data.
            labels_p1 = p1[0]
            data_p1 = p1[1]
            
            #Person 2 label and data.
            labels_p2 = p2[0]
            data_p2 = p2[1]

            # Training data
            labels = np.concatenate((labels_p1[:300], labels_p2[:300]), axis=0)
            data = np.concatenate((data_p1[:300], data_p2[:300]), axis=0)
            
            # Test data
            test_labels = np.concatenate((labels_p1[300:400], labels_p2[300:400]), axis=0)
            test_data = np.concatenate((data_p1[300:400], data_p2[300:400]), axis=0)

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
            #print "the highest value k is equal to :", k, "for testing."
            
            # Time to test out test_data/labels
            nn.train(data, labels, k)
            prediction = nn.predict(test_data)
            predictionAccuracy = '%f' % (np.mean(prediction == test_labels) )
            print"Test complete!\nThe accurracy is : ", predictionAccuracy, "between subjects : ", p, " and ", pp            
            accuracyPairs.append(predictionAccuracy)
