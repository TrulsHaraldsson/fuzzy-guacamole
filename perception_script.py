'''
Created on Dec 1, 2016

@author: eao
'''
from utilities import Utilities as util
from perceptron import *
from datahelper import *
from plotter import *
from preprocesser import *


# Just some constants
LABELS = 0
DATA = 1
filename = "DSL-StrongPasswordData.csv"

dm = DataManager()

# Read file
dm.pre_process(filename)

# Our super plotter
plotter = Plotter()


def prepare_for_back_propagation(batch_size, train_data, train_labels, valid_data, valid_labels):

    #print ("Creating data...")
    batched_train_data, batched_train_labels = util.create_batches(train_data, train_labels,
                                                                   batch_size,
                                                                   create_bit_vector=True)
    batched_valid_data, batched_valid_labels = util.create_batches(valid_data, valid_labels,
                                                                   batch_size,
                                                                   create_bit_vector=True)
    #print ("Done!")

    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels


indexOfPair = 1
for p in range(len(dm.subjects)):
    for pp in range(p + 1, len(dm.subjects)):

        # Extract labels and data from two subjects
        p1 = dm.get_subject(p)
        p2 = dm.get_subject(pp)

        labels_p1 = np.ones(400, dtype=int)
        data_p1 = p1[1]

        labels_p2 = np.zeros(400, dtype=int)
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

        # Preprocess copied data sets
        prepro = Preprocessor()
        pData = np.copy(data)
        test_pData = np.copy(test_data)
        prepro.preprocess(pData, test_pData)
        processed_data, processed_test_data = prepro.pca_whitening()  # PCA and Whitening
        #processed_data, processed_test_data = prepro.meansub_norm()  # Mean sub normalization

        batch_size = 100

        train_data, train_labels, valid_data, valid_labels = prepare_for_back_propagation(batch_size, processed_data, labels, processed_test_data, test_labels)

        mlp = MultiLayerPerceptron(layer_config=[31, 100, 50, 2], batch_size=batch_size)

        acc = mlp.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)
        plotter.add_y_data(acc)
        plotter.add_x_data(indexOfPair)
        print indexOfPair
        indexOfPair += 1

# Final plot
plotter.plot("Pair", "Accuracy", "Accuracies over pairs")
