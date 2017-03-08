import numpy as np
from collections import Counter


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, data, labels, k): 
        """ Train specific data, labels and k-value.
            :param data:
            :param labels:
            :param k:
        """
        self.labels = labels
        self.data = data
        self.k = k

    def cross_validation(self, train_data, train_labels, folds):
        """ Cross-validation for training_data, training_labels and for certain folds.
            :param train_data:
            :param train_labels:
            :param folds:
            :return highest accuracy:
        """
        #  Split labels and data into folds.
        label_folds = np.split(train_labels, folds)
        data_folds = np.split(train_data, folds)
        
        #  accuracies
        votingbooth = []
        
        #  Test different k-values
        iterationsOfK = [1, 2, 3, 4, 5, 6, 7, 8]
        
        #  merge sub-folds into training sets and validation sets.
        for i in xrange(folds): 
            train_sub_labels = None
            train_sub_data = None
            valid_sub_labels = None
            valid_sub_data = None
        
            for j in xrange(folds): 
                
                if j != i:
                    if train_sub_labels is None:
                        train_sub_labels = label_folds[j]
                        train_sub_data = data_folds[j]
                    else:
                        train_sub_labels = np.concatenate((train_sub_labels, label_folds[j]), axis=0)
                        train_sub_data = np.concatenate((train_sub_data, data_folds[j]), axis=0)
                else: 
                    valid_sub_labels = np.array(label_folds[j])
                    valid_sub_data = np.array(data_folds[j])
            
            #  Stores a list of accuracies for each iteration, so it can be calculated in votingbooth's best accuracy.
            listOfAccuracies = []
            
            #print "labels", len(valid_sub_labels), "training_labels", len(train_sub_labels)
            
            # Make a prediction for each iteration.
            for k in iterationsOfK:
                self.train(train_sub_data, train_sub_labels, k)
                prediction = self.predict(valid_sub_data)
                predictionAccuracy = '%f' % (np.mean(prediction == valid_sub_labels) )
                #print "K: ", k, " Acc: ", predictionAccuracy
                listOfAccuracies.append(predictionAccuracy)   
                    
            votingbooth.append(listOfAccuracies)
        
        #print "All the tests are now done!\nThe best results will now be calculated for best K value!"
        
        highestAccuracy = 0
        highestIteration = None
        
        #  Calculate the best k-value by adding votingbooth's values together for each fold.
        for i in iterationsOfK: 
            
            #  4 folds add 4 values.
            testAccuracy = float(votingbooth[0][i-1]) + float(votingbooth[1][i-1]) + float(votingbooth[2][i-1]) + float(votingbooth[3][i-1])
            #print "testAccuracy of the folds are : ", testAccuracy
            if highestAccuracy < testAccuracy:
                highestAccuracy = testAccuracy
                highestIteration = i
               
        return highestIteration   

    def predict(self, vData): 
        """Predict method that actually measures the distances between trained data and the validation data.
           :param vData: 
           :return prediction:
        """
        num_test = vData.shape[0]
        
        predictions = np.zeros(num_test, dtype = self.labels.dtype)
        
        for i in xrange(num_test):
            distances = np.linalg.norm(self.data - vData[i,:], axis=1)  # L2
            predictions[i] = self.vote(distances)
         
        return predictions

    def vote(self, distances):
        """ The method creates the opportunity to let the distance vectors vote on who they think they are related to (e.g. p1 or p2).
            This is because the distances is a random persons data that will be matched to either p1 or p2.
            :param distances:
            :return label vote:
        """
        
        distance = np.argsort(distances)
        votes = distance[:self.k]
        prediction = {} # There are two alternatives for the vote, person 1 or person 2.
        for i in votes:
            label = self.labels[i]
            prediction[i] = label
        
        cnt = Counter()
        for p in prediction:
            cnt[p] += 1
        maxVote = cnt.most_common(1)  # One is always best. Or at least got most votes (think Trump)..
        
        return self.labels[maxVote[0][0]]