import progressbar
import numpy as np

class NearestNeighbor():
    def __init__(self):
        pass
    
    def train(self, data, labels, k): # Store the training data that will be matched from validation data.
        self.labels = labels
        self.data = data
        self.k = k
    
    def cross_validation(self, train_data, train_labels, folds):
        
        # By splitting up the data into folds and comparing the different votes they produce, we can get a k best suited for all folds.            
        label_folds = np.split(train_labels, folds)
        data_folds = np.split(train_data, folds)
        
        # then to do the cross validation of training and validation folds and get all the best accuracies.
        train_sub_labels = None
        train_sub_data = None
        valid_sub_labels = None
        valid_sub_data = None
        
        #accuracies
        votingbooth = []
        
        #Test different k-values
        iterationsOfK = [1,2,4,8]
        
        # print "folds: ", folds, "label_folds: ", len(label_folds), "data_folds: ", len(data_folds)
        for i in xrange(folds): # Crossvalidation i folds
            for j in xrange(folds): # Folder j == i is the validation, rest is training.
                if(j != i):
                    if train_sub_labels is None:
                        train_sub_labels = label_folds[j]
                        train_sub_data = data_folds[j]
                    else:
                        train_sub_labels = np.concatenate((train_sub_labels, label_folds[j]), axis = 0) #concatenate works fine.
                        train_sub_data = np.concatenate((train_sub_data, data_folds[j]), axis = 0) #concatenate works fine.
                else: 
                    valid_sub_labels = np.array(label_folds[j])
                    valid_sub_data = np.array(data_folds[j])
                    #print "valid_sub_length: ", len(valid_sub_data), " ", len(valid_sub_labels)
                    
            listOfAccuracies = []
            
            print "labels", len(valid_sub_labels), "training_labels", len(train_sub_labels)
            
            for k in iterationsOfK:
                self.train(train_sub_data, train_sub_labels, k)
                prediction = self.predict(valid_sub_data)
                predictionAccuracy = '%f' % (np.mean(prediction == valid_sub_labels) )
                print "K: ", k, " Acc: ", predictionAccuracy
                listOfAccuracies.append(predictionAccuracy)       
            votingbooth.append(listOfAccuracies)
            
        highestAccuracy = 0
        highestIteration = None
        
        for i in iterationsOfK: # adding fold accuracys to see total best value of k.
            testAccuracy = float(votingbooth[0][i]) + float(votingbooth[1][i]) + float(votingbooth[2][i]) 
            if(highestAccuracy < testAccuracy): 
                highestAccuracy = testAccuracy
                highestIteration = i
        
        return highestIteration     
                

    def predict(self, test_data): 
        num_test = test_data.shape[0]
        bar = progressbar.ProgressBar(maxval=num_test,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        
        # lets make sure that the output type matches the input type
        predictions = np.zeros(num_test, dtype = self.labels.dtype)
        
        print"self.data length: ", len(self.data[0]), "test_data length: ", len(test_data[0])    
        print "type:", type(self.data[0][0]), "type:", type(test_data[0][0])
        print "self.data: ", self.data[0][0], "test_data: ", test_data[0][0]
        #print int(self.data[0][0] - test_data[0][0])
        
        #print float(self.data[0][0]) + float(test_data[0][0])
        
        for i in xrange(num_test):
            distances = np.linalg.norm(self.data - test_data[i,:], axis=1)  # L2
            predictions[i] = self.vote(distances)
            bar.update(i+1)
        bar.finish()
         
        return predictions

    def vote(self, distances):
        """ The method creates the opportunity to let the distance vectors vote on who they think they are related to (e.g. p1 or p2).
            This is because the distances is a random persons data that will be matched to either p1 or p2.
            :param distances:
            :return:
        """
        distance = np.argsort(distances)
        votes = distance[:self.k]
        prediction = np.zeros(2) # There are two alternatives for the vote, person 1 or person 2.
        for i in votes:
            label = self.labels[int(i)]
            prediction[label] += 1
        return np.argmax(prediction)