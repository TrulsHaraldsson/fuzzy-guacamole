import progressbar
import numpy as np

class NearestNeighbor():
    def __init__(self):
        pass
    
    def train(self, data, labels, k): # Store the training data that will be matched from validation data.
        self.labels = labels
        self.data = data
        self.k = k
    
    def cross_validation(self, train_labels, train_data, folds):
        
        # By splitting up the data into folds and comparing the different votes they produce, we can get a k best suited for all folds.            
        votingbooth = np.zeros(folds)
        label_folds = np.split(train_labels, folds)
        data_folds = np.split(train_data, folds)
        
        # then to do the cross validation of training and validation folds and get all the best accuracies.
        train_sub_labels = None
        train_sub_data = None
        valid_sub_labels = None
        valid_sub_data = None
        
        for i in xrange(folds): # Crossvalidation i folds
            for j in xrange(folds): # Folder j == i is the validation, rest is training.
                if(j != i):
                    if train_sub_labels is None:
                        train_sub_labels = label_folds[j]
                        train_sub_data = data_folds[j]
                    else:
                        train_sub_labels = np.append(train_sub_labels, label_folds[j])
                        train_sub_data = np.append(train_sub_data, data_folds[j], axis=0)
                    #train_sub_labels.append(validation_labels[j])
                    #train_sub_data.append(validation_data[j])
                else: 
                    valid_sub_labels = np.array(label_folds[j])
                    valid_sub_data = np.array(data_folds[j])
                    
            #Now that we have train_sub and valid_sub we do the prediction.
            iterationsOfK = [1,2,4,8]
            listOfAccuracies = []
            print "labels", len(valid_sub_labels), "training_labels", len(train_sub_labels)
            for k in iterationsOfK:
                self.train(train_sub_labels, train_sub_data, k)
                prediction = self.predict(valid_sub_data)
                predictionAccuracy = '%f' % (np.mean(prediction == valid_sub_labels) )
                print "K: ", k, " Acc: ", predictionAccuracy
                listOfAccuracies.append(predictionAccuracy)
                
            votingbooth[i] = iterationsOfK[np.argmax(listOfAccuracies)]
            
        return np.mean(votingbooth)    
                
    def predict(self, test_data):
        num_test = test_data.shape[0]
        bar = progressbar.ProgressBar(maxval=num_test,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        
        # lets make sure that the output type matches the input type
        predictions = np.zeros(num_test, dtype=self.labels.dtype)
            
        for i in xrange(num_test):
            distances = np.linalg.norm(self.data - test_data[i, :], axis=1)  # L2
            predictions[i] = self.vote(distances)
            bar.update(i+1)
        bar.finish()
        
        return predictions

    def vote(self, distances):
        """
        The method creates the opportunity to let the distance vectors vote on who they think they are related to (e.g. p1 or p2).
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