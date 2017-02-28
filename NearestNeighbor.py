import progressbar
import numpy as np

class NearestNeighbor(Object):
    def __init__(self):
        pass
    
    def train(self, labels, data, k): # Store the training data that will be matched from validation data.
        self.labels = labels
        self.data = data
        self.k = k
    
    def folds(self, train_labels, train_data, folds):
        
        # By splitting up the data into folds and comparing the different votes they produce, we can get a k best suited for all folds.            
        votingbooth = np.zeros(folds)
        validation_labels = np.split(train_labels, folds)
        validation_data = np.split(train_data, folds)
        
        # then to do the cross validation of training and validation folds and get all the best accuracies.
        train_sub_labels = None
        train_sub_data = None
        valid_sub_labels = None
        valid_sub_data = None
        
        for i in xrange(folds): # Crossvalidation i folds
            for j in xrange(folds): # Folder j == i is the validation, rest is training.
                if(j != i):  
                    train_sub_labels.append(validation_labels[j])
                    train_sub_data.append(validation_data[j])
                else: 
                    valid_sub_labels.append(validation_labels[j])
                    valid_sub_data.append(validation_data[j])
                    
            #Now that we have train_sub and valid_sub we do the prediction.
            iterationsOfK = [1,2,4,8]
            listOfAccuracies = []
            
            for k in iterationsOfK:
                self.train(train_sub_labels, train_sub_data, k)
                prediction = self.predict(valid_sub_data)
                predictionAccuracy = '%f' % (np.mean(prediction == valid_sub_labels) )
                listOfAccuracies.append(predictionAccuracy)
                
            votingbooth[i] = iterationsOfK[np.argmax(listOfAccuracies)]
            
        return np.mean(votingbooth)    
                
    def predict(self, validation_data):
        """ validation_data is N x D where each row is an example we wish to predict label for """
        num_test = validation_data.shape[0]
        
        bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
            
        for i in xrange(num_test):
        
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # L1
            distances = np.linalg.norm(self.Xtr - X[i,:], axis = 1) # L2
            
            Ypred[i] = self.vote(distances)
            
            bar.update(i+1)
        bar.finish()
        
        return Ypred

    # Method vote creates the opportunity to let the distance vectors vote on who they think they are related to (e.g. p1 or p2).
    # This is because the distances is a random persons data that will be matched to either p1 or p2.
    def vote(self, distances): 
        distance = np.argsort(distances)
        votes = distance[:self.k]
        prediction = np.zeros(2) # There are two alternatives for the vote, person 1 or person 2.

    
