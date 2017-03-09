import numpy as np

class Preprocessor():
    
    def __init__ (self):
        pass
    
    def preprocess(self, train_data, validation_data):
        self.tData = train_data
        self.vData = validation_data
        
    def meansub_norm(self):
        mean = np.mean(self.tData, axis = 0)
        self.tData -= mean
        self.vData -= mean
        
        std = np.std(self.data, axis = 0) 
        self.tData /= std
        self.vData /= std
        
        return self.tData, self.vData
    
    