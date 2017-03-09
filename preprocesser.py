import numpy as np

class Preprocessor():
    
    def __init__ (self):
        pass
    
    def preprocess(self, data):
        self.data = data
        
    
    def meansub_norm(self):
        self.data -= np.mean(self.data, axis = 0)
        self.data /= np.std(self.data, axis = 0) 
        
        return self.data
     
       
    