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
        
        std = np.std(self.tdata, axis = 0) 
        self.tData /= std
        self.vData /= std
        
        return self.tData, self.vData
    
    def pca_whitening(self):
        mean = np.mean(self.tData, axis = 0)
        self.tData -= mean
        self.vData -= mean
        
        covariance = np.dot(self.tData.T, self.tData) / self.tData.shape[0]
        U,S,V = np.linalg.svd(covariance)
        Xrot_t = np.dot(self.tData, U)
        Xrot_v = np.dot(self.vData, U)
        
        Xwhite_t = Xrot_t / np.sqrt(S + 1e-5)
        Xwhite_v = Xrot_v / np.sqrt(S + 1e-5)
        
        return Xwhite_t, Xwhite_v
        
        