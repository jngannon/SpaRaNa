import numpy as np
import cupy as cp
from sparana.model import model

class model_splitter:
    
    def __init__(model, data):
        self._model = model
        self._data = data
        # Get the MAAVs here
        self._maavs = []
        
    def split_model(size):
        # Get the model
        newmodel = [layer1, layer2]
        #assign weights
        
        #get new bias values
        have I saved this?
        #assign biases
        
        return ('here is a model')
    
    def split_sp_model(size):
        split_model()
        #now add the indices to train over
        return('here is another model')
    