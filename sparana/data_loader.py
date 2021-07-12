import numpy as np
import gzip

class loader:
    
    def __init__(self, training_data, training_labels, test_data, test_labels):
        ''' I will need to feed in extracted data, data might be stored in different formats.
        This is for shuffling and tracking training.'''
        self._training_data = training_data
        self._training_labels = training_labels
        self._test_data = test_data
        self._test_labels = test_labels
        self._index_list = np.arange(len(self._training_data))
        np.random.shuffle(self._index_list)
        # Now for some things to keep track of
        self._minibatches = 0
        self._epochs = 0
        self._minibatch_index = 0
        
    def random_minibatch(self, batch_size):
        ''' This just selects a random minibatch from the whole training set, doesn't track epochs'''
        np.random.shuffle(self._index_list)
        data = self._training_data[self._index_list[:batch_size]]
        labels = self._training_labels[self._index_list[:batch_size]]
        return data, labels
    
    def minibatch(self, batch_size):
        ''' This takes minibatches from a shuffled training data set, tracks epochs, 
        and reshuffles the training data when the epoch is complete'''
        if self._minibatch_index + batch_size > len(self._training_data):
            np.random.shuffle(self._index_list)
            self._epochs += 1
            self._minibatch_index = 0
        data = self._training_data[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        labels = self._training_labels[self._index_list[self._minibatch_index : self._minibatch_index + batch_size]]
        self._minibatch_index += batch_size
        self._minibatches += 1
        return data, labels
    
    def print_stats(self):
        ''' Prints any useful information'''
        print('Epochs: ', self._epochs)
        print('Minibatches: ', self._minibatches)
        return
    
    def test_data(self):
        '''This seems pretty pointless until I am loading files directly with this class.'''
        return self._test_data


    def test_labels(self):
        return self._test_labels
