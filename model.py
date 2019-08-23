import numpy as np
import cupy as cp
import scipy
from sparana.layers import sparse_relu_layer
from sparana.layers import sparse_linear_layer
from cupy.sparse import coo_matrix
from cupy.sparse import csr_matrix


class model:
    
    def __init__(self, input_size, layers, dropout = None, comp_type = 'CPU'):
        self._input_size = input_size
        self._layers = layers
        # Can set the dropout value for the whole network.
        self._dropout = dropout
        for layer in self._layers:
            layer._dropout = self._dropout
        self._comp_type = comp_type
        self._depth = len(layers)
        # Infer layers type from first layer, important here for transposing the input/output
        # Not planning to use mixed(full, and sparse) layers.
        self._layer_type = layers[0]._layer_type
    
    @property
    def layers(self):
        return self._layers
    
    def outputs(self, inputs):
        if self._layer_type == 'Sparse':
            this_layer_inputs = inputs.transpose()
        if self._layer_type == 'Full':
            this_layer_inputs = inputs
        if self._comp_type == 'GPU':
            this_layer_inputs = cp.array(this_layer_inputs)
        output = None
        for layer in self._layers:
            output = layer.activate(this_layer_inputs)
            this_layer_inputs = output
        if self._layer_type == 'Sparse':
            output = output.transpose()
        return output
    
    def softmax_outputs(self, inputs):
        # Returns a vector of outputs that have been passed through a softmax function
        return
    
    def get_accuracy(self, inputs, labels):
        for layer in self._layers:
            layer._dropout = None
        if self._comp_type == 'GPU':
            answers = np.argmax(self.outputs(inputs), axis = 1)
            correct = np.argmax(cp.array(labels), axis = 1)
        if self._comp_type == 'CPU':
            answers = np.argmax(self.outputs(inputs), axis = 1)
            correct = np.argmax(labels, axis = 1)
        accuracy = sum([i == j for i, j in zip(answers, correct)])/len(correct)
        for layer in self._layers:
            layer._dropout = self._dropout
        return accuracy
    
    def one2ratio(self, inputs):
        # Returns a vector of 1-2 ratios
        return
    
    def all_outputs(self, inputs):
        # Returns all 3, only do inference once.
        return
    
    def load_weights(self, weight_list):
        """ Load from a list of [(weight, bias), (weight, bias)]"""
        return
    
    def initialize_weights(self, init_method, bias_constant = None):
        """ This initializes all weights, I might add a module to initialize based on a list of
        values(normal SDs), one for each layer, but for now I am using Xavier initialization for everything."""
        if self._comp_type == 'CPU':
            print('Initalizing CPU weights')
            for i in range(len(self._layers)):
                if i == 0:
                    shape = (self._input_size, self._layers[i].size())
                else:
                    shape = (self._layers[i-1].size(), self._layers[i].size())
                if init_method == 'Xavier':
                    self._layers[i]._weights = np.random.normal(0, np.sqrt(3. / sum(shape)), shape)
                    self._layers[i]._biases = np.full(self._layers[i].size(), bias_constant)
                if type(init_method) == float:
                    self._layers[i]._weights = np.random.normal(0, init_method)
                    self._layers[i]._biases = np.full(self._layers[i].size(), bias_constant)
                if type(init_method) == list:
                    self._layers[i]._weights = init_method[i][0]
                    self._layers[i]._biases = init_method[i][1]
                    
        if self._comp_type == 'GPU':
            print('Initalizing GPU weights')
            for i in range(len(self._layers)):
                self._layers[i]._comp_type = 'GPU'
                if i == 0:
                    shape = (self._input_size, self._layers[i]._size)
                else:
                    shape = (self._layers[i-1]._size, self._layers[i]._size)
                if init_method == 'Xavier':
                    self._layers[i]._weights = cp.random.normal(0, np.sqrt(3. / sum(shape)), shape, dtype = np.float32)
                    self._layers[i]._biases = cp.full(self._layers[i]._size, bias_constant, dtype = np.float32)
                if type(init_method) == float:
                    self._layers[i]._weights = cp.random.normal(0, init_method)
                    self._layers[i]._biases = cp.full(self._layers[i]._size, bias_constant)
                
                ### TODO
                ### Check this, load into gpu.
                
            if type(init_method) == list:
                self._layers[i]._weights = init_method[i][0]
                self._layers[i]._biases = init_method[i][1]
        return
    
    def initialize_sparse_weights(self, density, init_method = 'Xavier', bias_constant = None):
        """ This initializes all weights, I might add a module to initialize based on a list of
        values(normal SDs), one for each layer, but for now I am using Xavier initialization for everything."""
        print('Initalizing sparse weights')
        for i in range(len(self._layers)):
            if i == 0:
                shape = (self._layers[i]._size, self._input_size)
            else:
                shape = (self._layers[i]._size, self._layers[i-1]._size)
            if init_method == 'Xavier':
                self._layers[i]._weights = csr_matrix(scipy.sparse.random(shape[0], shape[1], density = density, format = 'csr', dtype = np.float32, data_rvs=np.random.randn)*np.sqrt(3. / sum(shape)))
                self._layers[i]._biases = cp.full(self._layers[i]._size, bias_constant)
            # This is just rescaling the initialization by the density. 
            if init_method == 'Xavier_2':
                self._layers[i]._weights = csr_matrix(scipy.sparse.random(shape[0], shape[1], density = density, format = 'csr', dtype = np.float32, data_rvs=np.random.randn)*np.sqrt(3. / sum(shape))/density)
                self._layers[i]._biases = cp.full(self._layers[i]._size, bias_constant)
            if type(init_method) == float:
                self._layers[i]._weights = csr_matrix(scipy.sparse.random(shape[0], shape[1], density = density, format = 'csr', dtype = np.float32, data_rvs=np.random.randn)*init_method)
                self._layers[i]._biases = cp.full(self._layers[i]._size, bias_constant)
                    

        for layer in self.layers:
            layer.get_coordinates()
        return
    
    def convert_comp_type(self):
        
        if self._comp_type == 'GPU':
            old_comp_type = 'GPU'
            self._comp_type = 'CPU'
        if self._comp_type == 'CPU':
            old_comp_type = 'CPU'
            self._comp_type = 'GPU'
        
        print('Converting model from', old_comp_type, 'to', self._comp_type)
        
        for layer in self._layers:
            layer.convert_comp_type()
        
        return
    
    def convert_to_sparse(self):
        # This converts full layers to sparse layers, I don't have conv layers yet so not going to worry about that yet.
        if self._layer_type == 'Sparse':
            print('Model is already sparse')
            return
        if self._comp_type == 'CPU':
            print('Still have not added that feature')
            return
        if self._comp_type == 'GPU':
            for i in range(self._depth):
                rows = cp.repeat(cp.arange(self._layers[i]._weights.shape[0]), self._layers[i]._weights.shape[1])
                columns = cp.tile(cp.arange(self._layers[i]._weights.shape[1]), self._layers[i]._weights.shape[0])
                if self._layers[i]._activation_type == 'Relu':
                    self._layers[i] = sparse_relu_layer(size = self._layers[i]._size, weights = csr_matrix(self._layers[i]._weights.transpose()), biases = cp.array(self._layers[i]._biases))
                if self._layers[i]._activation_type == 'Linear':
                    self._layers[i] = sparse_linear_layer(size = self._layers[i]._size, weights = csr_matrix(self._layers[i]._weights.transpose()), biases = cp.array(self._layers[i]._biases))
            self._layer_type = 'Sparse'
            print('Model is now sparse')
            for layer in self.layers:
                layer.get_coordinates()
            
            return