import numpy as np
import cupy as cp
from cupy.sparse import coo_matrix

class full_relu_layer:
    
    def __init__(self, size, inputs = None, dropout = None, l2_constant = None, l1_constant = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Full'
        self._activation_type = 'Relu'
        self._weights = None
        self._biases = None
        self._dot_product = None
        self._add_biases = None
        self._relu = None
        self._outputs = None
        self._comp_type = 'CPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._learning_rate = learning_rate
        self._dropout = dropout
        
    def layer_type(self):
        return self._layer_type
    
    def size(self):
        return self._size
    
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            if self._dropout:
                self._dot_product = cp.dot(inputs, self._weights*cp.random.binomial(1, 1-self._dropout, size = self._weights.shape,))
            else:
                self._dot_product = cp.dot(inputs, self._weights)
        if self._comp_type == 'CPU':
            if self._dropout:
                self._dot_product = inputs@(self._weights*np.random.binomial(1, 1-self._dropout, size = self._weights.shape))
            self._dot_product = inputs@self._weights
        self._add_biases = self._dot_product + self._biases
        self._relu = self._add_biases>0
        self._outputs = self._add_biases*self._relu
        return self._outputs
    
    def activate_weights(self, inputs):
        if self._comp_type == 'GPU':
            return cp.multiply(self._weights, inputs[: , np.newaxis])
        if self._comp_type == 'CPU':
            return np.multiply(self._weights, inputs[: , cp.newaxis])
        
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        ''' Returns an array for weights, and biases, and one for the previous layer'''
        if self._comp_type == 'CPU':
            layer_error = layer_error*self._relu
            bias_gradients = np.sum(layer_error, axis = 0)
            if not self._l2_constant:
                weight_gradients = layer_inputs.transpose()@(layer_error)
            if self._l2_constant:
                weight_gradients = layer_inputs.transpose()@(layer_error) + self._l2_constant/(layer_error.shape[0])*self._weights
            if self._l1_constant:
                print('No l1 regularization yet')
                return
            previous_layer_error = layer_error@self._weights.transpose()
        if self._comp_type == 'GPU':
            layer_error = layer_error*self._relu
            bias_gradients = cp.sum(layer_error, axis = 0)
            if not self._l2_constant:
                weight_gradients = cp.dot(layer_inputs.transpose(), layer_error)
            if self._l2_constant:
                weight_gradients = cp.dot(layer_inputs.transpose(), layer_error) + self._l2_constant / (layer_error.shape[0]) * self._weights
            
            previous_layer_error = cp.dot(layer_error, self._weights.transpose())
            
        return weight_gradients, bias_gradients, previous_layer_error
        
    def convert_comp_type(self):
        if self._comp_type == 'GPU':
            self._comp_type = 'CPU'
            self._weights = cp.asnumpy(self._weights)
            self._biases = cp.asnumpy(self._biases)
        
        if self._comp_type == 'CPU':
            self._comp_type = 'GPU'
            self._weights = cp.array(self._weights)
            self._biases = cp.array(self._biases)
                
class full_linear_layer:
    
    def __init__(self, size, inputs = None, dropout = None, l2_constant = None, l1_constant = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Full'
        self._activation_type = 'Linear'
        self._weights = None
        self._biases = None
        self._dot_product = None
        self._add_biases = None
        self._relu = 1
        self._outputs = None
        self._comp_type = 'CPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._learning_rate = learning_rate
        self._dropout = dropout
        
    def layer_type(self):
        return self._layer_type
    
    def size(self):
        return self._size
    
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            if self._dropout:
                self._dot_product = cp.dot(inputs, self._weights*cp.random.choice([0, 1], size = self._weights.shape, p = [self._dropout, 1-self._dropout]))
            else:
                self._dot_product = cp.dot(inputs, self._weights)
        if self._comp_type == 'CPU':
            if self._dropout:
                self._dot_product = inputs@(self._weights*np.random.choice([0, 1], size = self._weights.shape, p = [self._dropout, 1-self._dropout]))
        self._add_biases = self._dot_product + self._biases
        self._outputs = self._add_biases
        return self._outputs
    
    def activate_weights(self, inputs):
        if self._comp_type == 'GPU':
            return cp.multiply(self._weights, inputs[: , np.newaxis])
        if self._comp_type == 'CPU':
            return np.multiply(self._weights, inputs[: , cp.newaxis])
        
    
    @property
    def weights(self):
        return self._weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        ''' Returns an array for weights, and biases, and one for the previous layer'''
        if self._comp_type == 'CPU':
            bias_gradients = np.sum(layer_error, axis = 0)
            if not self._l2_constant:
                weight_gradients = layer_inputs.transpose()@(layer_error)
            if self._l2_constant:
                weight_gradients = layer_inputs.transpose()@(layer_error) + self._l2_constant/(layer_error.shape[0])*self._weights
            if self._l1_constant:
                print('No l1 regularization yet')
                return
            previous_layer_error = layer_error@self._weights.transpose()
        if self._comp_type == 'GPU':
            bias_gradients = cp.sum(layer_error, axis = 0)
            if not self._l2_constant:
                weight_gradients = cp.dot(layer_inputs.transpose(), layer_error)
            if self._l2_constant:
                weight_gradients = cp.dot(layer_inputs.transpose(), layer_error) + self._l2_constant / (layer_error.shape[0]) * self._weights
            
            previous_layer_error = cp.dot(layer_error, self._weights.transpose())
            
        return weight_gradients, bias_gradients, previous_layer_error
        
    def convert_comp_type(self):
        if self._comp_type == 'GPU':
            self._comp_type = 'CPU'
            self._weights = cp.asnumpy(self._weights)
            self._biases = cp.asnumpy(self._biases)
        
        if self._comp_type == 'CPU':
            self._comp_type = 'GPU'
            self._weights = cp.array(self._weights)
            self._biases = cp.array(self._biases)    

class sparse_relu_layer:
    
    def __init__(self, size, weights = None, biases = None, inputs = None, dropout = None, l2_constant = None, l1_constant = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Sparse'
        self._activation_type = 'Relu'
        self._weights = weights
        self._biases = biases
        self._dot_product = None
        self._add_biases = None
        self._relu = None
        self._outputs = None
        # Default to running on GPU, if the sparse model isn't going to fit in GPU memory, you were fucked anyway.
        self._comp_type = 'GPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._rows = self._weights.tocoo().transpose().row
        self._columns = self._weights.tocoo().transpose().col
    
    @property    
    def get_inputs(self):
        return self._inputs
       
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            self._dot_product = self._weights.dot(inputs)
        if self._comp_type == 'CPU':
            # use the @ operator
            self._dot_product = inputs@self._weights
        self._add_biases = self._dot_product + self._biases[: , np.newaxis]
        self._relu = self._add_biases>0
        self._outputs = self._add_biases*self._relu
        return self._outputs

    @property
    def softmax_activate(self):
        dot_product = self._inputs@self._weights
        add_biases = dot_product + self._biases
        softmax = np.array([[np.exp(i)/sum([np.exp(j) for j in k]) for i in k] for k in add_biases])
        return softmax
        
    @property
    def weights(self):
        return self._weights
    
    def activate_weights(self, inputs):
        act_weights = self._weights.multiply(np.transpose(inputs))
        return act_weights
    
    @property
    def biases(self):
        return self._biases
    
    def get_gradients(self, layer_inputs, layer_error):
        grads_shape = self._weights.shape
        layer_error = layer_error*(self._relu.transpose())
        bias_gradients = cp.sum(layer_error, axis = 0)
        previous_layer_error = self._weights.transpose().dot(layer_error.transpose()).transpose()
        weight_gradients = sum(layer_inputs[self._rows,:].transpose()*layer_error[:,self._columns])
           
        return weight_gradients, bias_gradients, previous_layer_error

class sparse_linear_layer:
    
    def __init__(self, size, weights = None, biases = None, inputs = None, dropout = None, l2_constant = None, l1_constant = None, learning_rate = None):
        self._size = size
        self._layer_type = 'Sparse'
        self._activation_type = 'Linear'
        self._weights = weights
        self._biases = biases
        self._dot_product = None
        self._add_biases = None
        self._relu = None
        self._outputs = None
        # Default to running on GPU, if the sparse model isn't going to fit in GPU memory, you were fucked anyway.
        self._comp_type = 'GPU'
        # Regularization parameters, and learning rates can be set for layers individually
        self._l2_constant = l2_constant
        self._l1_constant = l1_constant
        self._learning_rate = learning_rate
        self._dropout = dropout
        self._rows = self._weights.transpose().tocoo().row
        self._columns = self._weights.transpose().tocoo().col
    
    @property    
    def get_inputs(self):
        return self._inputs
       
    def activate(self, inputs):
        if self._comp_type == 'GPU':
            self._dot_product = self._weights.dot(inputs)
        if self._comp_type == 'CPU':
            # use the @ operator
            self._dot_product = inputs@self._weights
        self._add_biases = self._dot_product + self._biases[: , np.newaxis]
        self._outputs = self._add_biases
        return self._outputs

    @property
    def softmax_activate(self):
        dot_product = self._inputs@self._weights
        add_biases = dot_product + self._biases
        softmax = np.array([[np.exp(i)/sum([np.exp(j) for j in k]) for i in k] for k in add_biases])
        return softmax
        
    @property
    def weights(self):
        return self._weights
    
    @property
    def activate_weights(self):
        act_weights = self._weights.multiply((np.transpose(self._inputs)))
        return act_weights
    
    @property
    def biases(self):
        return self._biases
               
    def get_gradients(self, layer_inputs, layer_error):
        grads_shape = self._weights.shape
        bias_gradients = cp.sum(layer_error, axis = 0)
        previous_layer_error = self._weights.transpose().dot(layer_error.transpose()).transpose()
        weight_gradients = sum(layer_inputs[self._rows,:].transpose()*layer_error[:,self._columns])
           
        return weight_gradients, bias_gradients, previous_layer_error

        
      