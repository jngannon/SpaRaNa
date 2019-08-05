import numpy as np
import cupy as cp

class sgd_optimizer:
    
    """ First attempt at building an optimizer, only uses quadratic cost function"""
    def __init__(self, model, learning_rate, l1_constant = None, l2_constant = None):
        self._model = model
        #if self._model.layers[0]._learning_rate != None:
        #    self._layer_learning_rates = True
        self._learning_rate = learning_rate
        self._gradients = []
        
    def get_gradients(self, inputs, labels):
        
        outputs = self._model.outputs(inputs)
        
        # This is hard coded quadratic error.
        gradients = []
        if self._model._comp_type == 'CPU':
            error = -(outputs - labels)
        if self._model._comp_type == 'GPU':
            error = -(outputs - cp.array(labels))
        for i in range(len(self._model.layers)):
            
            if i < len(self._model.layers)-1:
                # For the last layer, feed in the error calculated from the outputs, for the middle layers
                # feed in the error from the following layer, and outputs from the previous layer
                
                if self._model._layer_type == 'Full':
                    weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients( self._model.layers[-2-i]._outputs, error*self._model.layers[-1-i]._relu)
                if self._model._layer_type == 'Sparse':
                    if self._model.layers[-1-i]._activation_type == 'Relu':
                        weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients( self._model.layers[-2-i]._outputs, error*(self._model.layers[-1-i]._relu.transpose()))
                    if self._model.layers[-1-i]._activation_type == 'Linear':
                        weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients( self._model.layers[-2-i]._outputs, error)
                
                gradients.append((weight_gradients, bias_gradients))
            if i == len(self._model.layers)-1:
                # For the first layer, feed in the error from the following layer, and the inputs
                if self._model._comp_type == 'CPU':
                    weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients(inputs, error*self._model.layers[-1-i]._relu)
                if self._model._comp_type == 'GPU':
                    if self._model._layer_type == 'Full':
                        weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients(cp.array(inputs), error*self._model.layers[-1-i]._relu)
                    if self._model._layer_type == 'Sparse':
                        weight_gradients, bias_gradients, error = self._model.layers[-1-i].get_gradients(cp.array(inputs.transpose()), error*(self._model.layers[-1-i]._relu.transpose()))
                gradients.append((weight_gradients, bias_gradients))
        
        # Gradients are appended in reverse order, reverse thisto simplify applying training step
        gradients.reverse()
        
        return gradients
    

    def train_step(self, inputs, labels):
        grads = self.get_gradients(inputs, labels)
        for i in range(len(grads)):
            if self._model._layer_type == 'Full':
                self._model.layers[i]._weights +=  + self._learning_rate*grads[i][0]
                self._model.layers[i]._biases += self._learning_rate*grads[i][1]
            if self._model._layer_type == 'Sparse':
                self._model.layers[i]._weights = (self._model.layers[i]._weights + self._learning_rate*grads[i][0]).tocoo()
                self._model.layers[i]._biases += self._learning_rate*grads[i][1]
                