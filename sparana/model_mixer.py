import numpy as np
import cupy as cp

class model_mixer:
    
    ''' Functions for combining the weights of different models. The models should probably be based on an initial training run so that the majority of parameters are the same to begin with.'''
    
    def __init__(self):
        
        self._model_parameters = []
        self._parameter_indicies = []
        self._weight_stats = []        
        
    def add_model(self, model, replace = -1):
        '''Just add a model, stats will be added seperately'''
        if replace == -1:
            # This is only storing weights for now
            self._model_parameters.append([cp.asnumpy(i.weights) for i in model.layers])
        else:
            self._model_parameters[replace] = [cp.asnumpy(i.weights) for i in model.layers]
        return
    
    def add_stats(self, stats, replace = -1):
        '''One stats array per model, should add them with an index so that I can change them at any time. If replace does not specify a position, they will be added to the last set in the list. '''
        
        return
    
    def add_MAAV_stats(self, model, data, replace = -1):
        ''' Adds MAAV stats from another module I built.'''
        if replace == -1:
            self._weight_stats.append([np.zeros(i._weights.shape) for i in model.layers])
        else:
            self._weight_stats[replace] = [np.zeros(i._weights.shape) for i in model.layers]
            
        for layer in model.layers:
            layer._dropout = None
        for i in data:
            if model._layer_type == 'Sparse':
                this_layer_inputs = i.transpose()
            if model._layer_type == 'Full':
                this_layer_inputs = i
            if model._comp_type == 'GPU':
                this_layer_inputs = cp.array(this_layer_inputs)
            output = None
            layer_count = 0
            for layer in model.layers:
                # Get the activated values
                self._weight_stats[replace][layer_count] += abs(cp.asnumpy(layer.activate_weights(this_layer_inputs)))
                # Run a step forward in the model
                output = layer.activate(this_layer_inputs)
                # Store the output for the next layers input
                this_layer_inputs = output
                # Iterate through the layers
                layer_count += 1
            
        # Convert stuff here
        for i in self._weight_stats[replace]:
            i = i/len(data)
        for layer in model.layers:
            layer._dropout = model._dropout
        return
    
    def add_indices(self, indices, replace = None):
        ''' Adds the indices for one of the models, if replace does not specify a position, they will be added to the last set in the list.'''        
        return
    
    def add_absolute_values(self, model, replace = -1):
        ''' Adds the absolute values as the stats.'''
        if replace == -1:
            self._weight_stats.append([abs(cp.asnumpy(i._weights)) for i in model.layers])
        else:
            self._weight_stats[replace] = ([abs(cp.asnumpy(i._weights)) for i in model.layers])
        
        return
    
    def rank_and_zero(self, models = False, zero = False): 
        '''Ranks the stats arrays and zeros a ratio of them. Set zero= ratio for the ratio values you want set to 0.'''
        if models == False:
            models = np.arange(len(self._weight_stats))
        for i in models:
            for j in range(len(self._weight_stats[i])):
                self._weight_stats[i][j] = np.argsort(self._weight_stats[i][j], axis = None)
                temp = np.empty_like(self._weight_stats[i][j])
                temp[self._weight_stats[i][j]] = np.arange(len(self._weight_stats[i][j]))
                self._weight_stats[i][j] = np.reshape(temp, self._model_parameters[i][j].shape)+1
                
            #self._weight_stats[i] = [np.argsort(j, axis = None).reshape(j.shape)+1 for j in self._weight_stats[i]]
            if i != 0 and zero:
                for j in self._weight_stats[i]:
                    j[j < zero*(j.size)] = 0 
        return
    
    def replace_all(self, model, primary, secondary):
        ''' Replaces all of the parameters in the primary model with those in the secondary determined by non zero values.'''
        output_parameters = self._weight_stats[primary]
        output_parameters *= (self._weight_stats[secondary]!=0)
        output_parameters += self._weight_stats[secondary]
        #output is adding them together, putting them into the model. 
        for i in range(len(output_parameters)):
            if model._comp_type == 'CPU':
                model.layers[i]._weights = output_parameters[i]
            if model._comp_type == 'GPU':
                model.layers[i]._weights = cp.array(output_parameters[i])
            
        return
    
    def average_all(self, model):
        '''Returns a model with the average of all of the parameters, don't think it will work well in most situations.'''
        output_parameters = [np.zeros(i._weights.shape) for i in model.layers]        
        for i in range(len(self._model_parameters)):
            for j in range(len(output_parameters)):
                output_parameters[j] += self._model_parameters[i][j]
        output_parameters = [i/len(self._model_parameters) for i in output_parameters]
        # Now put them back in the model
        for i in range(len(output_parameters)):
            if model._comp_type == 'CPU':
                model.layers[i]._weights = output_parameters[i]
            if model._comp_type == 'GPU':
                model.layers[i]._weights = cp.array(output_parameters[i])
        return
    
    def importance_weighted_average(self, model):
        '''Averages parameters based on their relative importance according to the weight_stats arrays.'''
        allsums = [np.zeros(i._weights.shape) for i in model.layers]
        output_parameters = [np.zeros(i._weights.shape) for i in model.layers]
        for i in range(len(self._weight_stats)):
            for j in range(len(output_parameters)):
                allsums[j] += self._weight_stats[i][j]
        for i in range(len(self._weight_stats)):
            for j in range(len(output_parameters)):
                
                output_parameters[j] += self._model_parameters[i][j]*(self._weight_stats[i][j]/allsums[j])
            
        # Now put the parameters back in the model
        for i in range(len(output_parameters)):
            if model._comp_type == 'CPU':
                model.layers[i]._weights = output_parameters[i]
            if model._comp_type == 'GPU':
                model.layers[i]._weights = cp.array(output_parameters[i])
        return 
    
