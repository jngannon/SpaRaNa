import numpy as np
import cupy as cp
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


class lobotomizer:
    
    ''' All stats arrays, sparse or no will be stored on the CPU ram, otherwise this will simply double the GPU memory requirements.
    These operations would be sped up on a GPU, but are run much less than training.'''
    
    def __init__(self, model):
        self._model = model
        self._weight_stats = [coo_matrix(i._weights.shape) for i in self._model.layers]
                
    def get_MAV(self, data):
        ''' This will run and store the mean activated values in the metric matrices in the class, sorts the list or whatever'''
        
        for i in data:
            if self._model._layer_type == 'Sparse':
                this_layer_inputs = i.transpose()
            if self._model._layer_type == 'Full':
                this_layer_inputs = i
            if self._model._comp_type == 'GPU':
                this_layer_inputs = cp.array(this_layer_inputs)
            output = None
                                  
            layer_count = 0
            for layer in self._model.layers:
                output = layer.activate(this_layer_inputs)
                this_layer_inputs = output
                if self._model._layer_type == 'Sparse':
                    self._weight_stats[layer_count] += layer.activate_weights(this_layer_inputs)
                # Convert the activatedd full layers to sparse matrices.
                if self._model._layer_type == 'Full':
                    self._weight_stats[layer_count] += csr_matrix(layer.activate_weights(this_layer_inputs))
                layer_count += 1
            if self._layer_type == 'Sparse':
                output = output.transpose()
        self._weight_stats = [coo_matrix(i) for i in self._weight_stats]
        for i in self._weight_stats:
            i.data = abs(i/len(data))
        return
    
    def get_MAAV(self, data):
        ''' MAAV is mean absolutes activated values'''
        for i in data:
            if self._model._layer_type == 'Sparse':
                this_layer_inputs = i.transpose()
            if self._model._layer_type == 'Full':
                this_layer_inputs = i
            if self._model._comp_type == 'GPU':
                this_layer_inputs = cp.array(this_layer_inputs)
            output = None
                                  
            layer_count = 0
            for layer in self._model.layers:
                if self._model._layer_type == 'Sparse':
                    self._weight_stats[layer_count] += abs(layer.activate_weights(this_layer_inputs))
                # Convert the activatedd full layers to sparse matrices.
                if self._model._layer_type == 'Full':
                    self._weight_stats[layer_count] += abs(coo_matrix(cp.asnumpy(layer.activate_weights(this_layer_inputs))))
                output = layer.activate(this_layer_inputs)
                this_layer_inputs = output
                layer_count += 1
            if self._model._layer_type == 'Sparse':
                output = output.transpose()
        self._weight_stats = [coo_matrix(i) for i in self._weight_stats]
        for i in self._weight_stats:
            i = i/len(data)
        return

    
    def get_absolute_values(self):
        ''' Stores the sorted list or whatever, either of these will just replace what is already there'''
        if self._model._comp_type == 'GPU':
            self._weight_stats = [coo_matrix(abs(i.weights.get())) for i in self._model.layers]
        if self._model._comp_type == 'CPU':
            self._weight_stats = [coo_matrix(abs(i.weights)) for i in self._model.layers]
        return
    
    def prune_smallest(self, prune_ratio, print_stats = False, layers = None):
        ''' Prunes the weights in the model class.
        Using the smallest values from weight stats to prune.
        Sparse matrices will be reconstructed and assigned to the layer classes.
        Layers needs to be a list, all layers will be pruned to the prune_ratios'''
        
        # Sparse GPU weights need to be reassigned, dont support index based assignment, full GPU, and sparse, and full CPU 
        # can be assigned, I will need to run eliminate zeros.
        if layers:
            for i in layers:
                
                if self._model._layer_type == 'Sparse' and self._model._comp_type == 'GPU':
                    # Copy weight matrix to CPU ram as a COO matrix
                    cpu_coo_matrix = self._model.layers[i]._weights.get().tocoo()
                    # Number of parameters to be removed
                    remove = int(prune_ratio*cpu_coo_matrix.nnz)
                    if print_stats:
                        print('Pruning ', remove,' parameters from ', len(cpu_coo_matrix.data), ' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._layer_stats[i].data)[:remove]
                    # New COO matrix with parameters removed
                    cpu_coo_matrix = coo_matrix((cpu_coo_matrix.data[sortlist], (cpu_coo_matrix.row[sortlist], cpu_coo_matrix.col[sortlist])), shape = cpu_coo_matrix.shape)                                        
                    # Copy back to GPU in the layer class as the original CSR matrix
                    self._model.layers[i]._weights = cp.sparse.csr_matrix(cpu_coo_matrix)
                else:
                    # Number of parameters to be removed
                    remove = int(prune_ratio*self._layer_stats[i].nnz)
                    if print_stats:
                        print('Pruning ', remove,' parameters from ', cpu_coo_matrix.nnz, ' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._layer_stats[i].data)[:remove]
                    # Loop through and set weights to 0
                    for j in sortlist:
                        self._model.layers[i]._weights[self._layer_stats[i].row[j], self._layer_stats[i].col[j]] = 0
        
        if not layers:
            for i in range(len(self._model.layers)):
                
                if self._model._layer_type == 'Sparse' and self._model._comp_type == 'GPU':
                    # Copy weight matrix to CPU ram as a COO matrix
                    cpu_coo_matrix = self._model.layers[i]._weights.get().tocoo()
                    # Number of parameters to be removed
                    remove = int(prune_ratio*cpu_coo_matrix.nnz)
                    if print_stats:
                        print('Pruning ', remove,' parameters from ', len(cpu_coo_matrix.data), ' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._weight_stats[i].data)[:remove]
                    # New COO matrix with parameters removed
                    cpu_coo_matrix = coo_matrix((cpu_coo_matrix.data[sortlist], (cpu_coo_matrix.row[sortlist], cpu_coo_matrix.col[sortlist])), shape = cpu_coo_matrix.shape)                                        
                    # Copy back to GPU in the layer class as the original CSR matrix
                    self._model.layers[i]._weights = cp.sparse.csr_matrix(cpu_coo_matrix)
                else:
                    # Number of parameters to be removed
                    remove = int(prune_ratio*self._weight_stats[i].getnnz())
                    if print_stats:
                        print('Pruning ', remove,' parameters from ',' parameters in layer ', i)
                    # List of indices of parameters to be removed
                    sortlist = np.argsort(self._weight_stats[i].data)[:remove]
                    # Loop through and set weights to 0
                    for j in sortlist:
                        self._model.layers[i]._weights[self._weight_stats[i].row[j], self._weight_stats[i].col[j]] = 0
            
        return