# from https://github.com/ahmedfgad/TorchGA/blob/main/torchga.py
import copy
import numpy as np
import torch
from metric_utils import to_numpy

def model2vector(model, filter_function=None):
    
    weights_vector = []
    
    model_params = model.parameters()
    if filter_function is not None:
        model_params = filter_function(model_params)

    for curr_weights in model_params:
        # Calling detach() to remove the computational graph from the layer. 
        # numpy() is called for converting the tensor into a NumPy array.
        curr_weights = to_numpy(curr_weights)
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)

    return np.array(weights_vector)


def vector2model(model, weights_vector, filter_function=None):
    
    params = model.parameters()
    
    if filter_function is not None:
        params = filter_function(params)
    
    start = 0
    for p in params:
        
        layer_weights_shape = p.shape
        layer_weights_size = p.numel()

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        
        p.data = torch.tensor(layer_weights_matrix, device=p.device, dtype=p.dtype)

        start = start + layer_weights_size

# def model_weights_as_dict(model, weights_vector):
    
#     weights_dict = model.state_dict()

#     start = 0
    
#     for key in weights_dict:
#         # Calling detach() to remove the computational graph from the layer. 
#         # numpy() is called for converting the tensor into a NumPy array.
#         w_matrix = weights_dict[key].detach().numpy()
#         layer_weights_shape = w_matrix.shape
#         layer_weights_size = w_matrix.size

#         layer_weights_vector = weights_vector[start:start + layer_weights_size]
#         layer_weights_matrix = numpy.reshape(layer_weights_vector, newshape=(layer_weights_shape))
#         weights_dict[key] = torch.from_numpy(layer_weights_matrix)

#         start = start + layer_weights_size

# #     return weights_dict

# def predict(model, solution, data):
#     # Fetch the parameters of the best solution.
#     model_weights_dict = model_weights_as_dict(model=model,
#                                                weights_vector=solution)

#     # Use the current solution as the model parameters.
#     model.load_state_dict(model_weights_dict)

#     predictions = model(data)

#     return predictions



class TorchGA:

    def __init__(self, model, num_solutions, filter_function=None, weight_magnitude=1.):

        """
        Creates an instance of the TorchGA class to build a population of model parameters.
        model: A PyTorch model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """
        
        self.model = model
        self.num_solutions = num_solutions
        self.filter_function = filter_function
        self.weight_magnitude = weight_magnitude 

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). 
        Each element in the list holds a different weights of the PyTorch model.
        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model2vector(model=self.model,
                                            filter_function=self.filter_function)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions-1):

            net_weights = copy.deepcopy(model_weights_vector)
            net_weights = np.array(net_weights) + np.random.uniform(low=-1.0,
                                                                  high=1.0, 
                                                                  size=model_weights_vector.size)
            
            net_weights = np.clip(net_weights, -self.weight_magnitude, self.weight_magnitude)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights