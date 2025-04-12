# snnTorch Imports
from snntorch import surrogate

# Torch Imports
import torch
import torch.nn as nn

# Python Libary Imports
import random

# Script Imports
from EONS import EONS



def crossover(parent1: EONS, parent2: EONS):
        """
        This function picks a random spot in the network of parent1, then
        mixes the layers of both the parents to create the child layers.
        The child network structure is defined as:
                parent_first = 1:
                        - parent1 when index < crossover_point
                        - parent2 when index > crossover_point
                parent_first = 2:
                        - parent2 when index < crossover_point
                        - parent1 when index > crossover_point

        Currently, I am choosing to exclude having the crossover point
        be the input and output layers of the parent networks.

        The I/O between synapses is also fixed after the crossover has
        been completed. 
        """
        # Combine layers and weights of parent1 and parent2
        crossover_point = random.randint(2, len(parent1.template_network) - 2) # Exclude input and output layer
        parent_first = random.randint(1, 2) # Randomly decide which parent will be the first portion of the child network

        print(f"Changing at {crossover_point} with Parent{parent_first} first")

        # Putting the appropriate parent first
        if (parent_first == 1):
              child_layers = parent1.template_network[:crossover_point] + parent2.template_network[crossover_point:]
        else:
              child_layers = parent2.template_network[:crossover_point] + parent1.template_network[crossover_point:]

        # Convert the network to a list of layers
        layers = list(child_layers)
        # Fix I/O between synapses
        layer_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]

        # Do not run for last index b/c there are no more synapses to check (index out of range)
        for layer_idx in layer_indices[:-1]:
             # Get the out features of the first layer and see if it matches the in features of the next layer
             out_features = layers[layer_idx].out_features
             in_features = layers[layer_idx+2].in_features

             # Make in features of next layer equal out features of previous layer if they don't match
             if (out_features != in_features):
                modified_layer = nn.Linear(out_features, layers[layer_idx+2].out_features)

                # Copy the existing weights and biases to the new layer
                with torch.no_grad():
                    modified_layer.weight.data = layers[layer_idx + 2].weight.data.clone()
                    modified_layer.bias.data = layers[layer_idx + 2].bias.data.clone()

                layers[layer_idx+2] = modified_layer

        child_network = nn.Sequential(*layers)
        return child_network



# UNCOMMENT BELOW FOR SIMPLE TEST
# if __name__ == "__main__":
#     params = {
#         "nin": 20,
#         "nout": 40,
#         "rand_range_start": 1,
#         "rand_range_end": 10,
#         "beta": 0.95,
#         "spike_grad": surrogate.fast_sigmoid(slope=25),
#     } 
#     parent1 = EONS(params)
#     parent2 = EONS(params)

#     parent1.make_template_network()
#     parent2.make_template_network()

#     child = crossover(parent1, parent2)

#     print("Parent1 Network:")
#     print(parent1.template_network)
#     print("\n")

#     print("Parent2 Network:")
#     print(parent2.template_network)
#     print("\n")

#     print("Child Network")
#     print(child)
#     print("\n")