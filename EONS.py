# snnTorch Imports
import snntorch as snn
from snntorch import surrogate

# Torch Imports
import torch
import torch.nn as nn

# Python Libary Imports
import random


class EONS:
    def __init__(self,params):
        self.params = params
        self.template_network = None
        self.nin = params['nin']
        self.nout = params['nout']
        self.rand_range_start = params['rand_range_start']
        self.rand_range_end = params['rand_range_end']
        self.beta = params['beta']
        self.spike_grad = params['spike_grad']
        
    def make_template_network(self):
        hidden = random.randint(self.rand_range_start,self.rand_range_end)
        hidden2 = random.randint(self.rand_range_start,self.rand_range_end)
        hidden3 = random.randint(self.rand_range_start,self.rand_range_end)
        hidden4 = random.randint(self.rand_range_start,self.rand_range_end)
        hidden5 = random.randint(self.rand_range_start,self.rand_range_end)
        hidden6 = random.randint(self.rand_range_start,self.rand_range_end)
        net = nn.Sequential(nn.Linear(self.nin,hidden),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden,hidden2),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden2,hidden3),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden3,hidden4),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden4,hidden5),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden5,hidden6),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad),
                            nn.Linear(hidden6,self.nout),
                            snn.Leaky(beta=self.beta,spike_grad=self.spike_grad))
        self.template_network = net
        return None
    
    def add_node(self):
        """
        This method adds a neuron to a randomly selected hidden layer and updates the 
        in_features to the next layer accordingly.
        """
        # Don't do anything if the network didn't initialize properly
        if self.template_network is None:
            print("Template network not initialized")
            return
        
        # Find indicies for all linear layers
        layer_indices = [i for i, layer in enumerate(self.template_network) if isinstance(layer, nn.Linear)]

        # Exclude input and output layers
        if (len(layer_indices) <= 2):
               print("Not enough layers to modify (only input/output layers present)")
               return

        hidden_layer_indices = layer_indices[1:-1] # Only hidden layers
        selected_index = random.choice(hidden_layer_indices) # Select random layer from hidden layer list
        next_index = selected_index + 2 # Get the next layer so that we can update the input

        # Get the layers
        linear = self.template_network[selected_index]
        next_linear = self.template_network[next_index]


        in_feat= linear.in_features # Keeping in features the same
        new_out_feat = linear.out_features + 1 # Adding neuron to the out

        # Creating updated hidden layer
        new_linear = nn.Linear(in_feat, new_out_feat)

        # Repeat for the next hidden layer. This ensures that the output of prev layer matches input of next layer
        next_out = next_linear.out_features
        new_next_linear = nn.Linear(new_out_feat, next_out)    

        linear = self.fix_layer(linear)
        next_linear = self.fix_layer(next_linear)

        # print("==========================ADD NODE=========================================")
        # print(linear)
        # print(linear.weight.shape)
        # print(linear.weight.shape[0], linear.weight.shape[1])
        # print(linear.out_features, linear.in_features)
        # print(new_linear.weight.shape)
        # print(new_out_feat, in_feat)
        # print(new_linear.weight[:linear.weight.shape[0], :linear.weight.shape[1]], flush=True)
        # print(linear.weight, flush=True)
        # print(linear.out_features)
        # print(new_linear.bias[:linear.weight.shape[1]])
        # print(linear.bias)
        # print("\n\n")
        # print(new_linear)
        # print(new_linear.weight.shape)
        # print(new_linear.weight.shape[0], new_linear.weight.shape[1])
        # print(new_linear.out_features, new_linear.in_features)
        # print(new_linear.in_features)
        # print(new_next_linear.weight[:next_linear.weight.shape[0], :next_linear.weight.shape[1]], flush=True)
        # print(next_linear.weight, flush=True)
        # print(new_linear.out_features)
        # print(new_next_linear.bias[:next_linear.weight.shape[1]])
        # print(new_linear.bias)
        # print("==========================ADD NODE=========================================")
        
        # Ensuring it keeps the same weights and biases
        # Weight shape = [out_features, in_features]
        with torch.no_grad():
            new_linear.weight[:linear.weight.shape[0], :linear.weight.shape[1]].copy_(linear.weight)
            new_linear.bias[:linear.weight.shape[0]].copy_(linear.bias)
        
        with torch.no_grad():
            new_next_linear.weight[:next_linear.weight.shape[0], :next_linear.weight.shape[1]].copy_(next_linear.weight)
            new_next_linear.bias[:next_linear.weight.shape[0]].copy_(next_linear.bias)

        # Replacing both layers in the sequential
        layers = list(self.template_network)
        layers[selected_index] = new_linear
        layers[next_index] = new_next_linear 
        
        self.template_network = nn.Sequential(*layers)
        print(f"Added a neuron to output of layer {selected_index} and updated input to layer {next_index}")

    def remove_node(self):
        """
        This method removes a neuron from a randomly selected hidden layer.
        The in_features of the next layer is updated accordingly.
        If there isn't enough neurons in the layer to remove (if # of neurons is <= 1),
        the function simply returns and no mutation occurs. 
        """
        # Don't do anything if the network didn't initialize properly
        if self.template_network is None:
            print("Template network not initialized")
            return
        
        # Find indicies for all linear layers
        layer_indices = [i for i, layer in enumerate(self.template_network) if isinstance(layer, nn.Linear)]

        # Exclude input and output layers
        if (len(layer_indices) <= 2):
               print("Not enough layers to modify (only input/output layers present)")
               return

        hidden_layer_indices = layer_indices[1:-1] # Only hidden layers
        selected_index = random.choice(hidden_layer_indices) # Select random layer from hidden layer list
        next_index = selected_index + 2# Get the next layer so that we can update the input

        # Get the layers
        linear = self.template_network[selected_index]
        next_linear = self.template_network[next_index]

        # Recursively call method until there are more than one neuron in the out features
        if (linear.out_features <= 1):
            print(f"Not enough neurons in layer {selected_index}. Cannot remove neuron.")
            self.remove_node()
            return

        in_feat= linear.in_features # Keeping in features the same
        new_out_feat = linear.out_features - 1 # Removing neuron to the out

        # Creating updated hidden layer
        new_linear = nn.Linear(in_feat, new_out_feat)

        # Repeat for the next hidden layer. This ensures that the output of prev layer matches input of next layer
        next_out = next_linear.out_features
        new_next_linear = nn.Linear(new_out_feat, next_out) 

        linear = self.fix_layer(linear)
        next_linear = self.fix_layer(next_linear)


        # print("==========================REMOVE NODE=========================================")
        # print(linear)
        # print(linear.weight.shape)
        # print(linear.weight.shape[0], linear.weight.shape[1])
        # print(linear.out_features, linear.in_features)
        # print(new_linear.weight.shape)
        # print(new_out_feat, in_feat)
        # print(new_linear.weight, flush=True)
        # print(linear.weight[:new_linear.weight.shape[0], :new_linear.weight.shape[1]], flush=True)
        # print(linear.out_features)
        # print(new_linear.bias)
        # print(linear.bias)
        # print("\n\n")
        # print(new_linear)
        # print(new_linear.weight.shape)
        # print(new_linear.weight.shape[0], new_linear.weight.shape[1])
        # print(new_linear.out_features, new_linear.in_features)
        # print(new_linear.in_features)
        # print(new_next_linear.weight, flush=True)
        # print(next_linear.weight, flush=True)
        # print(new_linear.out_features)
        # print(next_linear.bias[:new_next_linear.out_features])
        # print(new_linear.bias)
        # print("==========================REMOVE NODE=========================================")

        # Ensuring it keeps the same weights and biases
        # Weight shape = [out_features, in_features]
        with torch.no_grad():
            new_linear.weight.copy_(linear.weight[:new_linear.weight.shape[0], :new_linear.weight.shape[1]])
            new_linear.bias.copy_(linear.bias[:new_linear.out_features])
        
        with torch.no_grad():
            new_next_linear.weight.copy_(next_linear.weight[:new_next_linear.weight.shape[0], :(new_next_linear.weight.shape[1])])
            new_next_linear.bias.copy_(next_linear.bias[:new_next_linear.out_features])

        # Replacing both layers in the sequential
        layers = list(self.template_network)
        layers[selected_index] = new_linear
        layers[next_index] = new_next_linear
        
        self.template_network = nn.Sequential(*layers) # Update network
        print(f"Removed a neuron to output of layer {selected_index} and updated input to layer {next_index}")

    def update_node_param(self):
        """
        This method udpates the weights and bias of one randomly selected neuron in a randomly selected
        hidden layer.

        NOTE: There are a couple of lines that are commented out. These lines were created for visualization purposes
              so that I could see if the neuron parameters were really updating. If you want to see those results,
              simply uncomment those lines and uncomment the necessary lines in the cell below this one. 
        """
        # Don't do anything if the network didn't initialize properly
        if self.template_network is None:
            print("Template network not initialized")
            return
        
        # Find indicies for all linear layers
        layer_indices = [i for i, layer in enumerate(self.template_network) if isinstance(layer, nn.Linear)]

        # Select random layer
        selected_index = random.choice(layer_indices) 
        linear = self.template_network[selected_index]

        # Select a random neuron and generate random weights/bias
        new_weight = torch.rand(1)
        new_bias = torch.rand(1)
        neuron_index = random.choice(range(linear.out_features))
        
        layers = list(self.template_network) # Convert network to list

        # Update neuron weights and bias
        with torch.no_grad():
            layers[selected_index].weight[neuron_index] = new_weight
            layers[selected_index].bias[neuron_index] = new_bias

        self.template_network = nn.Sequential(*layers) # Updated network

        print(f"Updated neuron {neuron_index} in layer {selected_index}")

    def add_edge(self):
        """
        Dynamically add a synapse (new connection) between two randomly selected layers in the network.
        This method adds a new nn.Linear layer between two existing layers.
        """
        if self.template_network is None:
            print("Template network not initialized")
            return
        
        # Convert the network to a list of layers
        layers = list(self.template_network)

        # Find indicies for all linear layers
        layer_indices = [i for i, layer in enumerate(self.template_network) if isinstance(layer, nn.Linear)]

        # Exclude input and output layers
        if (len(layer_indices) <= 2):
               print("Not enough layers to modify (only input/output layers present)")
               return

        hidden_layer_indices = layer_indices[1:-1] # Only hidden layers
        
        # Choose two random layers that are next to each other 
        selected_index_1 = random.choice(hidden_layer_indices)  # First layer
        selected_index_2 = selected_index_1 + 2 # Second layer
        
        new_leak = snn.Leaky(beta=self.beta,spike_grad=self.spike_grad) # Creating a leaky along with synapses
        
        # Insert the new synapse between the two selected layers (selected_index_1 --> new_synapse --> selected_index_2)
        # Get in_features of second layer and out_features of first layer
        out_features_1 = layers[selected_index_1].out_features 
        in_features_2 = layers[selected_index_2].in_features

        # Create new connection
        new_synapse = nn.Linear(out_features_1, in_features_2)
        
        # Insert new layer between the selected layers
        layers.insert(selected_index_2, new_synapse)
        layers.insert(selected_index_2 + 1, new_leak)

        print(f"Added new synapse at {selected_index_2}.")
        
        # Update the network with the new layers
        self.template_network = nn.Sequential(*layers)

    def remove_edge(self):
        """
        Dynamically remove a synapse (new connection) between two randomly selected layers in the network.
        This method removes a new nn.Linear layer between two existing layers.
        """
        if self.template_network is None:
            print("Template network not initialized")
            return
        
        # Convert the network to a list of layers
        layers = list(self.template_network)

        # Find indicies for all linear layers
        layer_indices = [i for i, layer in enumerate(self.template_network) if isinstance(layer, nn.Linear)]

        # Exclude input and output layers
        if (len(layer_indices) <= 2):
               print("Not enough layers to modify (only input/output layers present)")
               return

        hidden_layer_indices = layer_indices[1:-1] # Only hidden layers
        
        # Choose a random hidden layer to delete
        selected_index_1 = random.choice(hidden_layer_indices)  # Chosen layer
        selected_index_2 = selected_index_1 + 1 # Associated leaky layer
        
        # Remove chosen layer
        remove_layers = [selected_index_1, selected_index_2]
        layers = [layer for index, layer in enumerate(layers) if index not in remove_layers]
        
        # Fix I/O between synapses
        self.fixIO(layers)

        print(f"Removed a synapse at {selected_index_1}.")
        
        # Update the network with the new layers
        self.template_network = nn.Sequential(*layers)

    def update_edge_param(self):
        """
        This method randomly selects a synapse (i.e., a connection between two neurons),
        and updates the weight of that synapse.

        NOTE: The input neuron's parameters aren't directly affected by the synapse update because the input neuron doesn't learn its own weight 
              — it only passes its value to the next layer. Thus, only the weight (synapse) and output neuron's bias are updated.
        
        """
        if self.template_network is None:
            print("Template network not initialized")
            return

        # Convert the network to a list of layers
        layers = list(self.template_network)

        # Find all linear layers in the network
        layer_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]

        # Randomly select a layer (excluding the input and output layers)
        selected_index = random.choice(layer_indices)
        linear_layer = layers[selected_index]

        # Ensure there are enough input and output neurons before proceeding
        if linear_layer.in_features <= 1 or linear_layer.out_features <= 1:
            print(f"Layer at index {selected_index} does not have enough neurons. Trying again.")
            self.update_edge_param()
            return
        
        # Fixing linear layer in case of any weight to feature mismatches
        linear_layer = self.fix_layer(linear_layer)

        # Select random neuron indices for both the input and output neurons
        input_neuron_index = random.choice(range(linear_layer.in_features))  # Random input neuron
        output_neuron_index = random.choice(range(linear_layer.out_features))  # Random output neuron

        # These two if statements make sure the index stay within range
        # That is because random.choice picks a number between 2 digits, inclusive so [a, b]
        if (input_neuron_index == linear_layer.in_features):
            print(f'Input neuron index {input_neuron_index} is greater than {linear_layer.in_features}. Changing...')
            input_neuron_index -= 1

        if (output_neuron_index == linear_layer.out_features):
            print(f'Output neuron index {output_neuron_index} is greater than {linear_layer.out_features}. Changing...')
            output_neuron_index -= 1

        # print(linear_layer.weight.shape)
        # print(linear_layer.out_features, linear_layer.in_features)
        # print(output_neuron_index, input_neuron_index)

        # Update the synapse weight (i.e., the connection between input neuron and output neuron)
        new_weight = torch.rand(1)  # Random weight for the synapse
        with torch.no_grad():  
            # Accessing the specific weight between the two neurons and updating it
            linear_layer.weight[output_neuron_index, input_neuron_index] = new_weight

        # Update the bias for the both neurons
        new_bias = torch.rand(1)
        with torch.no_grad():  
            linear_layer.bias[output_neuron_index] = new_bias


        # Update the network with the new layers (though layers should already be updated)
        self.template_network = nn.Sequential(*layers)

        print(f"Updated the synapse connecting input neuron {input_neuron_index} to output neuron {output_neuron_index} "
            f"in layer {selected_index}.")
        
    
    def fixIO(self, layers: list[nn.Linear]):
        """
        This function loops through the layers, after they have been mutated,
        and checks to see if there are any mismatches between the out_features
        of layer 1 and the in_features of layer 2. If a mismatch is detected,
        then the in_features of layer 2 is updated so that it matches the
        out_features of layer 1.
        """
        # Fix I/O between synapses
        layer_indices = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]

        # Do not run for last index b/c there are no more synapses to check (index out of range)
        for layer_idx in layer_indices[:-1]:
            # Get the out features of the first layer and see if it matches the in features of the next layer
            out_features = layers[layer_idx].out_features # Layer 1
            in_features = layers[layer_idx+2].in_features # Layer 2

            # Make in features of next layer equal out features of previous layer if they don't match
            if (out_features != in_features):
                modified_layer = nn.Linear(out_features, layers[layer_idx+2].out_features) # (Layer 1 in_features, Layer 2 out_features)

                # Copy the existing weights and biases to the new layer
                with torch.no_grad():
                    modified_layer.weight.data = layers[layer_idx + 2].weight.data.clone()
                    modified_layer.bias.data = layers[layer_idx + 2].bias.data.clone()


                layers[layer_idx+2] = modified_layer

    def fix_layer(self, layer):
        """
        I was noticing that on some runs, the weight.shape matrix would not match
        the features that were in the actual linear layer. This function's purpose
        is to fix that mismatch.
        """
        correct_shape = (layer.out_features, layer.in_features)
        if layer.weight.shape != correct_shape:
            print(f"Fixing corrupted layer: {layer}", flush=True)
            new_layer = nn.Linear(layer.in_features, layer.out_features)
            with torch.no_grad():
                # Copy as much as fits
                rows = min(layer.weight.shape[0], new_layer.weight.shape[0])
                cols = min(layer.weight.shape[1], new_layer.weight.shape[1])
                new_layer.weight[:rows, :cols].copy_(layer.weight[:rows, :cols])
                new_layer.bias[:rows].copy_(layer.bias[:rows])
            return new_layer
        return layer
        


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

#     test_eons = EONS(params)
#     test_eons.make_template_network()
#     print(test_eons.template_network)