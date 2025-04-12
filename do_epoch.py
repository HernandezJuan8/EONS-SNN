# snnTorch Imporst
from snntorch import surrogate

# Torch Imports
import torch.nn as nn

# Python Library Imports
import configparser
import numpy as np

# Script Imporst
from test_config import getHyperParameters
from EONS import EONS
from crossover import crossover

    

def conduct_mutation(network: EONS, eons_params:dict) -> tuple[nn.Module, bool]:
    add_node_rate = eons_params['add_node_rate'] # Probability of adding a node
    delete_node_rate = eons_params['delete_node_rate'] # Probability of deleting a node
    add_edge_rate = eons_params['add_edge_rate'] # Probablity of adding a edge
    delete_edge_rate = eons_params['delete_edge_rate'] # Probability of deleting an edge
    node_param_rate = eons_params['node_param_rate'] # Probability of randomizing node weights
    edge_param_rate = eons_params['edge_param_rate'] # Probability of randomizing edge weights

    has_mutated = False
    mutated_network = network # See if there is a .copy() method when right data type is established

    add_node = np.random.choice([0, 1], 1, p=[1-add_node_rate, add_node_rate]).item() # Determine whether to add a node or not
    delete_node = np.random.choice([0, 1], 1, p=[1-delete_node_rate, delete_node_rate]).item() # Determine whether to delete a node or not
    add_edge = np.random.choice([0, 1], 1, p=[1-add_edge_rate, add_edge_rate]).item() # Determine whether to add an edge or not
    delete_edge = np.random.choice([0, 1], 1, p=[1-delete_edge_rate, delete_edge_rate]).item() # Determine whether to delete an edge or not
    update_node_param = np.random.choice([0, 1], 1, p=[1-node_param_rate, node_param_rate]).item() # Determine whether to change node weights or not
    update_edge_param = np.random.choice([0, 1], 1, p=[1-edge_param_rate, edge_param_rate]).item() # Determine whether to change edge weights or not


    if add_node:
        network.add_node()
        has_mutated = True

    if delete_node:
        network.remove_node()
        has_mutated = True

    if add_edge:
        network.add_edge()
        has_mutated = True

    if delete_edge:
        network.remove_edge()
        has_mutated = True

    if update_node_param:
        network.update_node_param()
        has_mutated = True

    if update_edge_param:
        network.update_edge_param()
        has_mutated = True

    return mutated_network, has_mutated


# I am currently assuming the data type of these variables, will edit them if needed
def do_epoch(pop: list[EONS], fits: list[float], eons_params:dict)->list[nn.Module]:
    """
    The purpose of this function is to determine the top k networks in the current population as
    well as apply any mutations

    Parameters:
        - pop (list[nn.Module]): A list of all of the current networks
        - fits (list[float]): A list containing the results from the fitness functions for each network
        - eons_params (dict): A dictionary containing the hyperparameters for this run

    Returns:
        - next_gen (list[nn.Module]): The next generation of the population
    """

    top_k = eons_params["num_best"] # Dynamically have top k number of networks selected
    num_selected = 0 # Will track the current number of networks selected
    best_score = 0 # Will track the current best score in iteration
    best_scores = [] # Will contain all the best scores so that there are no repeats
    next_gen = [] # Will contain the next generation
    

    # Iterate until top k networks have been chosen
    while num_selected < top_k:
        # Assuming that pop and fits are parallel, this will get the index of the best score and add the appropriate network to next_gen
        fits_index = 0 
        pop_index = 0

        # Looping through the values contained in the fitness function
        for score in fits:
            if score > best_score and score not in best_scores: # Ensuring that the current best score in the interation hasn't already been selected
                best_score = score
                pop_index = fits_index
            fits_index += 1

        
        best_scores.append(best_score)
        next_gen.append(pop[pop_index])
        num_selected += 1

    # Get necessary hyperparameters
    crossover_rate = eons_params['crossover_rate'] # Probability of mixing two networks together
    mutation_rate = eons_params['mutation_rate'] # Probability of causing a mutation (add/delete)
    num_mutations = eons_params['num_mutations'] # Number of mutations
    

    
    curr_idx = 0 # Track index of current network
    for network in next_gen:
        do_crossover = np.random.choice([0, 1], 1, p=[1-crossover_rate, crossover_rate]).item() # Determine whether to do crossover or not
        num_cross = 0 # Track how many crossovers have occurred

        # Conduct crossover randomly (Decided to not count is as a mutation for now)
        if do_crossover:
            while num_cross < 1:
                # Merge current network with a randomly selected network
                merge_with_idx = np.random.choice([x for x in range(len(next_gen))], 1, p=[1/len(next_gen) for x in range(len(next_gen))]).item()

                # Only conduct crossover if the randomly selected network is not the same as the current one
                if (merge_with_idx != curr_idx):
                    num_cross += 1
                    crossover(next_gen[curr_idx], next_gen[merge_with_idx])
                
        curr_idx += 1



    mutations_made = 0

    # Keep iterating until the number of mutations has been met
    while mutations_made < num_mutations:
        for network in next_gen:
            do_mutation = np.random.choice([0, 1], 1, p=[1-mutation_rate, mutation_rate]).item() # Determine whether to do mutation or not
            if do_mutation:
                mutated_network, has_mutated = conduct_mutation(network, eons_params)

                if has_mutated:
                    next_gen.append(mutated_network)
                    mutations_made += 1

                

    return next_gen


"""
What is currently done:
    - Can select the top k number of networks
    - Can get the network associated with the best fitness score (assuming lists are parallel)

What needs to be done:
    - Mutation functions
"""


if __name__ == "__main__":
    # Use this to test selection of best network based on fitness function
    """
    pop = [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12]
    fits = [0.35, 0.21, 0.62, 0.87, 0.12, 0.32, 0.55, 0.84, 0.52, 0.88, 0.11, 0.77]
    config = configparser.ConfigParser()
    config.read('config.ini')
    hyperparameters = getHyperParameters(config)

    pop = do_epoch(pop, fits, hyperparameters)

    print(pop)
 """
    params1 = {
        "nin": 40,
        "nout": 20,
        "rand_range_start": 1,
        "rand_range_end": 10,
        "beta": 0.95,
        "spike_grad": surrogate.fast_sigmoid(slope=25),
    }
    params2 = {
        "nin": 9,
        "nout": 5,
        "rand_range_start": 1,
        "rand_range_end": 10,
        "beta": 0.95,
        "spike_grad": surrogate.fast_sigmoid(slope=25),
    }
    params3 = {
        "nin": 27,
        "nout": 15,
        "rand_range_start": 1,
        "rand_range_end": 10,
        "beta": 0.95,
        "spike_grad": surrogate.fast_sigmoid(slope=25),
    }
    params4 = {
        "nin": 80,
        "nout": 10,
        "rand_range_start": 1,
        "rand_range_end": 10,
        "beta": 0.95,
        "spike_grad": surrogate.fast_sigmoid(slope=25),
    }
    params5 = {
        "nin": 60,
        "nout": 20,
        "rand_range_start": 1,
        "rand_range_end": 10,
        "beta": 0.95,
        "spike_grad": surrogate.fast_sigmoid(slope=25),
    }

    # Use this to test mutations
    eon1 = EONS(params1)
    eon1.make_template_network()

    eon2 = EONS(params2)
    eon2.make_template_network()

    eon3 = EONS(params3)
    eon3.make_template_network()

    eon4 = EONS(params4)
    eon4.make_template_network()

    eon5 = EONS(params5)
    eon5.make_template_network()

    pop = [eon1, eon2, eon3, eon4, eon5]
    fits = [0.35, 0.21, 0.62, 0.87, 0.12]
    config = configparser.ConfigParser()
    config.read('config.ini')
    hyperparameters = getHyperParameters(config)

    pop = do_epoch(pop, fits, hyperparameters)

    # for network in pop:
    #     print(network.template_network)

