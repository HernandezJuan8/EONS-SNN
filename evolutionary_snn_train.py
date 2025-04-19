import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
from snntorch import functional as SF
import torch.nn.init as init
from evolutionary_base import evaluate, n_weights,model
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# --- EVOLUTIONARY SETUP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # He initialization for conv layers
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  # He initialization for fully connected layers

# Create a function to initialize the weights of an individual (network)
def initialize_individual():
    # Initialize the network with the individual weights
    model.net.apply(init_weights)  # Apply your custom weight initialization
    weights = model.get_weights()
    return weights

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, initialize_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.4, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)

# --- RUN EVOLUTION ---
def main():
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=150,
                                   stats=stats, halloffame=hof, verbose=True)

    print("Best fitness:", hof[0].fitness.values[0])
    gen = log.select("gen")
    avg_acc = log.select("avg")

    # Plotting
    plt.figure()
    plt.plot(gen, avg_acc, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Average Accuracy")
    plt.title("Accuracy over Generations")
    plt.grid(True)
    return hof[0]

if __name__ == "__main__":
    best = main()
    np.save("best_weights.npy", np.array(best))
    plt.show()
    
