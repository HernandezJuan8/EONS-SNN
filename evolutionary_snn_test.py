import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
from snntorch import functional as SF
from evolutionary_base import input_size, output_size, hidden_size, evaluate
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# --- EVOLUTIONARY SETUP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

n_weights = input_size * hidden_size + hidden_size * output_size

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: np.random.uniform(-0.5, 0.5))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- RUN EVOLUTION ---
def main():
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100,
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
    
