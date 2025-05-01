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

# man what is this
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

#below is my junk
from torchvision import datasets, transforms
from evolutionary_base import EvoSNN_MNIST
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path='mnist'
batch_size = 128
n_samples = 128  # Use small set for speed
time_steps = 25  # Number of time steps per sample

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=n_samples, shuffle=True, drop_last=True)



### INITIALIZE TOOLBOX BELOW
toolbox = base.Toolbox()

def flatten_model(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

dummy_model = EvoSNN_MNIST().to(device)
n_weights = dummy_model.getTotalNumberOfWeights()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_float", random.uniform, -1.0, 1.0)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    
    weights = np.array(individual) # yuh the weights
    
    model = EvoSNN_MNIST(weights=weights).to(device) # yuh our model w/ weights
    
    #yoo our batch_accuracy in the cardiovascular_SNN_model.py
    accuracy = model.batch_accuracy(train_loader, num_steps=5,max_batches=5)
    
    return (accuracy.item(),)  # returned as a tuple

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
#Try increasing the tournament size from 3 (e.g., tournsize=5)?
toolbox.register("select", tools.selTournament, tournsize=5)

def main():
    pop = toolbox.population(n=50)  # set our population however u want
    hof = tools.HallOfFame(1)  # heh
    # for the statistics tool info here's the link:
    # https://deap.readthedocs.io/en/master/tutorials/basic/part3.html
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # 
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.4, ngen=100,
        stats=stats, halloffame=hof, verbose=True)
    
    print("Best fitness:", hof[0].fitness.values[0])
    
    gen = log.select("gen")
    avg_acc = log.select("avg")
    
    plt.figure()
    plt.plot(gen, avg_acc, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Average Accuracy")
    plt.title("Accuracy over Generations")
    plt.grid(True)

    #saving the plot
    plt.savefig('hunter_accuracy_plot.png')
    
    return hof[0] 

if __name__ == "__main__":
    best = main()
    np.save("best_weights.npy", np.array(best)) 
    plt.show()
