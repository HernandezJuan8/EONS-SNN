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
from cardiovascular_SNN_model import EvoSNN
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("cardio_train.csv", sep=';')

df.drop("id", axis=1, inplace=True) # dropping ID column cause we don't need it as a feature for pred

X = df.drop('cardio', axis=1).values  # drop cardio cause that's our target
y = df['cardio'].values               # this is our target we have the outputs: 1 (true) and 0 (false)

scaler = StandardScaler() # yuh standardizing a scalar
X = scaler.fit_transform(X) # fit transform

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# below we have X (the features) and y (the target) split into training and test sets
# where we have 0.8% of our dataset going to training and 0.2% going to testing (is it good)
# and ignore random_state it's just the minecraft seed for the # generator
# for reproducing the same fixed randomness for the dataset each code run (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)

INPUT_SIZE = 11 # we got 11 features
OUTPUT_SIZE = 2 # we got 2 outputs (1 and 0) for our target


### INITIALIZE TOOLBOX BELOW
toolbox = base.Toolbox()

def flatten_model(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])

dummy_model = EvoSNN().to(device)
n_weights = dummy_model.getTotalNumberOfWeights()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_float", random.uniform, -1.0, 1.0)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    
    weights = np.array(individual) # yuh the weights
    
    model = EvoSNN(weights=weights).to(device) # yuh our model w/ weights
    
    #yoo our batch_accuracy in the cardiovascular_SNN_model.py
    accuracy = model.batch_accuracy(train_loader, num_steps=5, max_batches=5)
    
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
