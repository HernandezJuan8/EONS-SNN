import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools, algorithms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


# --- Setup ---
# Define the neural network architecture
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Prepare MNIST Data ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)

batch_size = 2048

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader_temp = iter(train_loader)
# --- DEAP Setup ---
# Define fitness and individual classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    """Create a random individual (ANN with random weights)"""
    input_size = 28*28  # Flattened image size
    hidden_size = 128   # Hidden layer size
    output_size = 10    # Output size (10 for MNIST)
    
    # Create the network
    net = ANN(input_size, hidden_size, output_size)
    
    # Flatten the network parameters (weights)
    flattened_weights = np.concatenate([param.detach().numpy().flatten() for param in net.parameters()])
    
    # Return the flattened weights as the individual
    individual = flattened_weights.tolist()
    return individual

def get_next_batch():
    global train_loader_temp
    try:
        data, targets = next(train_loader_temp)
    except StopIteration:
        # Reset the iterator when exhausted
        train_loader_temp = iter(train_loader)
        data, targets = next(train_loader_temp)
    return data.to(device), targets.to(device)

def get_random_batch(dataset, batch_size):
    """Returns a random batch of data from the dataset."""
    indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size)
    for data, targets in dataloader:
        return data, targets

def evaluate(individual):
    """Evaluate the fitness of an individual (neural network)"""
    # Define the network's structure
    input_size = 28*28  # Flattened image size
    hidden_size = 128   # Hidden layer size
    output_size = 10    # Output size (10 for MNIST)
    
    # Rebuild the network from the individual's weights
    net = ANN(input_size, hidden_size, output_size).to(device)
    
    # Flatten the individual list into the network's parameters
    flattened_weights = np.array(individual)
    start = 0
    for param in net.parameters():
        param_length = param.numel()
        param.data = torch.from_numpy(flattened_weights[start:start + param_length].reshape(param.shape)).float().to(device)
        start += param_length

    # Test the network on the MNIST validation set
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        #for data, targets in iter(train_loader):
        data,targets = get_random_batch(train_dataset, batch_size)
        data = data.view(-1, 28*28).to(device)  # Flatten the input images
        targets = targets.to(device)
        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = correct / total  # Fitness is the accuracy
    return accuracy,

# --- Main Evolutionary Loop ---
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Run Evolution ---
def main():
    population = toolbox.population(n=50)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    verbose = True
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    cxpb = 0.6
    mutpb = 0.2
    ngen = 100
    
    # Run the genetic algorithm
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hall_of_fame is not None:
        hall_of_fame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hall_of_fame is not None:
            hall_of_fame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    # Output best individual
    print(f"Best individual: {hall_of_fame[0]}")
    return hall_of_fame[0]

if __name__ == "__main__":
    best_individual = main()
