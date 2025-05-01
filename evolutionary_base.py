# --- SNN MODEL ---
import random
import numpy as np
from snntorch import utils
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import functional as SF
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PARAMETERS ---
BATCH_SIZE = 32
SUBSET_SIZE = 128
cached_loader = None  # global for caching if needed
time_steps = 25  # Number of time steps per sample
firing_rates_log = []

class EvoSNN(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            snn.Leaky(beta=0.9, init_hidden=True),
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 10),
            snn.Leaky(beta=0.9, init_hidden=True, output=True)
        ).to(device)
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Increase filters, reduce kernel size
        #     nn.MaxPool2d(2),
        #     snn.Leaky(beta=0.9, init_hidden=True),

        #     nn.Conv2d(16, 64, kernel_size=3, padding=1),  # Keep kernel size 3x3
        #     nn.MaxPool2d(2),
        #     snn.Leaky(beta=0.9, init_hidden=True),

        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 7, 128),  # More neurons in dense layer
        #     snn.Leaky(beta=0.9, init_hidden=True),

        #     nn.Linear(128, 10),
        #     snn.Leaky(beta=0.9, init_hidden=True, output=True)
        # ).to(device)
        self.net.eval()
        if weights is not None:
            self.set_weights(weights)

    def getTotalNumberOfWeights(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward_pass(self, num_steps, data):
        spk_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net
        for step in range(num_steps):
            spk_out, _ = self.net(data)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

    def batch_accuracy(self, train_loader, num_steps):
        with torch.no_grad():
            correct = 0
            total = 0
            for data, targets in train_loader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec = self.forward_pass(num_steps, data)

                # Get predicted labels from spikes
                _, predicted = spk_rec.sum(0).max(1)  # Sum over time, then get argmax
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return correct / total


    def get_weights(self):
        weights = []
        for layer in self.net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight.data.flatten().cpu().numpy())
                if layer.bias is not None:
                    weights.append(layer.bias.data.flatten().cpu().numpy())
        return np.concatenate(weights)

    def set_weights(self, flat_weights):
        idx = 0
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    w_shape = layer.weight.shape
                    w_numel = layer.weight.numel()

                    new_weights = torch.tensor(flat_weights[idx:idx + w_numel], dtype=layer.weight.dtype).reshape(w_shape)
                    layer.weight.copy_(torch.tensor(new_weights, dtype=layer.weight.dtype))
                    idx += w_numel

                    if layer.bias is not None:
                        b_shape = layer.bias.shape
                        b_numel = layer.bias.numel()

                        new_bias = torch.tensor(flat_weights[idx:idx + b_numel], dtype=layer.bias.dtype).reshape(b_shape)
                        layer.bias.copy_(torch.tensor(new_bias, dtype=layer.bias.dtype))
                        idx += b_numel

# --- ENCODING FUNCTION ---
# def poisson_encode(img, time_steps):
#     return (torch.rand((time_steps,) + img.shape).to(device) < img).float()

# def repeat_encode(img, time_steps):
#     return img.repeat(time_steps, 1)

# def repeat_encode_batch(img, time_steps):
#     # img: [batch, input_size] => [1, batch, input_size]
#     # Then repeat along time: [time, batch, input_size]
#     return img.unsqueeze(0).repeat(time_steps, 1, 1)

# --- LOAD MNIST SUBSET ---
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
mnist_train = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transform, download=True)
#train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=n_samples, shuffle=True, drop_last=True)
model = EvoSNN(weights=None).to(device)
n_weights = model.getTotalNumberOfWeights()


# Cache for same batch reuse
cached_data = None
cached_targets = None
# train_loader_temp = iter(train_loader)

# def get_next_batch():
#     global train_loader_temp
#     try:
#         data, targets = next(train_loader_temp)
#     except StopIteration:
#         # Reset the iterator when exhausted
#         train_loader_temp = iter(train_loader)
#         data, targets = next(train_loader_temp)
#     return data.to(device), targets.to(device)

def evaluate(individual, use_same_data=True):
    global cached_loader

    model = EvoSNN(weights=individual).to(device)
    model.eval()

    total = 0
    correct = 0
    total_spikes = 0

    with torch.no_grad():
        if use_same_data and cached_loader is not None:
            loader = cached_loader
        else:
            # Sample 2048 random indices from full dataset
            indices = random.sample(range(len(mnist_train)), SUBSET_SIZE)
            subset = Subset(mnist_train, indices)
            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
            if use_same_data:
                cached_loader = loader

        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec = model.forward_pass(time_steps, data)

            _, predicted = spk_rec.sum(0).max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            total_spikes += spk_rec.sum().item()

    accuracy = correct / total
    return accuracy,