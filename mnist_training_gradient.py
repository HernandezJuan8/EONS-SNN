import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
data_path='mnist'
batch_size = 128


use_surrogate = True
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.9

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
# Load the network onto CUDA if available
net = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Increase filters, reduce kernel size
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Conv2d(16, 64, kernel_size=3, padding=1),  # Keep kernel size 3x3
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),  # More neurons in dense layer
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    nn.Linear(128, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)

# Temporal Dynamics
num_steps = 25
loss_fn = SF.ce_rate_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

# Outer training loop
for epoch in range(num_epochs):

    # Training loop
    for data, targets in iter(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)

        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 50 == 0:
            with torch.no_grad():
                net.eval()

                # Test set forward pass
                test_acc = batch_accuracy(test_loader, net, num_steps)
                print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())

        counter += 1
    


# Plot Loss
# fig = plt.figure(facecolor="w")
# plt.plot(counter, test_acc_hist)
# plt.title("Test Set Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")

# fig = plt.figure(facecolor="w")
# plt.plot(loss_hist)
# plt.title("Test Set Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.show()


weights = []
for layer in net:
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        weights.append(layer.weight.data.flatten().cpu().numpy())
        if layer.bias is not None:
            weights.append(layer.bias.data.flatten().cpu().numpy())
weights = np.concatenate(weights)
np.save("best_weights_gradient.npy", weights)