# --- SNN MODEL FOR TABULAR DATA ---
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class EvoSNN(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),  # input layer (we got 11 features in our dataset)
            snn.Leaky(beta=0.9, init_hidden=True),
            nn.Linear(64, 32),  # this the hidden layer
            snn.Leaky(beta=0.9, init_hidden=True),
            nn.Linear(32, 1),   # this the output layer (we got 2 outputs from target column 'cardio', 1 and 0)
            snn.Leaky(beta=0.9, init_hidden=True)  # Output is a spike
        ).to(device)
        
        self.net.eval()
        if weights is not None:
            self.set_weights(weights)

    def getTotalNumberOfWeights(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward_pass(self, num_steps, data):
        """
        Forward pass through the SNN, processing each time step sequentially.
        data: Shape (batch_size, num_steps, num_features)
        """
        batch_size, num_steps, _ = data.shape
        # Initialize the output tensor for the spike recordings
        spike_rec = torch.zeros(batch_size, num_steps, 1)
        utils.reset(self.net)
        # Process each time step in the sequence
        for t in range(num_steps):
            input_data = data[:, t, :]  # Shape: (batch_size, num_features)
            spike_rec[:, t, :] = self.net(input_data)  # Forward pass for the current time step

        return spike_rec
    
    def batch_accuracy(self, train_loader, num_steps, max_batches=5):
        with torch.no_grad():
            total = 0
            acc = 0
            train_loader = iter(train_loader)
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                data = data.to(device)
                targets = targets.to(device)
                spk_rec = self.forward_pass(num_steps, data)  # [batch, time, 1]

                spk_rec_sum = spk_rec.sum(dim=1).to(device)  # -> [batch, 1]

                # For binary classification, turn summed spikes into class predictions: 0 or 1
                preds = (spk_rec_sum > 0).squeeze()  # -> [batch]

                acc += (preds == targets).float().sum().item()
                total += targets.size(0)

            return acc / total if total > 0 else 0
        
    def get_weights(self):
        weights = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.data.flatten().cpu().numpy())
                if layer.bias is not None:
                    weights.append(layer.bias.data.flatten().cpu().numpy())
        return np.concatenate(weights)

    def set_weights(self, flat_weights):
        idx = 0
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    w_shape = layer.weight.shape
                    w_numel = layer.weight.numel()
                    layer.weight.copy_(torch.tensor(flat_weights[idx:idx + w_numel]).reshape(w_shape))
                    idx += w_numel

                    if layer.bias is not None:
                        b_shape = layer.bias.shape
                        b_numel = layer.bias.numel()
                        layer.bias.copy_(torch.tensor(flat_weights[idx:idx + b_numel]).reshape(b_shape))
                        idx += b_numel
    def forward(self, x):
      # One step forward through the SNN
      return self.net(x)