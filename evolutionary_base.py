# --- SNN MODEL ---
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PARAMETERS ---
n_samples = 100  # Use small set for speed
time_steps = 25  # Number of time steps per sample
input_size = 784
hidden_size = 100
output_size = 10

class EvoSNN(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.lif2 = snn.Leaky(beta=0.9)

        if weights is not None:
            self.set_weights(weights)

    def forward(self, x_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_sum = 0

        for t in range(x_seq.size(0)):
            cur_input = x_seq[t]
            cur_input = self.fc1(cur_input)
            spk1, mem1 = self.lif1(cur_input, mem1)

            cur_hidden = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur_hidden, mem2)
            spk2_sum += spk2

        return spk2_sum  # spike count output

    def get_weights(self):
        return torch.cat([self.fc1.weight.data.flatten(), self.fc2.weight.data.flatten()]).cpu().numpy()

    def set_weights(self, flat_weights):
        with torch.no_grad():
            w1_size = self.fc1.weight.numel()
            w2_size = self.fc2.weight.numel()

            fc1_weights = torch.tensor(flat_weights[:w1_size]).reshape(self.fc1.weight.shape).to(device)
            fc2_weights = torch.tensor(flat_weights[w1_size:w1_size + w2_size]).reshape(self.fc2.weight.shape).to(device)

            self.fc1.weight.copy_(fc1_weights)
            self.fc2.weight.copy_(fc2_weights)

# --- ENCODING FUNCTION ---
def poisson_encode(img, time_steps):
    return (torch.rand((time_steps,) + img.shape).to(device) < img).float()

def repeat_encode(img, time_steps):
    return img.unsqueeze(0).repeat(time_steps, 1, 1)

# --- LOAD MNIST SUBSET ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

x_data = []
y_data = []

for i, (x, y) in enumerate(train_loader):
    x_data.append(x.squeeze())
    y_data.append(y.item())
    if i + 1 >= n_samples:
        break

x_data = torch.stack(x_data).to(device)
y_data = torch.tensor(y_data).to(device)

model = EvoSNN(weights=None).to(device)
def evaluate(individual):
    model.set_weights(individual)
    correct = 0
    for i in range(n_samples):
        img = x_data[i]
        label = y_data[i]
        spikes = repeat_encode(img, time_steps)
        output_spikes = model(spikes)
        prediction = output_spikes.argmax()
        if prediction == label:
            correct += 1
    return (correct / n_samples,)
