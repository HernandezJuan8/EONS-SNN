# Assuming `best_individual` is a PyTorch model of the same architecture
import numpy as np
import torch
from evolutionary_base import evaluate, device

loaded_weights = np.load("best_weights.npy")  # Load weights from the file

# If you're using a tensor, you can convert the loaded weights into a tensor
loaded_weights_tensor = torch.tensor(loaded_weights).to(device)  # Move to GPU if needed

# Test or use the model for inference
test_accuracy = evaluate(loaded_weights_tensor)
print(f"Test accuracy of the best individual: {test_accuracy}")