import numpy as np
import torch
from evolutionary_base import device, model, train_loader

with torch.no_grad():
    # If you're using a tensor, you can convert the loaded weights into a tensor
    loaded_weights = np.load("best_weights.npy")  # Load weights from the file
    loaded_weights_tensor = torch.tensor(loaded_weights).to(device)  # Move to GPU if needed
    model.set_weights(loaded_weights_tensor)
    test_accuracy = model.batch_accuracy(train_loader, 25)
    print(f"Test accuracy of the best individual across all batches (esnn): {test_accuracy*100:.2f}%")

    loaded_weights = np.load("best_weights_gradient.npy")  # Load weights from the file
    loaded_weights_tensor = torch.tensor(loaded_weights).to(device)  # Move to GPU if needed
    model.set_weights(loaded_weights_tensor)
    test_accuracy = model.batch_accuracy(train_loader, 25)
    print(f"Test accuracy of the best individual across all batches (gradient snn): {test_accuracy*100:.2f}%")