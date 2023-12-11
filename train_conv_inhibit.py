import torch
import torch.nn as nn

from tqdm import tqdm

import layers
import models
import utils

num_epochs = 100
num_steps = 1000
batch_size = 1
shrink_factor = 1
device = 'cpu'

kernel = layers.create_gaussian_kernel(7, 1, sigma=1.4).unsqueeze(0).unsqueeze(0)
mnist_training_loader, mnist_test_loader = (
    utils.get_mnist_dataloaders(shrink_factor=shrink_factor,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0))

network = models.SimpleConvInhibit(batch_size=batch_size, kernel=kernel)

for epoch in range(num_epochs):
    progress_bar = tqdm(iter(mnist_training_loader), total=len(mnist_training_loader),
                        unit_scale=batch_size)

    for inputs, labels in progress_bar:
        # Move inputs and labels to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Reset spike accumulators
        output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

        for step in range(num_steps):
            output = network(inputs, labels, train=True)
            output_spike_accumulator += output



        # Update progress bar description
        progress_bar.set_description(
            f'Epoch: {epoch + 1}/{num_epochs}')
