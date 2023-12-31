import torch
import torch.nn as nn

from tqdm import tqdm

import layers
import models
import utils

num_epochs = 100
num_steps = 100
batch_size = 64
shrink_factor = 1

# Determine what device to use
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

kernel = layers.create_gaussian_kernel(7, 1, sigma=1.4).unsqueeze(0).unsqueeze(0)
mnist_training_loader, mnist_test_loader = (
    utils.get_mnist_dataloaders(shrink_factor=shrink_factor,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0))
torch.no_grad()

network = models.SimpleLocallyConnected(batch_size=batch_size, kernel=kernel, device=device)
for epoch in range(num_epochs):
    progress_bar = tqdm(iter(mnist_training_loader), total=len(mnist_training_loader),
                        unit_scale=batch_size, position=0)

    for inputs, labels in progress_bar:
        # Move inputs and labels to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        input_sum = inputs.sum()

        # Reset spike accumulators
        output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

        step_progress_bar = tqdm(range(num_steps), total=num_steps, position=1)
        for step in step_progress_bar:
            output = network(inputs, labels, train=True)
            output_spike_accumulator += output
            step_progress_bar.set_description(f'Step: {step + 1}/{num_steps}, Epoch: {epoch + 1}'
                                              f'/{num_epochs}, Total spikes: {output_spike_accumulator.sum()}')

        print(output_spike_accumulator)

        # Update progress bar description
        progress_bar.set_description(
            f'Epoch: {epoch + 1}/{num_epochs}')
