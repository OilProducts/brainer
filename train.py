import torch
# import intel_extension_for_pytorch as ipex
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import snntorch as snn
import snntorch.spikegen as spikegen

from tqdm import tqdm

import neurons
import layers
import models
import utils

import cProfile
import pstats

torch.set_float32_matmul_precision('high')

torch.set_printoptions(threshold=100000)

if torch.cuda.is_available():
    print(f'Cuda compute version: {torch.cuda.get_device_capability(0)}')
    device = torch.device("cuda")

    # if torch.cuda.get_device_capability(0)[0] < 7:
    #     print("GPU compute capability is too low, falling back to CPU")
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda")
    #     print("GPU is available and being used")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")


data_path = "~/robots/datasets/"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)


def pool_spikes(output_spikes):
    # print(f'Output spikes: {output_spikes}')
    # print(f'Output spikes shape: {output_spikes.shape}')
    pooled_spikes = output_spikes.view(-1, 10, 10).sum(1)  # Assuming output_spikes has shape [100]
    # print(f'Pooled spikes: {pooled_spikes}')
    # print(f'Pooled spikes shape: {pooled_spikes.shape}')
    return pooled_spikes


def validate(network, batch_size, num_steps, test_loader):
    total_correct = 0
    total = 0

    correct_counts = torch.zeros(10, device=device)
    total_counts = torch.zeros(10, device=device)

    with torch.no_grad():
        progress_bar = tqdm(iter(test_loader), total=len(test_loader), unit_scale=batch_size)

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(batch_size, -1)
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)
                output_spikes = network(in_spikes, labels, train=False)
                output_spike_accumulator += output_spikes

            _, predicted_classes = output_spike_accumulator.max(dim=1)
            # _, predicted_classes = output_spike_accumulator.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()

            for idx, correct in enumerate(correct_predictions):
                if correct == 1.0:
                    correct_counts[labels[idx]] += 1
                total_counts[labels[idx]] += 1

            label_strings = []
            for label in range(10):
                accuracy_label = (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
                label_string = f"{label}: {int(correct_counts[label])}/{int(total_counts[label])}"  # ({accuracy_label:.2f}%)"
                label_strings.append(label_string)
            total_correct += correct_predictions.sum().item()  # Summing over the batch

            # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)
            total += batch_size

            # Update progress bar description
            accuracy = total_correct / total * 100
            desc = f'Test Accuracy: {accuracy:.2f}% ({total_correct}/{total}) ' + ' '.join(
                label_strings)

            progress_bar.set_description(desc)

            network.reset_hidden_state()
            # correct += (predicted_class == labels).sum().item()
    return 100 * total_correct / total


def main():
    # Training param
    num_epochs = 100
    num_steps = 100
    plasticity_reward = 1
    plasticity_punish = 1
    batch_size = 64
    shrink_factor = 1

    mnist_training_loader, mnist_test_loader = (
        utils.get_mnist_dataloaders(shrink_factor=shrink_factor,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0))

    # network = models.SimpleConv(batch_size=batch_size,
    #                             a_pos=.00001,
    #                             a_neg=.00001,
    #                             plasticity_reward=plasticity_reward,
    #                             plasticity_punish=plasticity_punish,
    #                             device=device)

    network = models.SimpleLinear(batch_size=batch_size,
                                a_pos=.00005,
                                a_neg=.00005,
                                plasticity_reward=plasticity_reward,
                                plasticity_punish=plasticity_punish,
                                device=device)


    network.eval()

    # network = ipex.optimize(network)

    for epoch in range(num_epochs):
        num_correct = 0  # Number of correct predictions
        samples_seen = 0  # Total number of samples processed

        correct_counts = torch.zeros(10, device=device)
        total_counts = torch.zeros(10, device=device)
        progress_bar = tqdm(iter(mnist_training_loader), total=len(mnist_training_loader), unit_scale=batch_size)
        for inputs, labels in progress_bar:
            # Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Convert inputs to spike trains
            inputs = inputs.view(batch_size, -1)  # This will reshape the tensor to [batch_size, 784]
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            # Set label threshold to be lower
            network.layer_3.threshold_targets = utils.supervisory_threshold_modulation(10, labels)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)

                # Forward pass through the network
                output_spikes = network(in_spikes, labels)
                # print(f'output_spikes: {output_spikes}')

                # Accumulate spikes
                output_spike_accumulator += output_spikes


            # Determine the predicted class based on the accumulated spikes
            # print(f'output_spike_accumulator: {output_spike_accumulator}')
            # _, predicted_classes = output_spike_accumulator.max(dim=1)
            _, predicted_classes = pool_spikes(output_spike_accumulator).max(
                dim=1)
            #print(predicted_classes)

            correct_predictions = (predicted_classes == labels).float()
            for idx, correct in enumerate(correct_predictions):
                if correct == 1.0:
                    correct_counts[labels[idx]] += 1
                total_counts[labels[idx]] += 1

            # Construct the string for the tqdm description:
            label_strings = []
            for label in range(10):
                accuracy_label = (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
                label_string = f"{label}: {int(correct_counts[label])}/{int(total_counts[label])}"  # ({accuracy_label:.2f}%)"
                label_strings.append(label_string)

            num_correct += correct_predictions.sum().item()  # Summing over the batch
            reward_factors = correct_predictions * plasticity_reward + (1 - correct_predictions) * plasticity_punish
            for factor in reward_factors:
                network.apply_reward(factor)

            # Update statistics
            samples_seen += batch_size

            # Update progress bar description
            accuracy = num_correct / samples_seen * 100
            desc = f'Epoch: {epoch + 1}/{num_epochs} Accuracy: {accuracy:.2f}% ({num_correct}/{samples_seen}) Avg weight: {torch.mean(network.layer_3.weights)}' + ' '.join(
                label_strings)

            progress_bar.set_description(desc)
            network.reset_hidden_state()

        # After training for one epoch, validate the model
        network.layer_3.threshold_targets = torch.full((batch_size, 100), 1, dtype=torch.float, device=device)
        val_accuracy = validate(network, batch_size, num_steps, mnist_test_loader)
        # print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")


with torch.no_grad():
    # main()

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime').print_stats(50)
