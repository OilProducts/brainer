import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add a trace that increases with frequent activation but decays slower than the membrane,
# use that as a long term potentiation trace, ust STDP for direction and short term magnitude
class STDPConvInhibit(nn.Module):
    def __init__(self, in_dimensions,  # (x, y)
                 out_dimensions,  # (x, y)
                 num_channels,
                 batch_size,
                 inhibition_kernel,
                 stride=1,
                 membrane_decay=.99,
                 inhibition_decay=.9,
                 trace_decay=.6,
                 device='cpu'):
        super().__init__()

        self.in_dim = in_dimensions
        self.out_dim = out_dimensions
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.inhibition_kernel = inhibition_kernel
        self.kernel_size = inhibition_kernel.shape[0]
        self.stride = stride
        self.membrane_decay = membrane_decay
        self.trace_decay = trace_decay
        self.device = device

        # Define the weights for the locally connected layer
        self.weights = nn.Parameter(torch.randn(
            self.out_dim[0], self.out_dim[1],
            self.num_channels, self.kernel_size, self.kernel_size,
            self.num_channels, device=device) * (1 / (self.kernel_size * self.kernel_size)))

        # self.weights = torch.rand(self.in_dim[0], self.in_dim[1], device=device) * (1 / (
        #         self.in_dim[0] * self.in_dim[1]))
        self.membrane = torch.ones(self.in_dim[0], self.in_dim[1], device=device)

        # This is the inhibition vector that will be updated after every forward pass
        self.inhibition = torch.zeros(self.in_dim[0], self.in_dim[1], device=device)

        self.trace_pre = torch.ones(self.in_dim[0], self.in_dim[1], device=device)
        self.trace_post = torch.ones(self.in_dim[0], self.in_dim[1], device=device)
        self.padding = (self.in_dim[0] - self.out_dim[0],
                        self.in_dim[0] - self.out_dim[0],
                        self.in_dim[1] - self.out_dim[1],
                        self.in_dim[1] - self.out_dim[1])

        self.pad = nn.ReflectionPad2d(self.padding)

    def forward(self, in_spikes: torch.Tensor, train=True):
        """
        The locally connected implementation of a spiking convolutional kernel.
        Args:
            in_spikes:
            train:

        Returns:

        """

        # Initialize output tensor
        batch_size = in_spikes.shape[0]
        output = torch.zeros(batch_size, self.num_channels, *self.out_dim).to(in_spikes.device)

        # Apply the locally connected operation for each position
        for i in range(self.out_dim[0]):
            for j in range(self.out_dim[1]):
                for k in range(self.num_channels):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.kernel_size
                    end_j = start_j + self.kernel_size

                    # Extract the corresponding region from input and apply weights
                    region = in_spikes[:, :, start_i:end_i, start_j:end_j]
                    weights = self.weights[i, j, :, :, :, k]
                    output[:, k, i, j] = (region * weights).sum(
                        dim=(1, 2, 3))  # + self.bias[i, j, k]

        return output

    # def forward(self, in_spikes: torch.Tensor, train=True):
    #     weighted_spikes = torch.matmul(in_spikes, self.weights)
    #     membrane = self.membrane * self.membrane_decay + weighted_spikes
    #     spikes = (membrane > (1 + self.inhibition)).float()
    #
    #     # reset the membrane where spiked
    #     membrane = membrane * (1 - spikes)
    #
    #     # update the inhibition
    #     padded_spikes = self.pad(spikes)
    #     inhibition_overlay = F.conv2d(spikes, self.inhibition_kernel)
    #
    #     # update the traces
    #     self.trace_pre = self.trace_pre * self.trace_decay + torch.mean(in_spikes, dim=0)
    #     self.trace_post = self.trace_post * self.trace_decay + torch.mean(spikes, dim=0)
    #     return spikes


def create_gaussian_kernel(kernel_size, max_value, sigma):
    # Create a grid of coordinates centered at (0,0)
    center = (kernel_size - 1) / 2
    coords = (torch.stack(
        torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')) -
              center)

    # Compute the squared Euclidean distance from the center for each element
    distances_squared = torch.sum(coords ** 2, axis=0)

    # Calculate the kernel values using a 2D Gaussian function
    kernel = max_value * torch.exp(-distances_squared / (2 * sigma ** 2))
    return kernel


class LocallyConnectedLayer(nn.Module):
    def __init__(self, input_channels, output_channels, output_size, kernel_size, stride):
        super(LocallyConnectedLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size

        # Define the weights for the locally connected layer
        self.weights = nn.Parameter(torch.randn(
            output_size[0], output_size[1],
            input_channels, kernel_size, kernel_size,
            output_channels
        ))
        self.bias = nn.Parameter(torch.randn(output_size[0], output_size[1], output_channels))

    def forward(self, x):
        # Initialize output tensor
        output = torch.zeros(x.size(0), self.output_channels, *self.output_size).to(x.device)

        # Apply the locally connected operation
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                for k in range(self.output_channels):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.kernel_size
                    end_j = start_j + self.kernel_size

                    # Element-wise multiplication and sum
                    output[:, k, i, j] = (x[:, :, start_i:end_i, start_j:end_j] *
                                          self.weights[i, j, :, :, :, k]).sum(dim=(1, 2, 3)) + \
                                         self.bias[i, j, k]

        return output
