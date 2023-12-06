import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: Add a trace that increases with frequent activation but decays slower than the membrane,
# use that as a long term potentiation trace, ust STDP for direction and short term magnitude
class STDPConvInhibit(nn.Module):
    def __init__(self, in_dimensions,  # (x, y)
                 out_dimensions,  # (x, y)
                 batch_size,
                 inhibition_kernel,
                 membrane_decay=.99,
                 inhibition_decay=.9,
                 trace_decay=.6,
                 device='cpu'):
        super().__init__()

        self.in_dim = in_dimensions
        self.out_dim = out_dimensions
        self.batch_size = batch_size
        self.inhibition_kernel = inhibition_kernel
        self.membrane_decay = membrane_decay
        self.trace_decay = trace_decay
        self.device = device

        self.weights = torch.rand(self.in_dim[0], self.in_dim[1], device=device) * (1 / (
                self.in_dim[0] * self.in_dim[1]))
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
        weighted_spikes = torch.matmul(in_spikes, self.weights)
        membrane = self.membrane * self.membrane_decay + weighted_spikes
        spikes = (membrane > (1 + self.inhibition)).float()

        # reset the membrane where spiked
        membrane = membrane * (1 - spikes)

        # update the inhibition
        padded_spikes = self.pad(spikes)
        inhibition_overlay = F.conv2d(spikes, self.inhibition_kernel)

        # update the traces
        self.trace_pre = self.trace_pre * self.trace_decay + torch.mean(in_spikes, dim=0)
        self.trace_post = self.trace_post * self.trace_decay + torch.mean(spikes, dim=0)
        return spikes


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
