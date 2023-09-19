import torch
import torch.nn as nn
import torch.nn.functional as F

class STDPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 input_height, input_width,
                 stride=1, padding=0, dilation=1,
                 batch_size=128, threshold_reset=1, threshold_decay=.95,
                 membrane_reset=.1, membrane_decay=.99, a_pos=0.005,
                 a_neg=0.005, trace_decay=.95, plasticity_reward=1,
                 plasticity_punish=1, device='cpu'):
        super(STDPConv2d, self).__init__()

        # Convolutional weights
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels,
                                                kernel_size, kernel_size).to(device) * (
                                                1 / (in_channels * kernel_size * kernel_size)))

        # Shapes
        self.output_shape = (batch_size, out_channels,
                             (1 + (input_height + 2 * padding - kernel_size - (kernel_size - 1) * (
                                         dilation - 1)) // stride),
                             (1 + (input_width + 2 * padding - kernel_size - (kernel_size - 1) * (
                                         dilation - 1)) // stride))

        self.membrane = torch.ones(self.output_shape, device=device) * membrane_reset
        self.thresholds = torch.ones(self.output_shape, device=device) * threshold_reset

        self.out_spikes = torch.zeros(self.output_shape, device=device)

        # Training parameters
        self.batch_size = batch_size

        # Neuron parameters
        self.membrane_reset = membrane_reset
        self.membrane_decay = membrane_decay
        self.threshold_reset = threshold_reset
        self.threshold_decay = threshold_decay
        self.threshold_targets = torch.full(self.output_shape, threshold_reset, dtype=torch.float, device=device)

        # Plasticity parameters
        self.plasticity_reward = plasticity_reward
        self.plasticity_punish = plasticity_punish

        # STDP parameters
        self.a_pos = a_pos
        self.a_neg = a_neg

        # Initialize traces
        self.trace_pre = torch.ones((batch_size, in_channels, input_height, input_width), device=device)
        self.trace_post = torch.ones(self.output_shape, device=device)
        self.trace_decay = trace_decay

        self.device = device

    def forward(self, in_spikes, train=True):
        # Simulate the LIF neurons
        membrane_potentials = F.conv2d(in_spikes, self.weights, stride=1, padding=1)
        self.membrane = self.membrane * self.membrane_decay + membrane_potentials
        self.out_spikes = (self.membrane > self.thresholds).float()
        self.membrane = torch.where(self.out_spikes.bool(), self.membrane_reset, self.membrane)
        self.thresholds = torch.where(self.out_spikes.bool(), self.threshold_targets,
                                      self.thresholds * self.threshold_decay)

        if train:
            # Update traces
            self.trace_pre = self.trace_pre * self.trace_decay + in_spikes
            self.trace_post = self.trace_post * self.trace_decay + self.out_spikes

            torch.clamp(self.trace_pre, 0, 1, out=self.trace_pre)
            torch.clamp(self.trace_post, 0, 1, out=self.trace_post)

            # Compute STDP weight changes using traces
            weight_changes = self.compute_stdp_with_trace(self.trace_pre, self.trace_post)
            self.weights += weight_changes.sum(dim=(0, 2, 3)).reshape(self.weights.shape)

            self.weights = torch.clamp(self.weights, -0.1, .1)

        return self.out_spikes

    def compute_stdp_with_trace(self, trace_pre, trace_post):
        # Create empty tensors to store potentiation and depression values
        potentiation = torch.zeros_like(self.weights)
        depression = torch.zeros_like(self.weights)

        # Iterate over each channel
        # Iterate over each channel
        for i in range(self.weights.size(0)):
            avg_trace_pre = torch.mean(trace_pre[:, i:i + 1], dim=0).unsqueeze(0)
            avg_trace_post = torch.mean(trace_post[:, i:i + 1], dim=0).unsqueeze(0)

            trace_post_slice = trace_post[:, i:i + 1]
            print(trace_post_slice.shape)
            print((self.a_pos * avg_trace_pre).shape)

            # Compute potentiation for the current channel
            potentiation[i] = F.conv2d(trace_post_slice, self.a_pos * avg_trace_pre, padding=1)

            # Compute depression for the current channel
            depression[i] = F.conv2d(trace_post_slice, self.a_neg * avg_trace_post, padding=1)

        # Compute the net STDP-induced weight change
        return potentiation - depression

    # def compute_stdp_with_trace(self, trace_pre, trace_post):
    #     # Note: Here, we're simply using the convolution operation for STDP.
    #     # This is a naive implementation and may need more sophistication in a real-world scenario.
    #     potentiation = F.conv2d(trace_post, self.a_pos * trace_pre, padding=1)
    #     depression = F.conv2d(trace_pre, self.a_neg * trace_post, padding=1)
    #     return potentiation - depression

    def reset_hidden_state(self):
        self.membrane = torch.ones(self.output_shape, device=self.device) * self.membrane_reset
        self.trace_pre = torch.ones((self.batch_size, self.in_channels, 32, 32), device=self.device)
        self.trace_post = torch.ones(self.output_shape, device=self.device)

    def apply_reward(self, factor):
        """Modifies the last weight update based on a reward/punishment factor.

        Args:
        - factor (float): The factor by which to scale the last weight change.
                          A value > 1 indicates a reward, while a value < 1 indicates a punishment.
        """
        avg_last_weight_change = self.last_weight_change.sum(dim=(0, 2, 3)).reshape(self.weights.shape)
        self.weights += avg_last_weight_change * (factor - 1)
