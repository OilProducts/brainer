import torch
import torch.nn as nn


class STDPLocallyConnected(nn.Module):
    """
    This class represents a locally connected layer with STDP (Spike-Timing Dependent Plasticity).
    It inherits from the PyTorch nn.Module class.

    Attributes:
        in_dim (tuple): The dimensions of the input.
        out_dim (tuple): The dimensions of the output.
        num_channels (int): The number of channels.
        batch_size (int): The size of the batch.
        inhibition_kernel (torch.Tensor): The kernel used for inhibition.
        stride (int): The stride size for the convolution operation. Default is 1.
        membrane_decay (float): The decay rate of the membrane potential. Default is .99.
        inhibition_decay (float): The decay rate of the inhibition. Default is .9.
        trace_decay (float): The decay rate of the trace. Default is .6.
        device (str): The device to run the computations on. Default is 'cpu'.
        weights (nn.Parameter): The weights for the locally connected layer.
        membrane (torch.Tensor): The membrane potentials.
        inhibition (torch.Tensor): The inhibition values.
        trace_pre (torch.Tensor): The pre-synaptic trace values.
        trace_post (torch.Tensor): The post-synaptic trace values.
        padding (tuple): The padding for the input.
        pad (nn.ReflectionPad2d): The padding layer.
    """

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
        """
        The constructor for the STDPLocallyConnected class.

        Parameters:
            in_dimensions (tuple): The dimensions of the input.
            out_dimensions (tuple): The dimensions of the output.
            num_channels (int): The number of channels.
            batch_size (int): The size of the batch.
            inhibition_kernel (torch.Tensor): The kernel used for inhibition.
            stride (int): The stride size for the convolution operation. Default is 1.
            membrane_decay (float): The decay rate of the membrane potential. Default is .99.
            inhibition_decay (float): The decay rate of the inhibition. Default is .9.
            trace_decay (float): The decay rate of the trace. Default is .6.
            device (str): The device to run the computations on. Default is 'cpu'.
        """
        super().__init__()

        # Initialize class attributes
        self.in_dim = in_dimensions
        self.out_dim = out_dimensions
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.inhibition_kernel = inhibition_kernel
        self.kernel_size = inhibition_kernel.shape[3]
        self.stride = stride
        self.membrane_decay = membrane_decay
        self.inhibition_decay = inhibition_decay
        self.trace_decay = trace_decay
        self.device = device

        # Define the weights for the locally connected layer
        self.weights = nn.Parameter(torch.randn(
            self.out_dim[0], self.out_dim[1],
            self.num_channels, self.kernel_size, self.kernel_size,
            self.num_channels, device=device) * (1 / (self.kernel_size * self.kernel_size)))

        self.membrane = torch.ones(batch_size, num_channels, self.out_dim[0], self.out_dim[1], device=device)

        # This is the inhibition vector that will be updated after every forward pass
        self.inhibition = torch.zeros(self.in_dim[0], self.in_dim[1], device=device)

        self.trace_pre = torch.ones(self.in_dim[0], self.in_dim[1], device=device)
        self.trace_post = torch.ones(self.in_dim[0], self.in_dim[1], device=device)

        # padding needs to be sufficient to allow for inhibition kernel to be applied to all pixels
        self.padding = (3, 3, 3, 3) # hard coded, this should be dynamic based on the size of the
        # locally connected area


        self.pad = nn.ReflectionPad2d(self.padding)

    @torch.compile
    def forward(self, in_spikes: torch.Tensor, train=True):
        """
        The forward method for the STDPLocallyConnected class.

        Parameters:
            in_spikes (torch.Tensor): The input spikes.
            train (bool): Whether the model is in training mode. Default is True.

        Returns:
            output (torch.Tensor): The output of the forward pass.
        """

        # Initialize output tensor
        batch_size = in_spikes.shape[0]
        local_operation_result = torch.zeros(batch_size, self.num_channels, *self.out_dim).to(in_spikes.device)
        in_spikes = self.pad(in_spikes)
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
                    local_operation_result[:, k, i, j] = (region * weights).sum(
                        dim=(1, 2, 3))  # + self.bias[i, j, k]

        self.membrane = self.membrane * self.membrane_decay + local_operation_result
        spike = (self.membrane > 1).float()
        self.membrane = torch.where(spike.bool(), 1, self.membrane)

        return spike