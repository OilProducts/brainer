import torch
import torch.nn as nn
import layers


class SimpleLocallyConnected(nn.Module):
    def __init__(self, batch_size, kernel, device='cpu'):
        super(SimpleLocallyConnected, self).__init__()

        self.kernel = kernel
        self.batch_size = batch_size

        self.layer_1 = layers.STDPLocallyConnected((28, 28), (24, 24), 1, self.batch_size, kernel,
                                                   device=device)
        self.layer_2 = layers.STDPLocallyConnected((24, 24), (20, 20), 1, self.batch_size, kernel,
                                                   device=device)
        self.layer_3 = layers.STDPLocallyConnected((20, 20), (16, 16), 1, self.batch_size, kernel,
                                                   device=device)
        self.layer_4 = layers.STDPLocallyConnected((16, 16), (12, 12), 1, self.batch_size, kernel,
                                                   device=device)
        self.layer_5 = layers.STDPLocallyConnected((12, 12), (10, 10), 1, self.batch_size, kernel,
                                                   device=device)
        self.layer_6 = layers.STDPLocallyConnected((10, 10), (10, 10), 1, self.batch_size, kernel,
                                                   device=device)

    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)
        layer_3_out = self.layer_3(layer_2_out, train=train)
        layer_4_out = self.layer_4(layer_3_out, train=train)
        layer_5_out = self.layer_5(layer_4_out, train=train)
        layer_6_out = self.layer_6(layer_5_out, train=train)

        return layer_6_out.flatten(start_dim=1)
