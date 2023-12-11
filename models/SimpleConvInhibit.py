import torch
import torch.nn as nn
import layers

# a 7 x 7 inhibition kernel
torch.Tensor([[0.0526, 0.0714, 0.0909, 0.1000, 0.0909, 0.0714, 0.0526],
              [0.0714, 0.1111, 0.1667, 0.2000, 0.1667, 0.1111, 0.0714],
              [0.0909, 0.1667, 0.3333, 0.5000, 0.3333, 0.1667, 0.0909],
              [0.1000, 0.2000, 0.5000, 1.0000, 0.5000, 0.2000, 0.1000],
              [0.0909, 0.1667, 0.3333, 0.5000, 0.3333, 0.1667, 0.0909],
              [0.0714, 0.1111, 0.1667, 0.2000, 0.1667, 0.1111, 0.0714],
              [0.0526, 0.0714, 0.0909, 0.1000, 0.0909, 0.0714, 0.0526]])


class SimpleConvInhibit(nn.Module):
    def __init__(self, batch_size, kernel):
        super(SimpleConvInhibit, self).__init__()

        self.kernel = kernel
        self.batch_size = batch_size

        self.layer_1 = layers.STDPConvInhibit((28, 28), (24, 24), 1, self.batch_size, kernel)
        self.layer_2 = layers.STDPConvInhibit((24, 24), (20, 20), 1, self.batch_size, kernel)
        self.layer_3 = layers.STDPConvInhibit((20, 20), (16, 16), 1, self.batch_size, kernel)
        self.layer_4 = layers.STDPConvInhibit((16, 16), (12, 12), 1, self.batch_size, kernel)
        self.layer_5 = layers.STDPConvInhibit((12, 12), (10, 10), 1, self.batch_size, kernel)
        self.layer_6 = layers.STDPConvInhibit((10, 10), (10, 10), 1, self.batch_size, kernel)

    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)
        layer_3_out = self.layer_3(layer_2_out, train=train)
        layer_4_out = self.layer_4(layer_3_out, train=train)
        layer_5_out = self.layer_5(layer_4_out, train=train)
        layer_6_out = self.layer_6(layer_5_out, train=train)

        return layer_6_out.flatten(start_dim=1)
