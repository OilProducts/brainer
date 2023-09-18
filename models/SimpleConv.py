import torch.nn as nn
import layers

class SimpleConv(nn.Module):
    def __init__(self, batch_size=128, a_pos=.005, a_neg=.005, plasticity_reward=1, plasticity_punish=1, device='cuda'):
        super(SimpleConv, self).__init__()
        self.a_pos = a_pos
        self.a_neg = a_neg

        self.conv_1 = layers.STDPConv2d(1, 32, 3,
                                        a_pos=self.a_pos,
                                        a_neg=self.a_neg,
                                        plasticity_reward=plasticity_reward,
                                        plasticity_punish=plasticity_punish,
                                        device=device,
                                        batch_size=batch_size)
        self.conv_2 = layers.STDPConv2d(32, 64, 3,
                                        a_pos=self.a_pos,
                                        a_neg=self.a_neg,
                                        plasticity_reward=plasticity_reward,
                                        plasticity_punish=plasticity_punish,
                                        device=device,
                                        batch_size=batch_size)
        self.conv_3 = layers.STDPConv2d(64, 64, 3,
                                        a_pos=self.a_pos,
                                        a_neg=self.a_neg,
                                        plasticity_reward=plasticity_reward,
                                        plasticity_punish=plasticity_punish,
                                        device=device,
                                        batch_size=batch_size)

        self.fc_1 = layers.STDPLinear(64 * 4 * 4, 512,
                                        a_pos=self.a_pos,
                                        a_neg=self.a_neg,
                                        plasticity_reward=plasticity_reward,
                                        plasticity_punish=plasticity_punish,
                                        device=device,
                                        batch_size=batch_size)
        self.fc_2 = layers.STDPLinear(512, 100,
                                        a_pos=self.a_pos,
                                        a_neg=self.a_neg,
                                        plasticity_reward=plasticity_reward,
                                        plasticity_punish=plasticity_punish,
                                        device=device,
                                        batch_size=batch_size)





    def forward(self, x, labels, train=True):

        layer_1_out = self.conv_1(x, train=train)
        layer_2_out = self.conv_2(layer_1_out, train=train)
        layer_3_out = self.conv_3(layer_2_out, train=train)

        layer_3_out = layer_3_out.view(-1, 64 * 4 * 4)

        layer_4_out = self.fc_1(layer_3_out, train=train)
        layer_5_out = self.fc_2(layer_4_out, train=train)

        return layer_5_out


    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)

    def reset_hidden_state(self):

        self.conv_1.reset_hidden_state()
        self.conv_2.reset_hidden_state()
        self.conv_3.reset_hidden_state()

        self.fc_1.reset_hidden_state()
        self.fc_2.reset_hidden_state()
