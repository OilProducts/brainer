import torch.nn as nn
import layers

class SimpleLinear(nn.Module):
    def __init__(self, batch_size=128, a_pos=.005, a_neg=.005, plasticity_reward=1, plasticity_punish=1, device='cuda'):
        super(SimpleLinear, self).__init__()
        self.a_pos = a_pos
        self.a_neg = a_neg

        self.layer_1 = layers.STDPLinear(784, 500,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)
        self.layer_2 = layers.STDPLinear(500, 200,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)
        self.layer_3 = layers.STDPLinear(200, 100,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)

    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)
        layer_3_out = self.layer_3(layer_2_out, train=train)
        # print(f'layer_2 threshold targets: {self.layer_2.threshold_targets}')

        return layer_3_out

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)

    def reset_hidden_state(self):
        self.layer_1.reset_hidden_state()
        self.layer_2.reset_hidden_state()
        self.layer_3.reset_hidden_state()