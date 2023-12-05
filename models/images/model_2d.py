
from .ConvLSTM import ConvLSTM
from .ConvGRU import ConvGRU
import torch
import torch.nn as nn


ConvLSTM_structure = {1: [[128, 64, 64, 32, 32], (7, 7), False],
                      # [hidden_channels, kernel_size, attention]
                      2: [[128, 64, 64, 32, 32], (7, 7), True],
                      3: [[128, 64, 64, 64, 32], (5, 5), True],
                      4: [[128, 64, 32], (3, 3), True],
                      5: [[128, 64, 32, 32, 32], (5, 5), True],
                      6: [[128, 64, 32], (3, 3), False],
                      7: [[128, 128, 128, 64, 32], (7, 7), False],
                      8: [[128, 128, 128, 64, 32], (7, 7), True],
                      9: [[128, 128, 128, 64, 32], (3, 3), True],
                      10: [[128, 32], (3, 3), False],
                      }

ConvGRU_structure = {1: [[128, 64, 64, 32, 32], (5,5), False],
                     2: [[128, 64, 64, 32, 32], (5,5), True],
                     }


TCN_structure = {1:[[6, 6, 6, 6], 5, 2],
                 # [channel_size, kernel_size, dilation]
                 2: [[3, 4, 5, 6], 5, 4],
                 3: [[6, 6, 6, 6], 7, 2],
                 4: [[12, 12, 12, 12], 7, 2],
                 5: [[8, 16, 8, 16], 7, 2],
                 6: [[6, 6, 6, 6], 7, 4],
                 7: [[6, 6, 6, 6], 7, 8],
                 8: [[6, 6, 6, 6], 7, 1],
                 }


class ImageTemporalModels(nn.Module):
    def __init__(self, temporal_module, struct_num, input_channels, n_classes, device):
        super().__init__()
        self.device = device
        if temporal_module == "ConvLSTM":
            [hidden_channel, kernel_size, attention] = ConvLSTM_structure[struct_num]
            self.model = ConvLSTM(input_size=(int(input_channels/2), 2),
                             input_dim=1,
                             hidden_dim=hidden_channel,
                             kernel_size=kernel_size,
                             num_layers=len(hidden_channel),
                             num_classes=n_classes,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False,
                             attention=attention)
        elif temporal_module == "ConvGRU":
            [hidden_channel, kernel_size, attention] = ConvGRU_structure[struct_num]
            self.model = ConvGRU(input_size=(int(input_channels / 2), 2),
                                 input_dim=1,
                                 hidden_dim=hidden_channel,
                                 kernel_size=kernel_size,
                                 num_layers=len(hidden_channel),
                                 num_classes=n_classes,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False,
                                 attention=attention)
        else:
            raise ValueError("temporal_module must be ConvLSTM or ConvGRU")

    def forward(self, x):
        return self.model(x)
