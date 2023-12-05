import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSequenceModel(nn.Module):
    def __init__(self, num_classes, input_dim, temporal_module, hidden_dims, num_rnn_layers, attention=None):
        super(TemporalSequenceModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.rnn_layers = num_rnn_layers
        self.attention = attention
        self.temporal_module, self.Bi = self.tm_select(temporal_module)
        self.build_model()

    def tm_select(self, temporal_module):
        if temporal_module == 'BiLSTM':
            return nn.LSTM, True
        elif temporal_module == 'LSTM':
            return nn.LSTM, False
        elif temporal_module == 'GRU':
            return nn.GRU, False
        elif temporal_module == 'BiGRU':
            return nn.GRU, True
        else:
            raise NotImplementedError

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = self.temporal_module(input_size=self.input_dim,
                                hidden_size=self.hidden_dims,
                                num_layers=self.rnn_layers,
                                batch_first=True,
                                bidirectional=self.Bi)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.atten_fc_out = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dims, self.num_classes)
        )
        feat_dim = self.hidden_dims*2 if self.Bi else self.hidden_dims
        self.fc_out = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dims, self.num_classes))

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        #print('h',h.shape)
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=0)
        #print('lstm_hidden', lstm_hidden.shape)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        # if self.Bi:
        if isinstance(self.lstm_net, nn.GRU) and self.Bi:
            output, final_hidden_state = self.lstm_net(x)
        else:
            output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        # else:
        #     output, final_hidden_state = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        # final_hidden_state = final_hidden_state#.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        if self.attention:
            atten_out = self.attention_net_with_w(output, final_hidden_state)
            return self.atten_fc_out(atten_out)
        else:
            return self.fc_out(output[:,-1,:])


if __name__ == '__main__':
    device = "cpu"
    input_tensor = torch.randn(2, 30, 34).to(device)
    model = TemporalSequenceModel(num_classes=2,
                 input_dim=34,
                 hidden_dims=64,
                 temporal_module='BiLSTM',
                 num_rnn_layers=2,
                 attention=True).to(device)
    output = model(input_tensor)
    print(output)
