import torch
from models.sequence.model_1d import TemporalSequenceModel
import numpy as np

device = "cpu"
temporal_structure = {1: [64, 2, False],
                    2: [64, 2, True],
                    }

kps_num = 34
frame_length = 20


class SequenceTemporalPredictor:
    def __init__(self, model_name, n_classes, struct_num, temporal_module):
        # struct_num = int(model_name.split('/')[-1].split('_')[1][6:])
        [hidden_dims, num_rnn_layers, attention] = temporal_structure[struct_num]
        self.model = TemporalSequenceModel(num_classes=n_classes, input_dim=kps_num, hidden_dims=hidden_dims,
                            num_rnn_layers=num_rnn_layers, attention=attention, temporal_module=temporal_module)
        self.model.load_state_dict(torch.load(model_name))
        self.model.to(device)
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        return data  # (1, 30, 34)

    def predict(self, data):
        data = self.get_input_data(data.reshape(frame_length, kps_num))
        output = self.model(data)
        pred = output.data.max(1, keepdim=True)[1]
        return pred


if __name__ == '__main__':
    model_path = 'exp/TCN/model.pth'
    temporal_module = "TCN"
    data_path = "data/kps_data/input2/equal/data.txt"
    inps = np.loadtxt(data_path).astype(np.float32)
    prediction = SequenceTemporalPredictor(model_path, 2, struct_num=1, temporal_module=temporal_module)
    for i in inps:
        res = prediction.predict(i)
        print(res)
